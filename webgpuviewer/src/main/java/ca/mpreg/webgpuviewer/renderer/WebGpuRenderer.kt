package ca.mpreg.webgpuviewer.renderer

import android.annotation.SuppressLint
import android.util.Log
import android.view.Surface
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.FeatureLevel
import androidx.webgpu.FeatureName
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPUInstanceDescriptor
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPUSurface
import androidx.webgpu.GPUSurfaceConfiguration
import androidx.webgpu.GPUSurfaceDescriptor
import androidx.webgpu.GPUSurfaceSourceAndroidNativeWindow
import androidx.webgpu.GPUTexture
import androidx.webgpu.SurfaceGetCurrentTextureStatus
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.initLibrary
import ca.mpreg.webgpuviewer.filter.FilterChain
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.mutex
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.withContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors

enum class FrameResult {
    Drawn,
    Retry,
    Unavailable,
}

class WebGpuRenderer {
    companion object {
        const val MIN_SURFACE_DIMENSION = 8

        const val MAX_SURFACE_DIMENSION = 8192

        lateinit var instance: GPUInstance
        lateinit var adapter: GPUAdapter
        lateinit var device: GPUDevice
        private val mutex = Mutex()

        var offsetX: Float = 0f
        var offsetY: Float = 0f

        /** Set if [init]'s adapter/device request throws, so a callsite can fail soft instead. */
        @Volatile
        var initError: Throwable? = null
            private set

        @Volatile
        private var deviceLost = false

        /**
         * Called once, on whichever thread the driver reports it from, when the device is lost.
         * Nothing renders after that for the life of the process, so a host should move to a
         * viewer that does not need WebGPU.
         */
        @Volatile
        var onDeviceLost: (() -> Unit)? = null

        val isAvailable: Boolean
            get() = initError == null && !deviceLost &&
                    ::instance.isInitialized && ::adapter.isInitialized && ::device.isInitialized

        val unavailableReason: String?
            get() = when {
                initError != null -> "WebGPU failed to initialize: ${initError?.message}"
                deviceLost -> "WebGPU device lost"
                !::instance.isInitialized || !::adapter.isInitialized ||
                        !::device.isInitialized -> "WebGPU never initialized"

                else -> null
            }

        fun requireAvailable() {
            check(isAvailable) {
                "WebGPU not available" + (initError?.let { ": ${it.message}" }
                    ?: if (deviceLost) ": device lost" else "")
            }
        }

        val dispatcher = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "WebGPU-Render-Thread")
        }.asCoroutineDispatcher()

        // Frame time profiling
        var profilingEnabled = false

        /** Guards every field below - written on the render thread, read from wherever polls them. */
        private val profilingLock = Any()
        private var frameCount = 0L
        private var totalFrameTimeNs = 0L
        private var minFrameTimeNs = Long.MAX_VALUE
        private var maxFrameTimeNs = 0L
        private var lastFrameTimeNs = 0L
        private val recentFrameTimes = LongArray(60)
        private var recentFrameIndex = 0

        val lastFrameTimeMs: Float get() = synchronized(profilingLock) { lastFrameTimeNs / 1_000_000f }
        val avgFrameTimeMs: Float
            get() = synchronized(profilingLock) {
                if (frameCount > 0) totalFrameTimeNs / frameCount / 1_000_000f else 0f
            }
        val minFrameTimeMs: Float
            get() = synchronized(profilingLock) {
                if (minFrameTimeNs == Long.MAX_VALUE) 0f else minFrameTimeNs / 1_000_000f
            }
        val maxFrameTimeMs: Float get() = synchronized(profilingLock) { maxFrameTimeNs / 1_000_000f }
        val recentAvgFrameTimeMs: Float
            get() = synchronized(profilingLock) {
                val count = minOf(frameCount.toInt(), 60)
                if (count == 0) return@synchronized 0f
                var sum = 0L
                for (i in 0 until count) {
                    sum += recentFrameTimes[i]
                }
                sum.toFloat() / count / 1_000_000f
            }
        val estimatedFps: Float
            get() = synchronized(profilingLock) {
                if (lastFrameTimeNs > 0) 1_000_000_000f / lastFrameTimeNs else 0f
            }

        fun resetProfiling() = synchronized(profilingLock) {
            frameCount = 0
            totalFrameTimeNs = 0
            minFrameTimeNs = Long.MAX_VALUE
            maxFrameTimeNs = 0
            lastFrameTimeNs = 0
            recentFrameIndex = 0
            recentFrameTimes.fill(0)
        }

        internal fun recordFrameTime(timeNs: Long) {
            if (!profilingEnabled) return
            synchronized(profilingLock) {
                frameCount++
                totalFrameTimeNs += timeNs
                lastFrameTimeNs = timeNs
                if (timeNs < minFrameTimeNs) minFrameTimeNs = timeNs
                if (timeNs > maxFrameTimeNs) maxFrameTimeNs = timeNs
                recentFrameTimes[recentFrameIndex] = timeNs
                recentFrameIndex = (recentFrameIndex + 1) % 60
            }
        }

        init {
            runBlocking {
                try {
                    initLibrary()

                    instance = createInstance(GPUInstanceDescriptor())

                    adapter =
                        instance.requestAdapter(GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility))

                    val requiredFeatures =
                        if (adapter.hasFeature(FeatureName.TimestampQuery)) {
                            intArrayOf(FeatureName.TimestampQuery)
                        } else {
                            intArrayOf()
                        }

                    device = adapter.requestDevice(
                        GPUDeviceDescriptor(
                            deviceLostCallback = DeviceLostCallback { lost, reason, message ->
                                val first = !deviceLost
                                deviceLost = true
                                Log.e(
                                    "WebGpuRenderer",
                                    "WebGPU device lost reason=$reason: $message device=$lost"
                                )
                                if (first) onDeviceLost?.invoke()
                            },
                            deviceLostCallbackExecutor = Executor(Runnable::run),
                            uncapturedErrorCallback = UncapturedErrorCallback { _, type, message ->
                                Log.e(
                                    "WebGpuRenderer",
                                    "Uncaptured WebGPU error type=$type: $message"
                                )
                            },
                            uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                            requiredFeatures = @SuppressLint("WrongConstant") requiredFeatures,
                        )
                    )
                } catch (e: Throwable) {
                    // Fails soft via initError - a driver init failure shouldn't poison the whole class.
                    Log.e("WebGpuRenderer", "Failed to initialize WebGPU", e)
                    initError = e
                }
            }
        }

        @JvmStatic
        suspend fun <R> withContext(block: suspend CoroutineScope.(GPUDevice) -> R): R {
            requireAvailable()
            return withContext(dispatcher) {
                mutex.withLock {
                    block(this, device)
                }
            }
        }

        /**
         * Run [block] on the GPU thread *without* taking the render mutex.
         *
         * For long resource work that yields as it goes. [withContext] would defeat that: a
         * [render] call woken by the yield would just block on the mutex and hand the thread
         * straight back, so the work would still run to completion before the next frame. Without
         * the mutex the yield actually lets a frame through.
         *
         * Only safe for work that either owns its resources outright (an image still being built
         * and not yet reachable from a page) or that cannot be observed mid-flight. Anything that
         * has to appear atomically to the renderer belongs in [withContext].
         */
        @JvmStatic
        suspend fun <R> onDispatcher(block: suspend CoroutineScope.(GPUDevice) -> R): R {
            requireAvailable()
            return withContext(dispatcher) {
                block(this, device)
            }
        }
    }

    @Volatile
    private var surface: GPUSurface? = null

    /** The window reference [windowFromSurface] took for [surface], released in [cleanup]; see [NativeWindow]. */
    private var window = 0L

    /**
     * The format the swapchain is currently configured as, so [render] can notice when
     * [Hdr.frameFormat] has moved and rebuild it.
     */
    private var configuredFormat: Int = 0

    /**
     * Post-processing over the finished frame - see [FilterChain]. Empty by default, in which
     * case [render] hands the swapchain texture straight to its caller as it always did.
     */
    val filters = FilterChain()

    var width: Int = 0
    var height: Int = 0

    private var scope: CoroutineScope? = null

    private var pendingSurface: Surface? = null

    @Volatile
    private var resizePending = false

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        if (!isAvailable) {
            Log.w("WebGpuRenderer", "init called but WebGPU not available: $unavailableReason")
            return
        }
        this.scope = scope
        this.pendingSurface = surface
        this.width = width.coerceIn(0, MAX_SURFACE_DIMENSION)
        this.height = height.coerceIn(0, MAX_SURFACE_DIMENSION)
        createSurface()
    }

    @Synchronized
    fun resize(width: Int, height: Int) {
        if (!isAvailable) return
        val w = width.coerceIn(0, MAX_SURFACE_DIMENSION)
        val h = height.coerceIn(0, MAX_SURFACE_DIMENSION)
        if (w == this.width && h == this.height && surface != null) return
        this.width = w
        this.height = h

        if (surface == null) {
            createSurface()
        } else {
            // Not here: [render] holds [mutex] across a suspending draw, so blocking on it from
            // this thread deadlocks against the frame that owns it.
            resizePending = true
        }
    }

    /** Dispatches only when not already on the render thread, which would deadlock. */
    private fun onRenderThread(block: () -> Unit) {
        if (Thread.currentThread().name == "WebGPU-Render-Thread") {
            block()
        } else {
            runBlocking(dispatcher) { block() }
        }
    }

    private fun createSurface() {
        if (surface != null) return
        val pending = pendingSurface ?: return

        // A transient layout pass can hand over a near-zero size; [resize] picks it up later.
        if (width < MIN_SURFACE_DIMENSION || height < MIN_SURFACE_DIMENSION) {
            Log.w("WebGpuRenderer", "surface deferred at undersized ${width}x${height}")
            return
        }

        val initSurface = initSurface@{
            // A cleanup queued first let [pending] go; a surface from it would leak.
            if (surface != null || pendingSurface !== pending) return@initSurface
            val created = instance.createSurface(
                GPUSurfaceDescriptor(
                    surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                        windowFromSurface(pending).also { acquired -> window = acquired }
                    )
                )
            )
            try {
                created.apply {
                    // Before the first latch, so the swapchain is configured knowing
                    // whether float is even available.
                    Hdr.resolve(this, adapter)
                    // No images exist yet, so a previous session's stranded retains can go
                    // without discarding a live one.
                    Hdr.resetContent()
                    Hdr.latchFrameFormat()
                    configure(
                        GPUSurfaceConfiguration(
                            device,
                            this@WebGpuRenderer.width,
                            this@WebGpuRenderer.height,
                            Hdr.frameFormat,
                            TextureUsage.RenderAttachment
                        )
                    )
                    this@WebGpuRenderer.configuredFormat = Hdr.frameFormat
                }
            } catch (e: Throwable) {
                // Unassigned, so cleanup would never free either.
                created.close()
                if (window != 0L) {
                    NativeWindow.release(window)
                    window = 0L
                }
                throw e
            }
            this@WebGpuRenderer.surface = created
        }

        onRenderThread(initSurface)
    }

    suspend fun render(fn: suspend (GPUCommandEncoder, GPUTexture) -> Unit): FrameResult {
        if (!isAvailable) return FrameResult.Unavailable
        val startTime = if (profilingEnabled) System.nanoTime() else 0L

        mutex.withLock {
            // Only [init]/[resize] can build one, so a redraw alone accomplishes nothing.
            val surface = surface ?: return FrameResult.Unavailable
            // A stale surface from before a resize shrunk below this would otherwise still reach
            // getCurrentTexture - [createSurface]/[reconfigure] refuse to configure one this small.
            if (width < MIN_SURFACE_DIMENSION || height < MIN_SURFACE_DIMENSION) {
                return FrameResult.Unavailable
            }

            // Between frames and under the lock - the only place a rebuild is safe.
            if (resizePending) {
                resizePending = false
                reconfigure(surface)
            }

            // An HDR image arriving, or the last one leaving, changes what the swapchain should
            // be. Here rather than at the decode: the format can only change between frames, and
            // this is the one place guaranteed to be between them - hence the latch too.
            try {
                Hdr.latchFrameFormat()
                if (Hdr.frameFormat != configuredFormat) {
                    reconfigure(surface)
                    Hdr.syncPresentation()
                } else if (Hdr.consumePresentationDirty()) {
                    // Same format, but a brighter image arrived (or the brightest was freed), so
                    // the headroom asked of the display has moved.
                    Hdr.syncPresentation()
                }
            } catch (e: Exception) {
                // Escaping would end the frame loop.
                Log.e("WebGpuRenderer", "HDR presentation update failed", e)
            }

            val current = try {
                surface.getCurrentTexture()
            } catch (e: Exception) {
                Log.w("WebGpuRenderer", "Failed to get current texture", e)
                return FrameResult.Retry
            }

            // A non-success status hands back a null texture, and every GPUTexture read goes
            // straight through its handle - so one segfaults rather than throws. Outdated means
            // a window resize under a frame already in flight.
            val texture = current.texture
            if (!current.status.isSurfaceSuccess() || texture.handle == 0L) {
                Log.w(
                    "WebGpuRenderer",
                    "No surface texture: ${SurfaceGetCurrentTextureStatus.toString(current.status)}"
                )
                // Lost needs a new surface: park until one arrives.
                if (texture.handle != 0L) texture.close()
                if (current.status == SurfaceGetCurrentTextureStatus.Lost) {
                    return FrameResult.Unavailable
                }
                reconfigure(surface)
                return FrameResult.Retry
            }

            var encoder: GPUCommandEncoder? = null
            try {
                encoder = device.createCommandEncoder()
                // Draws into an offscreen texture when filters are enabled; endFrame runs them
                // over it and lands the result on the swapchain.
                fn(encoder, filters.beginFrame(texture))
                filters.endFrame(encoder, texture)
                device.queue.submitAndRelease(encoder)
                encoder = null
                surface.present()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e("WebGpuRenderer", "Render error", e)
                // Don't rethrow - allow the app to continue rendering next frame
            } catch (e: OutOfMemoryError) {
                Log.e("WebGpuRenderer", "Out of memory rendering", e)
            } finally {
                // Unsubmitted if the frame threw.
                encoder?.close()
                // getCurrentTexture hands out a new reference every frame - see [endAndRelease].
                texture.close()
            }
        }

        if (profilingEnabled) {
            val frameTime = System.nanoTime() - startTime
            recordFrameTime(frameTime)
            Log.d(
                "WebGpuRenderer", "Frame: %.2fms | Avg: %.2fms | FPS: %.1f".format(
                    frameTime / 1_000_000f, recentAvgFrameTimeMs, estimatedFps
                )
            )
        }

        return FrameResult.Drawn
    }

    /** Rebuild the swapchain at the size [init] was last given. Must hold [mutex]. */
    private fun reconfigure(surface: GPUSurface) {
        if (width < MIN_SURFACE_DIMENSION || height < MIN_SURFACE_DIMENSION) return
        try {
            surface.configure(
                GPUSurfaceConfiguration(
                    device, width, height, Hdr.frameFormat, TextureUsage.RenderAttachment
                )
            )
            configuredFormat = Hdr.frameFormat
        } catch (e: Exception) {
            Log.w("WebGpuRenderer", "Failed to reconfigure surface", e)
        }
    }

    fun cleanup() {
        // Check if already on dispatcher thread to avoid deadlock
        val isOnDispatcherThread = Thread.currentThread().name == "WebGPU-Render-Thread"

        val doCleanup: suspend () -> Unit = {
            mutex.withLock {
                // Once the device is lost there is nothing to free, and calling into its objects
                // crashed in native code.
                if (!deviceLost) {
                    filters.cleanup()
                    surface?.close()
                }
                surface = null
                // After the surface: Vulkan holds its own reference while its surface lives, so ours is the last.
                if (window != 0L) {
                    NativeWindow.release(window)
                    window = 0L
                }
                pendingSurface = null
                resizePending = false
            }
        }

        if (isOnDispatcherThread && mutex.isLocked) {
            // The mutex holder is on this thread: blocking would deadlock.
            CoroutineScope(dispatcher).launch { doCleanup() }
        } else if (isOnDispatcherThread) {
            // Already on dispatcher, run synchronously
            runBlocking {
                doCleanup()
            }
        } else {
            runBlocking(dispatcher) {
                doCleanup()
            }
        }
    }
}

/** Suboptimal still draws - it only asks to be reconfigured eventually. */
private fun Int.isSurfaceSuccess(): Boolean =
    this == SurfaceGetCurrentTextureStatus.SuccessOptimal ||
            this == SurfaceGetCurrentTextureStatus.SuccessSuboptimal
