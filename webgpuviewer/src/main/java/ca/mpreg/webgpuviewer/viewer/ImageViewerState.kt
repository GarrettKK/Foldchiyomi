package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.util.Log
import android.view.Surface
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.runtime.MonotonicFrameClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.Density
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDepthStencilAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.filter.FilterChain
import ca.mpreg.webgpuviewer.renderer.Downscaler
import ca.mpreg.webgpuviewer.renderer.DownscalerBox
import ca.mpreg.webgpuviewer.renderer.FrameResult
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.renderer.Rescaler
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.Upscaler
import ca.mpreg.webgpuviewer.renderer.UpscalerArtCnn
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.dispatcher
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.transition.Transition
import ca.mpreg.webgpuviewer.transition.TransitionBasic
import ca.mpreg.webgpuviewer.transition.TurnGesture
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

open class ImageViewerState(var isVertical: Boolean = false, var isReversed: Boolean = false) {
    val renderer = WebGpuRenderer()

    internal val tiles = TileRenderer { invalidate() }

    /**
     * How a high-quality tile that magnifies the page is resized - see [Rescaler]. Assigning
     * drops the tiles already generated. [UpscalerArtCnn] doubles through a network first.
     */
    var upscaler: Upscaler
        get() = tiles.upscaler
        set(value) {
            tiles.upscaler = value
        }

    /** How a high-quality tile that shrinks the page is resized. [DownscalerBox] is the only one. */
    var downscaler: Downscaler
        get() = tiles.downscaler
        set(value) {
            tiles.downscaler = value
        }

    /**
     * Post-processing over the finished frame - assign [FilterChain.filters] to run some. Wired
     * to [invalidate] here, so changing a filter's settings redraws by itself.
     */
    val filters: FilterChain = renderer.filters.also { chain ->
        chain.onInvalidate = { invalidate() }
    }

    var scope: CoroutineScope? = null

    var animationJob: Job? = null

    val width get() = renderer.width
    val height get() = renderer.height

    /** Top padding clearing the display cutout, in pixels. Set by ImageViewer while [avoidCutout]. */
    var cutoutTopPx: Float = 0f

    val viewportHeight: Float
        get() = height - if (avoidCutout) cutoutTopPx else 0f

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f
    var density: Density =
        Density(density = Resources.getSystem().displayMetrics.density, fontScale = 1f)

    /**
     * Whether a double tap zooms. Off leaves the gesture inert - the page neither zooms in nor
     * returns home - for a reader that would rather double tap did nothing at all.
     */
    var doubleTapZoomEnabled: Boolean = true

    /** Whether two fingers scale the page. Off leaves the page at whatever scale it is on. */
    var pinchZoomEnabled: Boolean = true

    /** When true, images will be positioned/scaled to avoid the display cutout. */
    var avoidCutout: Boolean by mutableStateOf(false)

    /** Shift below the cutout always, rather than only when the image would overlap it. */
    var alwaysAvoidCutout: Boolean by mutableStateOf(false)

    private var suppressPageChange = false

    var pageOffset = 0f
        set(value) {
            if (value.isNaN() || value.isInfinite()) return
            var v = value
            var pageDelta = 0

            if (!suppressPageChange) {
                // Guards a haveNext/havePrev stuck true against a pathological page provider.
                var guard = 0
                while (v >= 1f && haveNext && guard++ < 1000) {
                    pageDelta += 1
                    v -= 1f
                }
                guard = 0
                while (v <= -1f && havePrev && guard++ < 1000) {
                    pageDelta -= 1
                    v += 1f
                }
            }

            // Numeric test first: haveNext/havePrev reach fetchPage, which resolves a page and
            // may trim a cache to do it, and every animation frame sets this with |v| under 1.
            if (v > 1f && !haveNext) v = 1f
            if (v < -1f && !havePrev) v = -1f

            val settling = field != 0f && v == 0f

            field = v

            if (pageDelta != 0) {
                dropBubble()
                onPageChange?.runCatching { invoke(if (isReversed) -pageDelta else pageDelta) }
            }

            // Rotate rather than invalidate: onPageChange already moved what backs getPage, so a
            // slot often holds a valid render of the new current page.
            if (settling) {
                val current = getPage(0)
                if (current != null) Transition.rotateCacheOnPageChange(current) else Transition.invalidateCache()
            }
        }

    private fun setPageOffsetDirect(value: Float) {
        suppressPageChange = true
        pageOffset = value
        suppressPageChange = false
    }

    private var turn = 0

    fun animatePageTurn(direction: Int) {
        animationJob?.cancel()
        // cancel() doesn't wait: a turn started while the last one unwinds would otherwise have
        // its own transitionFromPage - set just before this call - cleared by that finally.
        val id = ++turn
        dropBubble()
        turnHeld = false
        turnAuto = true
        animationJob = scope?.launch {
            setPageOffsetDirect(direction.toFloat())
            invalidate()
            try {
                Animatable(direction.toFloat()).animateTo(
                    0f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
                ) {
                    setPageOffsetDirect(value)
                    invalidate()
                }
            } finally {
                // Cancelled or not: getPage provides the right page from here.
                if (turn == id) transitionFromPage = null
            }
            // Outside the try - a cancelled turn leaves the offset to its successor.
            setPageOffsetDirect(0f)
            invalidate()
        }
    }

    val havePrev get() = getPage(if (isReversed) 1 else -1) != null
    val haveNext get() = getPage(if (isReversed) -1 else 1) != null

    var fetchPage: ((Int) -> ImagePage?)? = null

    /** Once per outage, on the render dispatcher - see [FrameResult.Unavailable]. */
    var onRenderUnavailable: ((String?) -> Unit)? = null

    var onPageChange: ((Int) -> Unit)? = null
    var onTap: ((Offset) -> Unit)? = null
    var onLongTap: ((Offset) -> Unit)? = null

    /** Override for the "from" page during far navigation animation */
    var transitionFromPage: ImagePage? = null

    /** Whether a double tap on a speech bubble opens it enlarged - see [showBubble]. */
    var bubbleZoomEnabled: Boolean = true

    /** The speech bubble open over the page, if any - see [BubbleZoom]. */
    @Volatile
    internal var bubble: BubbleOverlay? = null
        private set

    private var bubbleJob: Job? = null

    /** True while a bubble is open, or closing: the next touch only closes it. */
    val bubbleShown: Boolean get() = bubble != null

    /**
     * Opens the speech bubble under [tap] (normalised screen coordinates) enlarged over the page,
     * as Google Play Books does. False where there is none - the page isn't at rest, or nothing
     * there looks like a bubble - so the caller can zoom as usual instead.
     */
    suspend fun showBubble(tap: Offset): Boolean {
        if (!bubbleZoomEnabled || pageOffset != 0f || bubble != null) return false
        val page = getPage(0) as? ImagePage.ImageSingle ?: return false
        val w = width
        val h = height
        if (w <= 0 || h <= 0) return false
        val found = withContext(Dispatchers.Default) {
            runCatching { BubbleZoom.detect(page, w, h, tap.x, tap.y) }
                .onFailure { Log.w("ImageViewerState", "Bubble detection failed", it) }
                .getOrNull()
        } ?: return false
        // The page may have turned while it looked.
        if (getPage(0) !== page || pageOffset != 0f || bubble != null) return false
        bubble = found
        animateBubble(found, 1f)
        return true
    }

    /** Closes the open bubble back into the page. */
    fun dismissBubble() {
        val current = bubble ?: return
        animateBubble(current, 0f) {
            if (bubble === current) {
                bubble = null
                releaseBubble(current)
            }
        }
    }

    /** Drops the open bubble at once, for a page that is going away. */
    private fun dropBubble() {
        val current = bubble ?: return
        bubbleJob?.cancel()
        bubble = null
        releaseBubble(current)
        invalidate()
    }

    private fun animateBubble(target: BubbleOverlay, to: Float, then: () -> Unit = {}) {
        bubbleJob?.cancel()
        bubbleJob = scope?.launch {
            Animatable(target.progress).animateTo(
                to, animationSpec = spring(
                    dampingRatio = if (to > 0f) 0.78f else 1f,
                    stiffness = Spring.StiffnessMediumLow,
                    visibilityThreshold = 0.002f,
                )
            ) {
                target.progress = value
                invalidate()
            }
            then()
            invalidate()
        }
    }

    /** Its GPU resources go on the render thread, after any frame already drawing with them. */
    private fun releaseBubble(overlay: BubbleOverlay) {
        post {
            overlay.release()
            BubbleZoom.releaseTarget()
        }
    }

    // One instance for this state's lifetime, so [cleanup] can tell its own from a successor's.
    private val invalidateCallback: () -> Unit = { invalidate() }

    /**
     * The page [index] steps from current. [isReversed] plays no part here - [fetchPage] and
     * [onPageChange] are what decide what a step actually means.
     */
    fun getPage(index: Int): ImagePage? {
        return fetchPage?.invoke(index)?.also { it.attach(this, scope, invalidateCallback) }
    }

    /**
     * What the last [captureRenderState] drew with, and so what [ImagePage.isOnScreen] answers
     * from. Off the last frame, not re-fetched: changing which pages show draws its own frame.
     */
    @Volatile
    protected var onScreenPages: List<ImagePage> = emptyList()

    internal fun isOnScreen(page: ImagePage): Boolean = onScreenPages.any { it.covers(page) }

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.renderer.init(scope, surface, width, height)
        this.scope = scope
        // The paged viewer only - the continuous one has no bubble zoom.
        if (bubbleZoomEnabled && this !is ImageViewerContinuousState) post { BubbleZoom.prewarm() }
        Hdr.requestFrame = invalidateCallback

        // On [dispatcher] and drained under the lock, as [post] itself would have run them.
        scope.launch(dispatcher) {
            val pending = synchronized(this@ImageViewerState) {
                _postInit.toList().also { _postInit.clear() }
            }
            pending.forEach { it() }
        }
    }

    @Synchronized
    fun resize(width: Int, height: Int) {
        renderer.resize(width, height)
        invalidate()
    }

    var firstPos = Offset.Zero
    var currentPos = Offset.Zero

    /** A finger is on the page turn right now - [currentPos] is where it is. */
    @Volatile
    var turnHeld = false

    /** [pageOffset] when the finger last let go of a turn - where its settle started from. */
    @Volatile
    var turnReleaseOffset = 0f

    /** The turn in flight came from [animatePageTurn] (a tap or key), not a drag. */
    @Volatile
    var turnAuto = false

    var transition: Transition = if (isVertical) TransitionBasic.Vertical else TransitionBasic

    // Anything changed since the last frame, however many invalidates said so.
    private val dirty = AtomicBoolean(true)

    @Volatile
    private var reportedUnavailable = false

    @Volatile
    private var retries = 0

    // Wakes [collect] when there is nothing to draw. Buffered, so no send races [dirty]'s check.
    private val renderWake = Channel<Unit>(Channel.CONFLATED)

    fun invalidate() {
        dirty.set(true)
        renderWake.trySend(Unit)
    }

    /**
     * Draw at most one frame per display frame, for as long as anything is [dirty].
     *
     * A frame loop, not a pass per invalidate: a fling step, a fade step, a decode and the tile
     * worker all land in one frame. [renderWake] carries no count, so a draw clearing [dirty]
     * doesn't leave that frame's remaining invalidates to wait out a frame each.
     *
     * Nothing may be awaited between the frame wait and the capture, the draw included: this
     * registers as an awaiter mid-draw to land in the next frame's batch, behind the animation,
     * and registering after that batch has gone out costs a frame every frame.
     */
    suspend fun collect() = coroutineScope {
        val frameClock = currentCoroutineContext()[MonotonicFrameClock]
        var drawing: Job? = null
        while (true) {
            // Paced either way: called without a clock, the loop would spin on whatever is dirty.
            if (frameClock != null) frameClock.withFrameNanos { } else delay(8)
            // Still drawing: leave [dirty] set for the next frame.
            if (drawing?.isActive == true) continue
            // Anything invalidating from here belongs to the next frame.
            if (!dirty.getAndSet(false)) {
                renderWake.receive()
                continue
            }
            // Drop the wake this frame's invalidate left, or going idle costs a spurious one.
            renderWake.tryReceive()
            // Captured here, drawn on the GPU thread below - see this function's doc.
            val snapshot = captureRenderState() ?: continue
            drawing = launch(dispatcher) {
                when (renderer.render { encoder, texture ->
                    renderSnapshot(encoder, texture, snapshot)
                }) {
                    FrameResult.Drawn -> {
                        reportedUnavailable = false
                        retries = 0
                    }

                    // Past the cap, park until something else invalidates.
                    FrameResult.Retry -> if (++retries <= MAX_RETRIES) invalidate()

                    // No invalidate: leaving [dirty] clear parks the loop on [renderWake] rather
                    // than spinning the frame clock, and a later resize still wakes it.
                    FrameResult.Unavailable -> if (!reportedUnavailable) {
                        reportedUnavailable = true
                        val reason = WebGpuRenderer.unavailableReason ?: "no render surface"
                        Log.e("ImageViewerState", "Nothing can be drawn: $reason")
                        onRenderUnavailable?.runCatching { invoke(reason) }
                    }
                }
            }
        }
    }

    protected open fun captureRenderState(): Any? {
        val currentPage = getPage(0) ?: run {
            onScreenPages = emptyList()
            return EmptySnapshot
        }
        val offset = pageOffset
        val adjacentPage = when {
            offset == 0f -> null
            // Use override if set (for far navigation)
            transitionFromPage != null -> transitionFromPage
            offset > 0f -> getPage(if (isReversed) -1 else 1)
            else -> getPage(if (isReversed) 1 else -1)
        }
        // Only prewarms while at rest - see renderSnapshot - so a turn in flight needs no lookup.
        val nextPage = if (offset == 0f) getPage(1) else null
        onScreenPages = listOfNotNull(currentPage, adjacentPage)
        return RenderSnapshot(
            currentPage, adjacentPage, nextPage, offset, transition, firstPos, currentPos,
            TurnGesture(turnHeld, turnAuto, turnReleaseOffset),
            bubble?.takeIf { offset == 0f && it.page === currentPage },
            bubble?.progress ?: 0f,
        )
    }

    private object EmptySnapshot

    private companion object {
        const val MAX_RETRIES = 10
    }

    private class RenderSnapshot(
        val currentPage: ImagePage,
        val adjacentPage: ImagePage?,
        val nextPage: ImagePage?,
        val offset: Float,
        val transition: Transition,
        val firstPos: Offset,
        val currentPos: Offset,
        val gesture: TurnGesture,
        val bubble: BubbleOverlay?,
        val bubbleProgress: Float,
    )

    /**
     * Run [block] against a render pass over [texture], ending it afterwards either way - a draw
     * that throws still closes the pass, and [WebGpuRenderer.render] makes it a dropped frame.
     * Owned here because only the whole frame's contents decide where a pass starts and ends.
     *
     * Always clears: `getCurrentTexture` rotates buffers, so loading would show stale content.
     */
    protected fun renderPass(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        clearColor: Int = 0,
        block: (GPURenderPassEncoder) -> Unit
    ) {
        val targetView = texture.createView()
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = targetView,
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(
                            ((clearColor shr 16) and 0xFF) / 255.0,
                            ((clearColor shr 8) and 0xFF) / 255.0,
                            (clearColor and 0xFF) / 255.0,
                            ((clearColor ushr 24) and 0xFF) / 255.0,
                        )
                    )
                ),
                // Fresh each frame: the tile blit marks what it covered so masked draws skip
                // re-shading it. Discarded afterward - nothing reads it across frames.
                depthStencilAttachment = GPURenderPassDepthStencilAttachment(
                    view = tiles.stencilViewFor(texture),
                    stencilLoadOp = LoadOp.Clear,
                    stencilStoreOp = StoreOp.Discard,
                    stencilClearValue = 0,
                )
            )
        )
        try {
            block(pass)
        } finally {
            pass.endAndRelease(targetView)
        }
    }

    protected open suspend fun renderSnapshot(
        encoder: GPUCommandEncoder, texture: GPUTexture, snapshot: Any
    ) {
        if (snapshot === EmptySnapshot) {
            Draw.clear(encoder, texture, 0)
            return
        }
        val s = snapshot as RenderSnapshot
        tiles.newFrame()
        val page = s.currentPage

        if (s.adjacentPage != null && s.offset != 0f) {
            s.transition.render(
                page, s.adjacentPage, encoder, texture, s.offset, s.firstPos, s.currentPos, tiles,
                s.gesture,
            )
            return
        }

        val covered = page.drawLive(encoder, texture, tiles)
        s.bubble?.let { BubbleZoom.draw(encoder, texture, it, s.bubbleProgress) }

        // Once the current page's tiles settle, prewarm the next page's, so a transition into it
        // starts mostly sharp. Gated on atHome: the tile cache is keyed by (x, y, scale).
        if (covered && page is ImagePage.ImageSingle && page.atHome) {
            val next = s.nextPage as? ImagePage.ImageSingle
            if (next != null && next.highQuality && !next.isAnimated && next.atHome) {
                tiles.prewarm(next, texture)
            }
        }
    }

    private val _postInit = mutableListOf<(suspend () -> Unit)>()

    @Synchronized
    fun post(fn: suspend () -> Unit) {
        val activeScope = scope
        if (activeScope?.isActive == true) {
            activeScope.launch(dispatcher) {
                fn()
            }
        } else {
            _postInit.add(fn)
        }
    }

    fun cleanup() {
        animationJob?.cancel()
        bubbleJob?.cancel()
        bubble?.let { overlay ->
            bubble = null
            overlay.release()
        }
        onScreenPages = emptyList()
        // Held by an object that outlives this state, so it has to be dropped by hand - but only
        // if it is still ours: a replacement viewer inits before the one it replaces cleans up.
        if (Hdr.requestFrame === invalidateCallback) Hdr.requestFrame = null
        Transition.releasePagesOf(this)
        tiles.cleanup()
        renderer.cleanup()
    }
}