package ca.mpreg.webgpuviewer.renderer

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.pm.ActivityInfo
import android.hardware.DataSpace
import android.os.Build
import android.os.Looper
import android.util.Log
import android.view.Display
import android.view.SurfaceControl
import android.view.View
import android.view.Window
import androidx.annotation.RequiresApi
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUSurface
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.renderer.Hdr.MAX_HEADROOM_RATIO
import ca.mpreg.webgpuviewer.renderer.Hdr.RESOLVE_TIMEOUT_MS
import ca.mpreg.webgpuviewer.renderer.Hdr.applyExtendedRange
import ca.mpreg.webgpuviewer.renderer.Hdr.attachColorModeHost
import ca.mpreg.webgpuviewer.renderer.Hdr.attachDisplay
import ca.mpreg.webgpuviewer.renderer.Hdr.colorModeApplied
import ca.mpreg.webgpuviewer.renderer.Hdr.desiredHeadroomRatio
import ca.mpreg.webgpuviewer.renderer.Hdr.displayPeakRatio
import ca.mpreg.webgpuviewer.renderer.Hdr.displaySupported
import ca.mpreg.webgpuviewer.renderer.Hdr.hdrLock
import ca.mpreg.webgpuviewer.renderer.Hdr.headroomRatioOverride
import ca.mpreg.webgpuviewer.renderer.Hdr.latchFrameFormat
import ca.mpreg.webgpuviewer.renderer.Hdr.liveHdrClaims
import ca.mpreg.webgpuviewer.renderer.Hdr.liveHeadroomRatio
import ca.mpreg.webgpuviewer.renderer.Hdr.maxPeakValue
import ca.mpreg.webgpuviewer.renderer.Hdr.peakWeight
import ca.mpreg.webgpuviewer.renderer.Hdr.presentFormat
import ca.mpreg.webgpuviewer.renderer.Hdr.presentPeak
import ca.mpreg.webgpuviewer.renderer.Hdr.requestHdrColorMode
import ca.mpreg.webgpuviewer.renderer.Hdr.resetContent
import ca.mpreg.webgpuviewer.renderer.Hdr.retainHdrImage
import ca.mpreg.webgpuviewer.renderer.Hdr.supportedByDevice
import ca.mpreg.webgpuviewer.renderer.Hdr.syncColorMode
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeoutOrNull
import java.lang.ref.WeakReference
import java.util.function.Consumer
import kotlin.math.log2
import kotlin.math.pow

/**
 * Whether the viewer is presenting HDR, and the texture format that follows from it.
 *
 * Two independent questions, and conflating them is the mistake to avoid. [supportedByDevice] is
 * fixed once a surface exists and is what the *decoder* keys off, because tone mapping at decode
 * is irreversible and a decoded page outlives whatever is on screen. [presentFormat] follows the
 * content, so an SDR-only stretch of a session costs what it always did.
 *
 * Both formats hold the same values - sRGB-encoded, 1.0 at white - so float merely stops clamping
 * and nothing downstream needs a linear-light variant. That encoding is Android's extended sRGB
 * and is what `ImageDecoder.PixelFormat.RGBA16F` hands over.
 *
 * A float swapchain alone is still a wider SDR surface: [applyExtendedRange] and the window's
 * colour mode ([attachColorModeHost]) are both required, and the first needs a `SurfaceControl`,
 * which only a `SurfaceView` has and only from API 34.
 */
object Hdr {
    private const val TAG = "Hdr"

    private const val RESOLVE_TIMEOUT_MS = 2000L

    /**
     * Necessary but not sufficient: a driver hands out a float swapchain on a panel with no HDR
     * at all, where it costs double the bandwidth to show the same picture. Hence
     * [displaySupported], which is independent of it.
     */
    @Volatile
    private var surfaceSupported: Boolean? = null

    @Volatile
    private var displaySupported: Boolean? = null

    private val resolved = CompletableDeferred<Boolean>()

    val supportedByDevice: Boolean
        get() = surfaceSupported == true && displaySupported == true

    /**
     * From `getHighestHdrSdrRatio`, not `getHdrSdrRatio` - the latter tracks current screen
     * brightness and would mean redoing every decode whenever it changed. See [liveHeadroomRatio].
     *
     * Below API 36, fall back to [maxPeakValue].
     */
    @Volatile
    private var displayPeakRatio: Float? = null

    /** Weak - [Hdr] outlives every display it has seen. */
    @Volatile
    private var displayRef: WeakReference<Display>? = null

    /** The panel's headroom right now - unlike [displayPeakRatio], shrinks as brightness rises. */
    @Volatile
    private var liveHeadroomRatio: Float? = null

    /** Kept to unregister from the previous display when [attachDisplay] gets a new one. */
    @Volatile
    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private var hdrSdrRatioListener: Consumer<Display>? = null

    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private fun readLiveHeadroom(display: Display): Float? = try {
        display.takeIf { it.isHdrSdrRatioAvailable }?.hdrSdrRatio
    } catch (e: Exception) {
        null
    }?.takeIf { it.isFinite() && it > 0f }

    /** Raising it trades highlight compression for brightness on a panel that can take it. */
    @Volatile
    var maxPeakValue: Float = 4f

    /**
     * A ceiling, not a target: content below it keeps its own peak, so a 500-nit image stays a
     * 500-nit image. Only what the panel cannot show is compressed.
     */
    val presentPeak: Float
        get() {
            val ceiling = maxPeakValue
            if (!ceiling.isFinite()) return 1f
            return sane(minOf(displayPeakRatio ?: ceiling, ceiling))
        }

    /**
     * `1.0` - no compression - is what content already inside [presentPeak] gets.
     *
     * Log space rather than a clamp, as libultrahdr and the browsers do it, so highlights
     * compress smoothly instead of flattening at a ceiling.
     *
     * Both HDR paths come through here - a gain map weighted as it is applied, PQ and HLG
     * rescaled afterwards - so the two cannot drift apart.
     */
    fun peakWeight(contentStops: Float): Float {
        if (contentStops <= 0f) return 1f
        return (log2(presentPeak) / contentStops).coerceIn(0f, 1f)
    }

    /** As [peakWeight], general to a `minContentBoost` other than 1.0 (`minStops` other than 0). */
    fun peakWeight(minStops: Float, maxStops: Float): Float {
        if (maxStops <= minStops) return 1f
        val d = log2(presentPeak).coerceIn(minStops, maxStops)
        return ((d - minStops) / (maxStops - minStops)).coerceIn(0f, 1f)
    }

    /**
     * Called before the renderer initialises, so the decode path has the answer as early as it
     * can. Always false below API 34, where the surface calls that reach the display do not exist.
     */
    internal fun attachDisplay(display: Display?) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            displaySupported = false
            publishIfResolved()
            return
        }

        val available = display?.isHdrSdrRatioAvailable == true

        displayPeakRatio = if (available && Build.VERSION.SDK_INT >= Build.VERSION_CODES.BAKLAVA) {
            display?.highestHdrSdrRatio?.takeIf { it.isFinite() && it > 1f }
        } else {
            null
        }

        displaySupported = available

        displayRef?.get()?.let { old ->
            hdrSdrRatioListener?.let {
                try {
                    old.unregisterHdrSdrRatioChangedListener(it)
                } catch (e: Exception) {
                    Log.w(TAG, "Could not unregister HDR/SDR ratio listener", e)
                }
            }
        }
        displayRef = display?.let { WeakReference(it) }
        liveHeadroomRatio = display?.let { readLiveHeadroom(it) }

        hdrSdrRatioListener = if (display != null) {
            val listener = Consumer<Display> { d ->
                liveHeadroomRatio = readLiveHeadroom(d)
                markPresentationDirty()
            }
            try {
                display.registerHdrSdrRatioChangedListener({ it.run() }, listener)
                listener
            } catch (e: Exception) {
                Log.w(
                    TAG,
                    "Could not register HDR/SDR ratio listener - won't track brightness live",
                    e
                )
                null
            }
        } else {
            null
        }

        Log.i(TAG, "Display HDR: supported=$displaySupported peak=$displayPeakRatio")
        publishIfResolved()
    }

    private fun publishIfResolved() {
        if (surfaceSupported != null && displaySupported != null && !resolved.isCompleted) {
            resolved.complete(supportedByDevice)
        }
    }

    /**
     * [supportedByDevice], waiting for a surface rather than answering no because none has
     * arrived yet. The decode path can reach an image first, and answering wrong bakes
     * irreversible SDR into the early pages of a session.
     *
     * False after [RESOLVE_TIMEOUT_MS], which is also what an app driving its own surface and
     * never calling [attachDisplay] gets.
     */
    suspend fun awaitSupportedByDevice(): Boolean {
        if (surfaceSupported != null && displaySupported != null) return supportedByDevice
        return withTimeoutOrNull(RESOLVE_TIMEOUT_MS) { resolved.await() } ?: run {
            Log.w(TAG, "No surface after ${RESOLVE_TIMEOUT_MS}ms - decoding as SDR")
            false
        }
    }

    /** Kept from the first surface to answer; a later one on the same device reports the same. */
    internal fun resolve(surface: GPUSurface, adapter: GPUAdapter) {
        if (surfaceSupported != null) return

        surfaceSupported = try {
            surface.getCapabilities(adapter).formats.any { it == TextureFormat.RGBA16Float }
        } catch (e: Exception) {
            Log.w(TAG, "Could not read surface capabilities - staying in SDR", e)
            false
        }

        Log.i(TAG, "HDR surface support = $surfaceSupported")
        publishIfResolved()
    }

    // ---- content tracking ----

    /**
     * Weak, so a claim cannot outlive its image. An image whose page was turned away from before
     * its decode finished is dropped without anyone releasing it, and a strong claim would then
     * pin HDR on for the rest of the session.
     */
    private class Claim(image: Image, val headroomStops: Float) {
        private val ref = WeakReference(image)
        val image: Image? get() = ref.get()
    }

    /**
     * One per HDR image *loaded*, not per visible one: a display takes tens of frames to ramp its
     * brightness, so following visibility would pulse the panel on every page turn through mixed
     * content.
     *
     * Headrooms rather than a count, because [desiredHeadroomRatio] has to cover the brightest
     * image loaded. Synchronised - decodes add from background threads, the render thread reads
     * it every frame.
     */
    private val liveHdrClaims = ArrayList<Claim>(4)

    private val hdrLock = Any()

    val presentFormat: Int
        get() = if (supportedByDevice && liveHdrCount > 0 && hdrRecentlyDrawn)
            TextureFormat.RGBA16Float
        else TextureFormat.RGBA8Unorm

    /**
     * Rendered frames drawing no HDR image before presentation drops back to SDR.
     *
     * A claim says an image is *loaded*; being drawn says it is on screen. Presentation follows
     * the second, because a claim is only as reliable as the chain of teardown paths that releases
     * it - a page a cache never got round to evicting kept HDR on for the whole session.
     *
     * Counted in frames, not wall clock: an idle loop draws nothing, and a clock would expire
     * while a static HDR page sat on screen, then clip it on the next unrelated redraw.
     */
    private const val HDR_IDLE_FRAMES = 120

    // Idle until an HDR image is drawn.
    @Volatile
    private var framesWithoutHdr = HDR_IDLE_FRAMES

    /** Whether the past frame drew HDR, so [latchFrameFormat] knows if it has to poll. */
    @Volatile
    private var hdrDrawn = false

    // While cutting tiles: drawn, not shown. Render thread only.
    internal var cuttingTiles = false

    private val hdrRecentlyDrawn: Boolean get() = framesWithoutHdr < HDR_IDLE_FRAMES

    /**
     * Called from the draw path for every HDR image actually drawn.
     *
     * Also re-establishes [image]'s claim if it went missing without the image itself being
     * released - [resetContent] does exactly that on a surface recreated in the same process
     * (switching apps and back), since a cached, already-decoded [Image] is redrawn without ever
     * going through [retainHdrImage] again.
     */
    internal fun noteHdrDrawn(image: Image? = null, headroomStops: Float = 0f) {
        if (image != null) reclaimIfMissing(image, headroomStops)
        if (cuttingTiles) return
        hdrDrawn = true
        framesWithoutHdr = 0
        // Drawn into an SDR frame: the next latches HDR.
        if (frameFormat != TextureFormat.RGBA16Float && presentFormat == TextureFormat.RGBA16Float) {
            requestFrame?.invoke()
        }
    }

    private fun reclaimIfMissing(image: Image, headroomStops: Float) {
        val added = synchronized(hdrLock) {
            pruneLocked()
            if (liveHdrClaims.any { it.image === image }) false
            else {
                liveHdrClaims.add(Claim(image, headroomStops))
                true
            }
        }
        if (added) {
            markPresentationDirty()
            Log.i(TAG, "HDR claim re-established for image drawn without one")
        }
    }

    private val liveHdrCount: Int get() = synchronized(hdrLock) { pruneLocked(); liveHdrClaims.size }

    /** Caller holds [hdrLock]. Read every frame by way of [presentFormat], so leaks self-correct. */
    private fun pruneLocked() {
        val before = liveHdrClaims.size
        liveHdrClaims.removeAll { it.image == null }
        val dropped = before - liveHdrClaims.size
        if (dropped > 0) Log.w(TAG, "$dropped HDR claim(s) collected without release")
    }

    /** A `SurfaceControl` transaction belongs between frames, so [WebGpuRenderer] runs it. */
    @Volatile
    private var presentationDirty = false

    /**
     * The render loop is invalidate-driven and only [WebGpuRenderer.render] re-declares the
     * swapchain and surface range, so content changing the answer has to wake it. Eviction is the
     * case that matters: it lands after a page turn's last frame, with the loop already parked.
     */
    @Volatile
    internal var requestFrame: (() -> Unit)? = null

    private fun markPresentationDirty() {
        presentationDirty = true
        requestFrame?.invoke()
    }

    val presenting: Boolean get() = presentFormat == TextureFormat.RGBA16Float

    /**
     * [presentFormat] latched for the current frame, and what every render target allocates
     * against. Decode threads move [presentFormat] mid-frame, and a target that outlives the
     * change is a use-after-free. Latched by [WebGpuRenderer.render].
     */
    @Volatile
    var frameFormat: Int = TextureFormat.RGBA8Unorm
        private set

    /**
     * Take [presentFormat] for the frame about to be drawn; true when it moved. Frame boundary
     * only - nothing may still hold a resource allocated against the previous value.
     */
    internal fun latchFrameFormat(): Boolean {
        val drew = hdrDrawn
        hdrDrawn = false
        if (drew) framesWithoutHdr =
            0 else if (framesWithoutHdr < HDR_IDLE_FRAMES) framesWithoutHdr++

        val next = presentFormat

        // Presenting HDR but nothing drew it: the loop is invalidate-driven, so it has to be
        // polled to reach the deadline. Only while idle - a frame that drew HDR implies the next
        // one is already coming, and polling then would render continuously for the whole time an
        // HDR page is open.
        if (!drew && next == TextureFormat.RGBA16Float) requestFrame?.invoke()
        if (next == frameFormat) return false
        frameFormat = next
        return true
    }

    /** Runs on whatever thread finished the decode, so the swapchain waits for the next frame. */
    internal fun retainHdrImage(image: Image, headroomStops: Float) {
        val (count, ratio) = synchronized(hdrLock) {
            pruneLocked()
            liveHdrClaims.add(Claim(image, headroomStops))
            liveHdrClaims.size to desiredHeadroomRatio
        }
        // Loaded isn't drawn: a prefetch mustn't switch (and wipe every target). See noteHdrDrawn.
        if (presenting) markPresentationDirty() else presentationDirty = true
        if (count == 1) {
            Log.i(TAG, "First HDR image loaded - HDR present, headroom ratio $ratio")
        }
        Log.d(TAG, "HDR images live: $count")
    }

    internal fun releaseHdrImage(image: Image) {
        val count = synchronized(hdrLock) {
            // By identity: by value would drop some other image's claim of the same headroom,
            // and every image compressed to the ceiling shares one.
            liveHdrClaims.removeAll { it.image === image }
            pruneLocked()
            liveHdrClaims.size
        }
        markPresentationDirty()
        Log.d(TAG, "HDR images live: $count")
        if (count == 0) Log.i(TAG, "Last HDR image freed - back to SDR present")
    }

    /**
     * [Hdr] is an object, so [liveHdrClaims] outlives any one viewer. Called when a renderer
     * starts, the one point where the true count is known to be zero.
     */
    internal fun resetContent() {
        framesWithoutHdr = HDR_IDLE_FRAMES
        hdrDrawn = false
        val stranded = synchronized(hdrLock) {
            val n = liveHdrClaims.size
            liveHdrClaims.clear()
            n
        }
        markPresentationDirty()
        if (stranded > 0) Log.w(TAG, "$stranded HDR image(s) never released - count reset")
    }

    internal fun consumePresentationDirty(): Boolean {
        if (!presentationDirty) return false
        presentationDirty = false
        return true
    }

    // ---- presentation ----

    /** Only guards against nonsense metadata; the compositor clamps to the panel regardless. */
    private const val MAX_HEADROOM_RATIO = 64f

    /** Overrides [desiredHeadroomRatio] when set, for an app that wants to pin the ratio. */
    @Volatile
    var headroomRatioOverride: Float? = null

    /**
     * The same [presentPeak] the content was scaled to, so declared and drawn agree by
     * construction. Declaring the content's own headroom asked for 49x on a PQ grade.
     */
    val desiredHeadroomRatio: Float
        get() {
            headroomRatioOverride?.let { return sane(it) }
            return sane(minOf(liveHdrPeak, presentPeak))
        }

    /**
     * The brightest live claim, as a multiple of SDR white; 1 when nothing HDR is loaded.
     *
     * Asked for rather than [presentPeak] because a compositor given a ratio dims SDR content to
     * afford it - so asking for four stops while the loaded images only reach one costs
     * brightness across the whole screen for headroom nothing draws into.
     */
    private val liveHdrPeak: Float
        get() = synchronized(hdrLock) {
            pruneLocked()
            var stops = 0f
            for (claim in liveHdrClaims) {
                if (claim.image != null && claim.headroomStops > stops) stops = claim.headroomStops
            }
            if (stops > 0f && stops.isFinite()) 2f.pow(stops) else 1f
        }

    /** [MAX_HEADROOM_RATIO] only guards nonsense metadata; the NaN check is the load-bearing half. */
    private fun sane(ratio: Float): Float =
        if (ratio.isFinite()) ratio.coerceIn(1f, MAX_HEADROOM_RATIO) else 1f

    @Volatile
    private var surfaceControl: SurfaceControl? = null

    internal fun attachSurfaceControl(control: SurfaceControl?) {
        surfaceControl = control
        syncPresentation()
    }

    internal fun syncPresentation() {
        syncColorMode()
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) return
        surfaceControl?.let { applyExtendedRange(it) }
    }

    /** Weak - [Hdr] outlives every view in the process. */
    @Volatile
    private var colorModeHost: WeakReference<View>? = null

    /** What the window was last told, so an unchanged frame posts nothing. */
    private var colorModeApplied = false

    /** Guards [colorModeApplied]: set from the render thread, cleared from the main one. */
    private val colorModeLock = Any()

    /**
     * Follow [view]'s window with the HDR colour mode, or stop when null. On with the first HDR
     * image and off with the last: held longer, the panel ramps for headroom nothing uses.
     */
    internal fun attachColorModeHost(view: View?) {
        if (view == null) {
            // Before the reference goes - setColorMode needs it to reach the window.
            setColorMode(false)
            colorModeHost = null
            return
        }
        colorModeHost = WeakReference(view)
        syncColorMode()
    }

    /** As attachColorModeHost(null), unless a newer view has since taken over. */
    internal fun detachColorModeHost(view: View) {
        if (colorModeHost?.get() === view) attachColorModeHost(null)
    }

    private fun syncColorMode() {
        if (presenting != colorModeApplied) setColorMode(presenting)
    }

    /**
     * Switching apps and back leaves [colorModeApplied] believing the window is still in whatever
     * mode it last set, but the system drops `COLOR_MODE_HDR` (and the surface's extended range)
     * when the window loses focus - so [syncColorMode]'s "unchanged" check would otherwise skip
     * reapplying it forever. Called when the window regains focus.
     */
    internal fun resyncPresentation() {
        synchronized(colorModeLock) { colorModeApplied = false }
        syncPresentation()
    }

    private fun setColorMode(wantHdr: Boolean) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val view = colorModeHost?.get() ?: return

        // Claimed under the lock: two threads reaching here with opposite intents could
        // otherwise both pass the check and leave the window disagreeing with the flag.
        synchronized(colorModeLock) {
            if (wantHdr == colorModeApplied) return
            colorModeApplied = wantHdr
        }

        val apply = Runnable {
            val window = view.context.findActivity()?.window ?: return@Runnable
            window.colorMode =
                if (wantHdr) ActivityInfo.COLOR_MODE_HDR else ActivityInfo.COLOR_MODE_DEFAULT
            Log.i(TAG, "Window colour mode: ${if (wantHdr) "HDR" else "default"}")
        }

        // Window is main-thread only. On arrives from the render thread; off from
        // onDetachedFromWindow, which must run now - View.post on a detaching view queues until a
        // reattach that may never come, leaving the panel ramped for a viewer that is gone.
        if (Looper.myLooper() == Looper.getMainLooper()) apply.run() else view.post(apply)
    }

    /**
     * Both halves are required. Without the extended data space a float swapchain is just a
     * wider SDR one and every highlight above white is discarded; without
     * `setExtendedRangeBrightness` the compositor has no reason to exceed SDR brightness.
     *
     * `DATASPACE_SCRGB` is sRGB primaries and transfer with 1.0 at white, matching what the
     * decoder produces. `SCRGB_LINEAR` has a linear transfer and is deliberately not it.
     */
    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private fun applyExtendedRange(surfaceControl: SurfaceControl): Boolean {
        val wantHdr = presenting

        return try {
            // bufferRatio: what's actually baked into the pixels. desiredRatio: what the panel can
            // deliver right now - lower at high brightness. Passing the same value for both (as
            // this used to) left the compositor no room to back off, so it clipped instead of
            // rolling off once brightness ate into the live headroom.
            val bufferRatio = if (wantHdr) desiredHeadroomRatio.coerceAtLeast(1f) else 1f
            val desiredRatio =
                if (wantHdr) minOf(bufferRatio, liveHeadroomRatio ?: bufferRatio) else 1f
            SurfaceControl.Transaction()
                .setDataSpace(
                    surfaceControl,
                    if (wantHdr) DataSpace.DATASPACE_SCRGB else DataSpace.DATASPACE_SRGB
                )
                .setExtendedRangeBrightness(surfaceControl, bufferRatio, desiredRatio)
                .apply()
            Log.i(
                TAG,
                "Surface range: ${if (wantHdr) "extended sRGB, buffer $bufferRatio desired $desiredRatio" else "sRGB"}"
            )
            true
        } catch (e: Exception) {
            Log.w(TAG, "Could not set surface range - HDR will be clamped", e)
            false
        }
    }

    /**
     * Not what the viewer uses - see [attachColorModeHost]. Kept for an app that wants to pin the
     * mode anyway, alongside [headroomRatioOverride].
     */
    fun requestHdrColorMode(context: Context): Boolean {
        val activity = context.findActivity() ?: return false
        return requestHdrColorMode(activity.window)
    }

    fun requestHdrColorMode(window: Window): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
        window.colorMode = ActivityInfo.COLOR_MODE_HDR
        return true
    }

    /** The counterpart to [requestHdrColorMode]; the mode costs power for as long as it is set. */
    fun clearHdrColorMode(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val activity = context.findActivity() ?: return
        activity.window.colorMode = ActivityInfo.COLOR_MODE_DEFAULT
    }

    private fun Context.findActivity(): Activity? {
        var context: Context? = this
        while (context != null) {
            if (context is Activity) return context
            context = (context as? ContextWrapper)?.baseContext
        }
        return null
    }
}
