package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTextureView
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer

/**
 * One post-processing step over the finished frame, run by [FilterChain] between the viewer's
 * draw and the swapchain - see that class for how the chain is wired up.
 *
 * A filter reads [run]'s `src` and writes its result to `dst`; the chain owns both and
 * ping-pongs them, so a filter never allocates its own input or output. A multi-pass filter
 * (an upscaling network, say) takes scratch textures for its intermediate passes from
 * [FilterChain.scratch] and writes only its last pass into `dst`.
 *
 * Fragment or compute is the filter's own choice. [FilterFullscreen] covers the fragment case,
 * which is what a per-pixel filter wants: it can write the swapchain texture directly, so the
 * last filter in the chain costs no extra copy. A compute filter must say so with [usesCompute]
 * - the swapchain has no storage binding, so the chain gives it an offscreen destination and
 * blits that to the screen afterwards.
 */
abstract class Filter {
    protected val device get() = WebGpuRenderer.device

    /** Ask for a redraw - set by [FilterChain] when this filter joins it. */
    internal var onUpdate: (() -> Unit)? = null

    // Set from wherever the app's settings live, read on the render thread.
    @Volatile
    var enabled: Boolean = true
        set(value) {
            if (field == value) return
            field = value
            invalidate()
        }

    /** Redraw with this filter's new settings. Call after anything that changes its output. */
    protected fun invalidate() {
        onUpdate?.invoke()
    }

    /**
     * Whether the chain runs this filter at all this frame - [enabled] plus whatever else the
     * filter needs to do anything, so one that is switched on but not yet configured costs
     * nothing. A chain whose filters are all inactive is skipped outright: no offscreen pass.
     */
    open val active: Boolean get() = enabled

    /** Shown in pass labels. */
    open val label: String get() = javaClass.simpleName

    /**
     * The format this filter writes. The chain's own textures all match the swapchain, which
     * is [Hdr.frameFormat]; a filter that needs headroom between its passes should say
     * [TextureFormat.RGBA16Float] here instead.
     *
     * Compute filters usually do need that, but only in SDR: [TextureFormat.RGBA8Unorm] is
     * storable behind an optional feature, whereas [TextureFormat.RGBA16Float] - which
     * [Hdr.frameFormat] already is while HDR is on screen - is storable as core.
     */
    open val outputFormat: Int get() = Hdr.frameFormat

    /** True when [run] writes `dst` from a compute pass, so it must be a storage texture. */
    open val usesCompute: Boolean get() = false

    /** Output size, for a filter that resamples - the input size by default. */
    open fun outputWidth(width: Int, height: Int): Int = width
    open fun outputHeight(width: Int, height: Int): Int = height

    abstract fun run(
        chain: FilterChain,
        encoder: GPUCommandEncoder,
        src: GPUTextureView,
        srcWidth: Int,
        srcHeight: Int,
        dst: GPUTextureView,
        dstWidth: Int,
        dstHeight: Int,
    )

    /** Release GPU resources. Called on the render thread by [FilterChain.cleanup]. */
    open fun cleanup() {}
}
