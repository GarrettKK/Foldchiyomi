package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView

/**
 * How [TileRenderer] resizes a high-quality tile - [Upscaler] magnifying, [Downscaler] shrinking.
 *
 * With no [factor] - the defaults - [RenderPage.render] resolves the tile in one step. A [factor]
 * above 1 splits that: [RenderPage.render] resolves at [firstStepScale], then this covers the
 * rest, once. The leftover is still the first step's filter doing it.
 *
 * Four calls per tile, in order, all on the tile worker: [input] for the `size`-square texture
 * the first step draws into, the caller drawing the page into it inset by [halo], [encode], then
 * [resolve].
 */
abstract class Rescaler {
    protected val device get() = WebGpuRenderer.device

    /** False when this device can't run it - [TileRenderer] then resolves in one step instead. */
    open val supported: Boolean get() = true

    /** How much one run resizes by. 1 means it does nothing, and the tile path skips it. */
    open val factor: Int get() = 1

    /**
     * The WGSL the first step resolves with, composed into [RenderPage.filtered]'s pipeline. An
     * [Upscaler] defines `resolve_magnify(uv) -> vec4<f32>`, a [Downscaler]
     * `resolve_minify(src_start, scale) -> vec4<f32>`, both against [RenderPage]'s header -
     * `transform`, `src_tex0..3`, `totalLoad`, `to_linear_exact`.
     */
    abstract val code: String

    /**
     * Input pixels of surrounding page each output pixel needs, cut off again by [resolve].
     * Without it a convolutional rescaler is wrong along every tile edge and the seams show.
     */
    open val halo: Int get() = 0

    /** The scale the first step resolves the tile at, leaving [factor] for this to cover. */
    abstract fun firstStepScale(scale: Float): Float

    /** The tile's own span measured in first-step pixels. */
    abstract fun firstStepSpan(tileSize: Int): Int

    /** True when a tile at [scale] has a whole [factor] of resizing to give this. */
    abstract fun appliesAt(scale: Float): Boolean

    /** True when a tile of [tileSize] divides the way [firstStepSpan] needs it to. */
    open fun fits(tileSize: Int): Boolean = true

    /** The square texture the first step renders into. Null if this rescaler can't run. */
    open fun input(size: Int, format: Int): GPUTexture? = null

    /** [input]'s view, kept rather than remade per tile. Valid only after [input] has answered. */
    open val inputView: GPUTextureView? get() = null

    /** Encode the resize of [input]'s current contents. */
    open fun encode(encoder: GPUCommandEncoder, size: Int) {}

    /** Draw the middle of the resized result - the tile itself, halo removed - into [pass]. */
    open fun resolve(pass: GPURenderPassEncoder, format: Int) {}

    open fun cleanup() {}
}

/**
 * A [Rescaler] for tiles that magnify the page, where a filter has to invent detail the source
 * doesn't have. [UpscalerCatmullRom] by default, [UpscalerArtCnn] the alternative.
 *
 * The first step resolves at `scale / factor`, so this only runs given a whole [factor] of zoom.
 * Below that the first step would shrink the page to make room, losing the detail this exists to
 * reconstruct.
 */
abstract class Upscaler : Rescaler() {
    /** Catmull-Rom, until a subclass says otherwise - see [UpscalerCatmullRom]. */
    override val code: String get() = UpscalerCatmullRom.CODE

    override fun firstStepScale(scale: Float): Float = scale / factor
    override fun firstStepSpan(tileSize: Int): Int = tileSize / factor
    override fun appliesAt(scale: Float): Boolean = scale >= factor

    /** The first step resolves a fraction of the tile, so it has to be a whole number of pixels. */
    override fun fits(tileSize: Int): Boolean = tileSize % factor == 0
}

/**
 * A [Rescaler] for tiles that shrink the page, averaging detail away without aliasing rather than
 * inventing any. [DownscalerBox] is the only one.
 *
 * The mirror of [Upscaler]: the first step resolves at `scale * factor`, larger than the tile,
 * and this reduces it the rest of the way.
 */
abstract class Downscaler : Rescaler() {
    /** A box filter, until a subclass says otherwise - see [DownscalerBox]. */
    override val code: String get() = DownscalerBox.CODE

    override fun firstStepScale(scale: Float): Float = scale * factor
    override fun firstStepSpan(tileSize: Int): Int = tileSize * factor
    override fun appliesAt(scale: Float): Boolean = scale <= 1f / factor
}
