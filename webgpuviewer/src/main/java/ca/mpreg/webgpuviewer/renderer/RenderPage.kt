package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.CompareFunction
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUDepthStencilState
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUStencilFaceState
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUVertexState
import androidx.webgpu.OptionalBool
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.renderer.RenderPage.MAGNIFY_MAIN
import ca.mpreg.webgpuviewer.renderer.RenderPage.TILE_SAMPLER_FS
import ca.mpreg.webgpuviewer.renderer.RenderPage.draw
import ca.mpreg.webgpuviewer.renderer.RenderPage.drawMaskedRect
import ca.mpreg.webgpuviewer.renderer.RenderPage.drawTile
import ca.mpreg.webgpuviewer.renderer.RenderPage.plainVariant
import ca.mpreg.webgpuviewer.renderer.RenderPage.render
import ca.mpreg.webgpuviewer.renderer.RenderPage.renderFast
import ca.mpreg.webgpuviewer.renderer.RenderPage.samplerVariant
import ca.mpreg.webgpuviewer.renderer.RenderPage.variantFor
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Draws a single image into a render pass. Every path that draws a page's live content comes
 * through here: the continuous viewer and the paged viewer when no page turn is in flight.
 * [ImagePage.ImageSingle.renderPage]/[ImagePage.ImageSingle.renderBackground] are the page-level
 * counterparts, sharing [variantFor]/[drawTile]/[drawMaskedRect] with this object.
 *
 * Three shaders, picked per call:
 *  - [render] - box filter minifying, Catmull-Rom magnifying, in linear light. Sharp and
 *    expensive, so bound to a fixed 2x2-tile window - safe only because its one caller,
 *    [TileRenderer]'s tile generation, always targets a single tile-sized destination.
 *  - [renderFast] - one bilinear tap per pixel, also linear-light via a cheap gamma-2.2
 *    approximation (so a [TileRenderer] tile popping in over it never shows a brightness seam,
 *    close enough that the curve mismatch isn't visible) unless called with `linear = false`,
 *    which skips the sRGB<->linear round trip for [ImagePage.ImageSingle.highQuality] false content where
 *    that correctness isn't worth the cost. Draws every tile the viewport overlaps separately,
 *    so the viewport can be any size or position without a window falling short.
 *
 * A cached transition's own snapshot seeds with [ImagePage.ImageSingle.renderPage] (`masked = false`),
 * then layers in [TileRenderer]'s tiles as they land - see [Transition.getCachedTexture].
 */
object RenderPage {
    private val device get() = WebGpuRenderer.device

    // Thread-local ByteBuffer to avoid per-frame allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    /**
     * One of the three shaders, with a pipeline built per target format on first use.
     *
     * Per format because a pipeline bakes its colour target's: the same shader draws an SDR page
     * into an 8-bit target and an HDR page into a float one, often in the same frame.
     */
    internal class Variant(build: (format: Int) -> GPURenderPipeline) {
        private val byFormat = FormatKeyed(build)
        fun pipeline(format: Int): GPURenderPipeline = byFormat[format]
    }

    // Every caller of renderFast is inside ImageViewerState/ImageViewerContinuousState's own
    // render pass, which always attaches TileRenderer's stencil buffer (see [stencilViewFor]) -
    // so samplerVariant's pipeline can carry the stencil test directly, skipping a pixel
    // TileRenderer's blit already wrote (stencil == 1) instead of shading it a second time.
    private val samplerVariant = Variant { format ->
        buildPipeline(
            TILE_HEADER + TILE_VS_MAIN + TILE_SAMPLER_FS, format,
            depthStencil = GPUDepthStencilState(
                format = TextureFormat.Stencil8,
                depthWriteEnabled = OptionalBool.False,
                depthCompare = CompareFunction.Always,
                stencilFront = GPUStencilFaceState(compare = CompareFunction.NotEqual),
                stencilBack = GPUStencilFaceState(compare = CompareFunction.NotEqual),
                stencilReadMask = 0xFF,
            )
        )
    }

    /** The two resolves in force. [render] picks between them once the mip level is known. */
    internal class Filtered(val magnify: Variant, val minify: Variant)

    /**
     * Pipelines for a pair of rescaler resolves, built on first use and kept.
     *
     * Keyed by the shader text, not the rescaler, and kept apart by direction: two
     * [UpscalerCatmullRom]s share a pipeline, and so does an [UpscalerArtCnn] with either, its own
     * leftover resolve being Catmull-Rom too. Only [TileRenderer] reaches this, and only when its
     * rescalers change, so hashing a few KB of source is not on any hot path.
     */
    private val magnifyVariants = HashMap<String, Variant>()
    private val minifyVariants = HashMap<String, Variant>()

    internal fun filtered(magnify: String, minify: String) = Filtered(
        magnifyVariants.getOrPut(magnify) {
            Variant { format -> buildPipeline(HEADER + VS_MAIN + magnify + MAGNIFY_MAIN, format) }
        },
        minifyVariants.getOrPut(minify) {
            Variant { format -> buildPipeline(HEADER + VS_MAIN + minify + MINIFY_MAIN, format) }
        },
    )

    // As samplerVariant, but stencil-free - Transition's cache-seed pass has none, and doesn't
    // need one: it fills once, then tiles blit on top in later passes via ordinary blending.
    private val samplerVariantUnmasked =
        Variant { format -> buildPipeline(TILE_HEADER + TILE_VS_MAIN + TILE_SAMPLER_FS, format) }

    // Used by Transition's cache seed for non-highQuality pages, whose pass has no stencil
    // attachment at all - must stay stencil-free. See [plainVariantMasked] for the
    // stencil-pass-compatible twin ImageViewerState/Continuous use instead.
    private val plainVariant =
        Variant { format -> buildPipeline(TILE_HEADER + TILE_VS_MAIN + TILE_PLAIN_FS, format) }

    // As plainVariant, but declares a no-op stencil state (always passes, never writes) purely so
    // it's valid to use within ImageViewerState/Continuous's stencil-attached pass alongside
    // samplerVariant/renderBackground's pipelines - it doesn't itself participate in masking.
    private val plainVariantMasked = Variant { format ->
        buildPipeline(
            TILE_HEADER + TILE_VS_MAIN + TILE_PLAIN_FS, format,
            depthStencil = GPUDepthStencilState(
                format = TextureFormat.Stencil8,
                depthWriteEnabled = OptionalBool.False,
                depthCompare = CompareFunction.Always,
            )
        )
    }

    private fun buildPipeline(
        code: String, format: Int, depthStencil: GPUDepthStencilState? = null
    ): GPURenderPipeline {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        return device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = format, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = BlendFactor.SrcAlpha,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
                depthStencil = depthStencil,
            )
        )
    }

    /**
     * As [ca.mpreg.webgpuviewer.draw.Draw.rect], but with a no-op stencil state declared so it's
     * valid to use inside ImageViewerState/Continuous's stencil-attached pass - the shared
     * [ca.mpreg.webgpuviewer.draw.Draw.rect] pipeline has none, and every other place it's used
     * (transitions, etc.) has no stencil attachment at all, so it can't just gain one here.
     * [ImagePage.ImageSingle.renderPage]'s background rect is the only thing that needs this twin.
     */
    private const val MASKED_RECT_SHADER = """
struct Params {
    rect: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let pos = positions[vertex_index];
    let x = mix(params.rect.x, params.rect.z, pos.x);
    let y = mix(params.rect.y, params.rect.w, pos.y);

    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return params.color;
}
"""

    private val maskedRectPipelines = FormatKeyed { format ->
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(MASKED_RECT_SHADER))
        )
        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(module = shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    module = shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = format, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = BlendFactor.SrcAlpha,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
                depthStencil = GPUDepthStencilState(
                    format = TextureFormat.Stencil8,
                    depthWriteEnabled = OptionalBool.False,
                    depthCompare = CompareFunction.Always,
                )
            )
        )
    }

    private val maskedRectByteBuffer = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    internal fun drawMaskedRect(
        pass: GPURenderPassEncoder,
        /** Format of [pass]'s colour attachment - see [Variant]. */
        format: Int,
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        color: Int
    ) {
        val r = ((color shr 16) and 0xFF) / 255f
        val g = ((color shr 8) and 0xFF) / 255f
        val b = (color and 0xFF) / 255f
        val a = ((color ushr 24) and 0xFF) / 255f

        val byteBuffer = maskedRectByteBuffer.get()
        byteBuffer.clear()
        byteBuffer.putFloat(x1)
        byteBuffer.putFloat(y1)
        byteBuffer.putFloat(x2)
        byteBuffer.putFloat(y2)
        byteBuffer.putFloat(r)
        byteBuffer.putFloat(g)
        byteBuffer.putFloat(b)
        byteBuffer.putFloat(a)
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(size = 32L, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        )
        device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

        val maskedRectPipeline = maskedRectPipelines[format]
        pass.setPipeline(maskedRectPipeline)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = maskedRectPipeline.groupLayout(), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = uniformBuffer)
                    )
                )
            )
        )
        pass.draw(6)
        uniformBuffer.close()
    }

    /** Uniforms, texture bindings and the vertex stage's view of the source, shared by both. */
    private const val HEADER = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    tile_size: f32,
    tiles_width: f32,
    tiles_height: f32,
    dst_width: f32,
    dst_height: f32,
}

@group(0) @binding(0) var<uniform> transform: Uniforms;
@group(0) @binding(1) var src_tex0: texture_2d<f32>;
@group(0) @binding(2) var src_tex1: texture_2d<f32>;
@group(0) @binding(3) var src_tex2: texture_2d<f32>;
@group(0) @binding(4) var src_tex3: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn totalDimensions() -> vec2<u32> {
    let w = i32(transform.tiles_width);
    let h = i32(transform.tiles_height);
    if (w <= 0 || h <= 0) {
        return vec2<u32>(0u);
    }

    let dim0 = textureDimensions(src_tex0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(src_tex1).x; }

    var height = dim0.y;
    if (h > 1) { height += textureDimensions(src_tex2).y; }

    return vec2<u32>(width, height);
}

// Shared by both fragment variants: the fast path also filters in linear light now, so both need
// the same sRGB<->linear conversion.
// Odd-symmetric as scRGB defines, so extended-sRGB negatives survive instead of being clamped.
fn to_linear_exact(srgb: vec4<f32>) -> vec4<f32> {
    let c = abs(srgb.rgb);
    let lower = c / vec3<f32>(12.92);
    let higher = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let cond = c <= vec3<f32>(0.04045);
    return vec4(sign(srgb.rgb) * select(higher, lower, cond), srgb.a);
}

fn to_srgb_exact(linear_rgb: vec4<f32>) -> vec4<f32> {
    let c = abs(linear_rgb.rgb);
    let lower = c * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let cond = c <= vec3<f32>(0.0031308);
    return vec4(sign(linear_rgb.rgb) * select(higher, lower, cond), linear_rgb.a);
}

fn tileLoad(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(src_tex0, pos, 0); }
    if (i == 1) { return textureLoad(src_tex1, pos, 0); }
    if (i == 2) { return textureLoad(src_tex2, pos, 0); }
    return textureLoad(src_tex3, pos, 0);
}

// Fetch by position across the whole quad, picking the tile the position falls in. This is what
// makes filtering work at a tile boundary: the four slots are separate textures, so anything
// that resolves within one tile has no access to its neighbour's edge texels.
fn totalLoad(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(transform.tile_size);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;

    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad(idx, pos0);
}
"""

    private const val VS_MAIN = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );

    let uv = uvs[vertex_index];

    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(totalDimensions());

    // Calculate destination canvas pixel position
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);

    // Convert pixel coordinate to WebGPU NDC Space:
    // X goes from [-1.0, 1.0] (left to right)
    // Y goes from [1.0, -1.0] (top to bottom)
    let ndc_x = (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / dst_size_f.y) * 2.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uv;
    return out;
}
"""

    /**
     * Uniforms, single-texture binding and vertex stage shared by [renderFast]/[ImagePage.ImageSingle.renderPage]'s
     * per-tile draws - no [tile_size]/[tiles_width]/[tiles_height] bookkeeping, since a draw
     * through here is always exactly one tile.
     */
    private const val TILE_HEADER = """
struct TileUniforms {
    offset: vec2<f32>,
    scale: f32,
    dst_width: f32,
    dst_height: f32,
}

@group(0) @binding(0) var<uniform> transform: TileUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;

struct TileVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn tile_to_linear(srgb: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow(max(srgb.rgb, vec3<f32>(0.0)), vec3<f32>(2.2)), srgb.a);
}

fn tile_to_srgb(linear_rgb: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow(max(linear_rgb.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), linear_rgb.a);
}
"""

    private const val TILE_VS_MAIN = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> TileVertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let uv = uvs[vertex_index];
    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(textureDimensions(src_tex));
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);

    var out: TileVertexOutput;
    out.position = vec4<f32>(
        (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0,
        1.0 - (pixel_pos.y / dst_size_f.y) * 2.0,
        0.0, 1.0
    );
    out.uv = uv;
    return out;
}
"""

    /**
     * Fragment stage for [renderFast]: one bilinear resolve per pixel, in approximate
     * (gamma-2.2) linear light - see [tile_to_linear]/[tile_to_srgb].
     */
    private const val TILE_SAMPLER_FS = """
@fragment
fn fs_main(in: TileVertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(src_tex));
    let pos = in.uv * size;
    let p = pos - 0.5;
    let base = floor(p);

    let max_coord = vec2<i32>(size) - 1;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), max_coord);
    let i1 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), max_coord);
    let f = p - base;

    let c00 = tile_to_linear(textureLoad(src_tex, vec2<i32>(i0.x, i0.y), 0));
    let c10 = tile_to_linear(textureLoad(src_tex, vec2<i32>(i1.x, i0.y), 0));
    let c01 = tile_to_linear(textureLoad(src_tex, vec2<i32>(i0.x, i1.y), 0));
    let c11 = tile_to_linear(textureLoad(src_tex, vec2<i32>(i1.x, i1.y), 0));

    let linear_col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    let col = tile_to_srgb(linear_col);
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /**
     * Fragment stage for `linear = false`: [TILE_SAMPLER_FS]'s bilinear tap with the sRGB<->linear
     * round trip removed - see the class doc for the tradeoff this makes.
     */
    private const val TILE_PLAIN_FS = """
@fragment
fn fs_main(in: TileVertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(src_tex));
    let pos = in.uv * size;
    let p = pos - 0.5;
    let base = floor(p);

    let max_coord = vec2<i32>(size) - 1;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), max_coord);
    let i1 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), max_coord);
    let f = p - base;

    let c00 = textureLoad(src_tex, vec2<i32>(i0.x, i0.y), 0);
    let c10 = textureLoad(src_tex, vec2<i32>(i1.x, i0.y), 0);
    let c01 = textureLoad(src_tex, vec2<i32>(i0.x, i1.y), 0);
    let c11 = textureLoad(src_tex, vec2<i32>(i1.x, i1.y), 0);

    let col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /**
     * Fragment stage for a magnifying draw: [Upscaler]'s `resolve_magnify`, and the premultiply
     * every draw through here ends with.
     */
    private const val MAGNIFY_MAIN = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col = resolve_magnify(in.uv);
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /** As [MAGNIFY_MAIN], for a minifying draw: [Downscaler]'s `resolve_minify`. */
    private const val MINIFY_MAIN = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // resolve_minify takes src_start, the footprint's position in source pixels, and the
    // footprint's own size - which is how many source pixels one destination pixel covers.
    let src_start = in.uv * vec2<f32>(totalDimensions());
    let col = resolve_minify(src_start, vec2<f32>(1.0 / transform.scale));
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /**
     * Draw an image into [pass] with the filtered shader. Takes a pass rather than an encoder so
     * a whole frame's draws can share one - a pass per image costs an attachment load/store each,
     * which dominates frame cost on tile-based GPUs. Opening/ending the pass is the caller's job.
     */
    internal fun render(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        filtered: Filtered,
    ) {
        val res = image.prepareForRender(dst, x, y, scale) ?: return
        // Decided here rather than per fragment: it is one value for the whole draw, and a shader
        // carrying both resolves needs registers for the union of them. Against
        // [Image.MipMapForDraw.scale], not the caller's - picking a mip level moves the scale the
        // draw resolves at, and moves it toward 1, the very boundary being tested.
        draw(pass, image, dst, res, if (res.scale < 1f) filtered.minify else filtered.magnify)
    }

    /** Picks one of the 4 tile pipelines - shared by [renderFast] and [ImagePage.ImageSingle.renderPage]. */
    internal fun variantFor(linear: Boolean, masked: Boolean): Variant = when {
        linear && masked -> samplerVariant
        linear -> samplerVariantUnmasked
        masked -> plainVariantMasked
        else -> plainVariant
    }

    /**
     * Draw an image into [pass], one bilinear tap per pixel. [linear] picks [samplerVariant]'s
     * linear-light gamma correction over [plainVariant]'s straight sRGB sampling - pass `false`
     * for [ImagePage.ImageSingle.highQuality] false content. [masked] picks the twin valid inside a
     * stencil-attached pass; pass `false` only when the pass has no stencil attachment.
     */
    internal fun renderFast(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        linear: Boolean = true,
        masked: Boolean = true
    ) = renderImageTiled(pass, image, dst, x, y, scale, variantFor(linear, masked))

    /** As [render], for [renderFast]/[ImagePage.ImageSingle.renderPage] - draws every tile separately. */
    private fun renderImageTiled(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        variant: Variant
    ) {
        for (tile in image.prepareTilesForRender(dst, x, y, scale)) {
            drawTile(pass, dst, tile, variant)
        }
    }

    private fun draw(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        res: Image.MipMapForDraw,
        variant: Variant
    ) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(res.x)
        byteBuffer.putFloat(res.y)
        byteBuffer.putFloat(res.scale)
        byteBuffer.putFloat(res.mipmap.tilesize.toFloat())
        byteBuffer.putFloat(res.mipmap.tilesCols.toFloat())
        byteBuffer.putFloat(res.mipmap.tilesRows.toFloat())
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.flip()

        device.queue.writeBuffer(image.buffer, 0, byteBuffer)

        val pipeline = variant.pipeline(dst.format)
        val entries = arrayOf(
            GPUBindGroupEntry(0, buffer = image.buffer),
        ).plus(res.quad.tileViews.mapIndexed { i, view ->
            GPUBindGroupEntry(1 + i, textureView = view)
        })

        pass.setPipeline(pipeline)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(layout = pipeline.groupLayout(), entries = entries)
            )
        )

        pass.draw(6)
    }

    /**
     * Draw one tile from [Image.prepareTilesForRender], into its own persistent uniform buffer
     * rather than the shared scratch one [draw] uses - see [Mipmap.TileRect]'s doc for why.
     */
    internal fun drawTile(
        pass: GPURenderPassEncoder, dst: GPUTexture, tile: Image.TileForDraw, variant: Variant
    ) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(tile.x)
        byteBuffer.putFloat(tile.y)
        byteBuffer.putFloat(tile.scale)
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.flip()

        device.queue.writeBuffer(tile.uniform, 0, byteBuffer)

        val pipeline = variant.pipeline(dst.format)
        pass.setPipeline(pipeline)
        // samplerVariant's stencil test reads against 1 - see [TileRenderer.blitPipelineStencilWrite],
        // the only thing that ever writes this attachment.
        if (variant === samplerVariant) pass.setStencilReference(1)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.groupLayout(), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = tile.uniform),
                        GPUBindGroupEntry(1, textureView = tile.view),
                    )
                )
            )
        )
        pass.draw(6)
    }
}
