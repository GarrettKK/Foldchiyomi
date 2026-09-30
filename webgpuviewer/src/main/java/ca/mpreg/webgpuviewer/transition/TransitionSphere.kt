package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Sphere rotation: both pages are mapped onto a sphere that turns half a revolution.
 *
 * Each page is rendered flat into a cached screen-sized texture first, then a hemisphere maps that
 * texture. [getCachedTexture] keys on the page's own transform, so the flat render happens once per
 * transition while only the rotation is per-frame.
 *
 * What the hemisphere wraps is the page's rect clipped to that surface. A page zoomed past the
 * surface clips to the whole of it, so the cache is taken exactly as drawn, pan and zoom included
 * - nothing undoes the page's transform. A page smaller than the surface wraps just its own rect,
 * so the sphere is the page rather than the letterbox around it.
 *
 * That rect is [ImagePage.wholeRect], not [ImagePage.pageRect], so a spread wraps as one sheet
 * with both its pages rather than turning one side and dropping the other.
 */
object TransitionSphere : Transition() {
    override val premultipliedOutput = true

    // Thread-local ByteBuffer to avoid per-frame allocation. Sized to the uniform exactly:
    // writeBuffer sends the whole capacity, so flip() does not shorten it.
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    private val sphereSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
            )
        )
    }

    override val code = """
struct Uniforms {
    // The page's rect inside the cached surface: (x1, y1, x2, y2), normalised. May reach outside
    // it, which is what the clip in the vertex stage is for.
    page_rect: vec4<f32>,
    dst_width: f32,
    dst_height: f32,
    transition: f32,
    is_second: f32,
}

@group(0) @binding(0) var<uniform> transform: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) sphere_z: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const COLS: u32 = 32u;
    const ROWS: u32 = 32u;
    let quad_index = vertex_index / 6u;
    let vert_in_quad = vertex_index % 6u;
    let col = quad_index % COLS;
    let row = quad_index / COLS;

    let x0 = f32(col) / f32(COLS);
    let x1 = f32(col + 1u) / f32(COLS);
    let y0 = f32(row) / f32(ROWS);
    let y1 = f32(row + 1u) / f32(ROWS);

    var uv: vec2<f32>;
    switch (vert_in_quad) {
        case 0u: { uv = vec2<f32>(x0, y0); }
        case 1u: { uv = vec2<f32>(x0, y1); }
        case 2u: { uv = vec2<f32>(x1, y0); }
        case 3u: { uv = vec2<f32>(x1, y0); }
        case 4u: { uv = vec2<f32>(x0, y1); }
        default: { uv = vec2<f32>(x1, y1); }
    }

    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let aspect = dst_size_f.x / dst_size_f.y;
    let sphere_r = 0.15;

    // Flat position: the page's rect where it fits inside the surface, the surface itself where
    // it does not. Clipping gives both without a branch - a page zoomed past the surface clips to
    // 0..1 and is taken as drawn, while a smaller one keeps its own rect and wraps the page
    // rather than the letterbox. uv runs 0..1 over whichever it is, and doubles as the texture
    // coordinate.
    let is_back = transform.is_second > 0.5;
    let clipped = clamp(transform.page_rect, vec4<f32>(0.0), vec4<f32>(1.0));
    let flat_pos = mix(clipped.xy, clipped.zw, uv);
    var flat_ndc = vec2<f32>(flat_pos.x * 2.0 - 1.0, 1.0 - flat_pos.y * 2.0);

    // Sphere position: map UV to sphere surface
    let theta = (uv.x - 0.5) * 3.14159265 + select(0.0, 3.14159265, is_back);
    let phi = (0.5 - uv.y) * 3.14159265;

    // 3D point on sphere
    var sp_x = sin(theta) * cos(phi);
    var sp_y = sin(phi);
    var sp_z = cos(theta) * cos(phi);

    let sx = sp_x * sphere_r * 2.0 / aspect;
    let sy = sp_y * sphere_r * 2.0;
    let sphere_ndc = vec2<f32>(sx, sy);

    // Determine phase and interpolation
    let t = transform.transition;
    var phase = 0.0;
    if (t < 1.0 / 3.0) {
        phase = t * 3.0;
    } else if (t < 2.0 / 3.0) {
        phase = 1.0;
    } else {
        phase = 1.0 - (t - 2.0 / 3.0) * 3.0;
    }

    // For phase 2, use the fully-rotated sphere position
    var target_sphere_ndc = sphere_ndc;
    if (t >= 2.0 / 3.0) {
        // After full PI rotation: rx = sp_x*cos(PI) + sp_z*sin(PI) = -sp_x
        let rotated_x = -sp_x * sphere_r * 2.0 / aspect;
        target_sphere_ndc = vec2<f32>(rotated_x, sy);
    }

    var final_ndc = mix(flat_ndc, target_sphere_ndc, vec2<f32>(phase));

    var sphere_z = sp_z;

    if (t >= 1.0 / 3.0 && t < 2.0 / 3.0) {
        let rot_phase = (t - 1.0 / 3.0) * 3.0;
        let rot_angle = -rot_phase * 3.14159265;
        let rx = sp_x * cos(rot_angle) + sp_z * sin(rot_angle);
        let rz = -sp_x * sin(rot_angle) + sp_z * cos(rot_angle);
        final_ndc = vec2<f32>(rx * sphere_r * 2.0 / aspect, sp_y * sphere_r * 2.0);
        sphere_z = rz;
    }

    var out: VertexOutput;
    out.position = vec4<f32>(final_ndc, 0.0, 1.0);
    out.uv = flat_pos;
    out.sphere_z = sphere_z;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) { discard; }

    let t = transform.transition;
    let is_second = transform.is_second > 0.5;

    // Phase 0: only image 1 visible
    // Phase 1: both visible based on sphere_z (front/back)
    // Phase 2: only image 2 visible
    if (t < 1.0 / 3.0 && is_second) { discard; }
    if (t >= 2.0 / 3.0 && !is_second) { discard; }
    if (t >= 1.0 / 3.0 && t < 2.0 / 3.0) {
        if (in.sphere_z < 0.0) { discard; }
    }

    // textureSampleLevel rather than textureSample: the discards above make this non-uniform
    // control flow, where implicit derivatives are not allowed. The cache is single-level, so an
    // explicit LOD of 0 loses nothing. Premultiplied already - see premultipliedOutput.
    return textureSampleLevel(src_tex, src_sampler, in.uv, 0.0);
}"""

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        // Draw single background rect that transitions between colors
        val t = if (frac > 0f) frac else -frac
        val bg1 = page1.backgroundColor ?: 0xFF000000.toInt()
        val bg2 = page2.backgroundColor ?: 0xFF000000.toInt()
        Draw.rect(encoder, dst, 0f, 0f, 1f, 1f, blendBackgroundColor(bg1, bg2, t))

        if (frac > 0f) {
            hemisphere(cached2, page2, encoder, dst, frac, 1f)
            hemisphere(cached1, page1, encoder, dst, frac, 0f)
        } else {
            hemisphere(cached1, page1, encoder, dst, 1f + frac, 1f)
            hemisphere(cached2, page2, encoder, dst, 1f + frac, 0f)
        }
    }

    private fun hemisphere(
        cachedView: GPUTextureView?,
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        transition: Float,
        isSecond: Float,
    ) {
        if (cachedView == null) return
        val rect = page.wholeRect(dst) ?: return

        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        for (v in rect) byteBuffer.putFloat(v)
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.putFloat(transition)
        byteBuffer.putFloat(isSecond)
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        )
        device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

        val targetView = dst.createView()
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = targetView,
                        loadOp = LoadOp.Load,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )

        val pipeline = pipelines[dst.format]
        pass.setPipeline(pipeline)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.groupLayout(), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = uniformBuffer),
                        GPUBindGroupEntry(1, textureView = cachedView),
                        GPUBindGroupEntry(2, sampler = sphereSampler),
                    )
                )
            )
        )

        pass.draw(6144)
        pass.endAndRelease(targetView, uniformBuffer)
    }
}
