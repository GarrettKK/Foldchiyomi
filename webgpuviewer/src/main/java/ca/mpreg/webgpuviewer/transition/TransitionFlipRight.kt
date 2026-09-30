package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.util.fastCoerceAtMost
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
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2

/**
 * Page flip in the other direction: the crease sweeps in from the left.
 *
 * Same shape as [TransitionFlipLeft] - each page is rendered flat into a cached screen-sized
 * texture once, and only the fold is per-frame.
 */
object TransitionFlipRight : Transition() {
    override val premultipliedOutput = true

    // Thread-local ByteBuffer to avoid per-frame allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    private val foldSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
            )
        )
    }

    override val code = """
struct Uniforms {
    // The page's rect inside the cached surface: (x1, y1, x2, y2), normalised.
    page_rect: vec4<f32>,
    page_flip: f32,
    fold_angle: f32,
    padding0: f32,
    padding1: f32,
}

@group(0) @binding(0) var<uniform> transform: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Tessellated quad: 32x16 grid for angled fold
    const COLS: u32 = 32u;
    const ROWS: u32 = 16u;
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

    // Flip horizontally when folding so fold comes from left
    if (transform.page_flip != 0.0) {
        uv.x = 1.0 - uv.x;
    }

    // uv runs 0..1 over the page, so the flat quad spans the page's rect within the surface and
    // the texture coordinate is that same position. Keeping uv page-relative is what makes the
    // fold below page-shaped rather than screen-shaped.
    let rect_min = transform.page_rect.xy;
    let rect_max = transform.page_rect.zw;
    let flat_pos = mix(rect_min, rect_max, uv);

    var ndc_x = flat_pos.x * 2.0 - 1.0;
    var ndc_y = 1.0 - flat_pos.y * 2.0;

    // Page fold effect: page folds back at an angled crease
    if (transform.page_flip != 0.0) {
        let flip = transform.page_flip;
        let norm_x = uv.x;
        let norm_y = uv.y;

        // Angled fold line: normal direction
        let fold_angle = transform.fold_angle;
        let nx = cos(fold_angle);
        let ny = sin(fold_angle);

        // Distance along fold normal from origin
        let max_dist = nx + abs(ny);
        let fold_pos = flip * max_dist;
        let dist = norm_x * nx + norm_y * ny;

        // The page's rect in NDC, which the reflection below is expressed in.
        let page_left = rect_min.x * 2.0 - 1.0;
        let page_width_ndc = (rect_max.x - rect_min.x) * 2.0;
        let page_top = 1.0 - rect_min.y * 2.0;
        let page_height_ndc = (rect_max.y - rect_min.y) * 2.0;

        if (dist < fold_pos) {
            let arc_len = fold_pos - dist;
            let radius = 0.15;
            let fold_len = 3.14159265 * radius;

            var folded_dist: f32;
            if (arc_len < fold_len) {
                let theta = arc_len / radius;
                folded_dist = fold_pos - radius * sin(theta);
            } else {
                folded_dist = fold_pos + (arc_len - fold_len);
            }

            // Reflect position across fold line
            let delta = folded_dist - dist;
            let new_norm_x = norm_x + delta * nx;
            let new_norm_y = norm_y + delta * ny;

            ndc_x = page_left + new_norm_x * page_width_ndc;
            ndc_y = page_top - new_norm_y * page_height_ndc;
        }
    }

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = flat_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The cache holds premultiplied alpha, so pass it straight through - see premultipliedOutput.
    return textureSample(src_tex, src_sampler, in.uv);
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

        val dy = pos2.y - pos1.y
        val dx = pos2.x - pos1.x
        val sign = if (dx < 0f) -1f else 1f
        val foldAngle = (sign * (atan2(dy, abs(dx)) / 2)).fastCoerceAtMost(0f)

        // Draw single background rect that transitions between colors
        val t = if (frac > 0f) frac else -frac
        val bg1 = page1.backgroundColor ?: 0xFF000000.toInt()
        val bg2 = page2.backgroundColor ?: 0xFF000000.toInt()
        Draw.rect(encoder, dst, 0f, 0f, 1f, 1f, blendBackgroundColor(bg1, bg2, t))

        if (frac > 0f) {
            fold(cached1, page1, encoder, dst, 0f, foldAngle)
            fold(cached2, page2, encoder, dst, 1f - frac, foldAngle)
        } else {
            fold(cached2, page2, encoder, dst, 0f, foldAngle)
            fold(cached1, page1, encoder, dst, -frac, foldAngle)
        }
    }

    private fun fold(
        cachedView: GPUTextureView?,
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        foldAngle: Float,
    ) {
        if (cachedView == null) return
        val rect = page.wholeRect(dst) ?: return

        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        for (v in rect) byteBuffer.putFloat(v)
        byteBuffer.putFloat(frac)
        byteBuffer.putFloat(foldAngle)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
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
                        GPUBindGroupEntry(2, sampler = foldSampler),
                    )
                )
            )
        )

        pass.draw(3072)
        pass.endAndRelease(targetView, uniformBuffer)
    }
}
