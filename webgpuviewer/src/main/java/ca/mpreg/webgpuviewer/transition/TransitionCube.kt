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
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.sin

/**
 * Cube rotation: the outgoing page is the front face, the incoming one the side face.
 *
 * Each page is rendered flat into a cached screen-sized texture first, then a face maps that
 * texture onto a rotating quad. [getCachedTexture] keys on the page's own transform, so the flat
 * render happens once per transition while only the rotation is per-frame.
 *
 * A face is the whole cached surface, so it is screen-shaped rather than page-shaped - unlike the
 * flips and the sphere, which map the page's rect. Each face also gets a background column behind
 * it, spanning its projected width and the full height of the surface.
 */
object TransitionCube : Transition() {
    override val premultipliedOutput = true

    // Thread-local ByteBuffer to avoid per-frame allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder())
    }

    private val faceSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
            )
        )
    }

    override val code = """
struct Uniforms {
    transform_mat: mat4x4<f32>,
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

    // The matrix maps the unit quad to clip space, perspective via W. The unit quad is the cached
    // surface, so the face is screen-shaped - the matrix is built from the surface's dimensions and
    // its vertical scale cancels against the screen aspect.
    //
    // uv doubles as the texture coordinate, since the face spans the whole surface.
    let local_pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    let transformed = transform.transform_mat * local_pos;

    var out: VertexOutput;
    out.position = vec4<f32>(transformed.xy / transformed.w, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) { discard; }

    // Back-face culling via UV winding
    let dudx = dpdx(in.uv);
    let dudy = dpdy(in.uv);
    if (dudx.x * dudy.y - dudx.y * dudy.x < 0.0) { discard; }

    // textureSampleLevel rather than textureSample: the discards above make this non-uniform
    // control flow, where implicit derivatives are not allowed. The cache is single-level, so an
    // explicit LOD of 0 loses nothing. Premultiplied already - see premultipliedOutput.
    return textureSampleLevel(src_tex, src_sampler, in.uv, 0.0);
}"""

    private const val HALF_PI = (Math.PI / 2.0).toFloat()
    private const val FOV = 4f
    private const val FACE_DEPTH = FOV / (FOV - 1f)

    private fun mat4(
        m00: Float, m01: Float, m02: Float, m03: Float,
        m10: Float, m11: Float, m12: Float, m13: Float,
        m20: Float, m21: Float, m22: Float, m23: Float,
        m30: Float, m31: Float, m32: Float, m33: Float,
    ) = floatArrayOf(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    )

    private fun rotateY(angle: Float): FloatArray {
        val c = cos(angle)
        val s = sin(angle)
        return mat4(
            c, 0f, -s, 0f,
            0f, 1f, 0f, 0f,
            s, 0f, c, 0f,
            0f, 0f, 0f, 1f,
        )
    }

    private fun translate(x: Float, y: Float, z: Float) = mat4(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        x, y, z, 1f,
    )

    private fun scale(x: Float, y: Float, z: Float) = mat4(
        x, 0f, 0f, 0f,
        0f, y, 0f, 0f,
        0f, 0f, z, 0f,
        0f, 0f, 0f, 1f,
    )

    private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val result = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                result[col * 4 + row] = sum
            }
        }
        return result
    }

    /**
     * Face transform for one side of the cube.
     *
     * [faceWidth] and [faceHeight] size the face so its content isn't distorted. The face is the
     * cached surface, so they are the surface's dimensions - the vertical scale then cancels
     * against [screenAspect] and the face comes out screen-shaped.
     */
    private fun buildFaceMatrix(
        rotAngle: Float,
        screenAspect: Float,
        faceWidth: Float,
        faceHeight: Float,
        isSide: Boolean,
    ): FloatArray {
        val faceScaleMat =
            scale(FACE_DEPTH, (faceHeight / faceWidth) * screenAspect * FACE_DEPTH, 1f)
        val baseMat = if (isSide) {
            multiply(rotateY(HALF_PI), multiply(translate(0f, 0f, FACE_DEPTH), faceScaleMat))
        } else {
            multiply(translate(0f, 0f, FACE_DEPTH), faceScaleMat)
        }
        val worldMat = multiply(rotateY(-rotAngle), baseMat)
        // Perspective projection: output.w = z + fov, ndc = xy / w
        val projMat = mat4(
            FOV, 0f, 0f, 0f,
            0f, FOV, 0f, 0f,
            0f, 0f, 1f, 1f,
            0f, 0f, 0f, FOV,
        )
        return multiply(projMat, worldMat)
    }

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

        val t = if (frac > 0f) frac else 1f + frac

        // Rotation runs the whole way across, with no held stages at either end. At 0 the front
        // face is exactly flat and full-screen, and at 90 degrees the side face is - which is what
        // FACE_DEPTH is chosen for - so there is nothing to ease into or out of.
        val rotAngle = t * HALF_PI

        val screenAspect = dst.width.toFloat() / dst.height.toFloat()

        // When frac > 0: page1 is front (rotating away), page2 is side (rotating in)
        // When frac < 0: page2 is front, page1 is side
        val frontPage: ImagePage
        val sidePage: ImagePage
        val frontFace: GPUTextureView?
        val sideFace: GPUTextureView?
        if (frac > 0f) {
            frontPage = page1
            sidePage = page2
            frontFace = cached1
            sideFace = cached2
        } else {
            frontPage = page2
            sidePage = page1
            frontFace = cached2
            sideFace = cached1
        }

        // A face is the whole cached surface, so both use the surface's dimensions.
        val faceWidth = dst.width.toFloat()
        val faceHeight = dst.height.toFloat()

        val frontMat = buildFaceMatrix(
            rotAngle, screenAspect, faceWidth, faceHeight, isSide = false,
        )
        val sideMat = buildFaceMatrix(
            rotAngle, screenAspect, faceWidth, faceHeight, isSide = true,
        )

        // Both faces load rather than clear, so that the second doesn't erase the first, and the
        // cube never covers the whole surface. Without a clear first, the area around it shows
        // whatever getCurrentTexture's rotating buffer held several frames ago.
        Draw.clear(encoder, dst, 0)

        // Render back-to-front for correct overlap
        if (t < 0.5f) {
            drawFace(sideFace, sidePage, encoder, dst, sideMat)
            drawFace(frontFace, frontPage, encoder, dst, frontMat)
        } else {
            drawFace(frontFace, frontPage, encoder, dst, frontMat)
            drawFace(sideFace, sidePage, encoder, dst, sideMat)
        }
    }

    /**
     * Render [page] flat into its cache slot, then map that onto a face by [matrix].
     *
     * Shared with [TransitionCubeOuter], which drives the same faces through a different rotation.
     * [isPage1] picks the cache slot, so the two pages must be given different values or they will
     * evict each other every frame.
     */
    internal fun face(
        page: ImagePage,
        isPage1: Boolean,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        matrix: FloatArray,
        tiles: TileRenderer,
    ) {
        val cached = getCachedTexture(page, isPage1, encoder, dst.width, dst.height, tiles)
        drawFace(cached, page, encoder, dst, matrix)
    }

    private fun drawFace(
        cachedView: GPUTextureView?,
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        matrix: FloatArray,
    ) {
        if (cachedView == null) return

        // Background behind the face, spanning its projected width and the full height. The face
        // itself carries the page's own background inside the cache, but only across the page's
        // band and rotated with it - this fills the column top to bottom.
        //
        // Project the face's left and right edges to find that width. Column-major, so index is
        // col * 4 + row; local_pos is (x, y, 0, 1) and neither x nor w takes a y term here, so
        // only column 0 and column 3 matter.
        val leftW = -matrix[3] + matrix[15]
        val rightW = matrix[3] + matrix[15]
        val leftX = (-matrix[0] + matrix[12]) / leftW
        val rightX = (matrix[0] + matrix[12]) / rightW
        page.backgroundColor?.let {
            Draw.rect(encoder, dst, (leftX + 1f) / 2f, 0f, (rightX + 1f) / 2f, 1f, it)
        }

        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        for (i in matrix.indices) {
            byteBuffer.putFloat(matrix[i])
        }
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(size = 64, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
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
                        GPUBindGroupEntry(2, sampler = faceSampler),
                    )
                )
            )
        )

        pass.draw(6144)
        pass.endAndRelease(targetView, uniformBuffer)
    }
}
