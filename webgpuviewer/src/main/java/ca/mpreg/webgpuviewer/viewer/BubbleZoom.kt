package ca.mpreg.webgpuviewer.viewer

import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.InkMap
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * A speech bubble lifted off the page and shown enlarged over it, the way Google Play Books does -
 * see [ImageViewerState.showBubble]. Everything a frame needs, fixed when the bubble is found.
 */
internal class BubbleOverlay(
    /** The page it was found on - it only draws over that page, at rest. */
    val page: ImagePage.ImageSingle,
    /** The bubble's shape: [maskWidth] x [maskHeight], 255 inside, over [rect]. */
    val mask: ByteArray,
    val maskWidth: Int,
    val maskHeight: Int,
    /** The bubble's bounds on screen, normalised (x1, y1, x2, y2). */
    val rect: FloatArray,
    /** Where it opens to: its centre on screen, and how much larger it gets. */
    val targetX: Float,
    val targetY: Float,
    val zoom: Float,
) {
    val centerX: Float get() = (rect[0] + rect[2]) * 0.5f
    val centerY: Float get() = (rect[1] + rect[3]) * 0.5f

    /** 0 closed to 1 open, animated by [ImageViewerState]. */
    @Volatile
    var progress: Float = 0f

    // Render thread only, like everything below.
    private var maskTexture: GPUTexture? = null
    private var maskView: GPUTextureView? = null

    /** Set once [release] has run: a frame captured before it is dropped rather than drawn. */
    var released = false
        private set

    fun maskView(): GPUTextureView {
        maskView?.let { return it }
        val device = WebGpuRenderer.device
        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(maskWidth, maskHeight),
                format = TextureFormat.R8Unorm,
                usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
            )
        )
        val data = ByteBuffer.allocateDirect(mask.size).order(ByteOrder.nativeOrder())
        data.put(mask).flip()
        device.queue.writeTexture(
            dataLayout = GPUTexelCopyBufferLayout(
                offset = 0L, bytesPerRow = maskWidth, rowsPerImage = maskHeight
            ),
            data = data,
            destination = GPUTexelCopyTextureInfo(texture = texture),
            writeSize = GPUExtent3D(maskWidth, maskHeight),
        )
        maskTexture = texture
        return texture.createView().also { maskView = it }
    }

    /** On the render thread. */
    fun release() {
        released = true
        maskView?.close()
        maskTexture?.destroyAndRelease()
        maskView = null
        maskTexture = null
    }
}

internal object BubbleZoom {
    /** A bubble is at least this bright where it was tapped, out of 255. */
    private const val MIN_SEED_LUMA = 165

    /** What counts as the bubble's own paper, relative to the brightness where it was tapped. */
    private const val PAPER_FRACTION = 0.82f

    /** How far a tap on the lettering looks for the paper around it, in map pixels. */
    private const val SEED_RADIUS = 10

    /** Largest share of the page a bubble may take - past this it's a white panel, not a bubble. */
    private const val MAX_AREA = 0.12f

    /** Share of a bubble's area its lettering takes, at least and at most. */
    private const val MIN_INK = 0.015f
    private const val MAX_INK = 0.3f

    /** Largest single mark inside a bubble, as a share of it - a letter or a line of them. */
    private const val MAX_MARK = 0.2f

    /** Map pixels added around the bubble, so its own outline comes with it. */
    private const val OUTLINE = 2

    /** How far the bubble grows at most, and at least for it to be worth opening. */
    private const val MAX_ZOOM = 3.2f
    private const val MIN_ZOOM = 1.3f

    /** How much of the screen an open bubble may fill across and down. */
    private const val FIT_WIDTH = 0.9f
    private const val FIT_HEIGHT = 0.62f

    /**
     * The bubble under the tap at ([tapX], [tapY]) - normalised screen coordinates on a
     * [screenWidth] x [screenHeight] surface - or null where there is none to open. Off the GPU
     * thread: it reads only the images' [InkMap]s and the page's placement.
     */
    fun detect(
        page: ImagePage.ImageSingle,
        screenWidth: Int,
        screenHeight: Int,
        tapX: Float,
        tapY: Float,
    ): BubbleOverlay? {
        var hit: Pair<Image, FloatArray>? = null
        page.forEachImage { image, offsetX, imageScale ->
            if (hit != null) return@forEachImage
            // As ImageSingle.forEachPlacedImage, at the page's own transform.
            val placeX = (page.x + offsetX / screenWidth + WebGpuRenderer.offsetX) / imageScale -
                    WebGpuRenderer.offsetX
            val placeY = (page.y + WebGpuRenderer.offsetY) / imageScale - WebGpuRenderer.offsetY
            val rect = image.placement(
                screenWidth, screenHeight, placeX, placeY, page.scale * imageScale
            )
            if (tapX >= rect[0] && tapX < rect[2] && tapY >= rect[1] && tapY < rect[3]) {
                hit = image to rect
            }
        }
        val (image, imageRect) = hit ?: return null
        val map = image.inkMap ?: return null

        val u = (tapX - imageRect[0]) / (imageRect[2] - imageRect[0])
        val v = (tapY - imageRect[1]) / (imageRect[3] - imageRect[1])
        val found = find(
            map,
            (u * map.width).toInt().coerceIn(0, map.width - 1),
            (v * map.height).toInt().coerceIn(0, map.height - 1),
        ) ?: return null

        // Map pixels to screen.
        val sx = (imageRect[2] - imageRect[0]) / map.width
        val sy = (imageRect[3] - imageRect[1]) / map.height
        val rect = floatArrayOf(
            imageRect[0] + found.x0 * sx,
            imageRect[1] + found.y0 * sy,
            imageRect[0] + (found.x0 + found.width) * sx,
            imageRect[1] + (found.y0 + found.height) * sy,
        )
        val bubbleWidth = (rect[2] - rect[0]) * screenWidth
        val bubbleHeight = (rect[3] - rect[1]) * screenHeight
        val zoom = min(
            MAX_ZOOM,
            min(FIT_WIDTH * screenWidth / bubbleWidth, FIT_HEIGHT * screenHeight / bubbleHeight),
        )
        if (zoom < MIN_ZOOM) return null

        // Opens where it is, pushed back on screen - centred when it takes most of the width.
        fun settle(center: Float, halfSize: Float, margin: Float): Float =
            if (halfSize + margin >= 0.5f) 0.5f else center.coerceIn(halfSize + margin, 1f - halfSize - margin)

        return BubbleOverlay(
            page = page,
            mask = found.mask,
            maskWidth = found.width,
            maskHeight = found.height,
            rect = rect,
            targetX = settle((rect[0] + rect[2]) * 0.5f, zoom * (rect[2] - rect[0]) * 0.5f, 0.03f),
            targetY = settle((rect[1] + rect[3]) * 0.5f, zoom * (rect[3] - rect[1]) * 0.5f, 0.05f),
            zoom = zoom,
        )
    }

    /** A bubble found on an [InkMap]: its bounds there, and its shape within them. */
    class Found(val x0: Int, val y0: Int, val width: Int, val height: Int, val mask: ByteArray)

    /**
     * The speech bubble around map pixel ([x], [y]): the bright paper there, flood-filled up to the
     * dark outline that closes it, its lettering filled in, its outline added back. Null for
     * anything that doesn't look like a bubble - open to the page's edge, too big, too thin, or
     * with nothing written in it.
     */
    fun find(map: InkMap, x: Int, y: Int): Found? {
        val w = map.width
        val h = map.height

        // A tap on the lettering starts from the brightest paper next to it.
        var seedX = x
        var seedY = y
        var seedLuma = map[x, y]
        if (seedLuma < MIN_SEED_LUMA) {
            for (dy in -SEED_RADIUS..SEED_RADIUS) for (dx in -SEED_RADIUS..SEED_RADIUS) {
                val px = x + dx
                val py = y + dy
                if (px !in 0 until w || py !in 0 until h) continue
                val l = map[px, py]
                if (l > seedLuma) {
                    seedLuma = l
                    seedX = px
                    seedY = py
                }
            }
            if (seedLuma < MIN_SEED_LUMA) return null
        }
        val paper = (seedLuma * PAPER_FRACTION).toInt()

        // Flood the paper, four-connected, giving up once it's too big to be a bubble.
        val maxArea = (MAX_AREA * w * h).toInt()
        val inside = BooleanArray(w * h)
        val queue = IntArray(maxArea + 1)
        var head = 0
        var tail = 0
        var x0 = seedX
        var x1 = seedX
        var y0 = seedY
        var y1 = seedY
        inside[seedY * w + seedX] = true
        queue[tail++] = seedY * w + seedX
        while (head < tail) {
            val i = queue[head++]
            val px = i % w
            val py = i / w
            // Open to the page's edge: a gutter or a borderless panel, not a bubble.
            if (px == 0 || py == 0 || px == w - 1 || py == h - 1) return null
            if (px < x0) x0 = px
            if (px > x1) x1 = px
            if (py < y0) y0 = py
            if (py > y1) y1 = py
            for (n in intArrayOf(i - 1, i + 1, i - w, i + w)) {
                if (inside[n] || (map.luma[n].toInt() and 0xFF) < paper) continue
                if (tail >= maxArea) return null
                inside[n] = true
                queue[tail++] = n
            }
        }
        val paperArea = tail

        val bw = x1 - x0 + 1
        val bh = y1 - y0 + 1
        if (bw < 0.03f * w || bh < 0.02f * h || paperArea < 200) return null

        // Fill the lettering in: everything in the bounds the outside can't reach is bubble.
        val outW = bw + 2 * (OUTLINE + 1)
        val outH = bh + 2 * (OUTLINE + 1)
        val ox = x0 - (OUTLINE + 1)
        val oy = y0 - (OUTLINE + 1)
        val shape = BooleanArray(outW * outH)
        for (sy in 0 until outH) for (sx in 0 until outW) {
            val mx = ox + sx
            val my = oy + sy
            if (mx in 0 until w && my in 0 until h && inside[my * w + mx]) shape[sy * outW + sx] = true
        }
        val outside = BooleanArray(outW * outH)
        val q2 = IntArray(outW * outH)
        head = 0
        tail = 0
        fun reach(i: Int) {
            if (!shape[i] && !outside[i]) {
                outside[i] = true
                q2[tail++] = i
            }
        }
        for (sx in 0 until outW) {
            reach(sx)
            reach((outH - 1) * outW + sx)
        }
        for (sy in 0 until outH) {
            reach(sy * outW)
            reach(sy * outW + outW - 1)
        }
        while (head < tail) {
            val i = q2[head++]
            val sx = i % outW
            val sy = i / outW
            if (sx > 0) reach(i - 1)
            if (sx < outW - 1) reach(i + 1)
            if (sy > 0) reach(i - outW)
            if (sy < outH - 1) reach(i + outW)
        }
        var filledArea = 0
        for (i in shape.indices) if (!outside[i]) filledArea++

        // Round enough to be a bubble, small enough not to be a panel, and with something
        // written in it - but only lettering, which is sparse and in small pieces. A panel's
        // artwork is dense, or has a large piece.
        if (filledArea < 0.4f * bw * bh || filledArea > maxArea) return null
        val ink = (filledArea - paperArea).toFloat() / filledArea
        if (ink < MIN_INK || ink > MAX_INK) return null
        if (largestMark(shape, outside, outW, outH) > MAX_MARK * filledArea) return null

        // Grow by the outline, square brush.
        val mask = ByteArray(outW * outH)
        for (sy in 0 until outH) for (sx in 0 until outW) {
            if (outside[sy * outW + sx]) continue
            for (dy in -OUTLINE..OUTLINE) for (dx in -OUTLINE..OUTLINE) {
                val tx = sx + dx
                val ty = sy + dy
                if (tx in 0 until outW && ty in 0 until outH) mask[ty * outW + tx] = 0xFF.toByte()
            }
        }
        return Found(ox, oy, outW, outH, mask)
    }

    /** Size of the largest connected mark inside the bubble: filled in, but not its paper. */
    private fun largestMark(paper: BooleanArray, outside: BooleanArray, w: Int, h: Int): Int {
        val seen = BooleanArray(w * h)
        val queue = IntArray(w * h)
        var largest = 0
        for (start in paper.indices) {
            if (paper[start] || outside[start] || seen[start]) continue
            var head = 0
            var tail = 0
            seen[start] = true
            queue[tail++] = start
            while (head < tail) {
                val i = queue[head++]
                val x = i % w
                val y = i / w
                for (n in intArrayOf(
                    if (x > 0) i - 1 else -1,
                    if (x < w - 1) i + 1 else -1,
                    if (y > 0) i - w else -1,
                    if (y < h - 1) i + w else -1,
                )) {
                    if (n < 0 || paper[n] || outside[n] || seen[n]) continue
                    seen[n] = true
                    queue[tail++] = n
                }
            }
            if (tail > largest) largest = tail
        }
        return largest
    }

    // Render thread only, from here down.

    private const val UNIFORM_SIZE = 64

    private val sampler by lazy {
        WebGpuRenderer.device.createSampler(
            GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
        )
    }

    private val byteBuffer = ByteBuffer.allocateDirect(UNIFORM_SIZE).order(ByteOrder.nativeOrder())

    private var zoomTexture: GPUTexture? = null
    private var zoomView: GPUTextureView? = null

    /** A surface-sized texture to draw the enlarged page into, kept while the size holds. */
    private fun zoomTarget(dst: GPUTexture): GPUTexture {
        zoomTexture?.let {
            if (it.width == dst.width && it.height == dst.height && it.format == dst.format) return it
        }
        releaseTarget()
        return WebGpuRenderer.device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(dst.width, dst.height),
                format = dst.format,
                usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding,
            )
        ).also {
            zoomTexture = it
            zoomView = it.createView()
        }
    }

    /** Frees the enlarged-page texture - on the render thread, once no bubble is open. */
    fun releaseTarget() {
        zoomView?.close()
        zoomTexture?.destroyAndRelease()
        zoomView = null
        zoomTexture = null
    }

    private val pipelines = FormatKeyed { format ->
        val device = WebGpuRenderer.device
        val module = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(SHADER))
        )
        // Premultiplied out, like the page it is drawn over.
        val over = GPUBlendComponent(
            srcFactor = BlendFactor.One,
            dstFactor = BlendFactor.OneMinusSrcAlpha,
            operation = BlendOperation.Add,
        )
        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(module, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    module, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = format, blend = GPUBlendState(color = over, alpha = over)
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    /**
     * Draws [bubble] over [dst], which already holds its page at rest: the page dimmed, then the
     * bubble growing from where it sits to where it opens, with a shadow under it.
     */
    fun draw(encoder: GPUCommandEncoder, dst: GPUTexture, bubble: BubbleOverlay, progress: Float) {
        if (bubble.released || progress <= 0f) return
        val page = bubble.page
        if (page.destroyed || page.scale <= 0f) return

        val t = progress.coerceAtLeast(0f)
        val zoom = 1f + (bubble.zoom - 1f) * t
        val cx = bubble.centerX + (bubble.targetX - bubble.centerX) * t
        val cy = bubble.centerY + (bubble.targetY - bubble.centerY) * t

        Draw.rect(encoder, dst, 0f, 0f, 1f, 1f, ((0.45f * t.coerceAtMost(1f) * 255).toInt() shl 24))

        // The page again, [zoom] times larger about the bubble, landing its centre on (cx, cy):
        // renderPage places at page.x + x scaled by page.scale * scale, so solve for x and y.
        val target = zoomTarget(dst)
        val targetView = target.createView()
        val zoomPass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = targetView,
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0),
                    )
                )
            )
        )
        try {
            val dx = (cx - 0.5f + zoom * (0.5f - bubble.centerX)) / (zoom * page.scale)
            val dy = (cy - 0.5f + zoom * (0.5f - bubble.centerY)) / (zoom * page.scale)
            page.renderPage(zoomPass, target, dx, dy, zoom, linear = true, masked = false)
        } finally {
            zoomPass.endAndRelease(targetView)
        }

        byteBuffer.clear()
        byteBuffer.putFloat(bubble.rect[0])
        byteBuffer.putFloat(bubble.rect[1])
        byteBuffer.putFloat(bubble.rect[2])
        byteBuffer.putFloat(bubble.rect[3])
        byteBuffer.putFloat(bubble.centerX)
        byteBuffer.putFloat(bubble.centerY)
        byteBuffer.putFloat(cx)
        byteBuffer.putFloat(cy)
        byteBuffer.putFloat(zoom)
        byteBuffer.putFloat(t.coerceAtMost(1f))
        byteBuffer.putFloat(dst.width.toFloat() / dst.height)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.flip()
        val device = WebGpuRenderer.device
        val uniforms = device.createBuffer(
            GPUBufferDescriptor(
                size = UNIFORM_SIZE.toLong(), usage = BufferUsage.Uniform or BufferUsage.CopyDst
            )
        )
        device.queue.writeBuffer(uniforms, 0, byteBuffer)

        val dstView = dst.createView()
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dstView,
                        loadOp = LoadOp.Load,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0),
                    )
                )
            )
        )
        try {
            val pipeline = pipelines[dst.format]
            pass.setPipeline(pipeline)
            pass.setTransientBindGroup(
                0, device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, buffer = uniforms),
                            GPUBindGroupEntry(1, textureView = zoomView!!),
                            GPUBindGroupEntry(2, textureView = bubble.maskView()),
                            GPUBindGroupEntry(3, sampler = sampler),
                        )
                    )
                )
            )
            pass.draw(6)
        } finally {
            pass.endAndRelease(dstView)
            uniforms.close()
        }
    }

    private const val SHADER = """
struct Uniforms {
    // The bubble's bounds on screen at rest.
    rect: vec4<f32>,
    // Its centre at rest, and where that centre is now.
    center: vec4<f32>,
    // Zoom now, how open (0-1), surface width over height, unused.
    params: vec4<f32>,
    unused: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var zoomed: texture_2d<f32>;
@group(0) @binding(2) var shape: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let pos = corners[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
    out.uv = pos;
    return out;
}

/// How much of the bubble covers screen point [uv] - traced back through the zoom to the page at
/// rest, where the shape was found. Sharpened, since the shape is far coarser than the screen.
fn coverage(uv: vec2<f32>) -> f32 {
    let rest = u.center.xy + (uv - u.center.zw) / u.params.x;
    let m = (rest - u.rect.xy) / (u.rect.zw - u.rect.xy);
    if (m.x < 0.0 || m.y < 0.0 || m.x > 1.0 || m.y > 1.0) { return 0.0; }
    return smoothstep(0.3, 0.7, textureSampleLevel(shape, samp, m, 0.0).r);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let open = u.params.y;
    let cover = coverage(in.uv);

    // A soft shadow below the bubble, lengthening as it lifts.
    let drop = vec2<f32>(0.0, 0.012 * open);
    let spread = vec2<f32>(0.01, 0.01 * u.params.z) * open;
    var shadow = 0.0;
    shadow += coverage(in.uv - drop);
    shadow += coverage(in.uv - drop + vec2<f32>(spread.x, 0.0));
    shadow += coverage(in.uv - drop - vec2<f32>(spread.x, 0.0));
    shadow += coverage(in.uv - drop + vec2<f32>(0.0, spread.y));
    shadow += coverage(in.uv - drop - vec2<f32>(0.0, spread.y));
    let shadow_alpha = 0.5 * open * shadow / 5.0;

    let page = textureSampleLevel(zoomed, samp, in.uv, 0.0);
    let front = page * cover;
    let alpha = front.a + (1.0 - front.a) * shadow_alpha;
    if (alpha <= 0.001) { discard; }
    return vec4<f32>(front.rgb, alpha);
}
"""
}
