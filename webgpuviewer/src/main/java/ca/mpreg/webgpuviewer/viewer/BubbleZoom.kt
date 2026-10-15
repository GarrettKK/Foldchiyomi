package ca.mpreg.webgpuviewer.viewer

import android.util.Log
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
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
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
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.InkMap
import ca.mpreg.webgpuviewer.renderer.UpscalerArtCnn
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.renderer.submitAndRelease
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

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
    /** Screen pixels per image pixel at rest - how much detail the scan has to give. */
    val screenPerImagePixel: Float,
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

    /**
     * The bubble's part of the page, enlarged once and kept while it is open - see
     * [BubbleZoom.enlarge]. [sourceScale] is its pixels per screen pixel at rest, from the
     * bubble's top-left corner.
     */
    var source: GPUTexture? = null
    var sourceView: GPUTextureView? = null
    var sourceScale: Float = 1f

    /** On the render thread. */
    fun release() {
        released = true
        maskView?.close()
        maskTexture?.destroyAndRelease()
        maskView = null
        maskTexture = null
        sourceView?.close()
        source?.destroyAndRelease()
        sourceView = null
        source = null
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
    private const val MAX_INK = 0.45f

    /** Largest single mark inside a bubble, as a share of it - a letter or a line of them. */
    private const val MAX_MARK = 0.2f

    /**
     * How round a bubble is at least, measured once [CLOSE] has smoothed it: 1 for a circle,
     * about 0.65 with a long tail, under 0.4 for a spiky radio bubble or two joined by a neck.
     * Lettering laid over artwork, flooded through its white halo, stays near 0.2.
     */
    private const val MIN_COMPACTNESS = 0.33f

    /** Share of its bounds a bubble fills at least - two joined by a neck fill about a third. */
    private const val MIN_FILL = 0.3f

    /**
     * Map pixels the shape is closed over before it is measured or cut out: comic lettering
     * often runs into the outline, and each letter that touches it would otherwise bite a notch
     * out of the bubble - out of its outline length, and out of what is enlarged.
     */
    private const val CLOSE = 3

    /** Map pixels added around the bubble, so its own outline comes with it. */
    private const val OUTLINE = 2

    /** Room around the bubble's bounds for [CLOSE] and [OUTLINE] to grow into. */
    private const val PAD = 4

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
        val hit = imageAt(page, screenWidth, screenHeight, tapX, tapY) ?: return null
        val image = hit.image
        val imageRect = hit.rect
        // In a spread, a bubble cut by the seam is still a bubble: the side's inner edge bounds
        // it instead of disqualifying it.
        val openEdge = when {
            page !is ImagePage.ImageSpread -> NO_EDGE
            hit.offsetX < 0f -> RIGHT_EDGE
            else -> LEFT_EDGE
        }
        val map = image.inkMap ?: return null

        val u = (tapX - imageRect[0]) / (imageRect[2] - imageRect[0])
        val v = (tapY - imageRect[1]) / (imageRect[3] - imageRect[1])
        val found = find(
            map,
            (u * map.width).toInt().coerceIn(0, map.width - 1),
            (v * map.height).toInt().coerceIn(0, map.height - 1),
            openEdge,
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
            screenPerImagePixel = (imageRect[2] - imageRect[0]) * screenWidth / image.width,
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

    /** One of [page]'s images and where it sits on screen - see [imageAt]. */
    class Hit(val image: Image, val rect: FloatArray, val offsetX: Float)

    /**
     * The image of [page] under the tap at ([tapX], [tapY]), normalised screen coordinates on a
     * [screenWidth] x [screenHeight] surface, with its placement there; null off every image.
     */
    fun imageAt(
        page: ImagePage.ImageSingle,
        screenWidth: Int,
        screenHeight: Int,
        tapX: Float,
        tapY: Float,
    ): Hit? {
        var hit: Hit? = null
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
                hit = Hit(image, rect, offsetX)
            }
        }
        return hit
    }

    /** A bubble found on an [InkMap]: its bounds there, and its shape within them. */
    class Found(val x0: Int, val y0: Int, val width: Int, val height: Int, val mask: ByteArray)

    /** Which side of the map, if any, a bubble may run off - the seam side of a spread's page. */
    private const val NO_EDGE = 0
    private const val LEFT_EDGE = 1
    private const val RIGHT_EDGE = 2

    /**
     * The speech bubble around map pixel ([x], [y]): the bright paper there, flood-filled up to the
     * dark outline that closes it, its lettering filled in, its outline added back. Null for
     * anything that doesn't look like a bubble - open to the page's edge, too big, too thin, or
     * with nothing written in it.
     */
    fun find(map: InkMap, x: Int, y: Int, openEdge: Int = NO_EDGE): Found? {
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
            // Open to the page's edge: a gutter or a borderless panel, not a bubble - except at
            // the open edge, which simply bounds it.
            val atLeft = px == 0
            val atRight = px == w - 1
            if (py == 0 || py == h - 1) return null
            if ((atLeft && openEdge != LEFT_EDGE) || (atRight && openEdge != RIGHT_EDGE)) return null
            if (px < x0) x0 = px
            if (px > x1) x1 = px
            if (py < y0) y0 = py
            if (py > y1) y1 = py
            for (n in intArrayOf(if (atLeft) -1 else i - 1, if (atRight) -1 else i + 1, i - w, i + w)) {
                if (n < 0 || inside[n] || (map.luma[n].toInt() and 0xFF) < paper) continue
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
        val outW = bw + 2 * PAD
        val outH = bh + 2 * PAD
        val ox = x0 - PAD
        val oy = y0 - PAD
        val shape = BooleanArray(outW * outH)
        for (sy in 0 until outH) for (sx in 0 until outW) {
            val mx = ox + sx
            val my = oy + sy
            if (mx in 0 until w && my in 0 until h && inside[my * w + mx]) shape[sy * outW + sx] = true
        }
        val outside = outsideOf(shape, outW, outH)
        var filledArea = 0
        for (i in shape.indices) if (!outside[i]) filledArea++

        // Round enough to be a bubble, small enough not to be a panel, and with something
        // written in it - but only lettering, which is in small pieces. A panel's artwork is
        // dense, or has a large piece.
        if (filledArea < MIN_FILL * bw * bh || filledArea > maxArea) return null
        val ink = (filledArea - paperArea).toFloat() / filledArea
        if (ink < MIN_INK || ink > MAX_INK) return null
        if (largestMark(shape, outside, outW, outH) > MAX_MARK * filledArea) return null

        // Closed over the notches lettering bites out of it, then filled again.
        val solid = BooleanArray(outW * outH) { !outside[it] }
        val closed = grow(grow(solid, outW, outH, CLOSE, dilate = true), outW, outH, CLOSE, dilate = false)
        val closedOutside = outsideOf(closed, outW, outH)

        // Compact, as a bubble or a caption box is - not the ragged white halo of lettering laid
        // over artwork, which floods into a shape all edge. Counted edges overshoot a smooth
        // outline's length by about 4/pi, hence the correction; a circle comes out near 1.
        var closedArea = 0
        var edges = 0
        for (sy in 0 until outH) for (sx in 0 until outW) {
            val i = sy * outW + sx
            if (!closedOutside[i]) closedArea++
            if (sx > 0 && closedOutside[i] != closedOutside[i - 1]) edges++
            if (sy > 0 && closedOutside[i] != closedOutside[i - outW]) edges++
        }
        val compactness = 64f * closedArea / (Math.PI.toFloat() * edges * edges)
        if (compactness < MIN_COMPACTNESS) return null

        // Grown by the outline.
        val grown = grow(BooleanArray(outW * outH) { !closedOutside[it] }, outW, outH, OUTLINE, dilate = true)
        val mask = ByteArray(outW * outH) { if (grown[it]) 0xFF.toByte() else 0 }
        return Found(ox, oy, outW, outH, soften(mask, outW, outH))
    }

    /** What of a [w] x [h] frame can be reached from its border without crossing [shape]. */
    private fun outsideOf(shape: BooleanArray, w: Int, h: Int): BooleanArray {
        val outside = BooleanArray(w * h)
        val queue = IntArray(w * h)
        var head = 0
        var tail = 0
        fun reach(i: Int) {
            if (!shape[i] && !outside[i]) {
                outside[i] = true
                queue[tail++] = i
            }
        }
        for (x in 0 until w) {
            reach(x)
            reach((h - 1) * w + x)
        }
        for (y in 0 until h) {
            reach(y * w)
            reach(y * w + w - 1)
        }
        while (head < tail) {
            val i = queue[head++]
            val x = i % w
            val y = i / w
            if (x > 0) reach(i - 1)
            if (x < w - 1) reach(i + 1)
            if (y > 0) reach(i - w)
            if (y < h - 1) reach(i + w)
        }
        return outside
    }

    /**
     * [shape] dilated or eroded by a square of radius [r], one axis at a time. Past the frame
     * counts as empty, which [PAD] keeps clear of the shape.
     */
    private fun grow(shape: BooleanArray, w: Int, h: Int, r: Int, dilate: Boolean): BooleanArray {
        fun pass(src: BooleanArray, horizontal: Boolean): BooleanArray = BooleanArray(w * h) { i ->
            val x = i % w
            val y = i / w
            var any = false
            var all = true
            for (d in -r..r) {
                val v = if (horizontal) {
                    val tx = x + d
                    tx in 0 until w && src[y * w + tx]
                } else {
                    val ty = y + d
                    ty in 0 until h && src[ty * w + x]
                }
                if (v) any = true else all = false
            }
            if (dilate) any else all
        }
        return pass(pass(shape, horizontal = true), horizontal = false)
    }

    /**
     * Blurs the shape's hard map-pixel steps into a smooth field, so the edge drawn at its
     * midpoint is a curve rather than a staircase once the bubble is enlarged. Two box passes
     * each way, radius 2 - near enough a Gaussian.
     */
    private fun soften(mask: ByteArray, w: Int, h: Int): ByteArray {
        var src = FloatArray(mask.size) { (mask[it].toInt() and 0xFF) / 255f }
        val radius = 2
        repeat(2) {
            val horizontal = FloatArray(src.size)
            for (y in 0 until h) for (x in 0 until w) {
                var sum = 0f
                for (d in -radius..radius) sum += src[y * w + (x + d).coerceIn(0, w - 1)]
                horizontal[y * w + x] = sum / (2 * radius + 1)
            }
            val vertical = FloatArray(src.size)
            for (y in 0 until h) for (x in 0 until w) {
                var sum = 0f
                for (d in -radius..radius) sum += horizontal[(y + d).coerceIn(0, h - 1) * w + x]
                vertical[y * w + x] = sum / (2 * radius + 1)
            }
            src = vertical
        }
        return ByteArray(src.size) { (src[it] * 255f + 0.5f).toInt().coerceIn(0, 255).toByte() }
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

    /** How long after the reader opens to warm up - past its first page's decode. */
    private const val WARM_UP_DELAY_MS = 1500L

    /** How long to give ArtCNN's background compile before finishing it in the warm-up. */
    private const val ASYNC_COMPILE_TIMEOUT_MS = 4000L

    /** Largest side of what ArtCNN is given, halo included - keeps its feature maps small. */
    private const val MAX_INPUT = 512

    /**
     * ArtCNN's input side is rounded up to this, so bubbles of about the same size reuse its
     * working textures instead of allocating new ones - see [enlarge].
     */
    private const val INPUT_STEP = 64

    /** How long a bubble waits for its enlargement to finish on the GPU before it opens anyway. */
    private const val GPU_WAIT_TIMEOUT_MS = 1500L

    /** Largest side of the enlarged bubble when ArtCNN can't run and it is drawn directly. */
    private const val MAX_DIRECT = 2048

    private val sampler by lazy {
        WebGpuRenderer.device.createSampler(
            GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
        )
    }

    private val byteBuffer = ByteBuffer.allocateDirect(UNIFORM_SIZE).order(ByteOrder.nativeOrder())

    /**
     * Its own instance, not the tile renderer's: that one runs on the tile worker. Kept for the
     * session, so its compiled network outlives any one bubble - see [prewarm].
     */
    private val artCnn by lazy { UpscalerArtCnn() }

    /** Whether [warmUp] has run its tiny pass of ArtCNN's whole path. */
    private var warmed = false

    /**
     * Runs [block] while handing Dawn's events their callbacks: an `...AndAwait` call resolves
     * only when something processes events, and between frames nothing else does - unpumped,
     * ArtCNN's background compile never finished and the warm-up compiled it all over again,
     * holding the render thread for a second or two.
     */
    private suspend fun <R> pumped(block: suspend () -> R): R = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1)
            }
        }
        try {
            block()
        } finally {
            pump.cancel()
        }
    }

    /**
     * Readies everything the first bubble needs, shortly after the reader opens: ArtCNN's network
     * compiled in the background where the device allows it, then one tiny run of the whole path
     * - its textures, its resolve pipeline, the first dispatch a driver compiles lazily, and this
     * overlay's own pipeline. Whatever of that still blocks then lands while a page is being
     * read, not after a tap.
     */
    suspend fun prewarm() {
        delay(WARM_UP_DELAY_MS)
        withTimeoutOrNull(ASYNC_COMPILE_TIMEOUT_MS) { pumped { artCnn.prewarm() } }
        runCatching { WebGpuRenderer.withContext { device -> warmUp(device) } }
            .onFailure { Log.w("BubbleZoom", "Warm-up failed", it) }
    }

    private fun warmUp(device: GPUDevice) {
        val format = Hdr.frameFormat
        pipelines[format]
        val art = artCnn
        if (!art.supported || warmed) return
        val side = 64
        val input = art.input(side, format) ?: return
        val inputView = art.inputView ?: return
        val encoder = device.createCommandEncoder()
        clearedPass(encoder, inputView) {}
        art.encode(encoder, side)
        val out = texture((side - 2 * art.halo) * art.factor, format)
        val outView = out.createView()
        clearedPass(encoder, outView) { pass -> art.resolve(pass, format) }
        device.queue.submitAndRelease(encoder)
        // Freed once the GPU is done with it - only the compiling and first dispatches mattered.
        outView.close()
        out.destroyAndRelease()
        warmed = true
    }

    /**
     * Enlarges [bubble] before its first frame, on the render thread between frames, so the
     * opening animation runs smoothly from its first step instead of stalling a few frames in.
     */
    fun prepare(bubble: BubbleOverlay, screenWidth: Int, screenHeight: Int) {
        if (bubble.released || bubble.sourceView != null) return
        val device = WebGpuRenderer.device
        val format = Hdr.frameFormat
        val encoder = device.createCommandEncoder()
        enlarge(encoder, screenWidth, screenHeight, format, bubble)
        device.queue.submitAndRelease(encoder)
        bubble.maskView()
        pipelines[format]
    }

    /**
     * Waits, off the render lock, for the GPU to finish the enlargement [prepare] submitted. The
     * network takes a few hundred milliseconds on some phones: opened straight away, the
     * animation's frames queued behind it and it froze for that long just short of full size.
     */
    suspend fun awaitPrepared() {
        withTimeoutOrNull(GPU_WAIT_TIMEOUT_MS) {
            WebGpuRenderer.onDispatcher { device -> pumped { device.queue.onSubmittedWorkDone() } }
        }
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

    private fun texture(side: Int, format: Int): GPUTexture =
        WebGpuRenderer.device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(side, side),
                format = format,
                usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding,
            )
        )

    /** A cleared pass over [view], for [block] to draw into. */
    private fun clearedPass(
        encoder: GPUCommandEncoder, view: GPUTextureView, block: (GPURenderPassEncoder) -> Unit
    ) {
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = view,
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0),
                    )
                )
            )
        )
        try {
            block(pass)
        } finally {
            pass.endAndRelease()
        }
    }

    /**
     * Draws [page] into [dst] ([side] square) at [scale] times its size at rest, with the
     * bubble's top-left corner at pixel ([inset], [inset]). renderPage places the page at
     * `page.x + x` in units of the target's own size, so the offset is solved for that size.
     */
    private fun drawCrop(
        pass: GPURenderPassEncoder,
        dst: GPUTexture,
        side: Int,
        page: ImagePage.ImageSingle,
        bubble: BubbleOverlay,
        screenWidth: Int,
        screenHeight: Int,
        scale: Float,
        inset: Int,
    ) {
        val s0 = page.scale
        fun offset(restPx: Float, screen: Int, pagePos: Float, globalOffset: Float): Float =
            (inset - scale * restPx - 0.5f * side + 0.5f * scale * screen +
                    s0 * scale * (pagePos + globalOffset) * (screen - side)) / (s0 * scale * side)
        val dx = offset(bubble.rect[0] * screenWidth, screenWidth, page.x, WebGpuRenderer.offsetX)
        val dy = offset(bubble.rect[1] * screenHeight, screenHeight, page.y, WebGpuRenderer.offsetY)
        page.renderPage(pass, dst, dx, dy, scale, linear = true, masked = false)
    }

    /**
     * The bubble's part of the page, enlarged once for as long as it is open: drawn at the scan's
     * own resolution where the zoom asks for more than that, then doubled by ArtCNN - the network
     * the reader already uses to sharpen zoomed line art - so lettering gains real edges rather
     * than blur. Where ArtCNN can't run, the page is simply drawn at the full zoom.
     */
    private fun enlarge(
        encoder: GPUCommandEncoder, w: Int, h: Int, format: Int, bubble: BubbleOverlay
    ): GPUTextureView? {
        bubble.sourceView?.let { return it }
        val page = bubble.page
        val bubbleSide = max((bubble.rect[2] - bubble.rect[0]) * w, (bubble.rect[3] - bubble.rect[1]) * h)
        if (bubbleSide < 1f) return null

        val art = artCnn
        if (art.supported) {
            val halo = art.halo
            // Native resolution where the zoom reaches past it, and never less than half the zoom
            // - ArtCNN doubles - but small enough to keep its working textures modest.
            val native = 1f / bubble.screenPerImagePixel
            var scale = native.coerceIn(bubble.zoom / art.factor, bubble.zoom)
            scale = min(scale, (MAX_INPUT - 2 * halo) / bubbleSide)
            // Rounded up, and so its working textures - kept from one bubble to the next - are
            // reallocated only when a bubble needs a different size of them.
            val side = min(
                MAX_INPUT,
                ((bubbleSide * scale + 2 * halo).toInt() + INPUT_STEP - 1) / INPUT_STEP * INPUT_STEP,
            )
            val input = art.input(side, format)
            val inputView = art.inputView
            if (input != null && inputView != null) {
                clearedPass(encoder, inputView) { pass ->
                    drawCrop(pass, input, side, page, bubble, w, h, scale, halo)
                }
                art.encode(encoder, side)
                if (art.supported) {
                    val outSide = (side - 2 * halo) * art.factor
                    val out = texture(outSide, format)
                    val outView = out.createView()
                    clearedPass(encoder, outView) { pass -> art.resolve(pass, format) }
                    bubble.source = out
                    bubble.sourceView = outView
                    bubble.sourceScale = scale * art.factor
                    return outView
                }
            }
        }

        val scale = min(bubble.zoom, MAX_DIRECT / bubbleSide)
        val side = (bubbleSide * scale).toInt().coerceAtLeast(1)
        val out = texture(side, format)
        val outView = out.createView()
        clearedPass(encoder, outView) { pass -> drawCrop(pass, out, side, page, bubble, w, h, scale, 0) }
        bubble.source = out
        bubble.sourceView = outView
        bubble.sourceScale = scale
        return outView
    }

    /**
     * Draws [bubble] over [dst], which already holds its page at rest: the page dimmed, then the
     * bubble growing from where it sits to where it opens, cleaned up, with a shadow under it.
     */
    fun draw(encoder: GPUCommandEncoder, dst: GPUTexture, bubble: BubbleOverlay, progress: Float) {
        if (bubble.released || progress <= 0f) return
        val page = bubble.page
        if (page.destroyed || page.scale <= 0f) return
        val source = enlarge(encoder, dst.width, dst.height, dst.format, bubble) ?: return
        val sourceSide = bubble.source?.width ?: return

        val t = progress.coerceAtLeast(0f)
        val zoom = 1f + (bubble.zoom - 1f) * t
        val cx = bubble.centerX + (bubble.targetX - bubble.centerX) * t
        val cy = bubble.centerY + (bubble.targetY - bubble.centerY) * t

        Draw.rect(encoder, dst, 0f, 0f, 1f, 1f, ((0.45f * t.coerceAtMost(1f) * 255).toInt() shl 24))

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
        byteBuffer.putFloat(bubble.sourceScale)
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.putFloat(sourceSide.toFloat())
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
                            GPUBindGroupEntry(1, textureView = source),
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
    // Zoom now, how open (0-1), surface width over height, source pixels per screen pixel.
    params: vec4<f32>,
    // Surface width and height in pixels, the source's side, unused.
    size: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
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

/// Screen point [uv] traced back through the zoom to the page at rest, where the bubble was found.
fn at_rest(uv: vec2<f32>) -> vec2<f32> {
    return u.center.xy + (uv - u.center.zw) / u.params.x;
}

/// How much of the bubble covers screen point [uv]. Sharpened, since the shape is far coarser
/// than the screen.
fn coverage(uv: vec2<f32>) -> f32 {
    let m = (at_rest(uv) - u.rect.xy) / (u.rect.zw - u.rect.xy);
    if (m.x < 0.0 || m.y < 0.0 || m.x > 1.0 || m.y > 1.0) { return 0.0; }
    return smoothstep(0.4, 0.6, textureSampleLevel(shape, samp, m, 0.0).r);
}

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

/// The enlarged page at screen point [uv], cleaned up the way lettering wants: a light unsharp
/// mask for crisp strokes, then paper nudged toward white and ink toward black - partly, so
/// lettering keeps its anti-aliasing, and only where the page is grey, so colour is left alone.
fn page_at(uv: vec2<f32>) -> vec4<f32> {
    let px = (at_rest(uv) - u.rect.xy) * u.size.xy * u.params.w;
    let st = px / u.size.z;
    let d = 1.0 / u.size.z;
    let c = textureSampleLevel(source, samp, st, 0.0);
    let around = (textureSampleLevel(source, samp, st + vec2<f32>(d, 0.0), 0.0) +
        textureSampleLevel(source, samp, st - vec2<f32>(d, 0.0), 0.0) +
        textureSampleLevel(source, samp, st + vec2<f32>(0.0, d), 0.0) +
        textureSampleLevel(source, samp, st - vec2<f32>(0.0, d), 0.0)) * 0.25;
    if (c.a <= 0.0) { return c; }
    // Premultiplied in, straight for the grading.
    var rgb = clamp((c.rgb + 0.15 * (c.rgb - around.rgb)) / c.a, vec3<f32>(0.0), vec3<f32>(1.0));
    let y = luma(rgb);
    let chroma = max(max(rgb.r, rgb.g), rgb.b) - min(min(rgb.r, rgb.g), rgb.b);
    let grey = 1.0 - smoothstep(0.06, 0.18, chroma);
    let graded = vec3<f32>(smoothstep(0.06, 0.94, y));
    rgb = mix(rgb, graded, 0.6 * grey);
    return vec4<f32>(rgb * c.a, c.a);
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

    var front = vec4<f32>(0.0);
    if (cover > 0.0) { front = page_at(in.uv) * cover; }
    let alpha = front.a + (1.0 - front.a) * shadow_alpha;
    if (alpha <= 0.001) { discard; }
    return vec4<f32>(front.rgb, alpha);
}
"""
}
