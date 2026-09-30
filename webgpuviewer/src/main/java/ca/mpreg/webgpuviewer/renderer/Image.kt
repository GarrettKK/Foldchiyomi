package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.ImageUtil
import ca.mpreg.webgpuviewer.Trim
import ca.mpreg.webgpuviewer.renderer.Image.Companion.invoke
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.round

const val BUFFER_SIZE = 96L

/**
 * An unapplied gain map, as [Image] wants it - fill from `ImageDecoder.DecodeResult.gainmap`.
 *
 * Its own type rather than the decoder's so this module keeps no dependency on it. See
 * [ImageUtil.applyGainmap] for what the fields mean and how they combine with the base.
 */
class GainmapInput(
    val pixels: ByteBuffer,
    val width: Int,
    val height: Int,
    val channels: Int,
    val gamma: FloatArray,
    val minContentBoost: FloatArray,
    val maxContentBoost: FloatArray,
    val offsetSdr: FloatArray,
    val offsetHdr: FloatArray,
) {
    /** Stops of headroom the map adds, from the largest [maxContentBoost]. */
    val headroomStops: Float
        get() = maxContentBoost.maxOrNull()?.takeIf { it > 1f }?.let { log2(it) } ?: 0f

    /** As [headroomStops], from the smallest [minContentBoost] - `Mmin` for [Hdr.peakWeight]. */
    val minHeadroomStops: Float
        get() = minContentBoost.minOrNull()?.takeIf { it > 0f }?.let { log2(it) } ?: 0f
}

class Image private constructor(
    val width: Int,
    val height: Int,
    /**
     * True when this image's tiles hold extended-sRGB half-float rather than 8-bit sRGB, i.e.
     * when HDR source pixels met a device that can present them. An image decoded on a device
     * that cannot is tone mapped at upload and reports false, having become an ordinary SDR one.
     *
     * Independent of whether HDR is on screen right now: tone mapping is irreversible, and this
     * image outlives any one frame. Presentation follows the count of these - see
     * [Hdr.retainHdrImage].
     */
    val isHdr: Boolean = false,
    /**
     * Stops of headroom above SDR white this image's pixels reach. Drives what the display is
     * asked for while it is loaded - see [Hdr.desiredHeadroomRatio] - so it is kept rather than
     * being only a decode-time argument.
     */
    val hdrHeadroom: Float = 0f,
) {

    var x: Float = 0f

    var y: Float = 0f

    var backgroundColor: Int = 0xFF000000.toInt()

    /**
     * Trim bounds detected from image content, or null if not trimmed.
     */
    var trim: Rect? = null

    companion object {
        suspend operator fun invoke(
            pixels: ByteBuffer, width: Int, height: Int,
            createMipMaps: Boolean = true,
            trimColors: List<FloatArray>? = null,
            trimThreshold: Float = 0.05f,
            backgroundColor: Int? = null,
            hdr: Boolean = false,
            hdrHeadroom: Float = 0f,
            gainmap: GainmapInput? = null,
        ): Image {
            require(width > 0 && height > 0) { "Image dimensions must be positive" }
            require(trimColors == null || trimColors.all { it.size >= 3 }) {
                "each trimColor must have at least 3 elements [r, g, b]"
            }

            // Everything HDR resolves to plain pixels here, so nothing downstream has to know
            // which kind arrived. An app can hand over HDR without checking whether the device
            // can show it.
            @Suppress("NAME_SHADOWING") var pixels = pixels
            var keepHdr = false
            var headroom = hdrHeadroom

            // A ceiling of 1.0 is a host asking for no headroom at all, and there is then nothing
            // to keep: the branches below fall through to the SDR ones, which is what makes that
            // an exact base rendition for a gain map and a real tone map for PQ or HLG, rather
            // than pixels scaled against a target they already sit above. Checked before the
            // await, which otherwise waits up to two seconds for a surface it will not use.
            val canHdr = (hdr || gainmap != null) &&
                    Hdr.presentPeak > 1f &&
                    Hdr.awaitSupportedByDevice()

            when {
                // A gain map is applied here rather than by the decoder: how much of it to use
                // is a display question, and the base has to survive intact for the SDR case.
                gainmap != null && canHdr -> {
                    pixels = withContext(Dispatchers.Default) {
                        ImageUtil.applyGainmap(
                            base = pixels,
                            width = width,
                            height = height,
                            gain = gainmap.pixels,
                            gainWidth = gainmap.width,
                            gainHeight = gainmap.height,
                            gainChannels = gainmap.channels,
                            gamma = gainmap.gamma,
                            minContentBoost = gainmap.minContentBoost,
                            maxContentBoost = gainmap.maxContentBoost,
                            offsetSdr = gainmap.offsetSdr,
                            offsetHdr = gainmap.offsetHdr,
                            // Scaled so the map's full boost lands on what is actually being
                            // presented, instead of wherever the file aimed.
                            weight = Hdr.peakWeight(
                                gainmap.minHeadroomStops,
                                gainmap.headroomStops
                            ),
                        )
                    }
                    keepHdr = true
                    // What the pixels reach, not what was asked for: a map that fits inside the
                    // ceiling is applied in full and never reaches it.
                    headroom = minOf(gainmap.headroomStops, log2(Hdr.presentPeak))
                }

                // No HDR to present, so the base is simply left alone - it already *is* the
                // file's SDR rendition, which is exact and free where tone mapping would be
                // neither. This is the payoff for the map arriving unapplied.
                gainmap != null -> headroom = 0f

                // PQ and HLG arrive normalised against SDR white and can reach far past any
                // panel. No map to weight, so the pixels themselves are scaled - against the
                // image's *measured* peak, not the format's, which is the difference between a
                // frame that fills the headroom and one five stops too dim.
                hdr && canHdr -> {
                    keepHdr = true
                    val peak = withContext(Dispatchers.Default) {
                        ImageUtil.scaleHdrPeakNative(pixels, width, height, Hdr.presentPeak)
                    }
                    headroom = log2(peak.coerceAtLeast(1f))
                }

                // Float pixels with nowhere to put them: tone map once, at upload.
                hdr -> {
                    pixels = withContext(Dispatchers.Default) {
                        ImageUtil.toneMapToSdr(pixels, width, height)
                    }
                    headroom = 0f
                }
            }

            // Which branch ran: a wrong-looking picture can't say whether HDR was applied,
            // refused, or never detected.
            if (hdr || gainmap != null) {
                Log.i(
                    "Renderer",
                    "HDR ${width}x$height hdr=$hdr gainmap=${gainmap != null} " +
                            "canHdr=$canHdr declared=${hdrHeadroom} stops " +
                            "-> keepHdr=$keepHdr headroom=$headroom stops target=${Hdr.presentPeak}"
                )
            }

            val image = Image(width, height, isHdr = keepHdr, hdrHeadroom = headroom)

            try {
                return finishImage(
                    image, pixels, width, height, keepHdr, headroom, createMipMaps,
                    trimColors, trimThreshold, backgroundColor,
                )
            } catch (e: Throwable) {
                // A cancelled decode (e.g. the viewer closing mid-load) still leaves the buffer
                // allocated above - nothing else holds a reference to release it.
                image.cleanup()
                throw e
            }
        }

        private suspend fun finishImage(
            image: Image,
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            keepHdr: Boolean,
            headroom: Float,
            createMipMaps: Boolean,
            trimColors: List<FloatArray>?,
            trimThreshold: Float,
            backgroundColor: Int?,
        ): Image {
            val tileFormat =
                if (keepHdr) TextureFormat.RGBA16Float else TextureFormat.RGBA8Unorm

            // Runs on a background dispatcher rather than as compute shaders - the GPU versions
            // would park on a buffer readback while holding the render thread, stalling every
            // queued frame (a stutter each time a page decodes).
            val (trim, background) = withContext(Dispatchers.Default) {
                measurePixels(
                    pixels,
                    width,
                    height,
                    keepHdr,
                    trimColors,
                    trimThreshold,
                    backgroundColor
                )
            }
            image.trim = trim
            background?.let { image.backgroundColor = it }

            val levels = listOf(Level(pixels, width, height, 1f)) +
                    if (createMipMaps) smallerLevels(
                        pixels,
                        width,
                        height,
                        keepHdr
                    ) else emptyList()

            // No render mutex: Mipmap.create yields between upload chunks so queued frames get
            // the thread back. Safe since the image isn't reachable from any page yet.
            WebGpuRenderer.onDispatcher { _ ->
                try {
                    for (level in levels) image.mipmaps.add(level.upload(tileFormat))
                } catch (e: Exception) {
                    Log.e("Renderer", "Error creating image", e)
                    image.mipmaps.forEach { it.cleanup() }
                    image.mipmaps.clear()
                    throw e
                }
            }

            if (keepHdr) Hdr.retainHdrImage(image, headroom)

            return image
        }

        /**
         * Trim and background colour for [pixels], null where none. Both read 8-bit sRGB, so kept
         * HDR is tone mapped first, only when needed.
         */
        private fun measurePixels(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            keepHdr: Boolean,
            trimColors: List<FloatArray>?,
            trimThreshold: Float,
            backgroundColor: Int?,
        ): Pair<Rect?, Int?> {
            var trim: Rect? = null
            var background = backgroundColor

            val trimWith = trimColors?.takeIf { it.isNotEmpty() }
            val sdrPixels = when {
                !keepHdr -> pixels
                trimWith != null || backgroundColor == null ->
                    ImageUtil.toneMapToSdr(pixels, width, height)

                else -> null
            }

            if (trimWith != null && sdrPixels != null) {
                // Find trim for each color and pick the smallest rect
                val rects = Trim.findAllCpu(sdrPixels, width, height, trimWith, trimThreshold)
                val best =
                    trimWith.zip(rects)
                        .minByOrNull { it.second.width().toLong() * it.second.height() }

                if (best != null) {
                    trim = best.second
                    // The winning trim colour, unless the caller named one.
                    if (background == null) {
                        val c = best.first
                        fun ch(v: Float) = (v * 255).toInt().coerceIn(0, 255)
                        background =
                            0xFF000000.toInt() or (ch(c[0]) shl 16) or (ch(c[1]) shl 8) or ch(c[2])
                    }
                }
            }

            // Probing the edges is only worth a pass when neither the caller nor trim has
            // already named a background colour.
            if (background == null && sdrPixels != null) {
                background = Trim.detectBackgroundCpu(sdrPixels, width, height, trimThreshold)
            }
            return trim to background
        }

        private const val TILESIZE = 2048

        private class Level(val pixels: ByteBuffer, val w: Int, val h: Int, val scale: Float) {
            suspend fun upload(format: Int) = Mipmap.create(pixels, w, h, scale, TILESIZE, format)
        }

        private suspend fun smallerLevels(
            pixels: ByteBuffer, width: Int, height: Int, keepHdr: Boolean,
        ): List<Level> {
            val levels = mutableListOf<Level>()
            var currentPixels = pixels
            var textureWidth = width
            var textureHeight = height
            var scale = 1f

            while (width * scale > TILESIZE || height * scale > TILESIZE) {
                scale /= 2
                val newWidth = floor(width * scale).toInt()
                val newHeight = floor(height * scale).toInt()
                if (newWidth < 1 || newHeight < 1) break
                Log.d("Renderer", "Create mipmap using CPU ${scale} ${newWidth} ${newHeight}")

                currentPixels = withContext(Dispatchers.Default) {
                    if (keepHdr) ImageUtil.resizeF16(currentPixels, textureWidth, textureHeight)
                    else ImageUtil.resize(currentPixels, textureWidth, textureHeight)
                }
                levels.add(Level(currentPixels, newWidth, newHeight, scale))
                textureWidth = newWidth
                textureHeight = newHeight
            }
            return levels
        }

        suspend operator fun invoke(width: Int, height: Int): Image {
            val image = Image(width, height)
            try {
                WebGpuRenderer.withContext { _ ->
                    try {
                        image.mipmaps.add(Mipmap(width, height))
                    } catch (e: Exception) {
                        Log.e("Renderer", "Error creating drawable image", e)
                        throw e
                    }
                }
                return image
            } catch (e: Throwable) {
                image.cleanup()
                throw e
            }
        }
    }

    /**
     * Rewrites this image in its own textures from [pixels], a full image of this size in its
     * texel format (half-float if [isHdr]) of which only [rect] changed. Smaller levels are
     * rebuilt whole. Chunked and yielding, so frames keep drawing. False if cleaned up part way.
     */
    suspend fun update(pixels: ByteBuffer, rect: Rect? = null): Boolean {
        requireFullImage(pixels)
        val levels = mipmaps.toList()
        val base = levels.firstOrNull() ?: return false
        val smaller =
            if (levels.size > 1) smallerLevels(pixels, width, height, isHdr) else emptyList()
        return WebGpuRenderer.onDispatcher { _ ->
            try {
                base.update(pixels, rect) &&
                        levels.drop(1).zip(smaller)
                            .all { (level, data) -> level.update(data.pixels) }
            } finally {
                contentVersion++
            }
        }
    }

    /** Bumped by [update]; part of the page's frameVersion, so a transition's cached copy re-seeds. */
    @Volatile
    var contentVersion: Int = 0
        private set

    /** Adds the smaller levels [invoke]'s createMipMaps makes, from [pixels] as in [update]. */
    suspend fun createMipMaps(pixels: ByteBuffer) {
        requireFullImage(pixels)
        val base = mipmaps.firstOrNull() ?: error("Image has no textures")
        if (mipmaps.size > 1) return
        val levels = smallerLevels(pixels, width, height, isHdr)
        if (levels.isEmpty()) return

        // Off the mutex: unreachable until added.
        val extra = mutableListOf<Mipmap>()
        try {
            WebGpuRenderer.onDispatcher { _ ->
                for (level in levels) extra.add(level.upload(base.format))
            }
            WebGpuRenderer.withContext { _ ->
                check(mipmaps.size == 1 && mipmaps[0] === base) { "Image was cleaned up" }
                mipmaps.addAll(extra)
            }
        } catch (e: Throwable) {
            if (extra.isNotEmpty() && mipmaps.none { it in extra }) {
                WebGpuRenderer.onDispatcher { _ -> extra.forEach { it.cleanup() } }
            }
            throw e
        }
    }

    /** A short buffer would have writeTexture read past it, and the native resize silently skip. */
    private fun requireFullImage(pixels: ByteBuffer) {
        val need = width.toLong() * height * (if (isHdr) 8 else 4)
        require(pixels.isDirect && pixels.capacity() >= need) {
            "pixels hold ${pixels.capacity()} B, ${width}x$height needs $need"
        }
    }

    /** Sets [trim] and [backgroundColor] from [pixels] as [invoke] would. */
    suspend fun measure(
        pixels: ByteBuffer,
        trimColors: List<FloatArray>? = null,
        trimThreshold: Float = 0.05f,
        backgroundColor: Int? = null,
    ) {
        val (newTrim, background) = withContext(Dispatchers.Default) {
            measurePixels(pixels, width, height, isHdr, trimColors, trimThreshold, backgroundColor)
        }
        WebGpuRenderer.withContext { _ ->
            trim = newTrim
            background?.let { this@Image.backgroundColor = it }
        }
    }

    private var _buffer: GPUBuffer? = WebGpuRenderer.device.createBuffer(
        GPUBufferDescriptor(size = BUFFER_SIZE, usage = BufferUsage.CopyDst or BufferUsage.Uniform)
    )

    val buffer: GPUBuffer
        get() = _buffer ?: error("Image buffer accessed after cleanup")

    val mipmaps: MutableList<Mipmap> = mutableListOf()

    /** Guards the [Hdr] count against a second [cleanup] on the same image. */
    @Volatile
    private var released = false

    /**
     * Idempotent, and separate from [cleanup] because that runs on the render dispatcher, which
     * a torn-down renderer never services - stranding the claim.
     */
    internal fun releaseHdr() {
        if (isHdr && !released) {
            released = true
            Hdr.releaseHdrImage(this)
        }
    }

    fun cleanup() {
        releaseHdr()
        mipmaps.forEach { it.cleanup() }
        mipmaps.clear()
        _buffer?.destroyAndRelease()
        _buffer = null
    }

    /**
     * Where this image's full extent lands in [dst], as normalised (x1, y1, x2, y2) surface
     * coordinates - the same placement [prepareForRender] resolves to, but without going through
     * a mip level or [Mipmap.getQuad]. For callers that only want geometry (a background rect,
     * [ca.mpreg.webgpuviewer.viewer.ImagePage.ImageSingle.pageRect]) with no reason to touch mip/tile
     * selection.
     */
    fun placement(dst: GPUTexture, x: Float, y: Float, scale: Float): FloatArray {
        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY
        val x1 = 0.5f + scale * (adjustedX - 0.5f * width / dst.width)
        val y1 = 0.5f + scale * (adjustedY - 0.5f * height / dst.height)
        return floatArrayOf(
            x1, y1, x1 + scale * width / dst.width, y1 + scale * height / dst.height
        )
    }

    class MipMapForDraw(
        val mipmap: Mipmap, val quad: Mipmap.Quad, val x: Float, val y: Float, val scale: Float
    )

    fun prepareForRender(dst: GPUTexture, x: Float, y: Float, scale: Float): MipMapForDraw? {
        if (mipmaps.isEmpty()) return null
        if (isHdr) Hdr.noteHdrDrawn(this, hdrHeadroom)

        var level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        // Scale alone isn't enough: getQuad only promises half a tile either side of the view
        // centre, so the viewport must fit in one tile's texels. A <=2x2 grid binds in one go
        // regardless, so only larger ones need checking.
        while (level < mipmaps.size - 1) {
            val m = mipmaps[level]
            if (m.tilesCols <= 2 && m.tilesRows <= 2) break

            // Source texels the viewport covers at this level.
            val visibleW = dst.width * m.scale / scale
            val visibleH = dst.height * m.scale / scale
            if (visibleW <= m.tilesize && visibleH <= m.tilesize) break

            level++
        }

        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        // View centre in this level's pixels: scale the level-0 offset by mipmap.scale before
        // adding the level's half-size, or the window lands up to 2^level too far out.
        val vx = round(-adjustedX * dst.width * mipmap.scale + mipmap.width / 2).toInt()
        val vy = round(-adjustedY * dst.height * mipmap.scale + mipmap.height / 2).toInt()

        val quad = mipmap.getQuad(vx, vy)

        return MipMapForDraw(
            mipmap,
            quad,
            (0.5f / scale + adjustedX) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width,
            (0.5f / scale + adjustedY) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height,
            scale / mipmap.scale
        )
    }

    /** One physical tile, already placed for a single draw call - see [prepareTilesForRender]. */
    class TileForDraw(
        val texture: GPUTexture,
        val view: GPUTextureView,
        val uniform: GPUBuffer,
        val x: Float,
        val y: Float,
        val scale: Float
    )

    /**
     * Every tile needed to cover the current viewport, each already placed for its own draw call
     * - the fast/plain paths' answer to [prepareForRender]'s fixed one-window quad, which can
     * silently drop content once the viewport needs more than that window covers. No coarse-level
     * guard is needed here since any viewport is just whichever tiles it happens to overlap.
     */
    fun prepareTilesForRender(
        dst: GPUTexture, x: Float, y: Float, scale: Float
    ): List<TileForDraw> {
        if (isHdr) Hdr.noteHdrDrawn(this, hdrHeadroom)
        if (mipmaps.isEmpty()) return emptyList()

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)
        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        // Same view-centre derivation as prepareForRender's vx/vy, kept unrounded since this is
        // now just a rect query rather than a single discrete window pick.
        val cx = -adjustedX * dst.width * mipmap.scale + mipmap.width / 2f
        val cy = -adjustedY * dst.height * mipmap.scale + mipmap.height / 2f
        val halfW = dst.width * mipmap.scale / (2f * scale)
        val halfH = dst.height * mipmap.scale / (2f * scale)

        return mipmap.tilesInRect(cx - halfW, cy - halfH, cx + halfW, cy + halfH).map { tile ->
            // Same reconstruction prepareForRender uses for quad.x/quad.y, evaluated at this
            // tile's own offset instead - the formula was already general, it just happened to
            // only ever be evaluated at one window's offset before.
            TileForDraw(
                tile.texture,
                tile.view,
                tile.uniform,
                (0.5f / scale + adjustedX) * mipmap.scale + (tile.x - 0.5f * mipmap.width) / dst.width,
                (0.5f / scale + adjustedY) * mipmap.scale + (tile.y - 0.5f * mipmap.height) / dst.height,
                scale / mipmap.scale
            )
        }
    }
}
