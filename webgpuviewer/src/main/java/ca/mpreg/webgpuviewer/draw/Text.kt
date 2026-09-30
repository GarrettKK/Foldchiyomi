package ca.mpreg.webgpuviewer.draw

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.createFontFamilyResolver
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUOrigin3D
import androidx.webgpu.GPUPrimitiveState
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
import androidx.webgpu.GPUVertexAttribute
import androidx.webgpu.GPUVertexBufferLayout
import androidx.webgpu.GPUVertexState
import androidx.webgpu.PrimitiveTopology
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.VertexFormat
import androidx.webgpu.VertexStepMode
import ca.mpreg.webgpuviewer.draw.Font.Companion.FIXED_RASTER_SIZE
import ca.mpreg.webgpuviewer.draw.Font.Companion.forFamily
import ca.mpreg.webgpuviewer.draw.Font.Companion.invoke
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

private val device get() = WebGpuRenderer.device

/**
 * A signed distance field font atlas - either a true multi-channel one, as produced by
 * msdf-atlas-gen's `-json` output alongside its atlas image
 * (https://github.com/Chlumsky/msdf-atlas-gen), or a single-channel one built on-device from a
 * Compose [FontFamily] (see [forFamily]). [Draw.text] samples either the same way, since a
 * single-channel value replicated into all 3 channels medians to itself.
 *
 * Doesn't chunk the initial atlas upload the way [ca.mpreg.webgpuviewer.renderer.Mipmap] does for
 * page images - a font atlas is expected to stay well under a single `writeTexture` call's limit.
 */
class Font private constructor(
    private var atlasTexture: GPUTexture,
    internal var atlasWidth: Int,
    internal var atlasHeight: Int,
    // Only set for a forFamily(...) font: lets ensureGlyph() rasterize+upload a missing glyph on
    // demand. Null for an invoke(...) (msdf-atlas-gen) font - that atlas is fixed, generated
    // offline, with no on-device rasterizer able to add to it.
    private val rasterizer: Rasterizer?,
    private val glyphs: MutableMap<Int, Glyph>,
    private val kerningPairs: Map<Long, Float>,
    internal val distanceRange: Float,
    internal val atlasFontSize: Float,
    /** Line-to-line baseline advance, in em units (multiply by draw [size] for pixels). */
    val lineHeight: Float,
    val ascender: Float,
    val descender: Float,
) {
    /**
     * One glyph's metrics. Plane bounds are in em units, relative to the glyph's own origin;
     * atlas bounds are raw atlas pixels (not normalised) so growing the atlas - which only ever
     * appends, never moves existing pixels - never invalidates an already-placed glyph.
     */
    internal class Glyph(
        val advance: Float,
        val planeLeft: Float = 0f,
        val planeBottom: Float = 0f,
        val planeRight: Float = 0f,
        val planeTop: Float = 0f,
        val atlasX: Int = 0,
        val atlasY: Int = 0,
        val atlasW: Int = 0,
        val atlasH: Int = 0,
        /** False for glyphs with no visible quad (e.g. space) - [Draw.text] skips the draw call. */
        val hasQuad: Boolean = false,
    )

    /** Live state for growing a [forFamily] font's atlas as new glyphs are first requested. */
    private class Rasterizer(
        val paint: Paint,
        val rasterSize: Float,
        val padding: Int,
        // CPU mirror of the atlas - needed to carry existing pixels into a regrown, bigger
        // texture (writeTexture has no texture-to-texture path here).
        var pixels: ByteArray,
        var shelfX: Int,
        var shelfY: Int,
        var shelfHeight: Int,
    )

    internal var atlasView: GPUTextureView = atlasTexture.createView()
        private set

    /**
     * [codepoint]'s glyph, rasterizing and uploading it first if this is a [forFamily] font that
     * doesn't have it yet. Null (permanently, for an [invoke] font) only if [codepoint] has no
     * glyph and can't ever get one.
     */
    @Synchronized
    internal fun ensureGlyph(codepoint: Int): Glyph? {
        glyphs[codepoint]?.let { return it }
        val rasterizer = rasterizer ?: return null
        val glyph = addGlyph(
            rasterizer,
            rasterizeGlyph(rasterizer.paint, codepoint, rasterizer.rasterSize, rasterizer.padding)
        )
        glyphs[codepoint] = glyph
        return glyph
    }

    private fun addGlyph(r: Rasterizer, raster: GlyphRaster): Glyph {
        if (!raster.hasQuad) return Glyph(advance = raster.advanceEm)

        if (r.shelfX + raster.width > atlasWidth) {
            r.shelfY += r.shelfHeight
            r.shelfX = 0
            r.shelfHeight = 0
        }
        val neededWidth = max(atlasWidth, raster.width)
        val neededHeight = max(atlasHeight, r.shelfY + raster.height)
        if (neededWidth > atlasWidth || neededHeight > atlasHeight) {
            growAtlas(r, neededWidth, max(neededHeight, atlasHeight * 2))
        }

        val originX = r.shelfX
        val originY = r.shelfY
        r.shelfX += raster.width
        r.shelfHeight = max(r.shelfHeight, raster.height)

        val tile = renderGlyphTile(raster, distanceRange)
        blitTile(r.pixels, atlasWidth, tile, raster.width, raster.height, originX, originY)
        device.queue.writeTexture(
            dataLayout = GPUTexelCopyBufferLayout(
                offset = 0L, bytesPerRow = raster.width * 4, rowsPerImage = raster.height
            ),
            data = tile.toDirectBuffer(),
            destination = GPUTexelCopyTextureInfo(
                texture = atlasTexture, origin = GPUOrigin3D(x = originX, y = originY)
            ),
            writeSize = GPUExtent3D(raster.width, raster.height),
        )

        return Glyph(
            advance = raster.advanceEm,
            planeLeft = raster.planeLeft,
            planeBottom = raster.planeBottom,
            planeRight = raster.planeRight,
            planeTop = raster.planeTop,
            atlasX = originX,
            atlasY = originY,
            atlasW = raster.width,
            atlasH = raster.height,
            hasQuad = true,
        )
    }

    /** Grows the atlas to at least [newWidth] x [newHeight] - existing glyphs keep their pixel offsets, so nothing about them needs recomputing. */
    private fun growAtlas(r: Rasterizer, newWidth: Int, newHeight: Int) {
        val newPixels = ByteArray(newWidth * newHeight * 4)
        for (row in 0 until atlasHeight) {
            System.arraycopy(
                r.pixels,
                row * atlasWidth * 4,
                newPixels,
                row * newWidth * 4,
                atlasWidth * 4
            )
        }
        r.pixels = newPixels
        atlasWidth = newWidth
        atlasHeight = newHeight

        // Closed, not destroyed: this frame may have recorded it.
        atlasView.close()
        atlasTexture.close()
        atlasTexture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(newWidth, newHeight),
                format = TextureFormat.RGBA8Unorm,
                usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
            )
        )
        device.queue.writeTexture(
            dataLayout = GPUTexelCopyBufferLayout(
                offset = 0L,
                bytesPerRow = newWidth * 4,
                rowsPerImage = newHeight
            ),
            data = r.pixels.toDirectBuffer(),
            destination = GPUTexelCopyTextureInfo(texture = atlasTexture),
            writeSize = GPUExtent3D(newWidth, newHeight),
        )
        atlasView = atlasTexture.createView()
    }

    internal fun kerning(first: Int, second: Int): Float =
        kerningPairs[(first.toLong() shl 32) or (second.toLong() and 0xFFFFFFFFL)] ?: 0f

    fun destroy() {
        atlasView.close()
        atlasTexture.destroyAndRelease()
    }

    companion object {
        /** Loads a font from an msdf-atlas-gen [json] layout and its already-decoded [bitmap] atlas. */
        operator fun invoke(bitmap: Bitmap, json: String): Font {
            val buf = ByteBuffer.allocateDirect(bitmap.byteCount)
            bitmap.copyPixelsToBuffer(buf)
            buf.rewind()
            return invoke(buf, bitmap.width, bitmap.height, json)
        }

        /** Loads a font from an msdf-atlas-gen [json] layout and raw RGBA8 [pixels]. */
        operator fun invoke(pixels: ByteBuffer, width: Int, height: Int, json: String): Font {
            val root = JSONObject(json)
            val atlas = root.getJSONObject("atlas")
            val metrics = root.optJSONObject("metrics")

            val distanceRange = atlas.optDouble("distanceRange", 4.0).toFloat()
            val atlasFontSize = atlas.optDouble("size", 32.0).toFloat()
            val yOriginBottom = atlas.optString("yOrigin", "bottom") != "top"

            val glyphs = HashMap<Int, Glyph>()
            root.optJSONArray("glyphs")?.let { array ->
                for (i in 0 until array.length()) {
                    val g = array.getJSONObject(i)
                    val unicode = g.getInt("unicode")
                    val advance = g.optDouble("advance", 0.0).toFloat()
                    val plane = g.optJSONObject("planeBounds")
                    val atlasBounds = g.optJSONObject("atlasBounds")
                    if (plane == null || atlasBounds == null) {
                        // No visible quad - e.g. space. Advance still applies.
                        glyphs[unicode] = Glyph(advance = advance)
                        continue
                    }
                    val left = atlasBounds.getDouble("left").toFloat()
                    val right = atlasBounds.getDouble("right").toFloat()
                    val bottom = atlasBounds.getDouble("bottom").toFloat()
                    val top = atlasBounds.getDouble("top").toFloat()
                    // atlasBounds are in atlas pixels with the tool's own y convention - flip to
                    // this texture's top-down origin if the atlas was generated y-up ("bottom"
                    // origin, msdf-atlas-gen's default).
                    val atlasY: Int
                    val atlasH: Int
                    if (yOriginBottom) {
                        atlasY = (height - top).roundToInt()
                        atlasH = (top - bottom).roundToInt()
                    } else {
                        atlasY = top.roundToInt()
                        atlasH = (bottom - top).roundToInt()
                    }
                    glyphs[unicode] = Glyph(
                        advance = advance,
                        planeLeft = plane.getDouble("left").toFloat(),
                        planeBottom = plane.getDouble("bottom").toFloat(),
                        planeRight = plane.getDouble("right").toFloat(),
                        planeTop = plane.getDouble("top").toFloat(),
                        atlasX = left.roundToInt(),
                        atlasY = atlasY,
                        atlasW = (right - left).roundToInt(),
                        atlasH = atlasH,
                        hasQuad = true,
                    )
                }
            }

            val kerningPairs = HashMap<Long, Float>()
            root.optJSONArray("kerning")?.let { array ->
                for (i in 0 until array.length()) {
                    val k = array.getJSONObject(i)
                    val first = k.getInt("unicode1")
                    val second = k.getInt("unicode2")
                    val advance = k.getDouble("advance").toFloat()
                    kerningPairs[(first.toLong() shl 32) or (second.toLong() and 0xFFFFFFFFL)] =
                        advance
                }
            }

            val texture = device.createTexture(
                GPUTextureDescriptor(
                    size = GPUExtent3D(width, height),
                    format = TextureFormat.RGBA8Unorm,
                    usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
                )
            )
            device.queue.writeTexture(
                dataLayout = GPUTexelCopyBufferLayout(
                    offset = 0L,
                    bytesPerRow = width * 4,
                    rowsPerImage = height,
                ),
                data = pixels,
                destination = GPUTexelCopyTextureInfo(texture = texture),
                writeSize = GPUExtent3D(width, height),
            )

            return Font(
                atlasTexture = texture,
                atlasWidth = width,
                atlasHeight = height,
                rasterizer = null,
                glyphs = glyphs,
                kerningPairs = kerningPairs,
                distanceRange = distanceRange,
                atlasFontSize = atlasFontSize,
                lineHeight = metrics?.optDouble("lineHeight", 1.2)?.toFloat() ?: 1.2f,
                ascender = metrics?.optDouble("ascender", 0.8)?.toFloat() ?: 0.8f,
                descender = metrics?.optDouble("descender", -0.2)?.toFloat() ?: -0.2f,
            )
        }

        private data class FamilyKey(
            val fontFamily: FontFamily,
            val weight: FontWeight,
            val style: FontStyle,
        )

        // Building a Font from a FontFamily rasterises + distance-transforms every requested
        // glyph - expensive enough that it must not repeat every call (e.g. every frame) for the
        // same (family, weight, style). Never evicted: a font tends to get reused for a whole
        // app's lifetime, so the memory is worth not stalling on a rebuild.
        private val familyCache = HashMap<FamilyKey, Font>()

        // Baked once, independent of any draw size - an msdf atlas stays crisp scaled well past
        // this in either direction (see TEXT_SHADER's screen_px_range), so there's no reason to
        // rebuild (or cache-key) a whole new atlas per on-screen size the way a plain raster font
        // would need to. Chosen as a size that stays sharp from small body text up through large
        // headings without the atlas ballooning.
        private const val FIXED_RASTER_SIZE = 64f

        /**
         * Builds (or reuses a cached) single-channel signed distance field atlas for [fontFamily]
         * at a [FIXED_RASTER_SIZE] resolution, resolved via Compose's [createFontFamilyResolver] -
         * unlike [invoke]'s msdf-atlas-gen atlas, this is generated on-device from the rasterised
         * glyphs (see [chamferDistance]), so it has no true multi-channel corner protection and no
         * kerning table, but needs no offline tool. [Draw.text]'s own `size` scales the result at
         * draw time - cheap, since it's just the quad/uniform math, not a re-rasterize - so this
         * never needs to bake more than once per (family, weight, style) no matter what sizes it's
         * later drawn at.
         *
         * [chars] is only the up-front set - the returned [Font] grows on demand (see
         * [ensureGlyph]) whenever [Draw.text] asks for a codepoint it doesn't have yet, so passing
         * a narrower [chars] just trades a slower first draw of an uncovered character for a
         * smaller initial atlas.
         */
        fun forFamily(
            context: Context,
            fontFamily: FontFamily,
            weight: FontWeight = FontWeight.Normal,
            style: FontStyle = FontStyle.Normal,
            chars: String = DEFAULT_CHARS,
        ): Font {
            val key = FamilyKey(fontFamily, weight, style)
            synchronized(familyCache) { familyCache[key]?.let { return it } }
            // Built outside the lock; racing builds keep the first.
            val font = buildFromFamily(context, fontFamily, weight, style, chars)
            val kept = synchronized(familyCache) { familyCache.getOrPut(key) { font } }
            if (kept !== font) font.destroy()
            return kept
        }

        private fun buildFromFamily(
            context: Context,
            fontFamily: FontFamily,
            weight: FontWeight,
            style: FontStyle,
            chars: String,
        ): Font {
            val resolver = createFontFamilyResolver(context)
            val typeface = resolver.resolve(fontFamily, weight, style).value as Typeface

            val rasterSize = FIXED_RASTER_SIZE
            // ~1/8th of the em size, the same ratio msdf-atlas-gen defaults to - wide enough for
            // the fragment shader's screen_px_range antialiasing skirt without ballooning padding.
            val distanceRange = (rasterSize / 8f).coerceAtLeast(2f)
            val padding = ceil(distanceRange).toInt() + 1

            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                this.typeface = typeface
                textSize = rasterSize
                color = -0x1 // opaque white; only the alpha mask is used
            }
            val fontMetrics = paint.fontMetrics

            val glyphRasters = chars.toSortedSet().map { ch ->
                rasterizeGlyph(paint, ch.code, rasterSize, padding)
            }

            val atlas = packAtlas(glyphRasters, distanceRange)

            val texture = device.createTexture(
                GPUTextureDescriptor(
                    size = GPUExtent3D(atlas.width, atlas.height),
                    format = TextureFormat.RGBA8Unorm,
                    usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
                )
            )
            device.queue.writeTexture(
                dataLayout = GPUTexelCopyBufferLayout(
                    offset = 0L,
                    bytesPerRow = atlas.width * 4,
                    rowsPerImage = atlas.height,
                ),
                data = atlas.pixels.toDirectBuffer(),
                destination = GPUTexelCopyTextureInfo(texture = texture),
                writeSize = GPUExtent3D(atlas.width, atlas.height),
            )

            val rasterizer = Rasterizer(
                paint = paint,
                rasterSize = rasterSize,
                padding = padding,
                pixels = atlas.pixels,
                shelfX = atlas.shelfX,
                shelfY = atlas.shelfY,
                shelfHeight = atlas.shelfHeight,
            )

            return Font(
                atlasTexture = texture,
                atlasWidth = atlas.width,
                atlasHeight = atlas.height,
                rasterizer = rasterizer,
                glyphs = atlas.glyphs.toMutableMap(),
                // Standard Paint has no public kerning-pair lookup - runtime SDF fonts draw
                // without kerning.
                kerningPairs = emptyMap(),
                distanceRange = distanceRange,
                atlasFontSize = rasterSize,
                lineHeight = (fontMetrics.descent - fontMetrics.ascent) / rasterSize,
                ascender = -fontMetrics.ascent / rasterSize,
                descender = -fontMetrics.descent / rasterSize,
            )
        }
    }
}

enum class TextAlign { Left, Center, Right }

private val DEFAULT_CHARS: String = (32..126).map { it.toChar() }.joinToString("")

// writeTexture's JNI binding needs a direct buffer to get a raw native pointer from - a plain
// ByteBuffer.wrap(array) is heap-backed and crashes the native side (see Mipmap's own uploads,
// which always allocateDirect for the same reason).
private fun ByteArray.toDirectBuffer(): ByteBuffer =
    ByteBuffer.allocateDirect(size).put(this).also { it.rewind() }

/** One rasterised glyph, not yet distance-transformed or placed in an atlas. */
private class GlyphRaster(
    val codepoint: Int,
    val width: Int,
    val height: Int,
    /** Row-major; true where the raster is "inside" the glyph's ink. */
    val mask: BooleanArray,
    val advanceEm: Float,
    val planeLeft: Float,
    val planeBottom: Float,
    val planeRight: Float,
    val planeTop: Float,
    val hasQuad: Boolean,
)

/** Rasterises one codepoint at [rasterSize] px, padded by [padding] px on every side. */
private fun rasterizeGlyph(
    paint: Paint,
    codepoint: Int,
    rasterSize: Float,
    padding: Int
): GlyphRaster {
    val str = String(Character.toChars(codepoint))
    val advancePx = paint.measureText(str)

    val bounds = Rect()
    paint.getTextBounds(str, 0, str.length, bounds)

    if (bounds.width() <= 0 || bounds.height() <= 0) {
        // No visible ink (e.g. space) - advance only, no quad/mask to build.
        return GlyphRaster(
            codepoint, 0, 0, BooleanArray(0), advancePx / rasterSize, 0f, 0f, 0f, 0f, false
        )
    }

    val width = bounds.width() + padding * 2
    val height = bounds.height() + padding * 2
    val originX = (padding - bounds.left).toFloat()
    val originY = (padding - bounds.top).toFloat()

    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    Canvas(bitmap).drawText(str, originX, originY, paint)
    val pixels = IntArray(width * height)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    bitmap.recycle()

    val mask = BooleanArray(pixels.size) { (pixels[it] ushr 24) and 0xFF >= 128 }

    return GlyphRaster(
        codepoint = codepoint,
        width = width,
        height = height,
        mask = mask,
        advanceEm = advancePx / rasterSize,
        planeLeft = (bounds.left - padding) / rasterSize,
        planeBottom = (-bounds.bottom - padding) / rasterSize,
        planeRight = (bounds.right + padding) / rasterSize,
        planeTop = (-bounds.top + padding) / rasterSize,
        hasQuad = true,
    )
}

/**
 * [raster]'s tile-local (not yet atlas-placed) RGBA8 pixels: [chamferDistance]'s signed result,
 * encoded into all three colour channels so the existing median-of-3 msdf fragment shader
 * reconstructs it identically to a true multi-channel atlas (median of three equal values is just
 * that value).
 */
private fun renderGlyphTile(raster: GlyphRaster, distanceRange: Float): ByteArray {
    val outsideDist = chamferDistance(raster.mask, raster.width, raster.height, seed = false)
    val insideDist = chamferDistance(raster.mask, raster.width, raster.height, seed = true)

    val tile = ByteArray(raster.width * raster.height * 4)
    for (i in raster.mask.indices) {
        val signedPx = if (raster.mask[i]) outsideDist[i] else -insideDist[i]
        val encoded = (0.5f + signedPx / distanceRange).coerceIn(0f, 1f)
        val channel = (encoded * 255f).roundToInt().coerceIn(0, 255).toByte()

        val pixelIndex = i * 4
        tile[pixelIndex] = channel
        tile[pixelIndex + 1] = channel
        tile[pixelIndex + 2] = channel
        tile[pixelIndex + 3] = -1 // 255
    }
    return tile
}

/** Copies [tile] ([tileWidth] x [tileHeight] RGBA8) into [dst] (row stride [dstWidth]) at ([originX], [originY]). */
private fun blitTile(
    dst: ByteArray,
    dstWidth: Int,
    tile: ByteArray,
    tileWidth: Int,
    tileHeight: Int,
    originX: Int,
    originY: Int
) {
    val rowBytes = tileWidth * 4
    for (row in 0 until tileHeight) {
        val srcOffset = row * rowBytes
        val dstOffset = ((originY + row) * dstWidth + originX) * 4
        System.arraycopy(tile, srcOffset, dst, dstOffset, rowBytes)
    }
}

private class PackedAtlas(
    val width: Int,
    val height: Int,
    val pixels: ByteArray,
    val glyphs: Map<Int, Font.Glyph>,
    // Final shelf-packing cursor, so a font built from this atlas can keep packing new glyphs
    // into it later instead of restarting from (0, 0) and overlapping what's already placed.
    val shelfX: Int,
    val shelfY: Int,
    val shelfHeight: Int,
)

/** Shelf-packs [rasters] into one atlas texture - see [renderGlyphTile] for the per-glyph pixels. */
private fun packAtlas(rasters: List<GlyphRaster>, distanceRange: Float): PackedAtlas {
    val visible = rasters.filter { it.hasQuad }.sortedByDescending { it.height }
    val totalArea = visible.sumOf { it.width * it.height }
    val width = max(64, ceil(sqrt(totalArea.toDouble()) * 1.3).toInt())

    var shelfX = 0
    var shelfY = 0
    var shelfHeight = 0
    val placedX = HashMap<Int, Int>()
    val placedY = HashMap<Int, Int>()
    for (raster in visible) {
        if (shelfX + raster.width > width) {
            shelfY += shelfHeight
            shelfX = 0
            shelfHeight = 0
        }
        placedX[raster.codepoint] = shelfX
        placedY[raster.codepoint] = shelfY
        shelfX += raster.width
        shelfHeight = max(shelfHeight, raster.height)
    }
    val height = max(1, shelfY + shelfHeight)

    val pixels = ByteArray(width * height * 4)
    val glyphs = HashMap<Int, Font.Glyph>()

    for (raster in rasters) {
        if (!raster.hasQuad) {
            glyphs[raster.codepoint] = Font.Glyph(advance = raster.advanceEm)
            continue
        }

        val originX = placedX.getValue(raster.codepoint)
        val originY = placedY.getValue(raster.codepoint)
        val tile = renderGlyphTile(raster, distanceRange)
        blitTile(pixels, width, tile, raster.width, raster.height, originX, originY)

        glyphs[raster.codepoint] = Font.Glyph(
            advance = raster.advanceEm,
            planeLeft = raster.planeLeft,
            planeBottom = raster.planeBottom,
            planeRight = raster.planeRight,
            planeTop = raster.planeTop,
            atlasX = originX,
            atlasY = originY,
            atlasW = raster.width,
            atlasH = raster.height,
            hasQuad = true,
        )
    }

    return PackedAtlas(width, height, pixels, glyphs, shelfX, shelfY, shelfHeight)
}

/**
 * Unsigned chamfer distance transform: for every pixel, its (approximate Euclidean)
 * distance to the nearest pixel where `mask == seed`. Two independent calls - one per seed value
 * - combine into a signed field in [renderGlyphTile], since chamfer distance itself is unsigned.
 * O(n) two-pass scan rather than a true (and pricier) Euclidean distance transform - the small
 * approximation error doesn't matter once it's re-thresholded through [TEXT_SHADER]'s
 * `screen_px_range` antialiasing.
 */
private fun chamferDistance(
    mask: BooleanArray,
    width: Int,
    height: Int,
    seed: Boolean
): FloatArray {
    val inf = 1e6f
    val dist = FloatArray(width * height) { if (mask[it] == seed) 0f else inf }
    val orthogonal = 1f
    val diagonal = 1.41421356f

    fun at(x: Int, y: Int) =
        if (x in 0 until width && y in 0 until height) dist[y * width + x] else inf

    for (y in 0 until height) for (x in 0 until width) {
        val i = y * width + x
        dist[i] = min(
            dist[i], min(
                min(at(x - 1, y) + orthogonal, at(x, y - 1) + orthogonal),
                min(at(x - 1, y - 1) + diagonal, at(x + 1, y - 1) + diagonal)
            )
        )
    }
    for (y in height - 1 downTo 0) for (x in width - 1 downTo 0) {
        val i = y * width + x
        dist[i] = min(
            dist[i], min(
                min(at(x + 1, y) + orthogonal, at(x, y + 1) + orthogonal),
                min(at(x + 1, y + 1) + diagonal, at(x - 1, y + 1) + diagonal)
            )
        )
    }
    return dist
}

private val pipelines = FormatKeyed { format ->
    val shaderModule = device.createShaderModule(
        GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(TEXT_SHADER))
    )
    device.createRenderPipeline(
        GPURenderPipelineDescriptor(
            vertex = GPUVertexState(
                module = shaderModule, entryPoint = "vs_main", buffers = arrayOf(
                    GPUVertexBufferLayout(
                        arrayStride = 32L,
                        stepMode = VertexStepMode.Instance,
                        attributes = arrayOf(
                            GPUVertexAttribute(
                                format = VertexFormat.Float32x4, offset = 0L, shaderLocation = 0
                            ),
                            GPUVertexAttribute(
                                format = VertexFormat.Float32x4, offset = 16L, shaderLocation = 1
                            ),
                        )
                    )
                )
            ),
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
            primitive = GPUPrimitiveState(topology = PrimitiveTopology.TriangleList)
        )
    )
}

// Linear filtering is safe (and standard) for an msdf atlas: bilinearly interpolating the
// encoded channel triplet still yields a locally-valid signed distance approximation.
private val sampler by lazy {
    device.createSampler(
        GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
    )
}

// One instance per glyph, its (dst_rect, uv_rect) coming from a per-instance vertex buffer rather
// than a storage buffer - see Draw.text: the whole string draws as a single instanced call
// instead of one draw (and one uniform buffer, one bind group) per glyph, which used to dominate
// the cost of drawing any non-trivial amount of text. A storage buffer would have been simpler
// (no vertex-buffer-layout boilerplate), but this adapter's maxStorageBuffersInVertexStage
// defaults to 0 - a per-adapter limit an app would have to opt into raising via requiredLimits at
// device creation - so a plain instanced vertex buffer (no such limit) is the portable choice.
private const val TEXT_SHADER = """
struct Params {
    color: vec4<f32>,
    screen_px_range: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) dst_rect: vec4<f32>,     // x1, y1, x2, y2 - normalised [0, 1] within the destination
    @location(1) uv_rect: vec4<f32>,      // u1, v1, u2, v2 - into the atlas texture
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let pos = positions[in.vertex_index];

    let x = mix(in.dst_rect.x, in.dst_rect.z, pos.x);
    let y = mix(in.dst_rect.y, in.dst_rect.w, pos.y);

    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(
        mix(in.uv_rect.x, in.uv_rect.z, pos.x),
        mix(in.uv_rect.y, in.uv_rect.w, pos.y)
    );
    return out;
}

fn median3(v: vec3<f32>) -> f32 {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let msdf = textureSample(atlas_tex, atlas_sampler, in.uv).rgb;
    var signed_dist = median3(msdf) - 0.5;
    signed_dist *= params.screen_px_range;
    let opacity = clamp(signed_dist + 0.5, 0.0, 1.0);
    return vec4<f32>(params.color.rgb, params.color.a * opacity);
}
"""

/**
 * Draws [text] into [pass] using [font]'s msdf atlas, staying crisp at any [size] despite the
 * atlas's own fixed resolution. [x]/[y] are in [dst]'s pixels: [y] is the first line's baseline,
 * [x] is the line's anchor edge, picked by [align]. `\n` starts a new line, advanced by
 * [Font.lineHeight] `* size`. A codepoint [font] doesn't have a glyph for yet is rasterized and
 * uploaded on the spot if [font] supports it (see [Font.forFamily]); otherwise it's skipped.
 *
 * One instanced draw call for the *whole string* (see [TEXT_SHADER]) rather than one draw call
 * per glyph - sets its own pipeline, so the caller must set theirs again before drawing something
 * else.
 *
 * Each `\n`-delimited line is word-wrapped to [maxWidth] pixels first (default: unbounded, so no
 * wrapping) - a lone word wider than [maxWidth] is hard-broken by character rather than left
 * overflowing, since there's no narrower unit to break it into.
 */
fun Draw.text(
    pass: GPURenderPassEncoder,
    dst: GPUTexture,
    font: Font,
    text: String,
    x: Float,
    y: Float,
    size: Float,
    color: Int,
    align: TextAlign = TextAlign.Left,
    maxWidth: Float = Float.POSITIVE_INFINITY,
) {
    if (text.isEmpty()) return

    val screenPxRange = (size / font.atlasFontSize) * font.distanceRange
    val dstWidth = dst.width.toFloat()
    val dstHeight = dst.height.toFloat()

    // 8 floats (dst_rect + uv_rect) per glyph - collected first so the whole string can go into
    // one storage buffer and one draw call instead of one of each per glyph.
    val instances = GlyphInstances(text.length)

    var penY = y
    for (rawLine in text.split("\n")) for (line in wrapLine(font, rawLine, size, maxWidth)) {
        val codepoints = codepointsOf(line)
        val startX = x - when (align) {
            TextAlign.Left -> 0f
            TextAlign.Center -> lineAdvance(font, codepoints) * size / 2f
            TextAlign.Right -> lineAdvance(font, codepoints) * size
        }

        var penX = startX
        var prev = -1
        for (cp in codepoints) {
            if (prev >= 0) penX += font.kerning(prev, cp) * size
            val glyph = font.ensureGlyph(cp)
            if (glyph != null) {
                if (glyph.hasQuad) {
                    addGlyphInstance(instances, font, glyph, penX, penY, size, dstWidth, dstHeight)
                }
                penX += glyph.advance * size
            }
            prev = cp
        }

        penY += font.lineHeight * size
    }

    if (instances.count == 0) return
    drawGlyphInstances(pass, dst.format, font, instances, color, screenPxRange)
}

/**
 * As the main [Draw.text], but resolves [fontFamily] (via Compose's [createFontFamilyResolver])
 * into a [Font] itself - see [Font.forFamily] for how that's built/cached. [size] doubles as the
 * atlas's bake resolution, so text drawn at other sizes through the same cached [Font] won't be
 * quite as crisp as this call's own [size].
 */
fun Draw.text(
    pass: GPURenderPassEncoder,
    dst: GPUTexture,
    context: Context,
    fontFamily: FontFamily,
    text: String,
    x: Float,
    y: Float,
    size: Float,
    color: Int,
    weight: FontWeight = FontWeight.Normal,
    style: FontStyle = FontStyle.Normal,
    align: TextAlign = TextAlign.Left,
    maxWidth: Float = Float.POSITIVE_INFINITY,
) {
    val font = forFamily(context, fontFamily, weight, style)
    text(pass, dst, font, text, x, y, size, color, align, maxWidth)
}

private fun codepointsOf(line: String): IntArray {
    val out = IntArray(line.codePointCount(0, line.length))
    var i = 0
    var j = 0
    while (i < line.length) {
        val cp = line.codePointAt(i)
        out[j++] = cp
        i += Character.charCount(cp)
    }
    return out
}

private fun lineAdvance(font: Font, codepoints: IntArray): Float {
    var width = 0f
    var prev = -1
    for (cp in codepoints) {
        if (prev >= 0) width += font.kerning(prev, cp)
        width += font.ensureGlyph(cp)?.advance ?: 0f
        prev = cp
    }
    return width
}

/** Splits [line] into pixel-width-[maxWidth]-or-narrower sublines, breaking at spaces. */
private fun wrapLine(font: Font, line: String, size: Float, maxWidth: Float): List<String> {
    if (!maxWidth.isFinite() || maxWidth <= 0f) return listOf(line)

    val spaceAdvance = (font.ensureGlyph(' '.code)?.advance ?: 0f) * size
    val result = mutableListOf<String>()
    var current = StringBuilder()
    var currentWidth = 0f

    fun flushCurrent() {
        if (current.isNotEmpty()) {
            result.add(current.toString())
            current = StringBuilder()
            currentWidth = 0f
        }
    }

    for (word in line.split(" ")) {
        if (word.isEmpty()) continue
        val wordWidth = lineAdvance(font, codepointsOf(word)) * size

        if (wordWidth > maxWidth) {
            // Doesn't fit on a line by itself even alone - hard-break by character instead of
            // overflowing [maxWidth].
            flushCurrent()
            var prev = -1
            for (cp in codepointsOf(word)) {
                val advance = (font.ensureGlyph(cp)?.advance ?: 0f) * size
                var kern = if (prev >= 0) font.kerning(prev, cp) * size else 0f
                if (current.isNotEmpty() && currentWidth + kern + advance > maxWidth) {
                    flushCurrent()
                    kern = 0f
                    prev = -1
                }
                current.appendCodePoint(cp)
                currentWidth += kern + advance
                prev = cp
            }
            continue
        }

        val withSpace =
            if (current.isEmpty()) wordWidth else currentWidth + spaceAdvance + wordWidth
        if (current.isNotEmpty() && withSpace > maxWidth) flushCurrent()
        if (current.isNotEmpty()) {
            current.append(' ')
            currentWidth += spaceAdvance
        }
        current.append(word)
        currentWidth += wordWidth
    }

    flushCurrent()
    if (result.isEmpty()) result.add("")
    return result
}

/**
 * Per glyph, normalised `dst_rect` then `uv_rect` in atlas pixels: a later glyph can grow the
 * atlas, so UVs are normalised at upload against the atlas bound.
 */
private class GlyphInstances(glyphs: Int) {
    var data = FloatArray(maxOf(glyphs, 1) * 8)
    var count = 0

    fun add(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        u1: Float,
        v1: Float,
        u2: Float,
        v2: Float
    ) {
        if ((count + 1) * 8 > data.size) data = data.copyOf(data.size * 2)
        val i = count * 8
        data[i] = x1; data[i + 1] = y1; data[i + 2] = x2; data[i + 3] = y2
        data[i + 4] = u1; data[i + 5] = v1; data[i + 6] = u2; data[i + 7] = v2
        count++
    }
}

private fun addGlyphInstance(
    instances: GlyphInstances,
    font: Font,
    glyph: Font.Glyph,
    penX: Float,
    baselineY: Float,
    size: Float,
    dstWidth: Float,
    dstHeight: Float,
) {
    // Plane bounds are y-up around the baseline (top positive); dst is y-down pixels.
    val x1 = (penX + glyph.planeLeft * size) / dstWidth
    val x2 = (penX + glyph.planeRight * size) / dstWidth
    val y1 = (baselineY - glyph.planeTop * size) / dstHeight
    val y2 = (baselineY - glyph.planeBottom * size) / dstHeight

    instances.add(
        x1, y1, x2, y2,
        glyph.atlasX.toFloat(), glyph.atlasY.toFloat(),
        (glyph.atlasX + glyph.atlasW).toFloat(), (glyph.atlasY + glyph.atlasH).toFloat(),
    )
}

// Reused per thread. writeBuffer sends the whole capacity: pass slice().
private val textStaging = ThreadLocal.withInitial {
    ByteBuffer.allocateDirect(4096).order(ByteOrder.nativeOrder())
}

private fun textStaging(size: Int): ByteBuffer {
    var b = textStaging.get()
    if (b.capacity() < size) {
        b = ByteBuffer.allocateDirect(Integer.highestOneBit(size - 1) shl 1)
            .order(ByteOrder.nativeOrder())
        textStaging.set(b)
    }
    b.clear()
    return b
}

/**
 * Uploads [instances] (8 floats per glyph: `dst_rect` then `uv_rect`) into one per-instance
 * vertex buffer and draws every glyph with a single instanced [GPURenderPassEncoder.draw] call,
 * rather than one buffer/bind group/draw call per glyph - see [TEXT_SHADER].
 */
private fun drawGlyphInstances(
    pass: GPURenderPassEncoder,
    /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
    format: Int,
    font: Font,
    instances: GlyphInstances,
    color: Int,
    screenPxRange: Float,
) {
    val glyphCount = instances.count
    val vertexSize = glyphCount * 8 * 4

    // Under ensureGlyph's lock, so another thread can't swap the atlas mid-bind.
    synchronized(font) {
        val atlasWidth = font.atlasWidth.toFloat()
        val atlasHeight = font.atlasHeight.toFloat()
        val vertexBytes = textStaging(vertexSize)
        val data = instances.data
        for (g in 0 until glyphCount) {
            val i = g * 8
            vertexBytes.putFloat(data[i]).putFloat(data[i + 1]).putFloat(data[i + 2])
                .putFloat(data[i + 3])
            vertexBytes.putFloat(data[i + 4] / atlasWidth).putFloat(data[i + 5] / atlasHeight)
            vertexBytes.putFloat(data[i + 6] / atlasWidth).putFloat(data[i + 7] / atlasHeight)
        }
        vertexBytes.flip()
        drawGlyphBatch(
            pass,
            format,
            font.atlasView,
            vertexBytes,
            vertexSize,
            glyphCount,
            color,
            screenPxRange
        )
    }
}

private fun drawGlyphBatch(
    pass: GPURenderPassEncoder,
    format: Int,
    atlasView: GPUTextureView,
    vertexBytes: ByteBuffer,
    vertexSize: Int,
    glyphCount: Int,
    color: Int,
    screenPxRange: Float,
) {
    val vertexBuffer = device.createBuffer(
        GPUBufferDescriptor(
            size = vertexSize.toLong(),
            usage = BufferUsage.Vertex or BufferUsage.CopyDst
        )
    )
    device.queue.writeBuffer(vertexBuffer, 0, vertexBytes.slice())

    val r = ((color shr 16) and 0xFF) / 255f
    val g = ((color shr 8) and 0xFF) / 255f
    val b = (color and 0xFF) / 255f
    val a = ((color ushr 24) and 0xFF) / 255f

    val paramsBytes = textStaging(32)
    paramsBytes.putFloat(r)
    paramsBytes.putFloat(g)
    paramsBytes.putFloat(b)
    paramsBytes.putFloat(a)
    paramsBytes.putFloat(screenPxRange)
    paramsBytes.putFloat(0f)
    paramsBytes.putFloat(0f)
    paramsBytes.putFloat(0f)
    paramsBytes.flip()
    val paramsBuffer = device.createBuffer(
        GPUBufferDescriptor(size = 32L, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
    )
    device.queue.writeBuffer(paramsBuffer, 0, paramsBytes.slice())

    val pipeline = pipelines[format]
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.setTransientBindGroup(
        0, device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.groupLayout(), entries = arrayOf(
                    GPUBindGroupEntry(0, buffer = paramsBuffer),
                    GPUBindGroupEntry(1, textureView = atlasView),
                    GPUBindGroupEntry(2, sampler = sampler),
                )
            )
        )
    )
    pass.draw(6, glyphCount)
    vertexBuffer.close()
    paramsBuffer.close()
}
