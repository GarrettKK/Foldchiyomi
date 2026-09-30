package ca.mpreg.webgpuviewer

import ca.mpreg.webgpuviewer.ImageUtil.applyGainmapNative
import ca.mpreg.webgpuviewer.ImageUtil.resize
import ca.mpreg.webgpuviewer.ImageUtil.resizeLinearAreaNative
import ca.mpreg.webgpuviewer.ImageUtil.rotateQuarterNative
import ca.mpreg.webgpuviewer.ImageUtil.toneMapToSdrNative
import java.nio.ByteBuffer

object ImageUtil {
    /** False on a packaging/ABI mismatch - loading a class that touches this shouldn't itself crash. */
    val isAvailable: Boolean

    init {
        isAvailable = try {
            System.loadLibrary("resize")
            true
        } catch (e: Throwable) {
            android.util.Log.e("ImageUtil", "Failed to load native resize library", e)
            false
        }
    }

    external fun resizeLinearAreaNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int
    )

    /** As [resizeLinearAreaNative], to [dstWidth] x [dstHeight] at 4 B/px. */
    external fun resizeLinearAreaToNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int,
        dstWidth: Int,
        dstHeight: Int
    )

    /**
     * [resizeLinearAreaNative] over extended-sRGB RGBA half-float, for an HDR image's mipmaps.
     *
     * Same half-resolution box filter in linear light, but without the clamp to 1.0 the 8-bit
     * path ends on - that clamp is exactly what would flatten every highlight.
     */
    external fun resizeLinearAreaNativeF16(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int
    )

    /** As [resizeLinearAreaNativeF16], to [dstWidth] x [dstHeight] at 8 B/px. */
    external fun resizeLinearAreaToNativeF16(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int,
        dstWidth: Int,
        dstHeight: Int
    )

    /**
     * For showing an HDR image on an SDR swapchain. The image's own brightest pixel lands on
     * white and the roll-off starts as late as that peak allows, so a dim HDR frame keeps
     * nearly all of its SDR range.
     */
    external fun toneMapToSdrNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int,
    )

    /**
     * Combine an SDR base with its unapplied gain map into extended-sRGB half-float.
     *
     * Deliberately here rather than in the decoder: how much of a gain map to apply is a display
     * question. It also lets the arithmetic follow libultrahdr's `applyGain` exactly - the map
     * value is used *raw*, with no transfer function undone, which is where libvips's own
     * three-channel path goes wrong and tints the whole picture.
     *
     * [weight] scales the gain in log space for a display with less headroom than the content
     * carries; 1.0 applies the map in full.
     */
    external fun applyGainmapNative(
        base: ByteBuffer,
        gain: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int,
        gainWidth: Int,
        gainHeight: Int,
        gainChannels: Int,
        gamma: FloatArray,
        minContentBoost: FloatArray,
        maxContentBoost: FloatArray,
        offsetSdr: FloatArray,
        offsetHdr: FloatArray,
        weight: Float,
    )

    /**
     * Scale extended-sRGB half-float [pixels] in place so the image's own brightest pixel lands
     * on [targetPeak]. Returns the peak it ends up with.
     *
     * For PQ and HLG. The peak is measured from the pixels rather than taken from the format:
     * PQ can encode 10000 nits and HLG 1000, but almost no real image comes near either, and
     * compressing from the format's ceiling left everything about five stops too dim.
     *
     * Values at or below SDR white are untouched, and an image already peaking under the target
     * is left alone rather than brightened to fill the headroom.
     */
    external fun scaleHdrPeakNative(
        pixels: ByteBuffer,
        width: Int,
        height: Int,
        targetPeak: Float,
    ): Float

    /**
     * Sized in [Long] and range-checked because the product overflows [Int] for a large enough
     * image, and an overflow landing on a small positive number is worse than one that throws:
     * the allocation succeeds undersized and the native pass writes past its end.
     */
    private fun directBuffer(width: Int, height: Int, bytesPerPixel: Int): ByteBuffer {
        val bytes = width.toLong() * height.toLong() * bytesPerPixel
        require(width > 0 && height > 0 && bytes in 1..Int.MAX_VALUE.toLong()) {
            "Cannot allocate ${width}x$height at $bytesPerPixel B/px"
        }
        return ByteBuffer.allocateDirect(bytes.toInt())
    }

    /**
     * A quarter turn of a packed image buffer. A page too wide for the screen reads better on its
     * side than shrunk to fit, and nothing between the decoded pixels and the screen can rotate,
     * so it happens here.
     */
    external fun rotateQuarterNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int,
        bytesPerPixel: Int,
        clockwise: Boolean,
    )

    /**
     * [rotateQuarterNative] into a freshly allocated buffer. The result is [height] x [width], so
     * the caller has to swap the two when it describes the image afterwards.
     */
    fun rotateQuarter(
        source: ByteBuffer,
        width: Int,
        height: Int,
        bytesPerPixel: Int = 4,
        clockwise: Boolean = true,
    ): ByteBuffer {
        val output = directBuffer(width, height, bytesPerPixel)
        rotateQuarterNative(source, output, width, height, bytesPerPixel, clockwise)
        return output
    }

    fun resize(source: ByteBuffer, width: Int, height: Int): ByteBuffer {
        // Half the width and half the height at 4 B/px, so one byte per source pixel.
        val output = directBuffer(width, height, 1)
        resizeLinearAreaNative(source, output, width, height)
        return output
    }

    /** [resize] to [dstWidth] x [dstHeight] rather than half. */
    fun resize(
        source: ByteBuffer, width: Int, height: Int, dstWidth: Int, dstHeight: Int
    ): ByteBuffer {
        val output = directBuffer(dstWidth, dstHeight, 4)
        resizeLinearAreaToNative(source, output, width, height, dstWidth, dstHeight)
        return output
    }

    /** [resize] for extended-sRGB half-float pixels, which are twice as wide. */
    fun resizeF16(source: ByteBuffer, width: Int, height: Int): ByteBuffer {
        // Half by half at 8 B/px, so two bytes per source pixel.
        val output = directBuffer(width, height, 2)
        resizeLinearAreaNativeF16(source, output, width, height)
        return output
    }

    /** [resizeF16] to [dstWidth] x [dstHeight] rather than half. */
    fun resizeF16(
        source: ByteBuffer, width: Int, height: Int, dstWidth: Int, dstHeight: Int
    ): ByteBuffer {
        val output = directBuffer(dstWidth, dstHeight, 8)
        resizeLinearAreaToNativeF16(source, output, width, height, dstWidth, dstHeight)
        return output
    }

    /**
     * [applyGainmapNative] into a freshly allocated RGBA16F buffer.
     *
     * [gainmap] is `ImageDecoder.DecodeResult.gainmap`, passed through as its parts so this
     * module doesn't have to depend on the decoder's types.
     */
    fun applyGainmap(
        base: ByteBuffer,
        width: Int,
        height: Int,
        gain: ByteBuffer,
        gainWidth: Int,
        gainHeight: Int,
        gainChannels: Int,
        gamma: FloatArray,
        minContentBoost: FloatArray,
        maxContentBoost: FloatArray,
        offsetSdr: FloatArray,
        offsetHdr: FloatArray,
        weight: Float = 1f,
    ): ByteBuffer {
        val output = directBuffer(width, height, 8)
        applyGainmapNative(
            base, gain, output, width, height, gainWidth, gainHeight, gainChannels,
            gamma, minContentBoost, maxContentBoost, offsetSdr, offsetHdr, weight,
        )
        return output
    }

    /** [toneMapToSdrNative] into a freshly allocated RGBA8 buffer. */
    fun toneMapToSdr(source: ByteBuffer, width: Int, height: Int): ByteBuffer {
        val output = directBuffer(width, height, 4)
        toneMapToSdrNative(source, output, width, height)
        return output
    }
}
