package ca.mpreg.webgpuviewer.renderer

import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * A downscaled greyscale copy of an image: [luma] is [width] x [height], one byte per pixel, and
 * [scale] maps image pixels onto it. Built with the darkest of each cell's samples rather than
 * their average, so a speech bubble's thin outline survives the shrink unbroken.
 */
class InkMap(val luma: ByteArray, val width: Int, val height: Int, val scale: Float) {

    operator fun get(x: Int, y: Int): Int = luma[y * width + x].toInt() and 0xFF

    companion object {
        /** Longest side of the map - about what a bubble needs to keep its outline and text. */
        private const val MAX_SIDE = 1280

        /** From 8-bit RGBA [pixels], [width] x [height]. */
        fun from(pixels: ByteBuffer, width: Int, height: Int): InkMap? {
            if (width <= 0 || height <= 0) return null
            if (pixels.capacity() < width.toLong() * height * 4) return null
            val scale = min(1f, MAX_SIDE.toFloat() / max(width, height))
            val w = max(1, (width * scale).roundToInt())
            val h = max(1, (height * scale).roundToInt())
            val out = ByteArray(w * h)
            // Each map pixel covers step source pixels; sample a 2x2 spread across it.
            val stepX = width.toFloat() / w
            val stepY = height.toFloat() / h
            for (y in 0 until h) {
                val sy0 = min(height - 1, (y * stepY).toInt())
                val sy1 = min(height - 1, ((y + 0.5f) * stepY).toInt())
                for (x in 0 until w) {
                    val sx0 = min(width - 1, (x * stepX).toInt())
                    val sx1 = min(width - 1, ((x + 0.5f) * stepX).toInt())
                    var darkest = 255
                    darkest = min(darkest, lumaAt(pixels, width, sx0, sy0))
                    darkest = min(darkest, lumaAt(pixels, width, sx1, sy0))
                    darkest = min(darkest, lumaAt(pixels, width, sx0, sy1))
                    darkest = min(darkest, lumaAt(pixels, width, sx1, sy1))
                    out[y * w + x] = darkest.toByte()
                }
            }
            return InkMap(out, w, h, w.toFloat() / width)
        }

        /** Symmetric in red and blue, so RGBA and BGRA read the same. */
        private fun lumaAt(pixels: ByteBuffer, width: Int, x: Int, y: Int): Int {
            val i = (y * width + x) * 4
            val r = pixels.get(i).toInt() and 0xFF
            val g = pixels.get(i + 1).toInt() and 0xFF
            val b = pixels.get(i + 2).toInt() and 0xFF
            return (r + 2 * g + b) shr 2
        }
    }
}
