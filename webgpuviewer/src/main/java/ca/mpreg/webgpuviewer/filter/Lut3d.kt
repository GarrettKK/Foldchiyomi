package ca.mpreg.webgpuviewer.filter

import java.io.BufferedInputStream
import java.io.ByteArrayInputStream
import java.io.EOFException
import java.io.InputStream

/**
 * A cubic colour lookup table: [size]^3 RGB triples in [data], red varying fastest, then green,
 * then blue - the layout a 3D texture wants, with red on x.
 *
 * Values are the mapped output for that input colour, normally in 0..1 but not clamped: a LUT
 * that overshoots keeps its overshoot until [FilterLut3d] writes an 8-bit frame.
 */
class Lut3d(val size: Int, val data: FloatArray, val limitedRange: Boolean = false) {
    init {
        require(size >= 2) { "LUT size $size" }
        require(data.size == size * size * size * 3) { "LUT data ${data.size} for size $size" }
    }

    companion object {
        /** The LUT that changes nothing. */
        fun identity(size: Int = 2): Lut3d {
            val data = FloatArray(size * size * size * 3)
            var i = 0
            for (b in 0 until size) for (g in 0 until size) for (r in 0 until size) {
                data[i++] = r.toFloat() / (size - 1)
                data[i++] = g.toFloat() / (size - 1)
                data[i++] = b.toFloat() / (size - 1)
            }
            return Lut3d(size, data)
        }

        /**
         * An Adobe/IRIDAS `.cube` 3D LUT. 1D cubes (`LUT_1D_SIZE`) are not 3D LUTs and are
         * rejected; `DOMAIN_MIN`/`DOMAIN_MAX` are honoured by rescaling into 0..1.
         */
        fun parseCube(lines: Sequence<String>): Lut3d {
            var size = 0
            val domainMin = floatArrayOf(0f, 0f, 0f)
            val domainMax = floatArrayOf(1f, 1f, 1f)
            var data: FloatArray? = null
            var count = 0

            for (raw in lines) {
                val line = raw.substringBefore('#').trim()
                if (line.isEmpty()) continue

                // A value line starts with a number - anything else is a keyword, which lets the
                // hot path skip the case fold and the keyword compares entirely.
                val first = line[0]
                if (first in '0'..'9' || first == '-' || first == '+' || first == '.') {
                    val target = data ?: throw IllegalArgumentException("LUT_3D_SIZE missing")
                    if (count + 3 > target.size) throw IllegalArgumentException("cube too long")
                    val parts = line.split(WHITESPACE, limit = 4)
                    require(parts.size >= 3) { "cube value line: $line" }
                    for (c in 0 until 3) {
                        val span = domainMax[c] - domainMin[c]
                        val v = parts[c].toFloat()
                        target[count + c] = if (span != 0f) (v - domainMin[c]) / span else v
                    }
                    count += 3
                    continue
                }

                val parts = line.split(WHITESPACE)
                when (parts[0].uppercase()) {
                    "LUT_1D_SIZE" -> throw IllegalArgumentException("1D cube, not a 3D LUT")
                    "LUT_3D_SIZE" -> {
                        size = parts[1].toInt()
                        require(size in 2..MAX_SIZE) { "LUT_3D_SIZE $size" }
                        data = FloatArray(size * size * size * 3)
                    }

                    "DOMAIN_MIN" -> for (c in 0 until 3) domainMin[c] = parts[c + 1].toFloat()
                    "DOMAIN_MAX" -> for (c in 0 until 3) domainMax[c] = parts[c + 1].toFloat()
                    else -> continue    // TITLE, and anything else this doesn't need
                }
            }

            val target = data ?: throw IllegalArgumentException("LUT_3D_SIZE missing")
            require(count == target.size) { "cube has $count of ${target.size} values" }
            // .cube entries run red fastest, the same order as [data].
            return Lut3d(size, target)
        }

        fun parseCube(text: String): Lut3d = parseCube(text.lineSequence())

        fun parseCube(stream: InputStream): Lut3d =
            stream.bufferedReader().useLines { parseCube(it) }

        /**
         * A madVR `.3dlut`: a 16KB header, then 256^3 entries of three little-endian 16-bit
         * samples in B, G, R order, blue varying fastest, over TV-level input and output.
         *
         * Resampled to [size]^3 rather than kept at the file's own 256 points, which would be
         * 200MB of floats on the heap and 134MB on the GPU. 64 is visually indistinguishable
         * for the smooth tables madVR writes; the cost is (size^3 * 20) bytes.
         *
         * Read as a stream, skipping the entries it doesn't need, so a 96MB file never lands in
         * memory whole - at the default size this touches about 6MB of it.
         */
        fun parseMadVr(stream: InputStream, size: Int = 64): Lut3d {
            require(size in 2..MAX_SIZE) { "madVR LUT size $size (limit $MAX_SIZE)" }

            val input = stream as? BufferedInputStream ?: BufferedInputStream(stream, 1 shl 16)
            skipFully(input, HEADER.toLong())

            val data = FloatArray(size * size * size * 3)
            val row = ByteArray(FULL * 6)
            var entry = 0L

            // Rows come out in file order, so every seek is a forward skip.
            for (r in 0 until size) {
                val fileR = axis(r, size)
                for (g in 0 until size) {
                    val start = (fileR.toLong() * FULL + axis(g, size)) * FULL
                    skipFully(input, (start - entry) * 6)
                    readFully(input, row)
                    entry = start + FULL

                    for (b in 0 until size) {
                        val at = axis(b, size) * 6
                        val i = ((b * size + g) * size + r) * 3
                        data[i] = sample(row, at + 4)
                        data[i + 1] = sample(row, at + 2)
                        data[i + 2] = sample(row, at)
                    }
                }
            }
            return Lut3d(size, data, limitedRange = true)
        }

        fun parseMadVr(bytes: ByteArray, size: Int = 64): Lut3d =
            parseMadVr(ByteArrayInputStream(bytes), size)

        private val WHITESPACE = Regex("\\s+")

        private const val HEADER = 16384
        private const val FULL = 256

        /**
         * Points per axis this will build. 128^3 is already 25MB of floats and 16MB on the GPU;
         * the file's own 256 would be eight times that and simply run out of memory.
         */
        const val MAX_SIZE = 128

        /** Nearest full-resolution index for step [i] of a [size]-point axis. */
        private fun axis(i: Int, size: Int): Int =
            if (size == FULL) i else (i * (FULL - 1) + (size - 1) / 2) / (size - 1)

        /** One 16-bit TV-level sample as 0..1 full range. */
        private fun sample(bytes: ByteArray, at: Int): Float {
            val v = (bytes[at].toInt() and 0xff) or ((bytes[at + 1].toInt() and 0xff) shl 8)
            return (v - 4096f) / 56064f
        }

        private fun skipFully(input: InputStream, count: Long) {
            var left = count
            while (left > 0) {
                val skipped = input.skip(left)
                if (skipped > 0) {
                    left -= skipped
                    continue
                }
                // skip() is allowed to do nothing; read to make progress, or give up at the end.
                if (input.read() < 0) throw EOFException("madVR LUT ends early")
                left--
            }
        }

        private fun readFully(input: InputStream, into: ByteArray) {
            var at = 0
            while (at < into.size) {
                val read = input.read(into, at, into.size - at)
                if (read < 0) throw EOFException("madVR LUT ends early")
                at += read
            }
        }
    }
}
