package ca.mpreg.webgpuviewer

import java.nio.ByteBuffer

/**
 * JNI bindings for the CPU trim and background-detection passes in `trim.cpp`.
 *
 * Both take the decoded pixels directly, so they can run on a background dispatcher
 * before the image is uploaded - unlike the compute-shader versions in [Trim], which
 * have to wait on a GPU readback from the render thread.
 *
 * Buffers must be direct and hold at least `width * height * 4` bytes of tightly
 * packed RGBA8.
 */
object TrimNative {
    /** False on a packaging/ABI mismatch - loading a class that touches this shouldn't itself crash. */
    val isAvailable: Boolean

    init {
        isAvailable = try {
            System.loadLibrary("resize")
            true
        } catch (e: Throwable) {
            android.util.Log.e("TrimNative", "Failed to load native resize library", e)
            false
        }
    }

    /**
     * Bounding box of the non-background pixels for each colour in [colors]
     * (`[r, g, b]` triples flattened, values 0..1).
     *
     * Writes `[minX, minY, maxX, maxY]` per colour into [outBounds], which must hold
     * at least `colors.size / 3 * 4` entries. Returns false without touching
     * [outBounds] if the arguments don't describe a readable image.
     */
    external fun findTrim(
        pixels: ByteBuffer,
        width: Int,
        height: Int,
        colors: FloatArray,
        threshold: Float,
        outBounds: IntArray
    ): Boolean

    /** Background colour sampled from the image edges, as 0xAARRGGBB. */
    external fun detectBackground(
        pixels: ByteBuffer,
        width: Int,
        height: Int,
        threshold: Float
    ): Int
}
