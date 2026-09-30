package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.TextureFormat

/**
 * A GPU object built once per target texture format and kept.
 *
 * A render pipeline bakes its colour target's format, so one cached pipeline serves one format -
 * and an SDR page and an HDR page can be drawn in the same frame. An SDR-only session still
 * builds exactly what it always did.
 *
 * Not thread-safe, and doesn't need to be: every pipeline is built and used on the render thread.
 */
class FormatKeyed<T>(private val build: (format: Int) -> T) {
    private val byFormat = HashMap<Int, T>(2)

    operator fun get(format: Int): T = byFormat.getOrPut(format) { build(format) }

    /** Every instance built so far, for cleanup. */
    val built: Collection<T> get() = byFormat.values

    fun clear() = byFormat.clear()
}

/** A storage texture's format is shader text, so it is interpolated in, not passed as a value. */
internal fun wgslStorageFormat(format: Int): String = when (format) {
    TextureFormat.RGBA16Float -> "rgba16float"
    TextureFormat.RGBA8Unorm -> "rgba8unorm"
    else -> error("No WGSL storage format for ${TextureFormat.toString(format)}")
}
