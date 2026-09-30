package ca.mpreg.webgpuviewer.renderer

/**
 * Releases the window reference that androidx.webgpu's `windowFromSurface` hands out. It is
 * `ANativeWindow_fromSurface` underneath, whose reference the caller owns, and nothing in androidx
 * or Dawn releases it: every surface kept its window, buffer queue and GPU buffers for good.
 */
internal object NativeWindow {
    init {
        System.loadLibrary("resize")
    }

    external fun release(window: Long)
}
