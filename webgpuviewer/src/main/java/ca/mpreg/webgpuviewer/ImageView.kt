package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.viewer.ImageViewer
import ca.mpreg.webgpuviewer.viewer.ImageViewerState

open class ImageView(
    context: Context,
    attrs: AttributeSet? = null,
    isVertical: Boolean = false,
    isReversed: Boolean = false,
) : AbstractComposeView(context, attrs) {
    constructor(context: Context, attrs: AttributeSet? = null) : this(
        context, attrs, context.obtainStyledAttributes(
            attrs, intArrayOf(android.R.attr.orientation)
        ).let {
            val orientation = it.getInt(0, 0)
            it.recycle()
            orientation == 1
        })

    open val state: ImageViewerState by lazy { ImageViewerState(isVertical, isReversed) }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        // Registered, not requested: [Hdr] sets the colour mode with the first HDR image and
        // drops it with the last. Held all session, it ramps panel brightness on SDR reading.
        Hdr.attachColorModeHost(this)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        Hdr.detachColorModeHost(this)
    }

    override fun onWindowFocusChanged(hasWindowFocus: Boolean) {
        super.onWindowFocusChanged(hasWindowFocus)
        // Switching apps and back doesn't detach the view, but the system silently drops the
        // window's HDR colour mode and the surface's extended range while unfocused.
        if (hasWindowFocus) Hdr.resyncPresentation()
    }

    @Composable
    override fun Content() {
        ImageViewer(state = state)
    }
}
