package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuous
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState

class ImageViewContinuous(context: Context, attrs: AttributeSet? = null) :
    ImageView(context, attrs) {
    override val state = ImageViewerContinuousState()

    @Composable
    override fun Content() {
        ImageViewerContinuous(state = state)
    }
}
