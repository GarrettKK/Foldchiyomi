package ca.mpreg.webgpuviewer.viewer

import android.os.Build
import android.view.Surface
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import androidx.compose.foundation.AndroidEmbeddedExternalSurface
import androidx.compose.foundation.AndroidExternalSurface
import androidx.compose.foundation.SurfaceCoroutineScope
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import ca.mpreg.webgpuviewer.renderer.Hdr

/**
 * A [SurfaceView] from API 34, where HDR needs a `SurfaceControl` and only a [SurfaceView] has
 * one; a TextureView below that, where there is no HDR to lose.
 *
 * The TextureView is not just the older default: [AndroidExternalSurface] renders behind the
 * activity window and relies on the HWUI hole-punch, which silently fails on some OEM builds -
 * every frame renders with no errors while the screen stays black (mihonapp/mihon#3773).
 */
@Composable
internal fun ViewerSurface(
    modifier: Modifier,
    onSurfaceReady: suspend SurfaceCoroutineScope.(Surface, Int, Int) -> Unit,
) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
        AndroidExternalSurface(modifier = modifier, isOpaque = false) {
            onSurface(onSurfaceReady)
        }
    } else {
        AndroidEmbeddedExternalSurface(modifier = modifier, isOpaque = false) {
            onSurface(onSurfaceReady)
        }
    }
}

internal fun attachHdrDisplay(view: View) {
    Hdr.attachDisplay(view.display)
}

internal fun attachHdrSurface(view: View?) {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) return
    Hdr.attachSurfaceControl(view?.findSurfaceView()?.surfaceControl)
}

/**
 * Compose builds it inside an `AndroidView` and does not hand it out. Null on the TextureView
 * path, and null if Compose ever stops backing [AndroidExternalSurface] with one - in which case
 * HDR stays off rather than breaking.
 */
private fun View.findSurfaceView(): SurfaceView? = when (this) {
    is SurfaceView -> this
    is ViewGroup -> (0 until childCount).asSequence()
        .mapNotNull { getChildAt(it).findSurfaceView() }
        .firstOrNull()

    else -> null
}
