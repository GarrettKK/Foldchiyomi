package ca.mpreg.webgpuviewer

import android.app.Activity
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.runtime.Composable
import androidx.compose.ui.MotionDurationScale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventTimeoutCancellationException
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.platform.LocalContext
import kotlin.math.abs

internal object NormalMotionDurationScale : MotionDurationScale {
    override val scaleFactor: Float = 1f
}

suspend fun AwaitPointerEventScope.waitForCleanUp(
    pointerId: PointerId, timeout: Long, touchSlop: Float
): PointerEvent? = try {
    withTimeout(timeout) {
        var acc = Offset.Zero

        while (true) {
            val event = awaitPointerEvent()

            if (event.changes.any { it.isConsumed }) {
                return@withTimeout null
            }

            val change = event.changes.firstOrNull { it.id == pointerId } ?: return@withTimeout null

            if (event.changes.any { it.id != pointerId && it.pressed }) {
                return@withTimeout null
            }

            acc += event.calculatePan()
            if (acc.getDistance() > touchSlop) {
                return@withTimeout null
            }
            if (change.changedToUp()) {
                return@withTimeout event
            }
        }
    }
} catch (e: PointerEventTimeoutCancellationException) {
    null
} as PointerEvent?

suspend fun AwaitPointerEventScope.waitForDown(timeout: Long) = try {
    withTimeout(timeout) {
        var down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        while (down == null) {
            down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        }
        down
    }
} catch (e: PointerEventTimeoutCancellationException) {
    null
}

fun Float.orZero(): Float = if (this.isNaN()) 0f else this

fun Float.closeTo(x: Float, eps: Float = 0.0001f): Boolean = abs(this - x) < eps

@Composable
fun RequestMaxRefreshRate() {
    val activity = LocalContext.current as? Activity
    activity?.window?.let { window ->
        val layoutParams = window.attributes

        val display = activity.windowManager.defaultDisplay
        val supportedModes = display.supportedModes

        val maxMode = supportedModes.maxByOrNull { it.refreshRate }

        if (maxMode != null) {
            layoutParams.preferredDisplayModeId = maxMode.modeId
            window.attributes = layoutParams
        }
    }
}
