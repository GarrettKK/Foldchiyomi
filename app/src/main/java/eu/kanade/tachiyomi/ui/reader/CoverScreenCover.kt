package eu.kanade.tachiyomi.ui.reader

import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.window.area.WindowAreaCapability
import androidx.window.area.WindowAreaController
import androidx.window.area.WindowAreaInfo
import androidx.window.area.WindowAreaPresentationSessionCallback
import androidx.window.area.WindowAreaSessionPresenter
import androidx.window.core.ExperimentalWindowApi
import coil3.imageLoader
import coil3.request.ImageRequest
import coil3.target.ImageViewTarget
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat
import tachiyomi.domain.manga.model.MangaCover
import tachiyomi.domain.manga.model.asMangaCover

/**
 * The cover of what you're reading on the phone's cover screen while it lies open, like the
 * cover of a book facing out - see [eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences.coverScreenCover].
 *
 * Android 14's dual-screen mode (Jetpack WindowManager's rear-facing window area): the reader
 * keeps the inner screen and presents a second window on the outer one. The system says
 * whether that is on offer at all - a foldable with a cover screen behind the inner one, and
 * only while it is open - and takes the window back when the phone folds, a power saver kicks
 * in, or the reader leaves the foreground. The cover goes back up whenever it is offered again.
 *
 * Nothing here may throw on the main thread: the session callbacks arrive there, and the
 * system's side of this is young. A window that can't be put up is given up and tried again
 * a moment later, a few times, then left until the offer changes.
 */
@OptIn(ExperimentalWindowApi::class)
class CoverScreenCover(
    private val activity: ReaderActivity,
    readerPreferences: ReaderPreferences,
) : WindowAreaPresentationSessionCallback {

    private val controller = WindowAreaController.getOrCreate()
    private val executor = ContextCompat.getMainExecutor(activity)

    private var area: WindowAreaInfo? = null
    private var available = false
    private var wanted = false
    private var cover: MangaCover? = null

    private var session: WindowAreaSessionPresenter? = null
    private var requested = false
    private var image: ImageView? = null

    private var retries = 0
    private var retryJob: Job? = null

    init {
        activity.lifecycleScope.launch {
            // STARTED only: the session is closed on stop and not re-opened until the reader is
            // back in front, where the window area is offered to it again.
            activity.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                try {
                    combine(
                        controller.windowAreaInfos
                            .map { infos -> infos.firstOrNull { it.type == WindowAreaInfo.Type.TYPE_REAR_FACING } }
                            .map { info ->
                                area = info
                                info?.let { it.getCapability(OPERATION).status }
                            },
                        readerPreferences.coverScreenCover.changes(),
                        activity.viewModel.state.map { it.manga?.asMangaCover() }.distinctUntilChanged(),
                    ) { status, enabled, cover -> Triple(status, enabled, cover) }
                        .catch { logcat(LogPriority.WARN, it) { "Cover screen availability failed" } }
                        .collect { (status, enabled, cover) ->
                            available = status == WindowAreaCapability.Status.WINDOW_AREA_STATUS_AVAILABLE ||
                                status == WindowAreaCapability.Status.WINDOW_AREA_STATUS_ACTIVE
                            wanted = enabled
                            this@CoverScreenCover.cover = cover
                            // A fresh offer: whatever went wrong before gets its tries back.
                            retries = 0
                            update()
                        }
                } finally {
                    // Leaving the foreground - the system may have ended it already.
                    close()
                }
            }
        }
    }

    private fun update() {
        val cover = cover
        if (wanted && cover != null && available && activity.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            if (session == null && !requested) {
                val token = area?.token ?: return
                requested = true
                runCatching {
                    controller.presentContentOnWindowArea(token, activity, executor, this)
                }.onFailure {
                    requested = false
                    logcat(LogPriority.WARN, it) { "Cover screen presentation failed to start" }
                    retryLater()
                }
            } else {
                image?.let { show(it, cover) }
            }
        } else {
            close()
        }
    }

    /**
     * Once more, a little later. The system takes the window back and offers it again around
     * a notification shade or an app switch, and a request that lands in between is answered
     * with a session whose display isn't there yet.
     */
    private fun retryLater() {
        if (retries >= MAX_RETRIES) return
        retries++
        retryJob?.cancel()
        retryJob = activity.lifecycleScope.launch {
            delay(RETRY_DELAY_MS * retries)
            // A request that never answered is not waited on past this.
            requested = false
            update()
        }
    }

    private fun close() {
        retryJob?.cancel()
        retryJob = null
        session?.let { runCatching { it.close() }.onFailure { e -> logcat(LogPriority.WARN, e) { "Cover screen close failed" } } }
        session = null
        image = null
    }

    private fun show(view: ImageView, cover: MangaCover) {
        ImageRequest.Builder(view.context)
            .data(cover)
            .target(ImageViewTarget(view))
            .build()
            .let(view.context.imageLoader::enqueue)
    }

    override fun onSessionStarted(session: WindowAreaSessionPresenter) {
        requested = false
        val context = session.context
        val view = ImageView(context).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER,
            )
        }
        val root = FrameLayout(context).apply {
            setBackgroundColor(Color.BLACK)
            addView(view)
        }
        try {
            session.setContentView(root)
        } catch (e: Throwable) {
            // InvalidDisplayException and kin: the session came without a display to show on.
            logcat(LogPriority.WARN, e) { "Cover screen window could not be added" }
            runCatching { session.close() }
            retryLater()
            return
        }
        this.session = session
        image = view
        retries = 0
        // The reader may have moved on, or the setting turned off, while the session started.
        update()
    }

    override fun onSessionEnded(t: Throwable?) {
        if (t != null) logcat(LogPriority.WARN, t) { "Cover screen presentation ended" }
        requested = false
        session = null
        image = null
        // Taken back by the system while still wanted: ask again once it has settled. A failed
        // window closes its session, which lands here too; its retry is already on the way.
        if (wanted && available && retryJob?.isActive != true) retryLater()
    }

    override fun onContainerVisibilityChanged(isVisible: Boolean) {}

    private companion object {
        val OPERATION = WindowAreaCapability.Operation.OPERATION_PRESENT_ON_AREA
        const val MAX_RETRIES = 3
        const val RETRY_DELAY_MS = 700L
    }
}
