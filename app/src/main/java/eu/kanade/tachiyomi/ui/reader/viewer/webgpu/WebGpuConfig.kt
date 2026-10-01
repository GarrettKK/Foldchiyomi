package eu.kanade.tachiyomi.ui.reader.viewer.webgpu

import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import eu.kanade.tachiyomi.ui.reader.viewer.ReaderPageImageView
import eu.kanade.tachiyomi.ui.reader.viewer.ViewerConfig
import eu.kanade.tachiyomi.ui.reader.viewer.ViewerNavigation
import eu.kanade.tachiyomi.ui.reader.viewer.navigation.DisabledNavigation
import eu.kanade.tachiyomi.ui.reader.viewer.navigation.EdgeNavigation
import eu.kanade.tachiyomi.ui.reader.viewer.navigation.KindlishNavigation
import eu.kanade.tachiyomi.ui.reader.viewer.navigation.LNavigation
import eu.kanade.tachiyomi.ui.reader.viewer.navigation.RightAndLeftNavigation
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach

/**
 * Configuration used by pager viewers.
 */
class WebGpuConfig(
    private val viewer: WebGpuViewer,
    scope: CoroutineScope,
    readerPreferences: ReaderPreferences,
) : ViewerConfig(readerPreferences, scope) {

    var theme = readerPreferences.readerTheme.get()
        private set

    var automaticBackground = false
        private set

    var dualPageSplitChangedListener: ((Boolean) -> Unit)? = null

    var imageScaleType = 1
        private set

    var imageZoomType = ReaderPageImageView.ZoomStartPosition.LEFT
        private set

    var imageCropBorders = false
        private set

    var navigateToPan = false
        private set

    var landscapeZoom = false
        private set

    var transitionAnimation = ReaderPreferences.TransitionAnimation.BASIC
        private set

    var transitionAnimationDual = ReaderPreferences.TransitionAnimation.BASIC
        private set

    var cutoutMode = ReaderPreferences.CutoutMode.AVOID
        private set

    var cutoutModeDual = ReaderPreferences.CutoutMode.AVOID
        private set

    var dualPageView = ReaderPreferences.DualPageView.NEVER
        private set

    var continuousMinWidth = 100
        private set

    var zoomOutDisabled = false
        private set

    var continuousGap = 10
        private set

    var bubbleZoom = true
        private set

    var bubbleZoomChangedListener: ((Boolean) -> Unit)? = null

    var doubleTapZoom = true
        private set

    /** Double tap to zoom, and how fast it zooms - see [ViewerConfig.doubleTapAnimDuration]. */
    var doubleTapChangedListener: (() -> Unit)? = null

    /** The webtoon modes read the webtoon section's settings, the paged ones the paged section's. */
    private val continuous = viewer is WebGpuViewerContinuous

    init {
        readerPreferences.readerTheme.register(
            {
                theme = it
                automaticBackground = it == 3
            },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.imageScaleType.register({ imageScaleType = it }, { imagePropertyChangedListener?.invoke() })

        readerPreferences.zoomStart.register({ zoomTypeFromPreference(it) }, { imagePropertyChangedListener?.invoke() })

        (if (continuous) readerPreferences.cropBordersWebtoon else readerPreferences.cropBorders)
            .register({ imageCropBorders = it }, { imagePropertyChangedListener?.invoke() })

        readerPreferences.navigateToPan.register({ navigateToPan = it })

        readerPreferences.landscapeZoom.register({ landscapeZoom = it }, { imagePropertyChangedListener?.invoke() })

        val navigationModePref =
            if (continuous) readerPreferences.navigationModeWebtoon else readerPreferences.navigationModePager
        navigationModePref.register({ navigationMode = it }, { updateNavigation(navigationMode) })

        val navInvertedPref = if (continuous) readerPreferences.webtoonNavInverted else readerPreferences.pagerNavInverted
        navInvertedPref.register({ tappingInverted = it }, { navigator.invertMode = it })
        navInvertedPref.changes().drop(1).onEach { navigationModeChangedListener?.invoke() }
            .launchIn(scope)

        readerPreferences.dualPageSplitPaged.register(
            { dualPageSplit = it },
            {
                imagePropertyChangedListener?.invoke()
                dualPageSplitChangedListener?.invoke(it)
            },
        )

        readerPreferences.dualPageInvertPaged.register(
            { dualPageInvert = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.dualPageRotateToFit.register(
            { dualPageRotateToFit = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.dualPageRotateToFitInvert.register(
            { dualPageRotateToFitInvert = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.transitionAnimation.register(
            { transitionAnimation = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.transitionAnimationDual.register(
            { transitionAnimationDual = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.cutoutMode.register(
            { cutoutMode = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.cutoutModeDual.register(
            { cutoutModeDual = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.dualPageView.register(
            { dualPageView = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.continuousMinWidth.register(
            { continuousMinWidth = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.webtoonDisableZoomOut.register(
            { zoomOutDisabled = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.continuousGap.register(
            { continuousGap = it },
            { imagePropertyChangedListener?.invoke() },
        )

        readerPreferences.bubbleZoom.register(
            { bubbleZoom = it },
            { bubbleZoomChangedListener?.invoke(it) },
        )

        (if (continuous) readerPreferences.webtoonDoubleTapZoomEnabled else readerPreferences.pagerDoubleTapZoomEnabled)
            .register({ doubleTapZoom = it }, { doubleTapChangedListener?.invoke() })
        readerPreferences.doubleTapAnimSpeed.changes().drop(1)
            .onEach { doubleTapChangedListener?.invoke() }
            .launchIn(scope)

        // "Animate page transitions" off turns every transition into a cut - see WebGpuViewer.
        readerPreferences.pageTransitions.changes().drop(1)
            .onEach { imagePropertyChangedListener?.invoke() }
            .launchIn(scope)
    }

    private fun zoomTypeFromPreference(value: Int) {
        imageZoomType = when (value) {
            // Auto
            1 -> if (viewer.isReversed) {
                ReaderPageImageView.ZoomStartPosition.RIGHT
            } else {
                ReaderPageImageView.ZoomStartPosition.LEFT
            }
            // Left
            2 -> ReaderPageImageView.ZoomStartPosition.LEFT
            // Right
            3 -> ReaderPageImageView.ZoomStartPosition.RIGHT
            // Center
            else -> ReaderPageImageView.ZoomStartPosition.CENTER
        }
    }

    override var navigator: ViewerNavigation = defaultNavigation()
        set(value) {
            field = value.also { it.invertMode = this.tappingInverted }
        }

    override fun defaultNavigation(): ViewerNavigation {
        return if (viewer.isVertical) {
            LNavigation()
        } else {
            RightAndLeftNavigation()
        }
    }

    override fun updateNavigation(navigationMode: Int) {
        navigator = when (navigationMode) {
            0 -> defaultNavigation()
            1 -> LNavigation()
            2 -> KindlishNavigation()
            3 -> EdgeNavigation()
            4 -> RightAndLeftNavigation()
            5 -> DisabledNavigation()
            else -> defaultNavigation()
        }
        navigationModeChangedListener?.invoke()
    }
}
