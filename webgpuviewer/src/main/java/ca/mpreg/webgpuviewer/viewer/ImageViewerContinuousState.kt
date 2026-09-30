package ca.mpreg.webgpuviewer.viewer

import android.graphics.Rect
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.closeTo
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.solveImagePlacement
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState.Companion.MAX_PAGE_WALK
import ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState.Companion.MAX_VISIBLE_PAGES
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlin.math.max
import kotlin.math.roundToInt

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    companion object {
        const val MAX_VISIBLE_PAGES = 24

        /** Caps a page walk against a provider that never reports null/zero-height. */
        private const val MAX_PAGE_WALK = 10_000
    }

    var backgroundColor: Int = 0
        set(value) {
            if (value == field) return
            field = value
            invalidate()
        }

    private fun Float.isSane() = !isNaN() && !isInfinite()

    private fun Double.isSane() = !isNaN() && !isInfinite()

    var scale = 1f
        set(value) {
            if (!value.isSane()) return
            field = value
        }

    var offsetX = 0f
        set(value) {
            if (!value.isSane()) return
            field = value
        }

    /**
     * How much of the viewport width a page fills when fully zoomed out, from 0 to 1.
     *
     * Only the zoom-out floor moves: pages are still laid out against the full width, so
     * [getPageHeight] and document space are unchanged. Setting it lifts a [scale] already
     * below the floor, so it applies without waiting for a gesture.
     */
    var homeScale: Float = 1f
        set(value) {
            val clamped = value.fastCoerceIn(0.01f, 1f)
            if (clamped == field) return
            field = clamped
            if (scale < clamped) scale = clamped
            invalidate()
        }

    /** Lowest [scale] a gesture may settle at - see [homeScale]. */
    var minScale = 0f
        get() {
            if (field > 0) return field
            return homeScale
        }

    val atHomeScale: Boolean
        get() = scale.closeTo(homeScale)

    val doubleTapScale: Float get() = homeScale * 2f

    val maxScale: Float get() = max(doubleTapScale * 2f, 4f)

    /**
     * True while a gesture is driving zoom (pinch, drag, fling, snap-back). Gates every visible
     * page's tile grid, as [ImagePage.isScaleAnimating] does for the paged viewer.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    /**
     * True while a plain (non-zoom) fling is scrolling - also set for [animateScroll], a tap
     * navigation's own scroll animation. Tile generation shares the render thread with the frame,
     * so it is held off while the camera moves fast. Separate from [isScaleAnimating] - different
     * gestures, either can be true alone.
     */
    @Volatile
    var isFlinging: Boolean = false

    /**
     * True while a one-or-more-finger drag is actively panning content (not yet released into a
     * fling). Along with [isFlinging], marks real scroll [onViewport] reports against - unlike
     * [animateSlideIn], whose [slideOffset] never touches [scrollY] at all.
     */
    @Volatile
    var isPanning: Boolean = false

    /** Set while [restorePosition] walks pages, so its intermediate steps don't reach the app. */
    @Volatile
    private var isRestoring: Boolean = false

    private val scrollLock = Any()

    /** [scrollY]/[anchorDocY]'s storage - double precision against long-document drift. */
    private var scrollYInternal: Double = 0.0
    private var anchorDocYInternal: Double = 0.0

    val scrollY: Float get() = synchronized(scrollLock) { scrollYInternal.toFloat() }

    /**
     * Document-space top of the page at [scrollY] == 0, in screen pixels at zoom 1. The only
     * position state kept across frames: every other visible page's is re-derived from it each
     * frame (see [captureRenderState]), never stored per page - a page's identity doesn't
     * survive a decode, so anything kept on it would be lost exactly when a placeholder
     * corrects to its real height. Written only by [scrollBy].
     */
    private val anchorDocY: Float get() = anchorDocYInternal.toFloat()

    /**
     * Visual-only slide, animated to 0 by [animateSlideIn]. Kept out of [scrollY], which would
     * walk into the page before it and report a page change.
     */
    private var slideOffset = 0f

    /**
     * Height of [page]'s own content in screen pixels.
     *
     * Measured the same decoded or not: a placeholder must occupy exactly the space its decoded
     * self will, or the pages below jump when it decodes.
     *
     * Only an [ImagePage.ImageSingle] fits the viewer's full width. A [ImagePage.Render] page is
     * reserved and drawn at its native size - see the matching pageScale in [renderSnapshot].
     */
    fun getPageHeight(page: ImagePage): Float {
        if (page !is ImagePage.ImageSingle) return page.height.toFloat()
        val crop = cropOf(page)
        val pageWidth = crop?.width() ?: page.width
        if (pageWidth <= 0 || width <= 0) return page.height.toFloat()
        return (crop?.height() ?: page.height) * (width.toFloat() / pageWidth)
    }

    /**
     * Cut each page to its measured trim: the trim fills the width, its slot is the trim's
     * height, and nothing outside it draws. Unlike the paged viewer, no zoom is involved.
     */
    var cropBorders: Boolean = false
        set(value) {
            if (field == value) return
            field = value
            currentPageHeight = null
            invalidate()
        }

    /** The part of [page] drawn, in its own pixels; null for all of it. */
    private fun cropOf(page: ImagePage.ImageSingle): Rect? {
        if (!cropBorders) return null
        val trim = page.image?.trim ?: return null
        return if (trim.width() > 0 && trim.height() > 0) Rect(trim) else null
    }

    /**
     * Empty space between pages, from 0 to 1 viewport heights. Resolved against [height], so it
     * lands in document space at zoom 1 and zooms with the content.
     *
     * Part of the page slot (see [getPageSlotHeight]), not a separate element: a page sits at
     * the top of its slot with the gap trailing below, so the first page starts flush against
     * the document's top. Document space measures slots, so nothing else knows the gap exists.
     */
    var pageGap: Float = 0f
        set(value) {
            val clamped = value.fastCoerceIn(0f, 1f)
            if (!clamped.isSane() || clamped == field) return
            field = clamped
            currentPageHeight = null
            invalidate()
        }

    /** [pageGap] in document-space pixels. 0 until the surface has a height to measure against. */
    private val pageGapPx: Float get() = pageGap * height

    /**
     * Height [page] reserves in document space: [getPageHeight] plus [pageGapPx]. This, not
     * [getPageHeight], is what document space is built from.
     */
    fun getPageSlotHeight(page: ImagePage): Float = getPageHeight(page) + pageGapPx

    /** Slot height page 0 was last measured at, to carry the position across a decode. */
    private var currentPageHeight: Float? = null

    /** Set by [savePosition], applied by [captureRenderState] once a page is actually available. */
    private var pendingRestore: ContinuousPosition? = null

    /**
     * [readThrough] is the deepest page whose bottom has reached the viewport's; where
     * [onPageChange] means "reached this page", this means "read past it" - so the document's
     * last page reads through exactly when its bottom comes on screen. Reported every frame while
     * [isFlinging] or [isPanning] - real scroll, not merely something else invalidating a frame -
     * not on a change alone: an edge that loses a race is lost for good, so diff it yourself.
     * Observation only - it never moves the scroll.
     */
    var onViewport: ((readThrough: ImagePage?) -> Unit)? = null

    /**
     * Pages the last frame reached below and above the current one. How many the viewport shows
     * depends on the zoom, so a decode window must follow this rather than a fixed count.
     */
    @Volatile
    var pagesBelow: Int = 0
        private set

    @Volatile
    var pagesAbove: Int = 0
        private set

    /**
     * Scroll by [deltaPixels], moving the current page as many times as the delta covers.
     *
     * One fling frame can cross several short pages, so both walks loop. Each stops on a
     * zero-height page, which would never advance the position; [MAX_PAGE_WALK] guards a
     * provider that never reports one.
     */
    fun scrollBy(deltaPixels: Float) {
        if (!deltaPixels.isSane()) return
        synchronized(scrollLock) {
            getPage(0) ?: return
            slideOffset = 0f

            scrollYInternal += deltaPixels.toDouble()

            // Backwards, above the current page top.
            var guard = 0
            while (scrollYInternal < 0.0 && guard++ < MAX_PAGE_WALK) {
                if (getPage(-1) == null) {
                    scrollYInternal = 0.0
                    break
                }
                if (!isRestoring) onPageChange?.runCatching { invoke(-1) }
                val newPage = getPage(0) ?: return
                val newHeight = getPageSlotHeight(newPage)
                anchorDocYInternal -= newHeight.toDouble()
                currentPageHeight = newHeight
                // Nothing to hold a position inside, so rest at its top - left above it, the
                // next scroll reads it as another step back.
                if (newHeight <= 0f) {
                    scrollYInternal = 0.0
                    break
                }
                scrollYInternal += newHeight.toDouble()
            }

            // Forwards, while it sits past the bottom. Stops at the last page rather than
            // stepping off the end.
            guard = 0
            while (guard++ < MAX_PAGE_WALK) {
                val page = getPage(0) ?: return
                val pageHeight = getPageSlotHeight(page)
                if (scrollYInternal <= pageHeight || pageHeight <= 0f) break
                if (getPage(1) == null) {
                    scrollYInternal = pageHeight.toDouble()
                    break
                }
                if (!isRestoring) onPageChange?.runCatching { invoke(1) }
                anchorDocYInternal += pageHeight.toDouble()
                val newPage = getPage(0) ?: return
                currentPageHeight = getPageSlotHeight(newPage)
                scrollYInternal -= pageHeight.toDouble()
            }

            clampToDocumentEnd()
            if (!scrollYInternal.isSane()) scrollYInternal = 0.0
            if (!anchorDocYInternal.isSane()) anchorDocYInternal = 0.0
        }
    }

    private val safeScale: Float get() = if (scale.isSane() && scale > 0f) scale else 1f

    /** Page-space height of the viewport. Runs from page-space 0 - the camera is top-anchored. */
    private val bandHeight: Double get() = height / safeScale.toDouble()

    /**
     * Furthest [scrollY] may go: the last page's bottom stops at the viewport's, or - zoomed
     * out past what [MAX_VISIBLE_PAGES] can measure - the last drawn page's does. Null when
     * there is provably content enough below.
     * Negative when the end falls above page 0's top - see [clampToDocumentEnd].
     *
     * Measured to the last page's content: past the end there is nothing for its [pageGap] to
     * separate, so the gap is neither scrollable nor content to fill a viewport with.
     */
    private fun maxScrollY(): Double? {
        val bottomEdge = bandHeight
        var slotTop = 0.0
        for (i in 0..MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: return max(0.0, slotTop - pageGapPx) - bottomEdge
            val contentHeight = getPageHeight(page).toDouble()
            if (contentHeight <= 0.0) return null
            // Enough content below to fill the viewport, whatever follows it.
            if (slotTop + contentHeight - bottomEdge > scrollYInternal) return null
            slotTop += contentHeight + pageGapPx
        }

        // Zoomed out past what [MAX_VISIBLE_PAGES] can measure. Bound by the last page it
        // reaches, not "no bound": past that is only blank, and unbounded scrolls off and snaps
        // back once the end comes into range.
        return max(0.0, slotTop - pageGapPx) - bottomEdge
    }

    /**
     * Hold [scrollY] at the document's end, which the walks above can overshoot. A last page
     * shorter than the viewport ends above page 0's top, and [scrollY] can't be negative (the
     * backward walk reads that as "step up"), so step back to a page that can hold it.
     */
    private fun clampToDocumentEnd() {
        var guard = 0
        while (guard++ < MAX_PAGE_WALK) {
            val max = maxScrollY() ?: return
            if (scrollYInternal <= max) return
            if (max >= 0.0) {
                scrollYInternal = max
                return
            }
            // Nothing above to measure from, so stop at the document's top.
            if (getPage(-1) == null) {
                scrollYInternal = 0.0
                return
            }
            if (!isRestoring) onPageChange?.runCatching { invoke(-1) }
            val newPage = getPage(0) ?: return
            val newHeight = getPageSlotHeight(newPage)
            anchorDocYInternal -= newHeight.toDouble()
            currentPageHeight = newHeight
            // No height yet to hold it either, so rest at its top.
            if (newHeight <= 0f) {
                scrollYInternal = 0.0
                return
            }
            // The same document position, measured off the page now at 0.
            scrollYInternal = max + newHeight.toDouble()
        }
    }

    /**
     * Document-space position of the viewport's top, in page-space pixels at zoom 1. Unlike
     * [scrollY] it survives a page crossing, so it is what to remember a position by and
     * [scrollTo] what to restore it with.
     */
    val documentY: Float get() = synchronized(scrollLock) { (anchorDocYInternal + scrollYInternal).toFloat() }

    /** As [documentY], at the precision [savePosition] actually keeps it in. */
    val documentYDouble: Double get() = synchronized(scrollLock) { anchorDocYInternal + scrollYInternal }

    /** Put the viewport's top at [docY] - see [documentY]. */
    fun scrollTo(docY: Float) {
        if (!docY.isSane()) return
        synchronized(scrollLock) { scrollBy((docY.toDouble() - (anchorDocYInternal + scrollYInternal)).toFloat()) }
    }

    /** As [scrollTo], at [documentYDouble]'s precision. */
    fun scrollToDouble(docY: Double) {
        if (!docY.isSane()) return
        synchronized(scrollLock) { scrollBy((docY - (anchorDocYInternal + scrollYInternal)).toFloat()) }
    }

    /** Move to the top of the page [getPage] now answers 0 with, after the app jumps pages. */
    fun resetScroll() {
        synchronized(scrollLock) {
            scrollYInternal = 0.0
            // A different page now: its height is the baseline, not the one left behind.
            currentPageHeight = null
            pendingRestore = null
        }
    }

    /**
     * A [documentY] plus enough to re-find it after a fresh set of pages replaces the ones it was
     * taken against: the raw number is only meaningful against *this* session's [anchorDocY], so
     * [pageIndexHint]/[fractionWithinPage] re-derive the place by page index.
     */
    data class ContinuousPosition(
        val documentY: Float,
        val scale: Float = 1f,
        val offsetX: Float = 0f,
        val pageIndexHint: Int = -1,
        val fractionWithinPage: Float = 0f,
    )

    /** Capture where the viewport is right now, to hand to [restorePosition] later. */
    fun savePosition(): ContinuousPosition = synchronized(scrollLock) {
        val docY = (anchorDocYInternal + scrollYInternal).toFloat()
        val page = getPage(0)
        val pageHeight = page?.let { getPageSlotHeight(it) } ?: 0f
        val fraction = if (pageHeight > 0f) (scrollYInternal / pageHeight).toFloat()
            .fastCoerceIn(0f, 1f) else 0f
        ContinuousPosition(
            documentY = docY,
            scale = scale,
            offsetX = offsetX,
            pageIndexHint = getCurrentPageIndexLocked() ?: -1,
            fractionWithinPage = fraction,
        )
    }

    /** Put the viewport back at [pos]. Deferred to [captureRenderState] if no page exists yet. */
    fun restorePosition(pos: ContinuousPosition) {
        if (!pos.documentY.isSane() || !pos.scale.isSane() || !pos.offsetX.isSane()) return
        synchronized(scrollLock) {
            if (getPage(0) == null) {
                pendingRestore = pos
                return
            }
            applyRestoreLocked(pos)
        }
    }

    private fun applyRestoreLocked(pos: ContinuousPosition) {
        isRestoring = true
        try {
            scale = pos.scale.fastCoerceIn(minScale, maxScale)
            val targetDocY = resolveDocumentYForRestoreLocked(pos)
            scrollBy((targetDocY - (anchorDocYInternal + scrollYInternal)).toFloat())
            val maxOffsetX = max(0f, (scale - 1f) / (2f * scale))
            offsetX = pos.offsetX.fastCoerceIn(-maxOffsetX, maxOffsetX)
            pendingRestore = null
        } finally {
            isRestoring = false
        }
        invalidate()
    }

    /** [pos]'s page/fraction hint when it resolves, its raw documentY otherwise. */
    private fun resolveDocumentYForRestoreLocked(pos: ContinuousPosition): Double {
        if (pos.pageIndexHint >= 0) {
            documentYForPageIndexLocked(
                pos.pageIndexHint,
                pos.fractionWithinPage
            )?.let { return it }
        }
        return pos.documentY.toDouble()
    }

    /** The page index [getPage] would need to answer 0 with to reach [documentY] - see [ContinuousPosition]. */
    private fun getCurrentPageIndexLocked(): Int {
        val docY = anchorDocYInternal + scrollYInternal
        var y = anchorDocYInternal
        var idx = 0
        var guard = 0
        while (guard++ < MAX_PAGE_WALK) {
            val page = getPage(idx) ?: break
            val h = getPageSlotHeight(page).toDouble()
            if (h <= 0.0) break
            if (docY < y + h) return idx
            y += h
            idx++
        }
        return 0
    }

    /** As [getCurrentPageIndexLocked], safe to call without already holding [scrollLock]. */
    fun getCurrentPageIndex(): Int = synchronized(scrollLock) { getCurrentPageIndexLocked() ?: 0 }

    /**
     * Document-space position [fraction] of the way down page [pageIndex] (relative to the page
     * [getPage] answers 0 with), or null if walking there runs off the pages available.
     */
    private fun documentYForPageIndexLocked(pageIndex: Int, fraction: Float): Double? {
        val clampedFraction = fraction.fastCoerceIn(0f, 1f)
        var docY = anchorDocYInternal
        if (pageIndex == 0) {
            val h = getPage(0)?.let { getPageSlotHeight(it).toDouble() } ?: return null
            return docY + h * clampedFraction
        }
        if (pageIndex > 0) {
            for (i in 0 until pageIndex) {
                val p = getPage(i) ?: return null
                docY += getPageSlotHeight(p).toDouble()
            }
            val target = getPage(pageIndex) ?: return null
            return docY + getPageSlotHeight(target).toDouble() * clampedFraction
        }
        for (i in pageIndex until 0) {
            val p = getPage(i) ?: return null
            docY -= getPageSlotHeight(p).toDouble()
        }
        val target = getPage(pageIndex) ?: return null
        return docY + getPageSlotHeight(target).toDouble() * clampedFraction
    }

    /** Slide the current page into place after a jump - [direction] 1 when it came from below. */
    fun animateSlideIn(direction: Int) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            try {
                animate(
                    direction * height / 2f, 0f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.5f
                    )
                ) { value, _ ->
                    slideOffset = value
                    invalidate()
                }
            } finally {
                // Not if cancelled: this resumes after its replacement set its own.
                if (animationJob === coroutineContext[Job]) {
                    slideOffset = 0f
                    invalidate()
                }
            }
        }
    }

    fun animateScroll(deltaPixels: Float) {
        if (!deltaPixels.isSane()) return
        animationJob?.cancel()
        animationJob = scope?.launch {
            isFlinging = true
            try {
                var lastValue = 0f
                animate(
                    0f, deltaPixels, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
                ) { value, _ ->
                    scrollBy(value - lastValue)
                    lastValue = value
                    invalidate()
                }
            } finally {
                isFlinging = false
                invalidate()
            }
        }
    }

    /**
     * One page visible this frame, at the document-space slot top [captureRenderState] found.
     * [pageHeight] is the slot, [contentHeight] the page drawn at its top.
     */
    private class VisiblePage(
        val page: ImagePage,
        val docTop: Float,
        val pageHeight: Float,
        val contentHeight: Float,
        /** As [getPageHeight] measured it, so the draw places what the layout sized. */
        val crop: Rect?,
    )

    private class ContinuousRenderSnapshot(
        val pages: List<VisiblePage>,
        val scale: Float,
        val offsetX: Float,
        /** Document position (see [anchorDocY]) currently at the viewport's vertical centre. */
        val cameraDocY: Float,
        /** [isScaleAnimating] or [isFlinging] - either means "don't generate tiles right now". */
        val suppressGeneration: Boolean,
        val backgroundColor: Int,
        val readThrough: ImagePage?,
    )

    private fun cropFor(page: ImagePage) = (page as? ImagePage.ImageSingle)?.let(::cropOf)

    override fun captureRenderState(): Any {
        val snapshot = captureLocked()

        if (isFlinging || isPanning) onViewport?.runCatching { invoke(snapshot.readThrough) }
        return snapshot
    }

    private fun captureLocked(): ContinuousRenderSnapshot = synchronized(scrollLock) {
        val screenH = height.toFloat()

        val page0 = getPage(0)

        pendingRestore?.let { pending ->
            if (getPage(0) != null) applyRestoreLocked(pending)
        }

        if (page0 != null) {
            val pageHeight = getPageSlotHeight(page0)
            // A decode correcting a placeholder's height holds the same fraction of the page.
            // Both heights must be measured - an unmeasured one is no baseline.
            val ratio = currentPageHeight
                ?.takeIf { it > 0f && pageHeight > 0f && pageHeight != it }
                ?.let { (pageHeight / it).toDouble() }
            val wasPinned = ratio != null && maxScrollY()?.let { scrollYInternal >= it } == true
            if (ratio != null) scrollYInternal *= ratio
            if (pageHeight > 0f) currentPageHeight = pageHeight
            // A decode can shorten the document under a position already at its end.
            clampToDocumentEnd()
            if (wasPinned) maxScrollY()?.let {
                if (it >= 0.0 && scrollYInternal < it) scrollYInternal = it
            }
        }

        // After the clamp, which can step the page at 0 back.
        val y0 = if (page0 != null) (-scrollYInternal + slideOffset).toFloat() else 0f

        val s = safeScale

        // The point both the fast path and TileRenderer's continuous overloads zoom around, so
        // they agree on where a page belongs.
        val cameraDocY = anchorDocY - y0 + 0.5f * screenH / s

        val pages = mutableListOf<VisiblePage>()
        // Undecoded ones too: [pages] skips them, and their decode's invalidate must still land.
        val visible = mutableListOf<ImagePage>()

        // Visible band in unscaled page space, from page 0's top - see cameraDocY.
        val visTop = 0f
        val screenBot = screenH / s
        // +1 tile, matching TileRenderer's prefetch ring, so a boundary tile just past the
        // viewport has its page already discovered.
        val visBot = screenBot + tiles.preferredTileSize / s

        // Read past, not merely reached - see [onViewport]. The bottom edge alone: a page
        // covering the viewport's top is one being read, and matching it would report it over
        // the finished pages below. Against the content, not the slot, or the trailing gap
        // keeps the last page from ever reporting - the end lands its bottom on screenBot.
        fun isScrolledThrough(top: Float, contentHeight: Float) =
            contentHeight > 0f && top + contentHeight <= screenBot + 0.5f

        var scrolledThrough: ImagePage? = null

        // Pages above page 0 - only ever there while [slideOffset] holds it down the screen,
        // since the band starts at its top. Mirrors the forward walk, toward negatives.
        var yTop = y0
        var iBack = -1
        var docTopBack = anchorDocY
        var above = 0
        while (yTop > visTop && iBack >= -MAX_VISIBLE_PAGES) {
            val page = getPage(iBack) ?: break
            above = -iBack
            val contentHeight = getPageHeight(page)
            val pageHeight = contentHeight + pageGapPx
            docTopBack -= pageHeight
            yTop -= pageHeight
            // Walking up, so the first match is the deepest one above page 0.
            if (scrolledThrough == null && isScrolledThrough(yTop, contentHeight)) {
                scrolledThrough = page
            }
            visible.add(page)
            // Walked upward, so each goes in front of the last - top to bottom, as the
            // forward walk appends.
            if (page.isDecoded) {
                pages.add(
                    0, VisiblePage(page, docTopBack, pageHeight, contentHeight, cropFor(page))
                )
            }
            if (pageHeight <= 0f) break
            iBack--
        }

        // Until the viewport (plus margin) is covered or MAX_VISIBLE_PAGES is reached: zoomed
        // far out, or with short pages, the document-space bound alone would walk on. Purely
        // local - nothing is written back to a page.
        var y = y0
        var i = 0
        var docTop = anchorDocY
        var prevHeight = 0f
        var hasPrev = false
        var below = 0
        while (y < visBot && i <= MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: break
            below = i
            // Anchored to the previous page in this walk, never frozen: an undecoded page's
            // height is a guess, and re-deriving it self-corrects once it decodes.
            if (hasPrev) docTop += prevHeight
            hasPrev = true
            val contentHeight = getPageHeight(page)
            val pageHeight = contentHeight + pageGapPx

            // Walking down, so a later match replaces whatever the backward walk found.
            if (isScrolledThrough(y, contentHeight)) scrolledThrough = page

            if (y + pageHeight > visTop) {
                visible.add(page)
                if (page.isDecoded) {
                    pages.add(VisiblePage(page, docTop, pageHeight, contentHeight, cropFor(page)))
                }
            }

            // A zero-height page never advances y, so stop.
            if (pageHeight <= 0f) break

            prevHeight = pageHeight
            y += pageHeight
            i++
        }

        onScreenPages = visible
        pagesBelow = below
        pagesAbove = above

        if (scrolledThrough == null) {
            scrolledThrough = getPage(-1)?.takeIf { getPageHeight(it) > 0f }
        }

        ContinuousRenderSnapshot(
            pages, scale, offsetX, cameraDocY, isScaleAnimating || isFlinging, backgroundColor,
            scrolledThrough
        )
    }

    /**
     * Limits [pass] to [vp]'s content band, where a cropped page's margins would otherwise land
     * on its neighbours. False when none of it is on screen.
     */
    private fun scissorToSlot(
        pass: GPURenderPassEncoder,
        anchorX: Float,
        anchorY: Float,
        vp: VisiblePage,
        scale: Float,
        dstW: Float,
        dstH: Float,
    ): Boolean {
        val l = (anchorX - scale * dstW / 2f).roundToInt().coerceIn(0, dstW.toInt())
        val r = (anchorX + scale * dstW / 2f).roundToInt().coerceIn(0, dstW.toInt())
        val t = (anchorY + scale * vp.docTop).roundToInt().coerceIn(0, dstH.toInt())
        val b = (anchorY + scale * (vp.docTop + vp.contentHeight)).roundToInt()
            .coerceIn(0, dstH.toInt())
        if (r <= l || b <= t) return false
        pass.setScissorRect(l, t, r - l, b - t)
        return true
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder, texture: GPUTexture, snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        tiles.newFrame()
        // Still clear: the swapchain rotates buffers, so leaving it would show a stale frame.
        if (s.pages.isEmpty()) {
            Draw.clear(encoder, texture, s.backgroundColor)
            return
        }

        // ImageSingle pages batch into one shared pass; a Render page has no image or tile to
        // draw, so it goes afterward through renderLoaded. renderLoaded loads rather than clears
        // (clearing this shared texture would blank the other pages), so something must clear it
        // first: the ImageSingle batch's pass, or Draw.clear when every page is a Render page.
        val hasImagePage = s.pages.any { it.page is ImagePage.ImageSingle }

        val dstW = texture.width.toFloat()
        val dstH = texture.height.toFloat()
        // Screen position of document space's origin - must mirror TileRenderer's continuous
        // anchor exactly, or the fast path, tile cache and Render pages disagree on placement.
        val anchorX = dstW / 2f + s.scale * (s.offsetX * dstW + WebGpuRenderer.offsetX * dstW)
        val anchorY = dstH / 2f - s.scale * s.cameraDocY + s.scale * WebGpuRenderer.offsetY * dstH

        if (hasImagePage) {
            renderPass(encoder, texture, s.backgroundColor) { pass ->
                s.pages.forEach { vp ->
                    val page = vp.page as? ImagePage.ImageSingle ?: return@forEach
                    // Captured on the main thread: the page can have been evicted since, its
                    // image buffers gone, and drawing one throws.
                    if (page.destroyed || !page.isDecoded || page.width <= 0) return@forEach

                    // A crop fills the width, and the page's own centre sits off the crop's.
                    val crop = vp.crop
                    val pageScale = dstW / (crop?.width() ?: page.width)
                    val shiftX = crop?.let { pageScale * (page.width / 2f - it.exactCenterX()) }
                        ?: 0f
                    val shiftY = crop?.let { pageScale * (page.height / 2f - it.exactCenterY()) }
                        ?: 0f
                    if (crop != null &&
                        !scissorToSlot(pass, anchorX, anchorY, vp, s.scale, dstW, dstH)
                    ) return@forEach

                    // Tiles first, marking the stencil; the sampler below shades only what is
                    // left, and nothing once the draw reports full coverage. Animated pages
                    // never get tiles.
                    val covered = !page.isAnimated && tiles.draw(
                        pass,
                        page,
                        texture,
                        s.cameraDocY,
                        vp.docTop,
                        vp.contentHeight,
                        s.offsetX,
                        s.scale,
                        s.suppressGeneration,
                        crop,
                    )
                    if (!covered) {
                        page.forEachImage { image, srcOffsetX, sideScale ->
                            if (image.mipmaps.isEmpty()) return@forEachImage
                            val imageScale = pageScale * s.scale * sideScale
                            val docCenterX =
                                shiftX + pageScale * (srcOffsetX + sideScale * image.x)
                            val docCenterY = vp.docTop + 0.5f * vp.contentHeight + shiftY +
                                    pageScale * sideScale * image.y
                            val targetX = anchorX + s.scale * docCenterX
                            val targetY = anchorY + s.scale * docCenterY
                            val (x, y) = solveImagePlacement(
                                targetX, targetY, imageScale, image, dstW, dstH
                            )
                            // Non-highQuality content skips linear light; an animated page
                            // swaps images every frame, so it takes the fast sampler too.
                            if (page.isAnimated || page.highQuality) {
                                RenderPage.renderFast(pass, image, texture, x, y, imageScale)
                            } else {
                                RenderPage.renderFast(
                                    pass, image, texture, x, y, imageScale, linear = false
                                )
                            }
                        }
                    }

                    // Last, and over this page's content band only - pages tile vertically, so
                    // veiling anything taller fades its neighbours too.
                    if (page.fade < 1f) {
                        val top = anchorY + s.scale * vp.docTop
                        page.drawFade(
                            pass,
                            texture.format,
                            (anchorX - s.scale * dstW / 2f) / dstW,
                            top / dstH,
                            (anchorX + s.scale * dstW / 2f) / dstW,
                            (top + s.scale * vp.contentHeight) / dstH
                        )
                    }
                    if (crop != null) pass.setScissorRect(0, 0, texture.width, texture.height)
                }
            }
        } else {
            Draw.clear(encoder, texture, s.backgroundColor)
        }

        s.pages.forEach { vp ->
            if (vp.page is ImagePage.ImageSingle) return@forEach
            // captureRenderState's isDecoded filter already excluded anything that isn't a
            // Render page with drawable content.
            val page = vp.page as ImagePage.Render
            if (page.destroyed) return@forEach

            // Render's x/y/scale are fractions of dst, not of this page's own width/height,
            // which is never stretched to the viewer's width (see [getPageHeight]). So no
            // dstW/page.width factor belongs here - folding one in scales its content by that
            // ratio. page.x/page.y are left out for the same reason: they're in that
            // dst-fraction unit and can't be added to vp.docTop's doc-space pixels.
            val renderScale = s.scale * page.scale
            val targetX = anchorX
            val targetY = anchorY + s.scale * (vp.docTop + 0.5f * vp.contentHeight)

            val x = (targetX - dstW / 2f) / (renderScale * dstW)
            val y = (targetY - dstH / 2f) / (renderScale * dstH)
            page.renderLoaded(encoder, x, y, renderScale, texture)
        }
    }
}
