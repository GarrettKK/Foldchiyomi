package ca.mpreg.webgpuviewer.viewer

import ca.mpreg.webgpuviewer.renderer.InkMap
import kotlin.math.max

/**
 * Finds the comic panel under a tap, for [ImageViewerState.zoomToPanel]: the page zooms to
 * frame it, rather than lifting it off the page as [BubbleZoom] does with a bubble.
 *
 * Panels are told apart by the paper between them: the gutters are white and run out to the
 * page's margins, so flooding the paper in from the edges paints every gutter at once, and what
 * is left unpainted is the panels. The one under the tap is cut out by its bounds.
 *
 * It gives up, and the reader zooms as usual, on what that can't separate: a borderless panel
 * whose white runs into the gutter, a dark page with black gutters, a page that is one panel.
 */
internal object PanelZoom {
    /** Paper at least this bright, out of 255, is gutter. Min-pooled luma runs a little dark. */
    private const val PAPER = 170

    /** A panel takes at least and at most this share of the page. */
    private const val MIN_AREA = 0.04f
    private const val MAX_AREA = 0.85f

    /** A panel fills its bounds at least this much - two joined at a corner fill far less. */
    private const val MIN_FILL = 0.55f

    /** Added around the panel's bounds, as a share of the page's side, to keep its border line. */
    private const val MARGIN = 0.012f

    /** How deep [panelOrder]'s cuts go - enough for any real layout. */
    private const val MAX_CUT_DEPTH = 6

    /** A gutter at least this share of the piece's side across splits it - a speck doesn't. */
    private const val MIN_GAP = 0.004f

    /** A line is a gutter when at least this share of it is paper - art may touch it. */
    private const val CLEAR_LINE = 0.97f

    /**
     * The panel under the tap at ([tapX], [tapY]) - normalised screen coordinates on a
     * [screenWidth] x [screenHeight] surface - as its bounds on screen (x1, y1, x2, y2), or null
     * where there is none to frame. Off the GPU thread, like [BubbleZoom.detect].
     */
    fun detect(
        page: ImagePage.ImageSingle,
        screenWidth: Int,
        screenHeight: Int,
        tapX: Float,
        tapY: Float,
    ): FloatArray? {
        val hit = BubbleZoom.imageAt(page, screenWidth, screenHeight, tapX, tapY) ?: return null
        val map = hit.image.inkMap ?: return null
        val imageRect = hit.rect
        val u = (tapX - imageRect[0]) / (imageRect[2] - imageRect[0])
        val v = (tapY - imageRect[1]) / (imageRect[3] - imageRect[1])
        val found = find(
            map,
            (u * map.width).toInt().coerceIn(0, map.width - 1),
            (v * map.height).toInt().coerceIn(0, map.height - 1),
        ) ?: return null
        val sx = (imageRect[2] - imageRect[0]) / map.width
        val sy = (imageRect[3] - imageRect[1]) / map.height
        return floatArrayOf(
            imageRect[0] + found[0] * sx,
            imageRect[1] + found[1] * sy,
            imageRect[0] + found[2] * sx,
            imageRect[1] + found[3] * sy,
        )
    }

    /** The gutters of [map]: paper reachable from the page's edges, one flag per map pixel. */
    private fun gutters(map: InkMap): BooleanArray {
        val w = map.width
        val h = map.height
        val luma = map.luma
        fun paper(i: Int) = (luma[i].toInt() and 0xFF) >= PAPER
        val gutter = BooleanArray(w * h)
        val queue = IntArray(w * h)
        var head = 0
        var tail = 0
        fun reach(i: Int) {
            if (!gutter[i] && paper(i)) {
                gutter[i] = true
                queue[tail++] = i
            }
        }
        for (px in 0 until w) {
            reach(px)
            reach((h - 1) * w + px)
        }
        for (py in 0 until h) {
            reach(py * w)
            reach(py * w + w - 1)
        }
        while (head < tail) {
            val i = queue[head++]
            val px = i % w
            val py = i / w
            if (px > 0) reach(i - 1)
            if (px < w - 1) reach(i + 1)
            if (py > 0) reach(i - w)
            if (py < h - 1) reach(i + w)
        }
        return gutter
    }

    /**
     * The page cut into panels in reading order, as rectangles (x0, y0, x1, y1; x1 and y1
     * exclusive) that between them cover the whole map. An XY-cut: the page is split along
     * every gutter that crosses it top to bottom or side to side, each piece is split again
     * the other way, and so on until nothing crosses. Bands read top to bottom; pieces across
     * a band read right to left where [rightToLeft], else left to right. A page with no
     * crossing gutter is one piece.
     */
    fun panelOrder(map: InkMap, rightToLeft: Boolean): List<IntArray> {
        val gutter = gutters(map)
        val out = ArrayList<IntArray>()
        cut(gutter, map.width, 0, 0, map.width, map.height, horizontal = true, depth = 0, rightToLeft, out)
        return out
    }

    private fun cut(
        gutter: BooleanArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int,
        horizontal: Boolean, depth: Int, rightToLeft: Boolean, out: MutableList<IntArray>,
    ) {
        if (depth < MAX_CUT_DEPTH) {
            var bands = split(gutter, w, x0, y0, x1, y1, horizontal)
            var axisHorizontal = horizontal
            if (bands.size <= 1) {
                bands = split(gutter, w, x0, y0, x1, y1, !horizontal)
                axisHorizontal = !horizontal
            }
            if (bands.size > 1) {
                if (!axisHorizontal && rightToLeft) bands = bands.reversed()
                for ((a, b) in bands) {
                    if (axisHorizontal) {
                        cut(gutter, w, x0, a, x1, b, horizontal = false, depth + 1, rightToLeft, out)
                    } else {
                        cut(gutter, w, a, y0, b, y1, horizontal = true, depth + 1, rightToLeft, out)
                    }
                }
                return
            }
        }
        out += intArrayOf(x0, y0, x1, y1)
    }

    /**
     * The spans along one axis between the gutters crossing the rectangle the other way, in
     * order along the axis and covering it end to end - rows for [horizontal], else columns.
     */
    private fun split(
        gutter: BooleanArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int, horizontal: Boolean
    ): List<Pair<Int, Int>> {
        val from = if (horizontal) y0 else x0
        val to = if (horizontal) y1 else x1
        val across = if (horizontal) x1 - x0 else y1 - y0
        val minGap = max(2, ((if (horizontal) y1 - y0 else x1 - x0) * MIN_GAP).toInt())
        // Where a line across the rectangle is (nearly) all gutter.
        fun clear(at: Int): Boolean {
            var gutterCount = 0
            if (horizontal) {
                for (px in x0 until x1) if (gutter[at * w + px]) gutterCount++
            } else {
                for (py in y0 until y1) if (gutter[py * w + at]) gutterCount++
            }
            return gutterCount >= CLEAR_LINE * across
        }
        val gaps = ArrayList<Pair<Int, Int>>()
        var at = from
        while (at < to) {
            if (!clear(at)) {
                at++
                continue
            }
            val gapStart = at
            while (at < to && clear(at)) at++
            // A margin at either end splits nothing.
            if (at - gapStart >= minGap && gapStart > from && at < to) gaps += gapStart to at
        }
        if (gaps.isEmpty()) return listOf(from to to)
        val bands = ArrayList<Pair<Int, Int>>()
        var start = from
        for ((a, b) in gaps) {
            val mid = (a + b) / 2
            bands += start to mid
            start = mid
        }
        bands += start to to
        return bands
    }

    /** The panel around map pixel ([x], [y]): its bounds (x0, y0, x1, y1), x1 and y1 exclusive. */
    fun find(map: InkMap, x: Int, y: Int): IntArray? {
        val w = map.width
        val h = map.height
        val gutter = gutters(map)

        // On the gutter, or on white open to it - a borderless panel - there is nothing to frame.
        val start = y * w + x
        if (gutter[start]) return null

        // The panel: what the tap's pixel reaches without crossing gutter.
        val inside = BooleanArray(w * h)
        val queue = IntArray(w * h)
        var head = 0
        var tail = 0
        inside[start] = true
        queue[tail++] = start
        var x0 = x
        var x1 = x
        var y0 = y
        var y1 = y
        while (head < tail) {
            val i = queue[head++]
            val px = i % w
            val py = i / w
            if (px < x0) x0 = px
            if (px > x1) x1 = px
            if (py < y0) y0 = py
            if (py > y1) y1 = py
            for (n in intArrayOf(
                if (px > 0) i - 1 else -1,
                if (px < w - 1) i + 1 else -1,
                if (py > 0) i - w else -1,
                if (py < h - 1) i + w else -1,
            )) {
                if (n < 0 || inside[n] || gutter[n]) continue
                inside[n] = true
                queue[tail++] = n
            }
        }
        val count = tail

        val bw = x1 - x0 + 1
        val bh = y1 - y0 + 1
        val area = bw.toFloat() * bh
        if (area < MIN_AREA * w * h || area > MAX_AREA * w * h) return null
        if (count < MIN_FILL * area) return null
        // Against three or more edges it is the page, not a panel on it.
        var edges = 0
        if (x0 == 0) edges++
        if (y0 == 0) edges++
        if (x1 == w - 1) edges++
        if (y1 == h - 1) edges++
        if (edges >= 3) return null

        val mx = (MARGIN * w).toInt()
        val my = (MARGIN * h).toInt()
        return intArrayOf(
            (x0 - mx).coerceAtLeast(0),
            (y0 - my).coerceAtLeast(0),
            (x1 + 1 + mx).coerceAtMost(w),
            (y1 + 1 + my).coerceAtMost(h),
        )
    }
}
