package ca.mpreg.webgpuviewer.viewer

import ca.mpreg.webgpuviewer.renderer.InkMap

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

    /** The panel around map pixel ([x], [y]): its bounds (x0, y0, x1, y1), x1 and y1 exclusive. */
    fun find(map: InkMap, x: Int, y: Int): IntArray? {
        val w = map.width
        val h = map.height
        val luma = map.luma
        fun paper(i: Int) = (luma[i].toInt() and 0xFF) >= PAPER

        // The gutters: paper reachable from the page's edges.
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

        // On the gutter, or on white open to it - a borderless panel - there is nothing to frame.
        val start = y * w + x
        if (gutter[start]) return null

        // The panel: what the tap's pixel reaches without crossing gutter.
        val inside = BooleanArray(w * h)
        head = 0
        tail = 0
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
