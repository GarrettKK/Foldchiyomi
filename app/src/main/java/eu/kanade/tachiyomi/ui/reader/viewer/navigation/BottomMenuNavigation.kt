package eu.kanade.tachiyomi.ui.reader.viewer.navigation

import android.graphics.RectF
import eu.kanade.tachiyomi.ui.reader.viewer.ViewerNavigation

/**
 * Visualization of default state without any inversion
 * +---+---+---+
 * | - | - | - |   P: Previous
 * +---+---+---+
 * | - | - | - |   M: Menu
 * +---+---+---+
 * | M | M | M |   N: Next
 * +---+---+---+
 *
 * Only the bottom edge opens the menu; a tap anywhere else does nothing, so that a double tap
 * or a long press that misses a speech bubble doesn't bring the menu up over the page.
 */
class BottomMenuNavigation : ViewerNavigation() {

    override var regionList: List<Region> = listOf(
        Region(
            rectF = RectF(0f, 0f, 1f, 0.88f),
            type = NavigationRegion.NONE,
        ),
    )
}
