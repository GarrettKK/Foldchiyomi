package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage

/**
 * Slide transition: both pages go to cached textures, then get blitted side by side at an offset.
 *
 * Drawing a page is [TileRenderer.renderFullyTiled]'s job; this only decides where they end up.
 *
 * [getCachedTexture] keys on the page's own transform, and a page turn only animates the offset,
 * so that render happens once per transition and every later frame is a cache hit plus a 1:1 blit.
 * Each page's background is drawn separately, live, at the same offset - see
 * [ImagePage.drawBackgroundColumns].
 */
object TransitionBasic : Transition() {
    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        val pass = beginClearedPass(encoder, dst)
        try {
            if (frac > 0f) {
                page2.drawBackgroundColumns(pass, dst, 1f - frac, 0f)
                page1.drawBackgroundColumns(pass, dst, -frac, 0f)
                blitCached(pass, dst.format, cached2, 1f - frac, 0f)
                blitCached(pass, dst.format, cached1, -frac, 0f)
            } else {
                page2.drawBackgroundColumns(pass, dst, -(frac + 1f), 0f)
                page1.drawBackgroundColumns(pass, dst, -frac, 0f)
                blitCached(pass, dst.format, cached2, -(frac + 1f), 0f)
                blitCached(pass, dst.format, cached1, -frac, 0f)
            }
        } finally {
            pass.endAndRelease()
        }
    }

    /** The same slide, vertically. */
    object Vertical : Transition() {
        override fun render(
            page1: ImagePage,
            page2: ImagePage,
            encoder: GPUCommandEncoder,
            dst: GPUTexture,
            frac: Float,
            pos1: Offset,
            pos2: Offset,
            tiles: TileRenderer,
        ) {
            val cached1 =
                getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

            val cached2 =
                getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

            val pass = beginClearedPass(encoder, dst)
            try {
                if (frac > 0f) {
                    page1.drawBackgroundColumns(pass, dst, 0f, -frac)
                    page2.drawBackgroundColumns(pass, dst, 0f, 1f - frac)
                    blitCached(pass, dst.format, cached1, 0f, -frac)
                    blitCached(pass, dst.format, cached2, 0f, 1f - frac)
                } else {
                    page2.drawBackgroundColumns(pass, dst, 0f, -(frac + 1f))
                    page1.drawBackgroundColumns(pass, dst, 0f, -frac)
                    blitCached(pass, dst.format, cached2, 0f, -(frac + 1f))
                    blitCached(pass, dst.format, cached1, 0f, -frac)
                }
            } finally {
                pass.endAndRelease()
            }
        }
    }
}
