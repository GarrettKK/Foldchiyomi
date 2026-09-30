package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.viewer.ImagePage

object TransitionStackRight : Transition() {
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
                page1.drawBackgroundColumns(pass, dst, 0f, 0f)
                blitCached(pass, dst.format, cached1, 0f, 0f)
                page2.drawBackgroundColumns(pass, dst, 1f - frac, 0f)
                blitCached(pass, dst.format, cached2, 1f - frac, 0f)
            } else {
                page2.drawBackgroundColumns(pass, dst, 0f, 0f)
                blitCached(pass, dst.format, cached2, 0f, 0f)
                page1.drawBackgroundColumns(pass, dst, -frac, 0f)
                blitCached(pass, dst.format, cached1, -frac, 0f)
            }
        } finally {
            pass.endAndRelease()
        }
    }
}
