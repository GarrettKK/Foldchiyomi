package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.transition.TransitionNone.render
import ca.mpreg.webgpuviewer.viewer.ImagePage

/**
 * No animation: [page1] stays on screen for the whole drag and the turn just jumps once it
 * commits (handled outside [render] - by the time [page2] would show, offset is back to 0 and
 * the normal live-render path takes over). [frac]/[page2] are unused since there's nothing to
 * blend or slide toward. Draws [page1] straight to [dst] via [ImagePage.renderCacheSeed] (which
 * opens its own pass), skipping the getCachedTexture/blitCached cache-texture indirection every
 * other transition uses.
 */
object TransitionNone : Transition() {
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
        page1.renderCacheSeed(encoder, dst, tiles)
    }
}
