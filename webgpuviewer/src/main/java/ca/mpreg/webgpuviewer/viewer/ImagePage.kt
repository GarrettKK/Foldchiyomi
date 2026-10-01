package ca.mpreg.webgpuviewer.viewer

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.compose.animation.core.AnimationSpec
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDepthStencilAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.closeTo
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.Font
import ca.mpreg.webgpuviewer.draw.TextAlign
import ca.mpreg.webgpuviewer.draw.circle
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.draw.text
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds

/**
 * A page in the viewer, with shared transform (x, y, scale), pan/zoom-to-fit bounds, and
 * animation.
 *
 * [ImageSingle] is the ordinary single-page case; [ImageSpread] composes two [ImageSingle] pages
 * side by side for a dual-page spread. [Dummy] is a placeholder with known dimensions but no content.
 * [Render]/[Progress] draw their own content via [renderWith] instead of blitting an image.
 *
 * Handles:
 * - Transform state (x, y, scale) and home position calculations
 * - Pan/zoom animation
 * - Lifecycle (cleanup)
 */
open class ImagePage {

    /** Placeholder page with dimensions but no image data */
    class Dummy(override val width: Int, override val height: Int) : ImagePage()

    /**
     * ImagePage whose content is supplied by an app-overridden [render] instead of a decoded
     * image - for rendering progress indicators or other app-drawn content with its own shader.
     * Called instead of blitting a texture wherever this page would otherwise be drawn into the
     * regular view or a [ca.mpreg.webgpuviewer.transition.Transition]'s cache texture - unlike
     * [ImageSingle], it has no [ImageSingle.highQuality] to opt into either path, so it always
     * draws this way.
     *
     * Opens one pass per [render] call (no stencil attachment - unlike [ImageSingle], nothing here
     * needs [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s masking), shared by every [rect]/
     * [circle] call inside it rather than each opening its own.
     *
     * [renderWith] clears that pass's destination first - right whenever this page owns the whole
     * thing (a rotated screen buffer, or a [ca.mpreg.webgpuviewer.transition.Transition]'s
     * per-page cache slot). [renderLoaded] is the one exception:
     * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] draws several pages into one
     * shared screen texture, so a Render page's destination there is that whole shared texture,
     * not a page-sized one of its own - clearing it would blank every other visible page too.
     *
     * Override [backgroundColor] to fill a background before [render] runs: when
     * [renderWith] owns the whole destination, that fill is just its clear color, so it covers
     * the full screen (letterboxing beyond this page's own footprint, same as a transition
     * blending toward it); [renderLoaded] instead fills just [fillPage]'s scoped rect, since it
     * can't clear the shared texture. Leave it null to paint nothing and rely entirely on [render].
     */
    open class Render(override val width: Int, override val height: Int) : ImagePage() {

        // Always has drawable content via render(), unlike Dummy - needed so
        // Transition.getCachedTexture doesn't skip this page as if it were undecoded.
        override val isDecoded: Boolean get() = true

        // Unlike the base ImagePage default (fixed "home" position, right for Dummy - it has
        // nothing pan-worthy anyway), a Render page's own render() gets its *live* pan/zoom
        // transform - the same x/y/scale gestures already drive on any ImagePage - so e.g. a
        // custom Render page's own drawn content can track a drag/pinch the same way an
        // ImageSingle page's would, rather than being stuck rendering at (0, 0, 1) forever.
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            renderWith(encoder, x, y, scale, dst)
            return false
        }

        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            renderWith(encoder, x, y, scale, tex)
        }

        // The rect [fillPage] fills, not all of [dst] - a page smaller than the surface would
        // otherwise warp stretched to a height it never asked for. Non-null, so transitions that
        // bail on a null pageRect still draw a Render page.
        override fun pageRect(dst: GPUTexture): FloatArray {
            val halfWidthFrac = scale * width / (2f * dst.width)
            val halfHeightFrac = scale * height / (2f * dst.height)
            val cx = 0.5f + scale * x
            val cy = 0.5f + scale * y
            return floatArrayOf(
                cx - halfWidthFrac, cy - halfHeightFrac, cx + halfWidthFrac, cy + halfHeightFrac
            )
        }

        // Set by renderWith right before calling render(), and only valid for the duration of
        // that call - rect/circle/text read it instead of taking a pass parameter, since there's
        // only ever one pass open per render() call (see the class doc).
        private lateinit var pass: GPURenderPassEncoder

        /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
        private var passFormat: Int = TextureFormat.RGBA8Unorm

        /** Draws this page's content. Use [rect]/[circle]/[text] to draw into the open pass. */
        open fun render(dst: GPUTexture, x: Float, y: Float, scale: Float) {}

        private val _renderVersion = AtomicInteger()

        /** Bumped by [invalidate], so a page turn sees the drawn content change. */
        override val frameVersion: Int
            get() = _renderVersion.get()

        /**
         * As [ImagePage.invalidate], bumping [frameVersion] so a page turn in flight re-seeds its
         * cache slot too - content that moves on its own (a progress ring, a countdown) would
         * otherwise freeze for the length of the turn.
         */
        override fun invalidate() {
            _renderVersion.incrementAndGet()
            super.invalidate()
        }

        protected fun rect(x1: Float, y1: Float, x2: Float, y2: Float, color: Int) =
            Draw.rect(pass, passFormat, x1, y1, x2, y2, color)

        /**
         * Fills this page's own [width] x [height] footprint with [color] - unlike [rect]'s raw
         * [dst]-relative `[0,1]` coordinates (which always cover the *entire* [dst], full stop),
         * this is sized and positioned from this page's own declared [width]/[height] plus the
         * same [x]/[y]/[scale] [render] itself received, rather than assuming this page fills the
         * whole of [dst] the way [renderWith]'s full-[dst] clear does. Matters once [dst] is
         * shared with other pages ([ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState])
         * instead of being this page's own - filling all of `[0,1]` there would blank every other
         * visible page too, and even a same-sized fill would come out the wrong aspect ratio
         * whenever [dst] (the real screen) isn't [width]x[height]'s own aspect.
         */
        protected fun fillPage(dst: GPUTexture, x: Float, y: Float, scale: Float, color: Int) {
            val halfWidthFrac = scale * width / (2f * dst.width)
            val halfHeightFrac = scale * height / (2f * dst.height)
            val cx = 0.5f + scale * x
            val cy = 0.5f + scale * y
            rect(
                cx - halfWidthFrac,
                cy - halfHeightFrac,
                cx + halfWidthFrac,
                cy + halfHeightFrac,
                color
            )
        }

        protected fun circle(cx: Float, cy: Float, radius: Float, color: Int) =
            Draw.circle(pass, passFormat, cx, cy, radius, color)

        protected fun text(
            dst: GPUTexture,
            font: Font,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(pass, dst, font, text, x, y, size, color, align, maxWidth)

        protected fun text(
            dst: GPUTexture,
            context: Context,
            fontFamily: FontFamily,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            weight: FontWeight = FontWeight.Normal,
            style: FontStyle = FontStyle.Normal,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(
            pass, dst, context, fontFamily, text, x, y, size, color, weight, style, align, maxWidth
        )

        final override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first - see
            // ImagePage.renderWith's default for why that's still needed for a page (like Dummy)
            // that doesn't override this at all. [dst] is this call's own - a rotated screen
            // buffer or a Transition's per-page cache slot - so nothing else on screen depends on
            // whatever was already there.
            openPassAndRender(encoder, x, y, scale, dst, clear = true)
        }

        /**
         * As [renderWith], but loads [dst] instead of clearing it -
         * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] uses this instead, since there
         * [dst] is one screen texture shared by several visible pages at once, and clearing it
         * would blank every other one. [render] is responsible for painting over every pixel of
         * its own footprint here - see [Progress] painting a full background rect before its
         * circle - since anything it doesn't touch keeps whatever [dst] already had (a decoded
         * neighbour's pixels, or simply undefined content the first time a fresh texture is used
         * - [ImageViewerContinuousState] clears once up front to guard against that).
         */
        internal fun renderLoaded(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            openPassAndRender(encoder, x, y, scale, dst, clear = false)
        }

        private fun argbToGPUColor(color: Int): GPUColor {
            val r = ((color shr 16) and 0xFF) / 255.0
            val g = ((color shr 8) and 0xFF) / 255.0
            val b = (color and 0xFF) / 255.0
            val a = ((color ushr 24) and 0xFF) / 255.0
            return GPUColor(r, g, b, a)
        }

        // Background fill driven by backgroundColor() (null by default; override it to
        // opt in - see Progress/an app's own transition placeholder) instead of a dedicated
        // property, since that's the exact same hook every Transition already reads for this
        // page's letterbox colour - one override covers both without the two ever disagreeing.
        private fun openPassAndRender(
            encoder: GPUCommandEncoder,
            x: Float,
            y: Float,
            scale: Float,
            dst: GPUTexture,
            clear: Boolean
        ) {
            val clearValue =
                backgroundColor?.let { argbToGPUColor(it) } ?: GPUColor(0.0, 0.0, 0.0, 0.0)

            val targetView = dst.createView()
            val openedPass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = targetView,
                            loadOp = if (clear) LoadOp.Clear else LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = clearValue
                        )
                    )
                )
            )
            pass = openedPass
            passFormat = dst.format
            try {
                // clear=true already painted the whole dst this colour via clearValue above - a
                // page-scoped fillPage on top would be redundant. clear=false (the shared-texture
                // continuous-mode pass) has no clear to fall back on, so paint the footprint here.
                if (!clear) {
                    backgroundColor?.let { fillPage(dst, x, y, scale, it) }
                }
                render(dst, x, y, scale)
            } finally {
                openedPass.endAndRelease(targetView)
            }
        }
    }

    /**
     * A page backed by a single decoded image, read straight off [currentImage]. [ImageSpread]
     * extends this to compose two side by side, overriding each member that needs both; pass
     * setup, tile cache and background fade math are shared as-is.
     */
    open class ImageSingle(val image: Image?) : ImagePage() {

        /** Animated from the start - no separate [startAnimationLoop] call needed. */
        constructor(frames: List<Pair<Image, Int>>) : this(frames.firstOrNull()?.first) {
            startAnimationLoop(frames)
        }

        /** If true, this page owns its image and will clean it up. If false, it's borrowed - see
         *  [ImageSpread], which composes existing pages without taking ownership of their images. */
        var ownsImage: Boolean = true

        /**
         * When false, this page skips [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s tile cache
         * entirely and its fast path renders through [renderPage] with `linear = false` instead of
         * the default `linear = true` - for content not worth either path's extra correctness or
         * sharpness, such as an app-drawn transition/error bitmap. A `var`, not a `val`, so
         * [ImageSpread] can override it to fan a set-through to both sides instead of just holding
         * its own copy.
         */
        open var highQuality: Boolean = true

        override val width: Int
            get() = image?.width ?: 0
        override val height: Int
            get() = image?.height ?: 0

        override val trimWidth: Int
            get() = image?.let { it.trim?.width() ?: it.width } ?: 0
        override val trimHeight: Int
            get() = image?.let { it.trim?.height() ?: it.height } ?: 0

        override fun xEdges(trimmed: Boolean): Pair<Float, Float> {
            if (!trimmed) return 0f to width.toFloat()
            val trim = image?.trim ?: return 0f to width.toFloat()
            return trim.left.toFloat() to trim.right.toFloat()
        }

        override fun yEdges(trimmed: Boolean): Pair<Int, Int>? {
            if (!trimmed) return null
            val trim = image?.trim ?: return null
            return trim.top to trim.bottom
        }

        private var animationLoop: Job? = null

        // Written by the animation loop on its own dispatcher, read by the render thread.
        @Volatile
        private var frames: List<Pair<Image, Int>>? = null

        @Volatile
        private var currentFrameImage: Image? = null

        /** True while an animation frame loop owns [currentImage]. The tile cache skips animated pages. */
        override val isAnimated: Boolean
            get() = frames != null

        private val _frameVersion = AtomicInteger()

        /** Incremented each time the animation frame changes. Used by the render cache to detect stale frames. */
        override val frameVersion: Int
            get() = _frameVersion.get() + contentVersion

        /** Uploads into the image(s) so far; unlike [frameVersion], a fade or redraw leaves it. */
        open val contentVersion: Int
            get() = image?.contentVersion ?: 0

        /** Current image for rendering (may change during animation) */
        open val currentImage: Image?
            get() = currentFrameImage ?: image

        /**
         * Runs [action] for each image drawn right now, with its pixel offset from the page
         * anchor - [currentImage] at 0 here, both sides for an [ImageSpread]. For callers that
         * place images with their own math ([ca.mpreg.webgpuviewer.renderer.TileRenderer],
         * [ImageViewerContinuousState]) instead of [renderPage].
         */
        internal open fun forEachImage(
            action: (image: Image, offsetX: Float, imageScale: Float) -> Unit
        ) {
            currentImage?.let { action(it, 0f, 1f) }
        }

        /** True once at least one of this page's images has been uploaded and can be drawn. */
        internal open val hasUploadedImage: Boolean
            get() = currentImage?.mipmaps?.isNotEmpty() == true

        override val isDecoded: Boolean
            get() = currentImage != null

        /**
         * Left/right extent from this page's own anchor, in raw pixels at scale 1 - symmetric
         * halves of [width] by default; [ImageSpread] overrides for its asymmetric seam-based
         * shape.
         */
        internal open fun horizontalExtent(): Pair<Float, Float> {
            val half = width / 2f
            return half to half
        }

        /** As [ImagePage.invalidate], over the same [frameVersion] the animation loop drives. */
        override fun invalidate() {
            _frameVersion.incrementAndGet()
            super.invalidate()
        }

        @Synchronized
        fun startAnimationLoop(frames: List<Pair<Image, Int>>) {
            // Nothing cancels the loop twice, so one started after cleanup holds every frame.
            if (destroyed) return

            animationLoop?.cancel()
            this.frames = frames
            currentFrameImage = frames.firstOrNull()?.first

            // Use the page's scope if available, otherwise use the shared background scope
            val loopScope = scope ?: cleanupScope
            animationLoop = loopScope.launch {
                var frameIndex = 0
                while (true) {
                    synchronized(this@ImageSingle) {
                        if (destroyed) null
                        else this@ImageSingle.frames?.getOrNull(frameIndex)
                            ?.also { currentFrameImage = it.first }
                    }?.let { (_, duration) ->
                        // Keeps running off screen - frames stay in step with their durations,
                        // and invalidate() asks for a redraw only while there is one to ask for.
                        invalidate()
                        if ((this@ImageSingle.frames?.size ?: 0) <= 1) return@launch
                        delay(duration.coerceAtLeast(MIN_FRAME_MILLIS).milliseconds)
                    } ?: break
                    frameIndex = (frameIndex + 1) % (this@ImageSingle.frames?.size ?: 1)
                }
            }
        }

        override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first. No
            // stencil attachment - this fallback never needs TileRenderer's masking, unlike the
            // overrides below that bypass it entirely.
            val targetView = dst.createView()
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = targetView,
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                renderPage(pass, dst, x, y, scale, linear = false, masked = false)
            } finally {
                pass.endAndRelease(targetView)
            }
        }

        /** Opens a `LoadOp.Clear` pass on [dst], with a stencil attachment for [TileRenderer]'s masking. */
        private fun beginLivePass(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): GPURenderPassEncoder {
            val targetView = dst.createView()
            // The pass holds its own reference to its attachment, so ours can go at once.
            return encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = targetView,
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    ), depthStencilAttachment = GPURenderPassDepthStencilAttachment(
                        view = tiles.stencilViewFor(dst),
                        stencilLoadOp = LoadOp.Clear,
                        stencilStoreOp = StoreOp.Discard,
                        stencilClearValue = 0,
                    )
                )
            ).also { targetView.close() }
        }

        /** Opens a `LoadOp.Clear` pass on [tex], no stencil attachment - a transition's cache is never masked. */
        private fun beginCachePass(
            encoder: GPUCommandEncoder,
            tex: GPUTexture
        ): GPURenderPassEncoder {
            val targetView = tex.createView()
            return encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = targetView,
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            ).also { targetView.close() }
        }

        /**
         * As [ImagePage.drawLive]. Animated frames always want the fast path regardless of
         * [highQuality] (never worth a tile cache that would just churn every frame); a
         * non-[highQuality], non-animated page falls back to the plain [renderWith] (via `super`);
         * everything else goes through the tile cache, backfilling with [renderPage] wherever it
         * isn't covered yet. Opens its own pass (with a stencil attachment, for [TileRenderer]'s
         * masking) rather than sharing one from the caller - see [Render] for why that split exists.
         */
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            if (isAnimated) {
                val pass = beginLivePass(encoder, dst, tiles)
                try {
                    renderBackground(pass, dst, 0f, 0f, 1f)
                    renderPage(pass, dst, 0f, 0f, 1f)
                } finally {
                    pass.endAndRelease()
                }
                return false
            }

            if (!highQuality) return super.drawLive(encoder, dst, tiles)

            val pass = beginLivePass(encoder, dst, tiles)
            try {
                // Background always drawn live first (its fades are position-dependent, never
                // from a stale tile) so it stays underneath everything else. Tiles draw next,
                // marking every pixel they cover in the stencil buffer tiles.draw() writes to;
                // renderPage then only shades what's left uncovered instead of the whole
                // viewport, since tiles.draw() already produced the right pixel wherever it drew.
                renderBackground(pass, dst, 0f, 0f, 1f)
                val covered = tiles.draw(pass, this, dst, 0f, 0f, 1f)
                if (!covered) {
                    renderPage(pass, dst, 0f, 0f, 1f)
                }
                if (fade < 1f) fadeRect(dst)?.let {
                    drawFade(
                        pass,
                        dst.format,
                        it[0],
                        it[1],
                        it[2],
                        it[3]
                    )
                }
                return covered
            } finally {
                pass.endAndRelease()
            }
        }

        /**
         * As [drawLive], seeding a transition's cache slot instead of the screen - same
         * isAnimated/highQuality precedence, but never with a stencil attachment (a transition's
         * cache is never stencil-masked either way).
         */
        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            if (isAnimated) {
                val pass = beginCachePass(encoder, tex)
                try {
                    renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                    if (fade < 1f) {
                        fadeRect(tex)?.let {
                            drawFade(
                                pass,
                                tex.format,
                                it[0],
                                it[1],
                                it[2],
                                it[3],
                                false
                            )
                        }
                    }
                } finally {
                    pass.endAndRelease()
                }
                return
            }

            if (!highQuality) {
                super.renderCacheSeed(encoder, tex, tiles)
                return
            }

            val pass = beginCachePass(encoder, tex)
            try {
                renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                tiles.blitAvailableTiles(pass, this, tex)
                // A fade re-seeds the cache every frame (frameVersion), which is what lets a
                // page fade in mid-turn at all.
                if (fade < 1f) {
                    fadeRect(tex)?.let {
                        drawFade(
                            pass,
                            tex.format,
                            it[0],
                            it[1],
                            it[2],
                            it[3],
                            false
                        )
                    }
                }
            } finally {
                pass.endAndRelease()
            }
        }

        override fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
            if (!highQuality || isAnimated) null else tiles.availableTileKeys(this, tex)

        override fun renderIntoCache(
            encoder: GPUCommandEncoder,
            tex: GPUTexture,
            tiles: TileRenderer,
            identityMatches: Boolean
        ) {
            if (!identityMatches) {
                renderCacheSeed(encoder, tex, tiles)
                return
            }
            val targetView = tex.createView()
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = targetView,
                            loadOp = LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                tiles.blitAvailableTiles(pass, this, tex)
            } finally {
                pass.endAndRelease(targetView)
            }
        }

        override fun pageRect(dst: GPUTexture): FloatArray? {
            val image = currentImage ?: return null
            if (image.mipmaps.isEmpty()) return null
            return image.placement(dst, x, y, scale)
        }

        override val backgroundColor: Int?
            get() = currentImage?.backgroundColor

        override fun drawBackgroundColumns(
            pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
        ) = forEachBackgroundColumn(dst) { color, x1, x2 ->
            Draw.rect(pass, dst.format, offsetX + x1, offsetY, offsetX + x2, offsetY + 1f, color)
        }

        /**
         * Draw this page into [pass], one bilinear tap per pixel. [linear]/[masked] pick the same
         * 4 pipelines as the image-level [RenderPage.renderFast]. Background handling is derived
         * from the pair rather than its own parameter: masked+linear skips it (that caller already
         * drew it via [renderBackground]), masked-only folds it into the masked draw, unmasked
         * always draws a plain one alongside.
         */
        fun renderPage(
            pass: GPURenderPassEncoder,
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            linear: Boolean = true,
            masked: Boolean = true
        ) {
            val variant = RenderPage.variantFor(linear, masked)
            if (!linear || !masked) {
                drawPageBackground(pass, dst, scale, maskedBackground = masked)
            }
            forEachPlacedImage(dst, x, y, scale) { image, _, placeX, placeY, placeScale ->
                for (tile in image.prepareTilesForRender(dst, placeX, placeY, placeScale)) {
                    RenderPage.drawTile(pass, dst, tile, variant)
                }
            }
        }

        /**
         * Draw just this page's per-image background colour, skipping the image itself - for
         * ImageViewerState's masked path, which draws this first (so it stays underneath
         * [ca.mpreg.webgpuviewer.renderer.TileRenderer.draw] and [renderPage]), or as the whole
         * draw once [ca.mpreg.webgpuviewer.renderer.TileRenderer] already covers the image itself.
         * The background's alpha depends on live pan/scale, so it's drawn every frame regardless
         * of tile coverage. Uses the stencil-pass-compatible rect pipeline - see
         * [RenderPage.drawMaskedRect] - since it always runs inside that pass.
         */
        fun renderBackground(
            pass: GPURenderPassEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float
        ) = drawPageBackground(pass, dst, scale, maskedBackground = true)

        /**
         * The background's fade, from the live pan/scale - it only shows near the edges of the
         * zoom/pan range. Page level, so a spread fades as one sheet.
         */
        private fun backgroundAlpha(scale: Float): Float {
            val parent = parent ?: return 1f
            val minScale = minScale
            val homeScale = homeScale
            val currentScale = this.scale * scale

            val fadeDistancePixels = 200f
            val imageSize = (width.coerceAtLeast(height)).toFloat()

            fun proximity(anchorScale: Float): Float {
                if (anchorScale <= 0f) return 0f
                val deltaPixels = abs(imageSize * (currentScale - anchorScale))
                return (1f - deltaPixels / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundProximity(value: Float, lo: Float, hi: Float, pixelsPerUnit: Float): Float {
                val overflow = when {
                    value < lo -> lo - value
                    value > hi -> value - hi
                    else -> return 1f
                }
                return (1f - overflow * pixelsPerUnit / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundsProximityAt(anchorScale: Float): Float {
                if (anchorScale <= 0f) return 0f
                val minX = minX(anchorScale)
                val maxX = maxX(anchorScale)
                val minY = minY(anchorScale)
                val maxY = maxY(anchorScale)
                val pixelsPerUnitX = parent.width.toFloat() * anchorScale
                val pixelsPerUnitY = parent.height.toFloat() * anchorScale
                return min(
                    boundProximity(this@ImageSingle.x, minX, maxX, pixelsPerUnitX),
                    boundProximity(this@ImageSingle.y, minY, maxY, pixelsPerUnitY)
                )
            }

            if (currentScale > minScale) return boundsProximityAt(currentScale)
            val homeProximity = min(proximity(homeScale), boundsProximityAt(homeScale))
            val minProximity = min(proximity(minScale), boundsProximityAt(minScale))
            return max(homeProximity, minProximity)
        }

        /** [color] at [backgroundAlpha]. The rect blends with SrcAlpha, so only alpha fades. */
        private fun drawBackgroundRect(
            pass: GPURenderPassEncoder,
            format: Int,
            color: Int,
            x1: Float,
            x2: Float,
            scale: Float,
            maskedBackground: Boolean,
        ) {
            // A column the seam clamp collapsed.
            if (x2 <= x1) return

            val a = (((color ushr 24) and 0xFF) * backgroundAlpha(scale)).toInt()
            if (a <= 0) return

            // Alpha only. The frame clears transparent, so fading this out crossfades to whatever
            // the app painted behind the surface - the backdrop this is meant to give way to.
            // Both pipelines blend with SrcAlpha already, so scaling rgb here applied the fade a
            // second time and took the crossfade through black on its way there.
            val bgColor = (a shl 24) or (color and 0xFFFFFF)
            if (maskedBackground) {
                RenderPage.drawMaskedRect(pass, format, x1, 0f, x2, 1f, bgColor)
            } else {
                Draw.rect(pass, format, x1, 0f, x2, 1f, bgColor)
            }
        }

        /** Each [forEachBackgroundColumn] column once - they tile, so nothing blends twice. */
        private fun drawPageBackground(
            pass: GPURenderPassEncoder, dst: GPUTexture, scale: Float, maskedBackground: Boolean
        ) {
            // Outside forEachPlacedImage's walk, so it needs that guard of its own.
            if (destroyed) return
            forEachBackgroundColumn(dst) { color, x1, x2 ->
                drawBackgroundRect(pass, dst.format, color, x1, x2, scale, maskedBackground)
            }
        }

        /** Walks this page's image(s) via [forEachImage], placing each for [action] to draw against. */
        private fun forEachPlacedImage(
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            action: (image: Image, rect: FloatArray, placeX: Float, placeY: Float, placeScale: Float) -> Unit
        ) {
            // The snapshot is captured on the main thread and drawn later, so the page may have
            // been evicted since - its images' buffers are gone, and touching one throws.
            if (destroyed) return

            forEachImage { img, srcOffsetX, imgScale ->
                if (img.mipmaps.isNotEmpty()) {
                    val placeX = (this.x + x + srcOffsetX / dst.width + WebGpuRenderer.offsetX) /
                            imgScale - WebGpuRenderer.offsetX
                    val placeY =
                        (this.y + y + WebGpuRenderer.offsetY) / imgScale - WebGpuRenderer.offsetY
                    val placeScale = this.scale * scale * imgScale
                    val rect = img.placement(dst, placeX, placeY, placeScale)
                    action(img, rect, placeX, placeY, placeScale)
                }
            }
        }

        @Synchronized
        override fun cleanup() {
            if (destroyed) return
            super.cleanup()

            animationLoop?.cancel()
            animationLoop = null

            // The HDR claim was taken when the image was built, so it goes back either way.
            if (!ownsImage) {
                frames?.forEach { it.first.releaseHdr() }
                image?.releaseHdr()
                frames = null
                currentFrameImage = null
                return
            }

            val framesToClean = frames
            frames = null
            currentFrameImage = null

            // [startAnimationLoop] takes any list, so [image] may not be among the frames.
            val imagesToClean = when (framesToClean) {
                null -> listOfNotNull(image)
                else -> (framesToClean.map { it.first } + listOfNotNull(image)).distinct()
            }

            // Before the launch, not inside it: work needing the render dispatcher is what kept
            // HDR on after the last HDR page was evicted.
            imagesToClean.forEach { it.releaseHdr() }

            if (imagesToClean.isNotEmpty()) {
                cleanupScope.launch {
                    try {
                        // Eviction fires exactly when the viewer reaches a new page, so freeing a
                        // page's textures competes with the frames that are drawing the new one.
                        // Yield between images and stay off the render mutex, for the same reason
                        // uploads do. Dawn keeps a destroyed texture alive until the command buffers
                        // referencing it retire, so a frame already in flight is unaffected.
                        WebGpuRenderer.onDispatcher {
                            imagesToClean.forEach { image ->
                                image.cleanup()
                                yield()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("ImagePage", "Cleanup error", e)
                    }
                }
            }
        }
    }

    /**
     * Two pages drawn side by side, sharing one pan/zoom transform and one
     * [ca.mpreg.webgpuviewer.renderer.TileRenderer] grid, so the seam bakes into whichever tile
     * straddles it instead of meeting two independently-snapped layers. Either side may be null,
     * e.g. a cover with no partner.
     *
     * Composes existing pages rather than owning decoded images: drawing and animation delegate to
     * whichever side is live via [ImageSingle.currentImage]/[isAnimated], so either can be an
     * animated GIF independently. Never cleans up [left]/[right] - whoever built them owns that.
     *
     * A side may also be a [Render] page. It has no image to place, so it sits out [forEachImage]
     * and the tile grid and is drawn into its half afterwards - see [drawRenderSides]. Everything
     * that measures a side reads the page, not a decoded image, so it lays out like an image one.
     */
    class ImageSpread(val left: ImagePage?, val right: ImagePage?) : ImageSingle(null) {

        /** Either side as an [ImageSingle] - null for a [Render] side, which has no image. */
        private val leftSingle: ImageSingle?
            get() = left as? ImageSingle
        private val rightSingle: ImageSingle?
            get() = right as? ImageSingle

        // Grows the shorter side to the taller one's height
        private fun sideScale(side: ImagePage?): Float {
            val h = side?.height ?: return 1f
            val tallest = max(left?.height ?: 0, right?.height ?: 0)
            return if (h <= 0 || tallest <= h) 1f else tallest.toFloat() / h
        }

        private val leftScale: Float
            get() = sideScale(left)
        private val rightScale: Float
            get() = sideScale(right)

        private fun sideWidth(side: ImagePage?): Float =
            (side?.width ?: 0) * sideScale(side)

        /** Runs [action] for each present side, with its pixel offset from the seam. */
        private inline fun forEachSide(action: (side: ImagePage, offsetX: Float, scale: Float) -> Unit) {
            left?.let { action(it, -0.5f * it.width * leftScale, leftScale) }
            right?.let { action(it, 0.5f * it.width * rightScale, rightScale) }
        }

        override var highQuality: Boolean
            get() = (leftSingle?.highQuality ?: true) && (rightSingle?.highQuality ?: true)
            set(value) {
                leftSingle?.highQuality = value
                rightSingle?.highQuality = value
            }

        override val isAnimated: Boolean
            get() = left?.isAnimated == true || right?.isAnimated == true

        override val frameVersion: Int
            get() = (left?.frameVersion ?: 0) + (right?.frameVersion ?: 0)

        override val contentVersion: Int
            get() = (leftSingle?.contentVersion ?: 0) + (rightSingle?.contentVersion ?: 0)

        /** No image of its own - [left]/[right] hold them, and [forEachImage] walks both. */
        override val currentImage: Image?
            get() = null

        /**
         * Each side sits half its own width out from the seam (the page anchor). A [Render] side
         * has no image to place and paints itself instead - see [drawRenderSides].
         */
        override fun forEachImage(
            action: (image: Image, offsetX: Float, imageScale: Float) -> Unit
        ) {
            leftSingle?.currentImage?.let { action(it, -0.5f * it.width * leftScale, leftScale) }
            rightSingle?.currentImage?.let { action(it, 0.5f * it.width * rightScale, rightScale) }
        }

        override val hasUploadedImage: Boolean
            get() = leftSingle?.hasUploadedImage == true || rightSingle?.hasUploadedImage == true

        override val isDecoded: Boolean
            get() = left?.isDecoded == true || right?.isDecoded == true

        /** As [ImageSingle.drawLive], then each [Render] side on top - see [drawRenderSides]. */
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            val covered = super.drawLive(encoder, dst, tiles)
            drawRenderSides(encoder, dst)
            return covered
        }

        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            super.renderCacheSeed(encoder, tex, tiles)
            drawRenderSides(encoder, tex)
        }

        /** [frameVersion] is the sides' sum, so this page's own has nothing to bump. */
        override fun invalidate() {
            left?.invalidate()
            right?.invalidate()
            if (isOnScreen) onInvalidate?.invoke()
        }

        override fun covers(other: ImagePage): Boolean =
            this === other || left === other || right === other

        override fun attach(
            parent: ImageViewerState, scope: CoroutineScope?, onInvalidate: () -> Unit
        ) {
            super.attach(parent, scope, onInvalidate)
            left?.attach(parent, scope, onInvalidate)
            right?.attach(parent, scope, onInvalidate)
        }

        /** True when either side paints itself rather than blitting a decoded image. */
        private val hasRenderSide: Boolean
            get() = left is Render || right is Render

        /**
         * A [Render] side repaints every frame, so [ImageSingle]'s incremental tile blit would
         * leave it stale for the whole transition. Reseed the slot outright instead.
         */
        override fun renderIntoCache(
            encoder: GPUCommandEncoder,
            tex: GPUTexture,
            tiles: TileRenderer,
            identityMatches: Boolean
        ) = super.renderIntoCache(encoder, tex, tiles, identityMatches && !hasRenderSide)

        /**
         * Draws each [Render] side into its own half of [dst], after the image sides.
         *
         * Needs a pass of its own ([Render.renderLoaded] opens one), since a pass cannot nest
         * inside the one the image sides just drew through. It loads rather than clears: [dst]
         * already holds the other side by now.
         *
         * Placed with the spread transform shifted by that side's seam offset, as [forEachImage]
         * places an image side, so [Render.render]/[Render.fillPage] land in the right half
         * instead of treating [dst] as a page of their own.
         */
        private fun drawRenderSides(encoder: GPUCommandEncoder, dst: GPUTexture) {
            // As forEachPlacedImage: the page can have been evicted since the snapshot was taken.
            if (destroyed) return
            forEachSide { side, offsetX, sideScale ->
                if (side is Render) {
                    // As forEachPlacedImage: render scales x/y by the scale it is given.
                    side.renderLoaded(
                        encoder,
                        (x + offsetX / dst.width) / sideScale,
                        y / sideScale,
                        scale * sideScale,
                        dst
                    )
                }
            }
        }

        /** Via [leafRect], so a spread of two [Render] sides still has a rect. */
        override fun pageRect(dst: GPUTexture): FloatArray? =
            leafRect(dst, true) ?: leafRect(dst, false)

        /** That side's own rect, so a spread turns one real page rather than half of a sheet. */
        override fun leafRect(dst: GPUTexture, left: Boolean): FloatArray? {
            val side = (if (left) this.left else this.right) ?: return null
            val sideScale = sideScale(side)
            val placeX = x + (if (left) -0.5f else 0.5f) * sideWidth(side) / dst.width
            (side as? ImageSingle)?.currentImage?.let { image ->
                if (image.mipmaps.isNotEmpty()) {
                    return image.placement(
                        dst,
                        (placeX + WebGpuRenderer.offsetX) / sideScale - WebGpuRenderer.offsetX,
                        (y + WebGpuRenderer.offsetY) / sideScale - WebGpuRenderer.offsetY,
                        scale * sideScale
                    )
                }
            }
            // A Render side has no image to place, so measure its own declared size instead.
            if (side !is Render) return null
            val cx = 0.5f + scale * (placeX + WebGpuRenderer.offsetX)
            val cy = 0.5f + scale * (y + WebGpuRenderer.offsetY)
            val hw = scale * 0.5f * sideWidth(side) / dst.width
            val hh = scale * 0.5f * side.height * sideScale / dst.height
            return floatArrayOf(cx - hw, cy - hh, cx + hw, cy + hh)
        }

        /** The seam, not the midpoint - the two sides can be different widths. */
        override fun spineX(dst: GPUTexture): Float? {
            leafRect(dst, true)?.let { return it[2] }
            leafRect(dst, false)?.let { return it[0] }
            return null
        }

        /** Both sides at once - the spread as one sheet, rather than [pageRect]'s single page. */
        override fun wholeRect(dst: GPUTexture): FloatArray? {
            val l = leafRect(dst, true)
            val r = leafRect(dst, false)
            if (l == null || r == null) return l ?: r
            return floatArrayOf(
                min(l[0], r[0]), min(l[1], r[1]), max(l[2], r[2]), max(l[3], r[3])
            )
        }

        override val backgroundColor: Int?
            get() = left?.backgroundColor ?: right?.backgroundColor

        /** Each side's own colour over its own half: seam to screen edge, not just its image. */
        override fun forEachBackgroundColumn(
            dst: GPUTexture, action: (color: Int, x1: Float, x2: Float) -> Unit
        ) {
            val leftColor = left?.backgroundColor
            val rightColor = right?.backgroundColor
            // Clamped: panned far enough, the seam leaves the screen and one half takes it all.
            val seam = spineX(dst)?.fastCoerceIn(0f, 1f)

            if (seam == null || leftColor == null || rightColor == null) {
                // One side, or nothing placed yet to find a seam by.
                action(leftColor ?: rightColor ?: return, 0f, 1f)
                return
            }

            action(leftColor, 0f, seam)
            action(rightColor, seam, 1f)
        }

        override fun horizontalExtent(): Pair<Float, Float> =
            sideWidth(left) to sideWidth(right)

        /** Total width (sum of both sides' widths) */
        override val width: Int
            get() = (sideWidth(left) + sideWidth(right)).roundToInt()

        /** Total height (max of both sides' heights) */
        override val height: Int
            get() = max(left?.height ?: 0, right?.height ?: 0)

        /**
         * Visible width after trim. Inner edges are ignored - a seam is never trimmed - and a
         * [Render] side has no trim at all, so it contributes its full width.
         */
        override val trimWidth: Int
            get() {
                val leftW =
                    leftSingle?.image?.let { it.width - (it.trim?.left ?: 0) } ?: left?.width ?: 0
                val rightW =
                    rightSingle?.image?.let { it.trim?.right ?: it.width } ?: right?.width ?: 0
                return (leftW * leftScale + rightW * rightScale).roundToInt()
            }

        /** Visible height after trim (max of trim heights) */
        override val trimHeight: Int
            get() = max(
                (left?.trimHeight ?: 0) * leftScale, (right?.trimHeight ?: 0) * rightScale
            ).roundToInt()

        override val isHalfWidth: Boolean
            get() = true

        /**
         * The sides hang off the seam - the anchor, at [width]/2 - by their own widths, so the
         * span runs [width]/2 - left width to [width]/2 + right width. That equals `0..width`
         * only when both sides are present and equally wide; otherwise it sits off-centre by the
         * difference. Reporting `0..width` regardless let a zoomed-in lone side pan half a page
         * past its own end, and cut off half a page early on the other.
         *
         * Trim plays no part: a seam is never trimmed, and [trimWidth] folds the outer edges in.
         */
        override fun xEdges(trimmed: Boolean): Pair<Float, Float> {
            val (leftWidth, rightWidth) = horizontalExtent()
            val anchor = width / 2f
            return anchor - leftWidth to anchor + rightWidth
        }

        /** Rests with the seam on the viewport centre - see [ImagePage.restingX]. */
        override fun restingX(scale: Float): Float = 0f

        override fun yEdges(trimmed: Boolean): Pair<Int, Int>? {
            if (!trimmed) return null
            val images = listOfNotNull(leftSingle?.image, rightSingle?.image)
            if (images.all { it.trim == null }) return null
            // An untrimmed side contributes its whole extent, so nothing of it is panned past.
            val trimTop = images.minOf { it.trim?.top ?: 0 }
            val trimBottom = images.maxOf { it.trim?.bottom ?: it.height }
            return trimTop to trimBottom
        }

        /**
         * Each side fit to its own half independently, since the two can differ in size - unlike
         * [ImagePage]'s default, which fits [width]/[height]'s combined span.
         */
        override fun halfWidthScale(halfWidth: Float, parentHeight: Float): Float =
            listOfNotNull(left, right).filter { it.width > 0 && it.height > 0 }
                .minOfOrNull { side ->
                    minOf(halfWidth / side.width, parentHeight / side.height)
                }?.coerceAtLeast(0.01f) ?: 0.01f
    }

    companion object {
        /**
         * Shared scope for fire-and-forget GPU cleanup work.
         * Lives for the application lifetime; individual cleanups are tiny and non-cancellable anyway.
         */
        internal val cleanupScope = CoroutineScope(Dispatchers.Default)

        /** Default [fadeIn] length. */
        const val FADE_MILLIS = 200

        /** Frame-duration floor: plenty of GIFs declare 0, which spins the loop on delay(0). */
        private const val MIN_FRAME_MILLIS = 10
    }

    /** True once page content has been decoded/is otherwise ready to draw. */
    open val isDecoded: Boolean get() = false

    /**
     * True once [cleanup] has run and the page's resources are gone or going.
     *
     * Volatile because it is set on whatever thread evicts the page but read on the GPU thread,
     * which uses it to skip drawing a page whose textures are being freed. A render snapshot is
     * captured on the main thread and drawn later, so it can outlive the page it names.
     */
    @Volatile
    var destroyed = false
        private set

    /** True while an animation frame loop owns the current frame. Only ever true for [Images]. */
    open val isAnimated: Boolean get() = false

    /**
     * Incremented each time this page's drawn content changes - an animated [Images] frame, or a
     * [Render] page's [Render.invalidate]. Read by a transition to spot a stale cache slot.
     */
    open val frameVersion: Int get() = 0

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (!x.isFinite() || !y.isFinite() || !scale.isFinite() || scale <= 0f) return
        if (this.x == x && this.y == y && this.scale == scale) return
        this.x = x
        this.y = y
        this.scale = scale
        onInvalidate?.invoke()
    }

    /**
     * Draws this page's content instead of blitting an image - see [Render] and [Images]. Given
     * the raw [encoder] rather than an already-open pass, so a page like [Render] that doesn't
     * need [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s stencil masking can open its own
     * pass(es) instead of being forced to match one it has no use for. Must clear [dst] itself -
     * this default does, in its own pass, since [Dummy] (which never overrides this) still needs
     * one: `getCurrentTexture` rotates buffers, so leaving it alone would show stale content from
     * several frames ago. [Render]/[Images] override this to clear as part of their own drawing
     * pass instead of paying for a separate one.
     */
    open fun renderWith(
        encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
    ) {
        Draw.clear(encoder, dst, 0)
    }

    /**
     * Draws this page's current live content into [dst] - the paged viewer's per-frame
     * (non-transition) path. Just [renderWith] at home position by default, which already does
     * the right thing for every non-[Images] page (including a no-op [Dummy]); [Images] overrides
     * this to add its [Images.highQuality] tile cache and animated-frame handling.
     *
     * Returns true if the page is now fully covered by sharp tiles, so [ImageViewerState] knows
     * it's safe to prewarm the next page - always false here, since only [Images] has a tile cache.
     */
    internal open fun drawLive(
        encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
    ): Boolean {
        renderWith(encoder, 0f, 0f, 1f, dst)
        return false
    }

    /**
     * As [drawLive], but seeding a [ca.mpreg.webgpuviewer.transition.Transition]'s cache slot
     * instead of the screen - see [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture].
     * Just [renderWith] by default; [Images] overrides this the same way it overrides [drawLive].
     */
    internal open fun renderCacheSeed(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
    ) {
        renderWith(encoder, 0f, 0f, 1f, tex)
    }

    /**
     * Tile keys newly available to blit since the last call, or null if this page is never tiled
     * (every non-[Images] page, or an [Images] one that isn't [Images.highQuality] or is animated)
     * - see [ca.mpreg.webgpuviewer.renderer.TileRenderer.availableTileKeys].
     */
    internal open fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
        null

    /**
     * Renders this page into a transition's cache slot - see
     * [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture]. [identityMatches] false
     * means a fresh slot ([renderCacheSeed] from scratch, `LoadOp.Clear`); true means an unchanged
     * one that just needs whatever's newly available layered on (`LoadOp.Load`). Just
     * [renderCacheSeed] by default, since a non-[Images] page never has anything incremental to
     * layer on top of - [Images] overrides this to blit instead when [identityMatches].
     */
    internal open fun renderIntoCache(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer, identityMatches: Boolean
    ) {
        renderCacheSeed(encoder, tex, tiles)
    }

    /**
     * This page's own rect within a flat render of it into [dst], as normalised (x1, y1, x2, y2)
     * surface coordinates - for a warp transition ([ca.mpreg.webgpuviewer.transition.TransitionSphere]/
     * flip) to map the page's actual rect rather than treating it as screen-shaped. Null if this
     * page has nothing to draw, which is always true for a non-[Images] page.
     */
    open fun pageRect(dst: GPUTexture): FloatArray? = null

    /**
     * One half of this page, normalised like [pageRect] - for
     * [ca.mpreg.webgpuviewer.transition.TransitionFlip]. [ImageSpread] answers with that side's
     * own rect, null where it has no page; everything else splits [pageRect] down the middle, which
     * is what lets a single page turn like a spread.
     */
    internal open fun leafRect(dst: GPUTexture, left: Boolean): FloatArray? {
        val r = pageRect(dst) ?: return null
        val mid = (r[0] + r[2]) * 0.5f
        return if (left) floatArrayOf(r[0], r[1], mid, r[3])
        else floatArrayOf(mid, r[1], r[2], r[3])
    }

    /** Where this page's two [leafRect] halves meet - the spine a page flip turns about. */
    internal open fun spineX(dst: GPUTexture): Float? {
        val r = pageRect(dst) ?: return null
        return (r[0] + r[2]) * 0.5f
    }

    /**
     * The color a transition should blend/fade toward for this page as a whole - [Images] reads
     * it from its first image's own [ca.mpreg.webgpuviewer.renderer.Image.backgroundColor];
     * [Render] also uses it (when overridden non-null) as its own background fill, both during a
     * transition and for its regular [Render.renderWith]/[Render.renderLoaded] draws - see
     * [Render]'s class doc. Null (nothing to fill/blend toward) by default.
     */
    open val backgroundColor: Int? = null

    /**
     * Background columns tiling the whole width, normalised to [dst]. One for a single page; a
     * spread gives each side its own, so neither crosses the seam nor leaves the edges bare.
     */
    internal open fun forEachBackgroundColumn(
        dst: GPUTexture, action: (color: Int, x1: Float, x2: Float) -> Unit
    ) {
        action(backgroundColor ?: return, 0f, 1f)
    }

    /**
     * [forEachBackgroundColumn] at [offsetX]/[offsetY], for a cached-surface slide - see
     * [ca.mpreg.webgpuviewer.transition.TransitionBasic]. A no-op for a non-[Images] page.
     */
    open fun drawBackgroundColumns(
        pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
    ) {
    }

    open val width: Int get() = 0
    open val height: Int get() = 0

    /** As [width]/[height], but after trim - defaults to the untrimmed size. */
    open val trimWidth: Int get() = width
    open val trimHeight: Int get() = height

    /** True if this page uses half-screen layout (dual page or single LEFT/RIGHT). */
    open val isHalfWidth: Boolean get() = false

    /** Left/right edges [minX]/[maxX] pan between: trim's edges once [trimmed], else raw span. */
    protected open fun xEdges(trimmed: Boolean): Pair<Float, Float> = 0f to width.toFloat()

    /**
     * As [xEdges], for top/bottom - null (not the raw fallback) so [nudgedYBounds] can tell real
     * trim edges (margin worth protecting) from the raw image's own (safe to pan past).
     */
    protected open fun yEdges(trimmed: Boolean): Pair<Int, Int>? = null

    var animationJob: Job? = null
    var animationTargetX: Float? = null
    var animationTargetY: Float? = null
    var animationTargetScale: Float? = null

    /**
     * True while [scale] is being animated - by [animateTo] or externally (e.g. fling-zoom decay,
     * which sets this directly). Gates the tile cache, which otherwise can't tell a settled scale
     * from a spring that's merely repeating a value for a frame mid-flight.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    /**
     * True while a plain (non-zoom) pan fling is actively decaying - set/cleared directly around
     * that decay in [ca.mpreg.webgpuviewer.viewer.ImageViewer]'s gesture handling. Checked there
     * so a tap landing while a fling from the previous gesture is still gliding doesn't also fire
     * [ImageViewerState.onTap] - the tap itself has no motion (that's what makes it a tap, not a
     * drag), so nothing else would otherwise tell the two apart.
     */
    @Volatile
    var isFlinging: Boolean = false

    var parent: ImageViewerState? = null
        set(value) {
            val wasNull = field == null
            field = value
            // Initialize to home when parent first set
            if (wasNull && value != null && x == 0f && y == 0f && scale == 1f) {
                x = homeX
                y = homeY
                scale = homeScale
            }
        }

    var scope: CoroutineScope? = null
    var onInvalidate: (() -> Unit)? = null

    /** How far this page has faded in: 1 is shown, less leaves a [backgroundColor] veil. */
    @Volatile
    var fade: Float = 1f
        private set

    private var fadeJob: Job? = null

    // Written by the thread that decoded the page, read by the draw that attaches it - a stale
    // start of 0 would read as a fade that began at boot.
    @Volatile
    private var fadeMillis = FADE_MILLIS

    @Volatile
    private var fadeStartMillis = 0L

    /** Waiting for a scope: a decode installs the page, the next frame attaches it. */
    @Volatile
    private var fadePending = false

    /**
     * Fade this page in instead of having it appear at once - for one swapped in behind a
     * placeholder that was on screen. Needs a [backgroundColor] to fade from, and runs on the
     * clock from here, so a fade nobody watches is over by the time it is reached.
     */
    @Synchronized
    fun fadeIn(durationMillis: Int = FADE_MILLIS) {
        if (destroyed || backgroundColor == null) return
        fadeMillis = durationMillis
        fadeStartMillis = SystemClock.uptimeMillis()
        fade = 0f
        fadePending = true
        startFade()
    }

    /** Pick the fade up wherever the clock has got to, or skip it if that is already past. */
    @Synchronized
    private fun startFade() {
        if (!fadePending) return
        val scope = scope ?: return
        fadePending = false
        fadeJob?.cancel()

        val elapsed = (SystemClock.uptimeMillis() - fadeStartMillis).toInt()
        if (elapsed >= fadeMillis) {
            fade = 1f
            return
        }
        val from = elapsed.toFloat() / fadeMillis
        fade = from

        fadeJob = scope.launch {
            // No finally: only another fadeIn cancels this, and it sets fade itself.
            animate(from, 1f, animationSpec = tween(fadeMillis - elapsed)) { value, _ ->
                fade = value
                // invalidate(), not onInvalidate: its frameVersion bump re-seeds a transition's
                // cached copy, which would otherwise hold the veil at whatever it was seeded with.
                invalidate()
            }
        }
    }

    /**
     * Everything this page covers, normalised like [pageRect]. An [ImageSpread] answers with both
     * sides together where [pageRect] gives whichever it finds first, so a transition treating the
     * page as one sheet wants this one.
     */
    internal open fun wholeRect(dst: GPUTexture): FloatArray? = pageRect(dst)

    /** What [drawFade] veils - the whole page, so a spread does not half-fade. */
    protected open fun fadeRect(dst: GPUTexture): FloatArray? = wholeRect(dst)

    /** Veil this page's rect, in fractions of the target, with what is left of the fade. */
    internal fun drawFade(
        pass: GPURenderPassEncoder,
        format: Int,
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        masked: Boolean = true
    ) {
        if (fade >= 1f) return
        val color = backgroundColor ?: return
        val alpha = (((color ushr 24) and 0xFF) * (1f - fade)).toInt().coerceIn(0, 255)
        val veil = (alpha shl 24) or (color and 0xFFFFFF)
        // Only a live draw's pass has the stencil attachment drawMaskedRect's pipeline declares.
        if (masked) RenderPage.drawMaskedRect(pass, format, x1, y1, x2, y2, veil)
        else Draw.rect(pass, format, x1, y1, x2, y2, veil)
    }

    /** True while the viewer is drawing this page, itself or as a side of a spread. */
    val isOnScreen: Boolean
        get() = parent?.isOnScreen(this) == true

    /** True when drawing this page draws [other] - itself, or a side [ImageSpread] overrides in. */
    internal open fun covers(other: ImagePage): Boolean = this === other

    /**
     * Adopts this page into [parent]'s viewer. [ImageSpread] passes it on to its sides, which the
     * viewer never fetches itself but which still need a scope to animate in and a way back to
     * the screen.
     */
    internal open fun attach(
        parent: ImageViewerState, scope: CoroutineScope?, onInvalidate: () -> Unit
    ) {
        if (this.parent !== parent) this.parent = parent
        if (this.scope !== scope) this.scope = scope
        if (this.onInvalidate !== onInvalidate) this.onInvalidate = onInvalidate
        if (fadePending) startFade()
    }

    /**
     * Redraws this page, if it is on screen to redraw. [Render] and [Images] also bump
     * [frameVersion] either way, so a page turn re-seeds the cached copy it is sampling rather
     * than showing the content as it was when the turn began.
     */
    open fun invalidate() {
        if (isOnScreen) onInvalidate?.invoke()
    }

    private val parentWidth: Float
        get() = parent?.width?.toFloat() ?: 0f

    private val parentHeight: Float
        get() = parent?.viewportHeight ?: 0f

    /**
     * [isHalfWidth]'s fit scale: each side of a spread can be a differently sized image, so
     * [Images] overrides this to fit each one independently rather than [width]/[height]'s
     * combined span - the default here is only ever exercised by a non-spread page.
     */
    protected open fun halfWidthScale(halfWidth: Float, parentHeight: Float): Float {
        if (width <= 0 || height <= 0) return 0.01f
        return minOf(halfWidth / width, parentHeight / height).coerceAtLeast(0.01f)
    }

    val atHome: Boolean
        get() = x.closeTo(homeX) && y.closeTo(homeY) && atHomeScale

    val atHomeScale: Boolean
        get() = scale.closeTo(homeScale)

    var homeScale: Float = -1f
        get() {
            if (field > 0) return field

            if (parentWidth <= 0f || parentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen, no trim
                return halfWidthScale(parentWidth / 2f, parentHeight)
            }

            // Single SINGLE page: fit trim to full screen
            val w = trimWidth.toFloat().takeIf { it > 0f } ?: return 0.01f
            val h = trimHeight.toFloat().takeIf { it > 0f } ?: return 0.01f
            return minOf(parentWidth / w, parentHeight / h).coerceAtLeast(0.01f)
        }

    var homeX: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxX(scale).fastCoerceIn(minX(scale), maxX(scale))
        }

    var homeY: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxY(scale).fastCoerceIn(minY(scale), maxY(scale))
        }

    var minScale = 0f
        get() {
            if (field > 0) return field
            if (parentWidth <= 0f || parentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen
                return halfWidthScale(parentWidth / 2f, parentHeight)
            }

            // Guarded as [homeScale] is: a page still sizing divides to Infinity, then NaN in x/y.
            val w = width.toFloat().takeIf { it > 0f } ?: return 0.01f
            val h = height.toFloat().takeIf { it > 0f } ?: return 0.01f
            return minOf(parentWidth / w, parentHeight / h).coerceAtLeast(0.01f)
        }

    var maxScale = 0f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

    var doubleTapScale: Float = 0f
        get() = if (field != 0f) field else max(minScale, homeScale) * 2

    // BOUNDS:
    // cutout ignore:
    //  with trim:
    //      >= homeScale: viewport pan over trimmed content
    //      < homeScale: viewport pan over untrimmed content
    //  without trim: viewport pan over content
    //
    // cutout avoid:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: nudge below cutout
    //
    // cutout shift:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: center in cut viewport

    /**
     * As [xBounds] but never collapsed (min > max when there's slack). [nudgedYBounds] needs
     * the true floor - the collapsed center sits below it, letting panning reveal past it.
     */
    private fun rawBounds(
        size: Int, nearEdge: Float, farEdge: Float, parentSize: Int, scale: Float
    ): Pair<Float, Float> {
        val maxV = (0.5f * size - nearEdge) / parentSize - 0.5f / scale
        val minV = (0.5f * size - farEdge) / parentSize + 0.5f / scale
        return minV to maxV
    }

    /** [minX]/[maxX] together, computing [homeScale] and [rawBounds] only once per call. */
    private fun xBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val (left, right) = xEdges(scale >= homeScale)
        val (minV, maxV) = rawBounds(width, left, right, parent.width, scale)
        if (minV > maxV) {
            // [maxV, minV] is every x showing the content whole - a lone spread side zoomed
            // past its half would otherwise stay pinned to the seam, hanging off screen.
            val rest = restingX(scale).fastCoerceIn(maxV, minV)
            return rest to rest
        }
        return minV to maxV
    }

    /**
     * Where the page rests horizontally once its content fits the viewport - the middle of
     * [xEdges]'s span, so the content sits centred.
     *
     * [ImageSpread] overrides it to rest on the seam instead. That is what keeps a lone left/right
     * side in its own half; centring the span - what the midpoint gives once one side is missing -
     * would pull it to the middle, indistinguishable from a page with no partner.
     */
    protected open fun restingX(scale: Float): Float {
        val parent = parent ?: return 0f
        val (near, far) = xEdges(scale >= homeScale)
        return (0.5f * width - 0.5f * (near + far)) / parent.width
    }

    fun minX(scale: Float): Float = xBounds(scale).first

    fun maxX(scale: Float): Float = xBounds(scale).second

    /**
     * "Ignore": [xBounds]'s plain collapse-when-it-fits.
     *
     * "Avoid"/"shift": near/top bound pushed further by the *full* [ImageViewerState.cutoutTopPx]
     * so the cut viewport (real viewport minus the cutout) can pan over all the content - unless
     * still fully visible there even after the push, which collapses to one rest point instead of
     * a pointless range: "shift" always rests centered in the cut viewport (half the push, since
     * shrinking the viewport only moves its center by half); "avoid" only nudges - just far enough
     * to clear the cutout, not all the way to centered-in-cut-viewport - and only when the plain
     * whole-screen-centered rest would actually overlap the cutout.
     */
    private fun nudgedYBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val trimmed = scale >= homeScale
        val (top, bottom) = yEdges(trimmed) ?: (0 to height)
        val (floor, natMax) = rawBounds(
            height, top.toFloat(), bottom.toFloat(), parent.height, scale
        )
        val slack = floor > natMax
        val center = (floor + natMax) / 2f

        if (!parent.avoidCutout || parent.cutoutTopPx <= 0f || parent.height <= 0) {
            return if (slack) center to center else floor to natMax
        }

        val fullPush =
            if (isHalfWidth && !trimmed) 0f else parent.cutoutTopPx / (scale * parent.height)
        val pushed = natMax + fullPush

        if (slack) {
            // "pushed" (natMax + fullPush) is exactly the rest position where the near edge sits
            // flush against the cutout - so center < pushed means the plain centered rest would
            // sit above that line, i.e. under the cutout.
            val overlapsCutout = center < pushed
            if (!parent.alwaysAvoidCutout && !overlapsCutout) return center to center
            val rest = if (parent.alwaysAvoidCutout) center + fullPush / 2f else pushed
            if (rest < floor) return rest to rest
        }
        return floor to pushed
    }

    fun minY(scale: Float): Float = nudgedYBounds(scale).first

    fun maxY(scale: Float): Float = nudgedYBounds(scale).second

    fun home() {
        animateTo(targetScale = homeScale)
    }

    fun animateTo(
        origin: Offset? = null,
        targetX: Float = homeX,
        targetY: Float = homeY,
        targetScale: Float = scale,
        /** How it moves there - a spring unless the caller says otherwise. */
        animationSpec: AnimationSpec<Float>? = null,
    ) {
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val targetScale = targetScale.fastCoerceIn(minScale, maxScale)

        val minX = minX(targetScale)
        val maxX = maxX(targetScale)
        val minY = minY(targetScale)
        val maxY = maxY(targetScale)

        val scaleChanging = targetScale != startScale
        val diffEnd = if (scaleChanging) 1 / targetScale - 1 / startScale else 1f

        val endX = when {
            origin != null && scaleChanging -> (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(
                minX, maxX
            )

            origin != null -> x.fastCoerceIn(minX, maxX)
            else -> targetX
        }
        val endY = when {
            origin != null && scaleChanging -> (startY + (origin.y - 0.5f) * diffEnd).fastCoerceIn(
                minY, maxY
            )

            origin != null -> y.fastCoerceIn(minY, maxY)
            else -> targetY
        }

        if (!scaleChanging && endX == startX && endY == startY) {
            animationJob = null
            return
        }

        animationJob = scope?.launch {
            animationTargetX = endX
            animationTargetY = endY
            animationTargetScale = targetScale
            if (scaleChanging) isScaleAnimating = true
            try {
                animate(
                    0f, 1f, animationSpec = animationSpec ?: spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
                ) { value, _ ->
                    val currentScale = startScale + (targetScale - startScale) * value
                    val c = if (scaleChanging) {
                        ((1 / currentScale - 1 / startScale) / diffEnd).fastCoerceIn(0f, 1f)
                    } else {
                        value
                    }

                    setPos(
                        (startX + (endX - startX) * c).orZero(),
                        (startY + (endY - startY) * c).orZero(),
                        currentScale
                    )
                }
            } finally {
                animationTargetX = null
                animationTargetY = null
                animationTargetScale = null
                isScaleAnimating = false
                // The last step drew with generation held off; settled, the tiles need a frame.
                if (scaleChanging) onInvalidate?.invoke()
            }
        }
    }

    @Synchronized
    open fun cleanup() {
        if (destroyed) return
        destroyed = true

        animationJob?.cancel()
        animationJob = null
        fadeJob?.cancel()
        fadeJob = null
        fadePending = false
    }

    /** Alias for cleanup() */
    fun destroy() = cleanup()
}
