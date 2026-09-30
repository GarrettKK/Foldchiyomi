package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blitCached
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blitCachedRegion
import ca.mpreg.webgpuviewer.transition.Transition.Companion.cacheLock
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.transition.Transition.Companion.invalidateCache
import ca.mpreg.webgpuviewer.viewer.ImagePage
import ca.mpreg.webgpuviewer.viewer.ImageViewerState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow

abstract class Transition {
    open val code: String = ""

    protected val device get() = WebGpuRenderer.device

    /**
     * True when [code]'s fragment stage already returns premultiplied alpha.
     *
     * A transition that samples a cached page texture is in that position: the cache was written
     * premultiplied, so re-multiplying by alpha on the way out would darken every edge. Such a
     * shader needs `One` for the colour source factor, the way [blitCached] does, rather than the
     * `SrcAlpha` that suits a shader resolving straight-alpha texels.
     */
    protected open val premultipliedOutput: Boolean = false

    protected open val pipelines = FormatKeyed { format ->
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = format, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = if (premultipliedOutput) BlendFactor.One
                                    else BlendFactor.SrcAlpha,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    internal abstract fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    )

    /**
     * As the plain [render], with what the finger is doing - for a transition that follows it, like
     * [TransitionCurl]. Everything else ignores [gesture].
     */
    internal open fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
        gesture: TurnGesture,
    ) = render(page1, page2, encoder, dst, frac, pos1, pos2, tiles)

    companion object {
        // Shared blit pipeline for all transitions
        private val blitPipelines = FormatKeyed { format ->
            val device = WebGpuRenderer.device
            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(BLIT_SHADER))
            )
            device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                    fragment = GPUFragmentState(
                        shaderModule, entryPoint = "fs_main", targets = arrayOf(
                            GPUColorTargetState(
                                format = format, blend = GPUBlendState(
                                    color = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    ), alpha = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    )
                                )
                            )
                        )
                    ),
                    primitive = GPUPrimitiveState(topology = TriangleList),
                )
            )
        }

        private const val BLIT_SHADER = """
struct Uniforms {
    offset: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    
    let pos = positions[vertex_index];
    
    // Apply offset to position
    let offset_pos = pos + uniforms.offset;
    
    // Convert to NDC
    let ndc_x = offset_pos.x * 2.0 - 1.0;
    let ndc_y = 1.0 - offset_pos.y * 2.0;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_sampler, in.uv);
}
"""

        /** As [blitPipeline], but drawing only a sub-rectangle - see [blitCachedRegion]. */
        private val regionPipelines = FormatKeyed { format ->
            val device = WebGpuRenderer.device
            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(REGION_SHADER))
            )
            device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                    fragment = GPUFragmentState(
                        shaderModule, entryPoint = "fs_main", targets = arrayOf(
                            GPUColorTargetState(
                                format = format, blend = GPUBlendState(
                                    color = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    ), alpha = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    )
                                )
                            )
                        )
                    ),
                    primitive = GPUPrimitiveState(topology = TriangleList),
                )
            )
        }

        private const val REGION_SHADER = """
struct Uniforms {
    rect: vec4<f32>,
    // Fade in x, the rest padding.
    fade: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let pos = mix(uniforms.rect.xy, uniforms.rect.zw, corners[vertex_index]);

    var out: VertexOutput;
    out.position = vec4<f32>(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
    // 1:1 - the cache renders the whole surface, so a region belongs at its own coordinates.
    out.uv = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Premultiplied, so the whole sample scales.
    return textureSample(src_tex, src_sampler, in.uv) * uniforms.fade.x;
}
"""

        private val regionByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
        }

        /**
         * Blit one region of a cached texture into [pass] at those same coordinates - one side of
         * a cached spread without the other, which [blitCached] cannot do. Null [cachedView] draws
         * nothing, as does [alpha] 0.
         */
        internal fun blitCachedRegion(
            pass: GPURenderPassEncoder,
            /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
            format: Int,
            cachedView: GPUTextureView?,
            x1: Float,
            y1: Float,
            x2: Float,
            y2: Float,
            alpha: Float = 1f,
        ) {
            if (cachedView == null || x2 <= x1 || y2 <= y1 || alpha <= 0f) return

            val byteBuffer = regionByteBuffer.get()
            byteBuffer.clear()
            byteBuffer.putFloat(x1)
            byteBuffer.putFloat(y1)
            byteBuffer.putFloat(x2)
            byteBuffer.putFloat(y2)
            byteBuffer.putFloat(alpha)
            byteBuffer.putFloat(0f)
            byteBuffer.putFloat(0f)
            byteBuffer.putFloat(0f)
            byteBuffer.flip()

            val uniformBuffer = WebGpuRenderer.device.createBuffer(
                GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
            )
            WebGpuRenderer.device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            val regionPipeline = regionPipelines[format]
            pass.setPipeline(regionPipeline)
            pass.setTransientBindGroup(
                0, WebGpuRenderer.device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = regionPipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, buffer = uniformBuffer),
                            GPUBindGroupEntry(1, textureView = cachedView),
                            GPUBindGroupEntry(2, sampler = blitSampler)
                        )
                    )
                )
            )
            pass.draw(6)
            uniformBuffer.close()
        }

        private val blitSampler by lazy {
            WebGpuRenderer.device.createSampler()
        }

        private val blitByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
        }

        // Cache for current transition - 2 slots for page1 and page2
        private val cacheLock = Any()

        // Textures - always non-null once first render happens at a given size
        private var texture1: GPUTexture? = null
        private var texture2: GPUTexture? = null
        private var view1: GPUTextureView? = null
        private var view2: GPUTextureView? = null

        // Cache validity tracking
        private var cachedPage1: ImagePage? = null
        private var cachedPage2: ImagePage? = null
        private var cachedX1 = 0f
        private var cachedY1 = 0f
        private var cachedScale1 = 0f
        private var cachedFrameVersion1 = -1
        private var cachedX2 = 0f
        private var cachedY2 = 0f
        private var cachedScale2 = 0f
        private var cachedFrameVersion2 = -1

        // Which tile keys [getCachedTexture] has actually blitted into each slot - compared
        // against [TileRenderer.availableTileKeys] to know exactly when a slot needs another
        // incremental blit, instead of a "done yet" boolean that has to agree with drawCore.
        private var blittedKeys1: Set<Long> = emptySet()
        private var blittedKeys2: Set<Long> = emptySet()

        private var cacheWidth = 0
        private var cacheHeight = 0
        private var cacheFormat = 0

        // Bumped on every swap or clear; a render records its metadata only if unchanged.
        private var cacheGeneration = 0

        // Textures pending destruction (deferred to avoid use-after-free), with the views over them:
        // a view is a Dawn handle of its own and only close() releases it.
        private var pendingDestroy1: GPUTexture? = null
        private var pendingDestroy2: GPUTexture? = null
        private var pendingView1: GPUTextureView? = null
        private var pendingView2: GPUTextureView? = null

        private fun ensureTexturesLocked(width: Int, height: Int) {
            // Destroy old pending textures (safe now - at least one frame has passed)
            pendingView1?.close()
            pendingView2?.close()
            pendingDestroy1?.destroyAndRelease()
            pendingDestroy2?.destroyAndRelease()
            pendingView1 = null
            pendingView2 = null
            pendingDestroy1 = null
            pendingDestroy2 = null

            // Recreate if the size changed, or if HDR came or went under us - these are
            // composited onto the swapchain, so they have to match it.
            val format = Hdr.frameFormat
            if (cacheWidth != width || cacheHeight != height || cacheFormat != format) {
                // Defer destruction of old textures
                pendingDestroy1 = texture1
                pendingDestroy2 = texture2
                pendingView1 = view1
                pendingView2 = view2

                // Create new textures
                texture1 = WebGpuRenderer.device.createTexture(
                    GPUTextureDescriptor(
                        size = GPUExtent3D(width, height),
                        format = format,
                        usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding
                    )
                )
                texture2 = WebGpuRenderer.device.createTexture(
                    GPUTextureDescriptor(
                        size = GPUExtent3D(width, height),
                        format = format,
                        usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding
                    )
                )
                view1 = texture1!!.createView()
                view2 = texture2!!.createView()

                // Invalidate cache
                cacheGeneration++
                cachedPage1 = null
                cachedPage2 = null
                blittedKeys1 = emptySet()
                blittedKeys2 = emptySet()
                cacheWidth = width
                cacheHeight = height
                cacheFormat = format
            }
        }

        private fun srgbToLinear(c: Float): Float =
            if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)

        private fun linearToSrgb(c: Float): Float =
            if (c <= 0.0031308f) c * 12.92f else 1.055f * c.pow(1f / 2.4f) - 0.055f

        /**
         * Blend [bg1] toward [bg2] by [t] in linear space - 50% between white and black should be
         * linear grey, not the lighter result a straight sRGB-byte lerp gives. [TransitionFade]'s
         * own shader mix matches this rate: it un-premultiplies before converting to linear and
         * re-premultiplies after converting back, rather than giving up on linear blending - so
         * both stay at the same perceptual pace without either one needing to give up correctness.
         */
        internal fun blendBackgroundColor(bg1: Int, bg2: Int, t: Float): Int {
            fun channel(shift: Int): Int {
                val c1 = srgbToLinear(((bg1 shr shift) and 0xFF) / 255f)
                val c2 = srgbToLinear(((bg2 shr shift) and 0xFF) / 255f)
                val blended = linearToSrgb(c1 + (c2 - c1) * t)
                return (blended * 255f).toInt().coerceIn(0, 255)
            }
            return 0xFF000000.toInt() or (channel(16) shl 16) or (channel(8) shl 8) or channel(0)
        }

        /**
         * Invalidate the transition cache. Call when transition ends.
         * Keeps textures allocated for reuse.
         */
        fun invalidateCache() {
            synchronized(cacheLock) {
                cacheGeneration++
                cachedPage1 = null
                cachedPage2 = null
                blittedKeys1 = emptySet()
                blittedKeys2 = emptySet()
            }
        }

        /**
         * Forget cached renders of [state]'s pages. The cache is static and a page reaches its
         * viewer through [ImagePage.parent], so a slot left pointing at a finished viewer's page
         * would keep that viewer and its host alive until another page takes the slot.
         */
        internal fun releasePagesOf(state: ImageViewerState) {
            synchronized(cacheLock) {
                cacheGeneration++
                if (cachedPage1?.parent === state) {
                    cachedPage1 = null
                    blittedKeys1 = emptySet()
                }
                if (cachedPage2?.parent === state) {
                    cachedPage2 = null
                    blittedKeys2 = emptySet()
                }
            }
        }

        /**
         * Called once a page turn settles on [newCurrentPage]. Slot 2 is often already a valid
         * render of it - prewarmed by [ImageViewerState] while it was still the *next* page - so
         * this swaps it into slot 1 instead of discarding it. Falls back to a full wipe (like the
         * unconditional [invalidateCache] this replaces) when neither slot matches.
         */
        fun rotateCacheOnPageChange(newCurrentPage: ImagePage) {
            synchronized(cacheLock) {
                cacheGeneration++
                when {
                    cacheHitLocked(newCurrentPage, true) -> {
                        cachedPage2 = null
                        blittedKeys2 = emptySet()
                    }

                    cacheHitLocked(newCurrentPage, false) -> {
                        val t = texture1; texture1 = texture2; texture2 = t
                        val v = view1; view1 = view2; view2 = v
                        cachedPage1 = cachedPage2
                        cachedX1 = cachedX2
                        cachedY1 = cachedY2
                        cachedScale1 = cachedScale2
                        cachedFrameVersion1 = cachedFrameVersion2
                        blittedKeys1 = blittedKeys2
                        cachedPage2 = null
                        blittedKeys2 = emptySet()
                    }

                    else -> {
                        cachedPage1 = null
                        cachedPage2 = null
                        blittedKeys1 = emptySet()
                        blittedKeys2 = emptySet()
                    }
                }
            }
        }

        /** Must hold [cacheLock]. Used by [getCachedTexture] to decide seed vs. incremental vs. skip. */
        private fun cacheHitLocked(page: ImagePage, isPage1: Boolean): Boolean {
            val cachedPage = if (isPage1) cachedPage1 else cachedPage2
            val cachedX = if (isPage1) cachedX1 else cachedX2
            val cachedY = if (isPage1) cachedY1 else cachedY2
            val cachedScale = if (isPage1) cachedScale1 else cachedScale2
            val cachedFrame = if (isPage1) cachedFrameVersion1 else cachedFrameVersion2
            return cachedPage === page && cachedX == page.x && cachedY == page.y && cachedScale == page.scale && cachedFrame == page.frameVersion
        }

        /**
         * Get cached texture view for a page, rendering into it as needed instead of requiring
         * full tile coverage up front:
         *  - Identity unchanged and the page never gets tiles ([ImagePage.newlyAvailableTileKeys]
         *    null) or matches what's tracked as blitted - return as-is.
         *  - Identity unchanged but something new is available - `LoadOp.Load` layers it on and
         *    that becomes the tracked set.
         *  - Identity changed - `LoadOp.Clear` and [ImagePage.renderCacheSeed] from scratch.
         *
         * Tracked key sets, not a derived "fully covered" boolean, so this can't desync from what
         * [TileRenderer.drawCore] actually blits. Never forces generation - whatever isn't cached
         * fills in on the background worker, and later calls pick up what's landed. Null only if
         * the page isn't [ImagePage.isDecoded].
         */
        internal fun getCachedTexture(
            page: ImagePage,
            isPage1: Boolean,
            encoder: GPUCommandEncoder,
            dstWidth: Int,
            dstHeight: Int,
            tiles: TileRenderer,
        ): GPUTextureView? {
            if (page.destroyed || !page.isDecoded) return null
            if (dstWidth <= 0 || dstHeight <= 0) return null

            // Lock only for metadata - GPU recording runs on the single GPU thread and doesn't
            // need it; cacheLock only guards against invalidateCache() from the UI thread.
            var generation = 0
            val (texture, view, identityMatches, blittedKeys) = synchronized(cacheLock) {
                ensureTexturesLocked(dstWidth, dstHeight)
                generation = cacheGeneration
                val texture = if (isPage1) texture1!! else texture2!!
                val view = if (isPage1) view1!! else view2!!
                val blitted = if (isPage1) blittedKeys1 else blittedKeys2
                // Never a hit for an animated page - it swaps images every frame, so every call
                // needs a fresh LoadOp.Clear + renderCacheSeed to blit whatever frame is current
                // right now, rather than relying on frameVersion happening to have ticked.
                val matches = !page.isAnimated && cacheHitLocked(page, isPage1)
                CacheReadResult(texture, view, matches, blitted)
            }

            // Null for a page that never gets tiles - not highQuality, animated, or not an Images
            // page at all - in which case there's nothing further to compare against blittedKeys.
            val available = page.newlyAvailableTileKeys(tiles, texture)

            if (identityMatches && (available == null || available == blittedKeys)) {
                return view
            }

            val pageX = page.x
            val pageY = page.y
            val pageScale = page.scale
            val pageFrameVersion = page.frameVersion

            // Record outside the lock - GPU thread is single-threaded. renderIntoCache opens its
            // own pass (Load or Clear, matching identityMatches) rather than sharing one from here.
            page.renderIntoCache(encoder, texture, tiles, identityMatches)

            val newBlitted =
                if (available == null) emptySet()
                else page.newlyAvailableTileKeys(tiles, texture) ?: emptySet()

            synchronized(cacheLock) {
                // Swapped mid-render: the view is still right, the metadata isn't.
                if (generation != cacheGeneration) return view
                if (isPage1) blittedKeys1 = newBlitted else blittedKeys2 = newBlitted
                if (!identityMatches) {
                    // Update cache metadata - only needed on a real identity change; an unchanged
                    // identity's metadata is already correct.
                    if (isPage1) {
                        cachedPage1 = page
                        cachedX1 = pageX
                        cachedY1 = pageY
                        cachedScale1 = pageScale
                        cachedFrameVersion1 = pageFrameVersion
                    } else {
                        cachedPage2 = page
                        cachedX2 = pageX
                        cachedY2 = pageY
                        cachedScale2 = pageScale
                        cachedFrameVersion2 = pageFrameVersion
                    }
                }
            }

            return view
        }

        /** [getCachedTexture]'s locked metadata read - a plain [Triple] runs out of slots. */
        private data class CacheReadResult(
            val texture: GPUTexture,
            val view: GPUTextureView,
            val identityMatches: Boolean,
            val blittedKeys: Set<Long>
        )

        /**
         * Open one cleared pass on [dst] for a transition's whole frame - [drawBackground] and
         * [blitCached] each used to open and close their own, costing an attachment load/store
         * apiece; sharing one pass across all of them instead is just as correct, since none of
         * them read back what an earlier one in the same frame wrote.
         */
        internal fun beginClearedPass(
            encoder: GPUCommandEncoder, dst: GPUTexture
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
                    )
                )
            ).also { targetView.close() }
        }

        /** Blit a cached texture into [pass] with an offset. Draws nothing if [cachedView] is null. */
        internal fun blitCached(
            pass: GPURenderPassEncoder,
            /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
            format: Int, cachedView: GPUTextureView?, offsetX: Float, offsetY: Float
        ) {
            if (cachedView == null) return

            val byteBuffer = blitByteBuffer.get()
            byteBuffer.clear()
            byteBuffer.putFloat(offsetX)
            byteBuffer.putFloat(offsetY)
            byteBuffer.flip()

            val uniformBuffer = WebGpuRenderer.device.createBuffer(
                GPUBufferDescriptor(size = 8, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
            )
            WebGpuRenderer.device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            val blitPipeline = blitPipelines[format]
            pass.setPipeline(blitPipeline)
            pass.setTransientBindGroup(
                0, WebGpuRenderer.device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = blitPipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, buffer = uniformBuffer),
                            GPUBindGroupEntry(1, textureView = cachedView),
                            GPUBindGroupEntry(2, sampler = blitSampler)
                        )
                    )
                )
            )
            pass.draw(6)
            uniformBuffer.close()
        }
    }
}

/** How a page turn is being driven this frame - see [Transition.render]. */
internal class TurnGesture(
    /** A finger holds the turn: pos2 is where it is now, pos1 where it took hold. */
    val held: Boolean,
    /** Started by a tap or key rather than a drag: no finger to follow at all. */
    val auto: Boolean,
    /** The frac the finger let go at, when neither [held] nor [auto]. */
    val releaseFrac: Float,
)
