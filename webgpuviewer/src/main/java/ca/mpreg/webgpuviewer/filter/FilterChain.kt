package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.AddressMode
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUSampler
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.renderer.Hdr
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease

/**
 * The output filter chain: the viewer draws its frame into an offscreen texture, each enabled
 * [Filter] runs over the result of the one before it, and the last one writes the swapchain.
 *
 * Held by [ca.mpreg.webgpuviewer.renderer.WebGpuRenderer] and reachable as
 * [ca.mpreg.webgpuviewer.viewer.ImageViewerState.filters]. With nothing enabled the chain steps
 * out of the way entirely - [beginFrame] hands back the swapchain texture itself, so a viewer
 * with no filters draws exactly as it did before.
 *
 * Textures are pooled and recycled within the frame, so a chain of any length runs on two
 * screen-sized textures rather than one per filter.
 */
class FilterChain {
    private val device get() = WebGpuRenderer.device

    /** Set by the viewer - a filter's settings change has to reach the screen somehow. */
    var onInvalidate: (() -> Unit)? = null

    private val invalidateCallback: () -> Unit = { onInvalidate?.invoke() }

    /** The chain, in the order they run. Assign to change it. */
    @Volatile
    var filters: List<Filter> = emptyList()
        set(value) {
            field.forEach { it.onUpdate = null }
            value.forEach { it.onUpdate = invalidateCallback }
            field = value
            invalidateCallback()
        }

    /** This frame's active filters, fixed at [beginFrame] so [endFrame] can't see a half-change. */
    private val active = ArrayList<Filter>()

    private var sceneSlot: Slot? = null

    private var poolWidth = 0
    private var poolHeight = 0

    /**
     * Where this frame should be drawn: an offscreen texture when any filter is enabled, and the
     * swapchain texture itself when none is. [endFrame] must follow on either path.
     */
    fun beginFrame(surface: GPUTexture): GPUTexture {
        // First, and nothing before it: with no filters this is the whole of the chain's work
        // per frame, and it should stay a list scan with no allocation and no GPU work at all.
        active.clear()
        val list = filters
        for (i in list.indices) if (list[i].active) active.add(list[i])

        if (active.isEmpty()) {
            sceneSlot = null
            // Two screen-sized textures is real memory to leave sitting behind a filter that is
            // switched off; turning one back on is a settings change and can afford to reallocate.
            if (pool.isNotEmpty()) destroyPool()
            return surface
        }

        // Slots a throwing frame never handed back - the pool only ever lives within one frame.
        releaseAll()
        frame++

        if (poolWidth != surface.width || poolHeight != surface.height) {
            destroyPool()
            poolWidth = surface.width
            poolHeight = surface.height
        }

        val slot = acquire(surface.width, surface.height, surface.format, false)
        sceneSlot = slot
        return slot.texture
    }

    /** Run the chain over what [beginFrame] handed out, ending on [surface]. */
    fun endFrame(encoder: GPUCommandEncoder, surface: GPUTexture) {
        val scene = sceneSlot ?: return
        sceneSlot = null

        var srcSlot: Slot? = scene
        var src: GPUTextureView = scene.view
        var width = surface.width
        var height = surface.height
        // Views over the frame's swapchain texture, made here rather than pooled. The pass each one is
        // attached to keeps its own reference, so ours goes once the chain has recorded everything.
        val surfaceViews = ArrayList<GPUTextureView>(2)

        try {
            for (i in active.indices) {
                val filter = active[i]
                val outWidth = filter.outputWidth(width, height)
                val outHeight = filter.outputHeight(width, height)
                val last = i == active.size - 1

                // The swapchain is a render attachment of one fixed format - a compute filter, or
                // one that resamples or wants headroom, has to land offscreen and be blitted.
                val direct = last && !filter.usesCompute &&
                        filter.outputFormat == surface.format &&
                        outWidth == surface.width && outHeight == surface.height

                val dstSlot = if (direct) null
                else acquire(outWidth, outHeight, filter.outputFormat, filter.usesCompute)
                val dst = dstSlot?.view ?: surface.createView().also { surfaceViews.add(it) }

                filter.run(this, encoder, src, width, height, dst, outWidth, outHeight)

                // Only after run() - until it returns, this is the texture it reads from.
                srcSlot?.let { it.inUse = false }
                srcSlot = dstSlot
                src = dst
                width = outWidth
                height = outHeight

                if (last && !direct) tailBlit.run(
                    this,
                    encoder,
                    src,
                    width,
                    height,
                    surface.createView().also { surfaceViews.add(it) },
                    surface.width,
                    surface.height
                )
            }
        } finally {
            srcSlot?.let { it.inUse = false }
            surfaceViews.forEach { it.close() }
            active.clear()
        }
    }

    /**
     * A pooled texture for a filter's own intermediate pass, free for the rest of this frame.
     * The caller must hand it back with [release] before returning from [Filter.run].
     */
    fun scratch(
        width: Int, height: Int, format: Int = Hdr.frameFormat, storage: Boolean = false
    ): GPUTextureView = acquire(width, height, format, storage).view

    /** Return a [scratch] texture to the pool. */
    fun release(view: GPUTextureView) {
        for (slots in pool.values) {
            for (slot in slots) {
                if (slot.view === view) {
                    slot.inUse = false
                    return
                }
            }
        }
    }

    fun cleanup() {
        filters.forEach { it.cleanup() }
        tailBlit.cleanup()
        destroyPool()
    }

    // ---- texture pool ----

    private class Slot(val texture: GPUTexture, val view: GPUTextureView) {
        var inUse = false
        var lastFrame = Long.MIN_VALUE

        /**
         * Both handles are AutoCloseable over a Dawn object with no finalizer, so a dropped slot keeps its
         * native texture and view until they are closed.
         */
        fun release() {
            texture.destroyAndRelease()
            view.close()
        }
    }

    private val pool = HashMap<Long, ArrayList<Slot>>()

    private var frame = 0L

    /** A filter whose output size varies frame to frame could otherwise pile up entries forever. */
    private var poolBytes = 0L
    private val maxPoolBytes = 48L * 1024 * 1024

    // Sizes are screen-scale and the format is a short enum, so one long holds the whole key.
    private fun key(width: Int, height: Int, format: Int, storage: Boolean): Long =
        (width.toLong() shl 44) or (height.toLong() shl 24) or
                (format.toLong() shl 1) or (if (storage) 1L else 0L)

    private fun acquire(width: Int, height: Int, format: Int, storage: Boolean): Slot {
        val slots = pool.getOrPut(key(width, height, format, storage)) { ArrayList() }

        var free: Slot? = null
        for (slot in slots) {
            if (!slot.inUse && (free == null || slot.lastFrame < free.lastFrame)) free = slot
        }

        // Rotate rather than reuse: writing the texture the previous frame is still reading from
        // makes the GPU serialize the two, the same hazard TileRenderer's stencil ring avoids.
        // Only up to [RING] of them, since a chain that needs several within one frame has to
        // come back round eventually.
        if (free != null && (free.lastFrame != frame - 1 || slots.size >= RING)) {
            free.inUse = true
            free.lastFrame = frame
            return free
        }

        if (poolBytes > maxPoolBytes) evictOldestUnused()

        var usage = TextureUsage.TextureBinding or TextureUsage.RenderAttachment
        if (storage) usage = usage or TextureUsage.StorageBinding

        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(width, height), format = format, usage = usage
            )
        )
        val slot = Slot(texture, texture.createView())
        slot.inUse = true
        slot.lastFrame = frame
        slots.add(slot)
        poolBytes += width.toLong() * height * 4
        return slot
    }

    /** The single oldest slot nothing is currently reading, across every size/format bucket. */
    private fun evictOldestUnused() {
        var oldestKey: Long? = null
        var oldestSlot: Slot? = null
        for ((k, slots) in pool) {
            for (slot in slots) {
                if (!slot.inUse && (oldestSlot == null || slot.lastFrame < oldestSlot.lastFrame)) {
                    oldestKey = k
                    oldestSlot = slot
                }
            }
        }
        val k = oldestKey ?: return
        val slot = oldestSlot ?: return
        val slots = pool[k] ?: return
        slots.remove(slot)
        if (slots.isEmpty()) pool.remove(k)
        poolBytes -= slot.texture.width.toLong() * slot.texture.height * 4
        slot.release()
    }

    private fun releaseAll() {
        sceneSlot = null
        for (slots in pool.values) for (slot in slots) slot.inUse = false
    }

    private fun destroyPool() {
        for (slots in pool.values) for (slot in slots) slot.release()
        pool.clear()
        poolBytes = 0L
        sceneSlot = null
    }

    // ---- tail blit ----

    // Linear, not the default nearest: the only filters that reach this path are the ones that
    // couldn't write the swapchain, which includes any that resampled to a different size.
    private val blitSampler: GPUSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
                addressModeU = AddressMode.ClampToEdge,
                addressModeV = AddressMode.ClampToEdge,
            )
        )
    }

    /**
     * Copies a filter's offscreen result to the swapchain - see [endFrame]'s `direct`. A filter
     * itself, so it inherits the pipeline, pass and per-texture bind group caching rather than
     * repeating them; it is never in [filters], and runs only when [endFrame] asks it to.
     */
    private val tailBlit = object : FilterFullscreen() {
        override val label get() = "FilterChain blit"
        override val code get() = BLIT_FS
        override fun entries(src: GPUTextureView) = arrayOf(
            GPUBindGroupEntry(0, textureView = src),
            GPUBindGroupEntry(1, sampler = blitSampler),
        )
    }

    private companion object {
        /** Textures kept per size and format, so consecutive frames don't share one. */
        const val RING = 3

        const val BLIT_FS = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(src, src_sampler, in.uv);
}
"""
    }
}
