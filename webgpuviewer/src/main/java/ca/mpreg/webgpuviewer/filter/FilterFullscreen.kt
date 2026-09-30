package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPUTextureView
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.Fullscreen
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout

/**
 * A [Filter] that is one fragment pass over the whole frame - the shape every per-pixel filter
 * takes, and the one that can write the swapchain directly.
 *
 * Subclasses supply [code] (a `fs_main` taking [Fullscreen.VERTEX]'s `VertexOutput`) and
 * [entries] (its group 0 bindings). Bind groups are kept per source texture rather than for the last one only,
 * since [FilterChain] rotates its textures across frames and a single-entry cache would then
 * miss every frame; a filter whose own bindings change - a new LUT, say - calls [rebind].
 *
 * There is no blend state: the pass covers every pixel and replaces it, and the frame it reads
 * already carries the alpha the swapchain needs.
 */
abstract class FilterFullscreen : Filter() {

    /**
     * WGSL fragment stage. [Fullscreen.VERTEX] is prepended, so `VertexOutput` and `in.uv` are
     * in scope.
     */
    protected abstract val code: String

    // Per format: [outputFormat] follows [Hdr.frameFormat], which an HDR toggle changes.
    private val pipelines = FormatKeyed { format -> Fullscreen.buildPipeline(code, format, label) }

    protected open val pipeline: GPURenderPipeline get() = pipelines[outputFormat]

    /** Group 0 bindings for this pass, with the chain's current input as [src]. */
    protected abstract fun entries(src: GPUTextureView): Array<GPUBindGroupEntry>

    // Keyed by the view's native handle, which outlives the wrapper object. Zero is free: a
    // live handle is never null.
    private val boundHandles = LongArray(CACHED_BIND_GROUPS)
    private val bindGroups = arrayOfNulls<GPUBindGroup>(CACHED_BIND_GROUPS)
    private var nextBindGroup = 0

    // The pipeline the cached groups were built against - an auto layout is per pipeline.
    private var boundPipeline: GPURenderPipeline? = null

    private fun dropBindGroups() {
        boundHandles.fill(0L)
        bindGroups.forEach { it?.close() }
        bindGroups.fill(null)
    }

    /** Drop the cached bind groups, for a filter whose own bindings have changed. */
    protected fun rebind() {
        dropBindGroups()
        invalidate()
    }

    override fun cleanup() = dropBindGroups()

    private fun bindGroupFor(src: GPUTextureView, pipeline: GPURenderPipeline): GPUBindGroup {
        if (pipeline !== boundPipeline) {
            dropBindGroups()
            boundPipeline = pipeline
        }
        val handle = src.handle
        for (i in boundHandles.indices) {
            if (boundHandles[i] == handle) bindGroups[i]?.let { return it }
        }

        val group = device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.groupLayout(), label = label, entries = entries(src)
            )
        )
        boundHandles[nextBindGroup] = handle
        // Evicted: a pass that used it holds its own reference, so ours can go.
        bindGroups[nextBindGroup]?.close()
        bindGroups[nextBindGroup] = group
        nextBindGroup = (nextBindGroup + 1) % CACHED_BIND_GROUPS
        return group
    }

    /** Prepare GPU state for this frame - uploads and the like, before the pass opens. */
    protected open fun prepare(srcWidth: Int, srcHeight: Int) {}

    override fun run(
        chain: FilterChain,
        encoder: GPUCommandEncoder,
        src: GPUTextureView,
        srcWidth: Int,
        srcHeight: Int,
        dst: GPUTextureView,
        dstWidth: Int,
        dstHeight: Int,
    ) {
        prepare(srcWidth, srcHeight)

        // Before the pass opens: prepare() may have replaced a binding and dropped the cache.
        val pipeline = pipeline
        val group = bindGroupFor(src, pipeline)

        val pass = Fullscreen.beginPass(encoder, dst, label)
        try {
            pass.setPipeline(pipeline)
            pass.setBindGroup(0, group)
            pass.draw(3)
        } finally {
            pass.endAndRelease()
        }
    }

    private companion object {
        /** Enough for [FilterChain]'s texture ring, so a steady chain builds none per frame. */
        const val CACHED_BIND_GROUPS = 4
    }
}
