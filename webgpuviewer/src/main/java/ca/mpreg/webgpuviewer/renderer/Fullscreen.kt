package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.Fullscreen.VERTEX

/**
 * One triangle covering the whole destination, for passes that are a function of every pixel
 * rather than a drawing of anything - [ca.mpreg.webgpuviewer.filter.FilterFullscreen]'s output
 * filters and [UpscalerArtCnn]'s halo crop.
 *
 * Here rather than beside the filters so both can reach it: the filter package already depends on
 * this one, and pointing it back would make a cycle.
 */
object Fullscreen {

    /**
     * Supplies `vs_main` and a `VertexOutput` carrying [0,1] uv, top-left to bottom-right - the
     * orientation the viewer's own draws use. Prepend it to a fragment stage.
     */
    const val VERTEX = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}
"""

    /**
     * Pipeline for [VERTEX] plus [code], writing [format]. No blend state: the triangle replaces
     * every pixel, and what these passes read already carries the alpha their destination needs.
     */
    fun buildPipeline(code: String, format: Int, label: String): GPURenderPipeline {
        val device = WebGpuRenderer.device
        val module = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(VERTEX + code))
        )
        return device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                label = label,
                vertex = GPUVertexState(module, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    module, entryPoint = "fs_main",
                    targets = arrayOf(GPUColorTargetState(format = format))
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    /** A pass over the whole of [dst]. Clears, since the triangle covers every pixel anyway. */
    fun beginPass(
        encoder: GPUCommandEncoder, dst: GPUTextureView, label: String
    ): GPURenderPassEncoder = encoder.beginRenderPass(
        GPURenderPassDescriptor(
            label = label, colorAttachments = arrayOf(
                GPURenderPassColorAttachment(
                    view = dst,
                    loadOp = LoadOp.Clear,
                    storeOp = StoreOp.Store,
                    clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                )
            )
        )
    )
}
