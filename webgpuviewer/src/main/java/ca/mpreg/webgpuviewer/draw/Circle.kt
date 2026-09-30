package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUVertexState
import androidx.webgpu.PrimitiveTopology
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import java.nio.ByteBuffer
import java.nio.ByteOrder

private val device get() = WebGpuRenderer.device

private val pipelines = FormatKeyed { format ->
    val shaderModule = device.createShaderModule(
        GPUShaderModuleDescriptor(
            shaderSourceWGSL = GPUShaderSourceWGSL(CIRCLE_SHADER)
        )
    )
    device.createRenderPipeline(
        GPURenderPipelineDescriptor(
            vertex = GPUVertexState(module = shaderModule, entryPoint = "vs_main"),
            fragment = GPUFragmentState(
                module = shaderModule, entryPoint = "fs_main", targets = arrayOf(
                    GPUColorTargetState(
                        format = format, blend = GPUBlendState(
                            color = GPUBlendComponent(
                                srcFactor = BlendFactor.SrcAlpha,
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
            primitive = GPUPrimitiveState(topology = PrimitiveTopology.TriangleList)
        )
    )
}

// Center/radius are in target pixels, matched against @builtin(position) - already window-space
// pixels - so the circle stays round regardless of the target's aspect ratio. The vertex stage
// just covers the whole clip space; the fragment stage discards everything outside the circle.
private const val CIRCLE_SHADER = """
struct Params {
    center: vec2<f32>,
    radius: f32,
    _pad: f32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.position.xy - params.center);
    let coverage = clamp(params.radius - dist + 0.5, 0.0, 1.0);
    let alpha = params.color.a * coverage;
    return vec4<f32>(params.color.rgb, alpha);
}
"""

// Thread-local ByteBuffer to avoid allocation per call
private val byteBufferLocal = ThreadLocal.withInitial {
    ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
}

/**
 * Draw a filled circle into an existing render pass, so it can share a pass with other draws.
 * Sets its own pipeline, so the caller must set theirs again before drawing something else.
 * [cx]/[cy]/[radius] are in the target's pixels, matching [GPURenderPassEncoder]'s coordinate
 * space directly.
 *
 * A fresh uniform buffer is allocated per call: several circles can share one pass, and
 * `queue.writeBuffer` is ordered against `submit` rather than against other writes, so a reused
 * buffer would give every circle in the batch the last colour written.
 */
fun Draw.circle(
    pass: GPURenderPassEncoder,
    /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
    format: Int,
    cx: Float,
    cy: Float,
    radius: Float,
    color: Int
) {
    val r = ((color shr 16) and 0xFF) / 255f
    val g = ((color shr 8) and 0xFF) / 255f
    val b = (color and 0xFF) / 255f
    val a = ((color ushr 24) and 0xFF) / 255f

    val byteBuffer = byteBufferLocal.get()
    byteBuffer.clear()
    byteBuffer.putFloat(cx)
    byteBuffer.putFloat(cy)
    byteBuffer.putFloat(radius)
    byteBuffer.putFloat(0f)
    byteBuffer.putFloat(r)
    byteBuffer.putFloat(g)
    byteBuffer.putFloat(b)
    byteBuffer.putFloat(a)
    byteBuffer.flip()

    val uniformBuffer = device.createBuffer(
        GPUBufferDescriptor(size = 32L, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
    )
    device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

    val pipeline = pipelines[format]
    pass.setPipeline(pipeline)
    pass.setTransientBindGroup(
        0, device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.groupLayout(), entries = arrayOf(
                    GPUBindGroupEntry(0, buffer = uniformBuffer)
                )
            )
        )
    )
    pass.draw(6)
    uniformBuffer.close()
}
