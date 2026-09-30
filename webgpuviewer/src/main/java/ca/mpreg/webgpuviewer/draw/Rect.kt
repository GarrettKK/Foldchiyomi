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
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import java.nio.ByteBuffer
import java.nio.ByteOrder

private val device get() = WebGpuRenderer.device

private val pipelines = FormatKeyed { format ->
    val shaderModule = device.createShaderModule(
        GPUShaderModuleDescriptor(
            shaderSourceWGSL = GPUShaderSourceWGSL(RECT_SHADER)
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

private const val RECT_SHADER = """
struct Params {
    rect: vec4<f32>,  // left, top, right, bottom in NDC
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Two triangles forming a quad
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );
    
    let pos = positions[vertex_index];
    
    // Interpolate between rect bounds (in normalized 0-1 coords stored in params)
    let x = mix(params.rect.x, params.rect.z, pos.x);
    let y = mix(params.rect.y, params.rect.w, pos.y);
    
    // Convert to NDC [-1, 1]
    let ndc_x = x * 2.0 - 1.0;
    let ndc_y = 1.0 - y * 2.0;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return params.color;
}
"""

// Thread-local ByteBuffer to avoid allocation per call
private val byteBufferLocal = ThreadLocal.withInitial {
    ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
}

/**
 * Draw a filled rectangle with the specified color.
 * Coordinates are in normalized [0, 1] range.
 */
fun Draw.rect(
    encoder: GPUCommandEncoder,
    texture: GPUTexture,
    x1: Float,
    y1: Float,
    x2: Float,
    y2: Float,
    color: Int
) {
    val targetView = texture.createView()
    val pass = encoder.beginRenderPass(
        GPURenderPassDescriptor(
            colorAttachments = arrayOf(
                GPURenderPassColorAttachment(
                    view = targetView,
                    loadOp = LoadOp.Load,
                    storeOp = StoreOp.Store,
                    clearValue = androidx.webgpu.GPUColor(0.0, 0.0, 0.0, 0.0)
                )
            )
        )
    )
    rect(pass, texture.format, x1, y1, x2, y2, color)
    pass.endAndRelease(targetView)
}

/**
 * Draw a filled rectangle into an existing render pass, so it can share a pass with other draws.
 * Sets its own pipeline, so the caller must set theirs again before drawing something else.
 *
 * A fresh uniform buffer is allocated per call: several rects can share one pass, and
 * `queue.writeBuffer` is ordered against `submit` rather than against other writes, so a reused
 * buffer would give every rect in the batch the last colour written.
 */
fun Draw.rect(
    pass: GPURenderPassEncoder,
    /** Format of [pass]'s colour attachment - see [FormatKeyed]. */
    format: Int,
    x1: Float,
    y1: Float,
    x2: Float,
    y2: Float,
    color: Int
) {
    val r = ((color shr 16) and 0xFF) / 255f
    val g = ((color shr 8) and 0xFF) / 255f
    val b = (color and 0xFF) / 255f
    val a = ((color ushr 24) and 0xFF) / 255f

    val byteBuffer = byteBufferLocal.get()
    byteBuffer.clear()
    byteBuffer.putFloat(x1)
    byteBuffer.putFloat(y1)
    byteBuffer.putFloat(x2)
    byteBuffer.putFloat(y2)
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
