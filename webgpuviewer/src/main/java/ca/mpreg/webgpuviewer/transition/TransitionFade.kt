package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TransitionFade : Transition() {
    private val blendByteBuffer = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(48).order(ByteOrder.nativeOrder())
    }

    private val blendSampler by lazy {
        WebGpuRenderer.device.createSampler()
    }

    private val blendPipelines = FormatKeyed { format ->
        val device = WebGpuRenderer.device
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(BLEND_SHADER))
        )
        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(format = format)
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    private const val BLEND_SHADER = """
struct Uniforms {
    blend: f32,
    bg1: vec4<f32>,
    bg2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var tex2: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;

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
    let ndc_x = pos.x * 2.0 - 1.0;
    let ndc_y = 1.0 - pos.y * 2.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = pos;
    return out;
}

fn to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb <= vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

fn to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear <= vec3<f32>(0.0031308);
    let lower = linear * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c1 = textureSample(tex1, tex_sampler, in.uv);
    let c2 = textureSample(tex2, tex_sampler, in.uv);

    let comp1 = uniforms.bg1.rgb * (1.0 - c1.a) + c1.rgb;
    let comp2 = uniforms.bg2.rgb * (1.0 - c2.a) + c2.rgb;

    let blended = mix(to_linear(comp1), to_linear(comp2), uniforms.blend);

    return vec4<f32>(to_srgb(blended), 1.0);
}
"""

    private fun putColor(buffer: ByteBuffer, color: Int) {
        buffer.putFloat(((color shr 16) and 0xFF) / 255f)
        buffer.putFloat(((color shr 8) and 0xFF) / 255f)
        buffer.putFloat((color and 0xFF) / 255f)
        buffer.putFloat(((color ushr 24) and 0xFF) / 255f)
    }

    private fun blendCached(
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        cachedView1: GPUTextureView?,
        cachedView2: GPUTextureView?,
        bg1: Int,
        bg2: Int,
        blend: Float
    ) {
        if (cachedView1 == null || cachedView2 == null) return

        val byteBuffer = blendByteBuffer.get()
        byteBuffer.clear()
        byteBuffer.putFloat(blend)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        putColor(byteBuffer, bg1)
        putColor(byteBuffer, bg2)
        byteBuffer.flip()

        val uniformBuffer = WebGpuRenderer.device.createBuffer(
            GPUBufferDescriptor(size = 48, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        )
        WebGpuRenderer.device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

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

        val blendPipeline = blendPipelines[dst.format]
        pass.setPipeline(blendPipeline)
        pass.setTransientBindGroup(
            0, WebGpuRenderer.device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = blendPipeline.groupLayout(),
                    entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = uniformBuffer),
                        GPUBindGroupEntry(1, textureView = cachedView1),
                        GPUBindGroupEntry(2, textureView = cachedView2),
                        GPUBindGroupEntry(3, sampler = blendSampler)
                    )
                )
            )
        )
        pass.draw(6)
        pass.endAndRelease(targetView, uniformBuffer)
    }

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

        // blend: 0 = fully page1, 1 = fully page2
        val blend = if (frac > 0f) frac else -frac

        val bg1 = page1.backgroundColor ?: 0xFF000000.toInt()
        val bg2 = page2.backgroundColor ?: 0xFF000000.toInt()

        blendCached(encoder, dst, cached1, cached2, bg1, bg2, blend)
    }
}
