package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.AddressMode
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUSampler
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureBindingViewDimension
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureDimension
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.TextureViewDimension
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicReference

/**
 * Applies a 3D colour lookup table to the finished frame - a display profile, a film look, or
 * anything else a [Lut3d] can express.
 *
 * One fragment pass, so as the last filter in the chain it writes the swapchain directly. The
 * table is a 3D texture sampled trilinearly, which is what makes a coarse LUT (a 33- or
 * 64-point cube) look smooth: the hardware interpolates between entries.
 *
 * The frame arrives premultiplied, so each pixel is divided back out by its alpha before the
 * lookup and multiplied by it again afterwards - a LUT is a curve on the colour, not on the
 * colour-times-coverage the swapchain holds.
 *
 * [lut] may be set from any thread; the upload happens on the render thread at the next frame.
 */
class FilterLut3d(lut: Lut3d? = null) : FilterFullscreen() {

    /** The table to apply. Null leaves the frame untouched, the same as disabling the filter. */
    @Volatile
    var lut: Lut3d? = null
        set(value) {
            if (field === value) return
            field = value
            pending = value
            limitedRange = value?.limitedRange ?: false
            invalidate()
        }

    /**
     * How far to apply the table, 0..1. Not a quality control - it interpolates toward the
     * original colour, so a partly applied display profile is a partly wrong one - but useful
     * for showing what a look is doing.
     */
    @Volatile
    var intensity: Float = 1f
        set(value) {
            if (field == value) return
            field = value
            uniformsDirty = true
            invalidate()
        }

    /**
     * Look the table up over TV levels (16..235) rather than the full 0..255 range. Set from
     * [Lut3d.limitedRange] whenever [lut] is assigned, so a madVR table gets this on its own.
     */
    @Volatile
    var limitedRange: Boolean = false
        set(value) {
            if (field == value) return
            field = value
            uniformsDirty = true
            invalidate()
        }

    /** Nothing to apply without a table - see [Filter.active]. */
    override val active: Boolean get() = enabled && lut != null

    override val code: String get() = FRAGMENT

    private val pendingRef = AtomicReference<Lut3d?>(null)

    private var pending: Lut3d?
        get() = pendingRef.get()
        set(value) = pendingRef.set(value)

    private var texture: GPUTexture? = null
    private var view: GPUTextureView? = null
    private var lutSize = 0

    @Volatile
    private var uniformsDirty = true

    private val uniforms: GPUBuffer by lazy {
        device.createBuffer(
            GPUBufferDescriptor(
                label = label, size = 16, usage = BufferUsage.Uniform or BufferUsage.CopyDst
            )
        )
    }

    // Trilinear between entries, clamped so the outermost half-texel doesn't wrap around.
    private val lutSampler: GPUSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
                addressModeU = AddressMode.ClampToEdge,
                addressModeV = AddressMode.ClampToEdge,
                addressModeW = AddressMode.ClampToEdge,
            )
        )
    }

    init {
        // Through the setter, so a LUT passed to the constructor uploads like any other.
        this.lut = lut
    }

    override fun prepare(srcWidth: Int, srcHeight: Int) {
        // [active] keeps this filter out of the chain until a table is set, but bind something
        // real regardless rather than leave the pass unbindable.
        val next = pendingRef.getAndSet(null) ?: if (texture == null) Lut3d.identity() else null
        if (next != null) upload(next)
        if (uniformsDirty) {
            uniformsDirty = false
            writeUniforms()
        }
    }

    override fun entries(src: GPUTextureView): Array<GPUBindGroupEntry> = arrayOf(
        GPUBindGroupEntry(0, buffer = uniforms),
        GPUBindGroupEntry(1, textureView = src),
        GPUBindGroupEntry(2, textureView = view!!),
        GPUBindGroupEntry(3, sampler = lutSampler),
    )

    override fun cleanup() {
        view?.close()
        texture?.destroyAndRelease()
        texture = null
        view = null
        // Not just the texture: upload() skips creating one when the size already matches.
        lutSize = 0
        pending = lut
        uniformsDirty = true
        rebind()
    }

    private fun upload(lut: Lut3d) {
        if (lutSize != lut.size) {
            view?.close()
            texture?.destroyAndRelease()
            texture = device.createTexture(
                GPUTextureDescriptor(
                    label = label,
                    size = GPUExtent3D(lut.size, lut.size, lut.size),
                    dimension = TextureDimension._3D,
                    format = TextureFormat.RGBA16Float,
                    usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
                    // Compat mode pins a texture to one view dimension; say which up front.
                    textureBindingViewDimension = GPUTextureBindingViewDimension(
                        TextureViewDimension._3D
                    ),
                )
            )
            view = texture!!.createView()
            lutSize = lut.size
        }

        val bytes = ByteBuffer.allocateDirect(lut.size * lut.size * lut.size * 8)
            .order(ByteOrder.nativeOrder())
        var i = 0
        while (i < lut.data.size) {
            bytes.putShort(half(lut.data[i]))
            bytes.putShort(half(lut.data[i + 1]))
            bytes.putShort(half(lut.data[i + 2]))
            bytes.putShort(ONE_HALF)
            i += 3
        }
        bytes.flip()

        device.queue.writeTexture(
            GPUTexelCopyTextureInfo(texture!!),
            bytes,
            GPUExtent3D(lut.size, lut.size, lut.size),
            GPUTexelCopyBufferLayout(
                bytesPerRow = lut.size * 8, rowsPerImage = lut.size
            )
        )

        // The bind group holds the old view when the table changed size.
        rebind()
        uniformsDirty = true
    }

    private val uniformBytes: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
    }

    private fun writeUniforms() {
        val bytes = uniformBytes
        bytes.clear()
        bytes.putFloat(lutSize.toFloat())
        bytes.putFloat(if (limitedRange) 1f else 0f)
        bytes.putFloat(intensity)
        bytes.putFloat(0f)
        bytes.flip()
        device.queue.writeBuffer(uniforms, 0, bytes)
    }

    private companion object {
        /** 1.0 as an IEEE half. */
        const val ONE_HALF: Short = 0x3c00

        /** Float to IEEE half, round-to-nearest, saturating rather than overflowing to inf. */
        fun half(value: Float): Short {
            val bits = java.lang.Float.floatToRawIntBits(value)
            val sign = (bits ushr 16) and 0x8000
            val magnitude = bits and 0x7fffffff

            if (magnitude >= 0x7f800000) {
                // NaN keeps a payload bit so it stays a NaN; infinity stays infinity.
                val nan = if (magnitude > 0x7f800000) 0x200 else 0
                return (sign or 0x7c00 or nan).toShort()
            }

            val rounded = magnitude + 0x1000
            if (rounded >= 0x47800000) return (sign or 0x7bff).toShort()   // saturate to 65504
            if (rounded >= 0x38800000) return (sign or ((rounded - 0x38000000) ushr 13)).toShort()
            if (magnitude < 0x33000000) return sign.toShort()              // rounds to zero

            // Subnormal: shift the implicit one back in by hand.
            val exponent = magnitude ushr 23
            val shift = 126 - exponent
            val mantissa = (magnitude and 0x7fffff) or 0x800000
            return (sign or ((mantissa + (1 shl (shift - 1))) ushr shift)).toShort()
        }

        const val FRAGMENT = """
struct Params {
    size: f32,
    limited: f32,
    intensity: f32,
    unused: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_3d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // This pass is 1:1 with its source, so take the texel outright - no sampler, no filtering
    // to soften what the viewer already resolved, and no coordinate rounding to get wrong.
    let texel = textureLoad(src, vec2<i32>(in.position.xy), 0);
    if (texel.a <= 0.0 || params.intensity <= 0.0) {
        return texel;
    }

    // The frame is premultiplied - undo that, so the table sees the colour itself.
    let colour = clamp(texel.rgb / texel.a, vec3<f32>(0.0), vec3<f32>(1.0));

    let levels = mix(vec3<f32>(16.0 / 255.0), vec3<f32>(235.0 / 255.0), colour);
    let lookup = mix(colour, levels, params.limited);

    // Entry centres, so the ends of the table land on the ends of the range.
    let half_texel = 0.5 / params.size;
    let coord = mix(vec3<f32>(half_texel), vec3<f32>(1.0 - half_texel), lookup);
    // Level 0 explicitly: the early return above makes this non-uniform control flow, where
    // textureSample's implicit derivatives are not allowed.
    let mapped = textureSampleLevel(lut, lut_sampler, coord, 0.0).rgb;

    return vec4<f32>(mix(colour, mapped, params.intensity) * texel.a, texel.a);
}
"""
    }
}
