package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferBindingType
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBindGroupLayoutDescriptor
import androidx.webgpu.GPUBindGroupLayoutEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferBindingLayout
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPipelineLayoutDescriptor
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.ShaderStage
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.FormatKeyed
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blendBackgroundColor
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blitCachedRegion
import ca.mpreg.webgpuviewer.transition.TransitionFlip.LIT_ENDS
import ca.mpreg.webgpuviewer.transition.TransitionFlip.blankAlpha
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * A dual-page book turn: one leaf lifts off the spine, curls, and lands on the other side.
 *
 * Forward, the leaf shows page 1's right half in front and page 2's left behind, each at its own
 * size; a side with no page is cut out of the frame where the pages carry no background of their
 * own - see [blankAlpha]. The halves that stay put come from
 * the caches, clipped at their spine - see [blitCachedRegion].
 *
 * Geometry is in width fractions from the surface centre, y over the aspect ratio. No depth
 * attachment: height rises with the tangent angle while it stays inside PI, so strips emitted
 * spine-outwards land back to front.
 */
object TransitionFlip : Transition() {
    override val premultipliedOutput = true

    private const val UNIFORM_SIZE = 96

    /** Total curl at the halfway point, in radians of tangent turn from spine to outer edge. */
    private const val BEND = 0.95f

    /** Never zero - the arc's radius is length/bend. */
    private const val MIN_BEND = 0.04f

    /** How much of each end eases back to flat lighting, to match the static halves. */
    private const val LIT_ENDS = 0.15f

    /** How much of each end the static halves fade over - opaque through the rest of the turn. */
    private const val FADE_ENDS = 0.25f

    // Along the leaf only - it does not bend vertically, so rows buy just a shorter diagonal.
    private const val COLS = 64
    private const val ROWS = 2
    private const val SHEET_VERTICES = COLS * ROWS * 6

    /** Shadow grid then leaf grid, in one draw - see `vs_main`. */
    private const val VERTICES = SHEET_VERTICES * 2

    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(UNIFORM_SIZE).order(ByteOrder.nativeOrder())
    }

    private val flipSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
        )
    }

    /** The turning leaf for one frame. */
    private class Leaf(
        /** The face it starts on, in page 1's cache: normalised (x1, y1, x2, y2). */
        val frontRect: FloatArray,
        /** The face it lands on, in page 2's cache. Mirrors [frontRect] when there is none. */
        val backRect: FloatArray,
        val spine: Float,
        /** +1 resting right of the spine, -1 left. */
        val dir: Float,
        /** Rotation about the spine, 0 (at rest) to PI (landed). */
        val phi: Float,
        /** Tangent turn from spine to outer edge - the curl. [phi] + this <= PI. */
        val bend: Float,
        /** Spine to the far edge of the sheet - both faces fit inside it. */
        val len: Float,
        val top: Float,
        val bottom: Float,
        val aspect: Float,
        /** Whether that face has a cached page to sample; blank if not. */
        val hasFront: Boolean,
        val hasBack: Boolean,
        /** Shading strength, 0 at either end - see [LIT_ENDS]. */
        val shading: Float,
    )

    /**
     * Whether the surface gets filled at all.
     *
     * A page's background is ARGB 0 unless one was asked for, and [blendBackgroundColor] forces its
     * result opaque - so filling regardless paints black over a surface meant to show through.
     */
    private fun surfaceFill(page1: ImagePage, page2: ImagePage): Boolean {
        fun asks(page: ImagePage): Boolean {
            val color = page.backgroundColor ?: return false
            return (color ushr 24) != 0
        }
        return asks(page1) || asks(page2)
    }

    /**
     * How opaque the leaf's blank face is - the one with no page behind it, on a first or last turn.
     *
     * 0 wherever the pages asked for no background, which cuts the face out of the frame instead of
     * painting it, so whatever is behind the surface shows through the turning sheet. A cut rather
     * than a blend, since blending cannot take back a page already drawn - see [punchPipeline].
     */
    private fun blankAlpha(page1: ImagePage, page2: ImagePage): Float =
        if (surfaceFill(page1, page2)) 1f else 0f

    /** [punchPipeline]'s bindings: the uniform, which is all `fs_punch` and `vs_main` read. */
    private val punchBindGroupLayout by lazy {
        device.createBindGroupLayout(
            GPUBindGroupLayoutDescriptor(
                entries = arrayOf(
                    GPUBindGroupLayoutEntry(
                        binding = 0,
                        visibility = ShaderStage.Vertex or ShaderStage.Fragment,
                        buffer = GPUBufferBindingLayout(type = BufferBindingType.Uniform)
                    )
                )
            )
        )
    }

    /**
     * The pipeline that cuts a blank face out of the frame - see [blankAlpha].
     *
     * Its own, because only the blend can do this: zero over the destination, where the usual
     * `src + dst * (1 - src.a)` has no output that returns an opaque page to transparent. Same
     * module and uniforms as the leaf, entry point `fs_punch`.
     *
     * Its layout is spelled out, not inferred: `fs_punch` samples nothing, so an inferred layout
     * holds the uniform alone and rejects the leaf's own bind group - which fails the pass, and
     * with it every draw in the frame.
     */
    private val punchPipelines = FormatKeyed { format ->
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )
        val zero = GPUBlendComponent(
            srcFactor = BlendFactor.Zero,
            dstFactor = BlendFactor.Zero,
            operation = BlendOperation.Add
        )
        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                layout = device.createPipelineLayout(
                    GPUPipelineLayoutDescriptor(bindGroupLayouts = arrayOf(punchBindGroupLayout))
                ),
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_punch", targets = arrayOf(
                        GPUColorTargetState(
                            format = format, blend = GPUBlendState(color = zero, alpha = zero)
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
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

        // Clamped: a fling can carry the offset past a page, and a negative bend inverts the arc.
        val t = abs(frac).coerceIn(0f, 1f)
        // frac > 0 brings page 2 in from the right, as TransitionBasic does: the right leaf turns left.
        val forward = frac > 0f
        val spine1 = page1.spineX(dst)
        val spine2 = page2.spineX(dst)

        val background = blendBackgroundColor(
            page1.backgroundColor ?: 0xFF000000.toInt(),
            page2.backgroundColor ?: 0xFF000000.toInt(),
            t,
        )

        // Faded only on the side that actually pokes out from behind the leaf's face there, so a
        // half neither pops in unblocked nor shows through the landed leaf. Opaque mid-turn.
        val fadeIn =
            if (sidePokesOut(page2, page1, dst, !forward)) smoothstep(t / FADE_ENDS) else 1f
        val fadeOut =
            if (sidePokesOut(page1, page2, dst, forward)) smoothstep((1f - t) / FADE_ENDS) else 1f

        val pass = beginClearedPass(encoder, dst)
        try {
            if (surfaceFill(page1, page2)) Draw.rect(pass, dst.format, 0f, 0f, 1f, 1f, background)
            // Clipped at each spine: page 1 keeps the side the leaf left, page 2 the one it
            // uncovers - fading in and out at the ends to match the leaf.
            if (forward) {
                spine1?.let { blitCachedRegion(pass, dst.format, cached1, 0f, 0f, it, 1f, fadeOut) }
                spine2?.let { blitCachedRegion(pass, dst.format, cached2, it, 0f, 1f, 1f, fadeIn) }
            } else {
                spine1?.let { blitCachedRegion(pass, dst.format, cached1, it, 0f, 1f, 1f, fadeOut) }
                spine2?.let { blitCachedRegion(pass, dst.format, cached2, 0f, 0f, it, 1f, fadeIn) }
            }
        } finally {
            pass.endAndRelease()
        }

        val leaf = leaf(page1, page2, dst, t, forward, spine1, spine2, cached1, cached2) ?: return
        // A blank face is never sampled, so the surviving view stands in - [leaf] rules out both.
        val front = cached1 ?: cached2 ?: return
        val back = cached2 ?: cached1 ?: return

        val targetView = dst.createView()
        val leafPass = encoder.beginRenderPass(
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
            val blank = blankAlpha(page1, page2)
            val uniforms = uniforms(leaf, background, blank)
            val cutting = blank <= 0f && (!leaf.hasFront || !leaf.hasBack)

            // The sheet folds over itself, so a pixel can be covered twice - by the part before
            // vertical showing the front face, and the part past it showing the back. The second is
            // always the nearer, height rising with the tangent angle.
            //
            // The cut belongs directly after whichever covers the pixel in front, which is the
            // blank face's own side of the fold: earlier and the near face paints over a hole that
            // should stay open, later and it takes a page really in front of that hole.
            val cutLast = !leaf.hasBack

            if (!cutting || cutLast) {
                attach(leafPass, pipelines[dst.format], uniforms, front, back)
                leafPass.draw(VERTICES)
                // After the leaf, so the near face goes with the hole it stands in.
                if (cutting) cut(leafPass, dst.format, uniforms)
            } else {
                // Shadow first, so the cut takes it too - one hanging in the hole is cast by a
                // sheet nobody can see.
                attach(leafPass, pipelines[dst.format], uniforms, front, back)
                leafPass.draw(SHEET_VERTICES)
                cut(leafPass, dst.format, uniforms)
                attach(leafPass, pipelines[dst.format], uniforms, front, back)
                leafPass.draw(SHEET_VERTICES, 1, SHEET_VERTICES)
            }
            uniforms.close()
        } finally {
            leafPass.endAndRelease(targetView)
        }
    }

    /**
     * The leaf at [t], or null when there is nothing to turn. One sheet big enough for both faces,
     * each drawn on it at its own size - so neither resizes into the other, nor snaps at its end.
     */
    private fun leaf(
        page1: ImagePage,
        page2: ImagePage,
        dst: GPUTexture,
        t: Float,
        forward: Boolean,
        spine1: Float?,
        spine2: Float?,
        cached1: GPUTextureView?,
        cached2: GPUTextureView?,
    ): Leaf? {
        // Page 1's hinge: blending toward page 2's would slide the fold across mid-turn.
        val spine = spine1 ?: spine2 ?: return null
        // Forward: front is page 1's right half, back is page 2's left. Backward mirrors both.
        val dir = if (forward) 1f else -1f
        val rawFront = page1.leafRect(dst, left = !forward)
        val rawBack = page2.leafRect(dst, left = forward)
        // A side with no page mirrors the other across the spine, to size its blank sheet by, and
        // with neither there the halves that stay put do it - otherwise nothing turns at all.
        val sized = rawFront ?: rawBack ?: page1.leafRect(dst, left = forward) ?: page2.leafRect(
            dst,
            left = !forward
        ) ?: return null
        val frontRect = if (rawFront != null) rawFront else mirror(sized, spine)
        val backRect = rawBack ?: mirror(frontRect, spine)

        val lenFront = if (forward) frontRect[2] - spine else spine - frontRect[0]
        val lenBack = if (forward) spine - backRect[0] else backRect[2] - spine
        val len = max(lenFront, lenBack)
        if (len <= 0f) return null

        // Flat at both ends, curliest halfway.
        val bend = MIN_BEND + (BEND - MIN_BEND) * sin(PI * t).toFloat()

        return Leaf(
            frontRect = frontRect,
            backRect = backRect,
            spine = spine,
            dir = dir,
            // Held back so the outer edge stops at PI - strip order needs height monotonic.
            phi = min((PI * t).toFloat(), PI.toFloat() - bend),
            bend = bend,
            len = len,
            top = min(frontRect[1], backRect[1]),
            bottom = max(frontRect[3], backRect[3]),
            aspect = dst.width.toFloat() / dst.height,
            hasFront = rawFront != null && cached1 != null,
            hasBack = rawBack != null && cached2 != null,
            shading = min(smoothstep(t / LIT_ENDS), smoothstep((1f - t) / LIT_ENDS)),
        )
    }

    private fun smoothstep(x: Float): Float {
        val e = x.coerceIn(0f, 1f)
        return e * e * (3f - 2f * e)
    }

    /** [rect] reflected across the spine - where the leaf's other face has to lie. */
    private fun mirror(rect: FloatArray, spine: Float) =
        floatArrayOf(2f * spine - rect[2], rect[1], 2f * spine - rect[0], rect[3])

    /**
     * Whether [big]'s own rect on this side reaches past [small]'s, so a face sized by [small]
     * would not fully cover it. Both pages' real rects, not a mirrored fallback - a missing side
     * always counts as uncovered.
     */
    private fun sidePokesOut(
        big: ImagePage, small: ImagePage, dst: GPUTexture, left: Boolean
    ): Boolean {
        val bigRect = big.leafRect(dst, left) ?: return true
        val smallRect = small.leafRect(dst, left) ?: return true
        fun span(rect: FloatArray, i: Int) = rect[i + 2] - rect[i]
        fun bigger(a: Float, b: Float) = a > b + 1e-4f
        return bigger(span(bigRect, 0), span(smallRect, 0)) || bigger(
            span(bigRect, 1), span(smallRect, 1)
        )
    }

    /**
     * This frame's uniforms. [blank] paints a face with no page, at [blankAlpha] - see [blankAlpha].
     */
    private fun uniforms(leaf: Leaf, blank: Int, blankAlpha: Float): GPUBuffer {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        for (v in leaf.frontRect) byteBuffer.putFloat(v)
        for (v in leaf.backRect) byteBuffer.putFloat(v)
        byteBuffer.putFloat(leaf.spine)
        byteBuffer.putFloat(leaf.dir)
        byteBuffer.putFloat(leaf.phi)
        byteBuffer.putFloat(leaf.bend)
        byteBuffer.putFloat(leaf.len)
        byteBuffer.putFloat(leaf.top)
        byteBuffer.putFloat(leaf.bottom)
        byteBuffer.putFloat(leaf.aspect)
        byteBuffer.putFloat(if (leaf.hasFront) 1f else 0f)
        byteBuffer.putFloat(if (leaf.hasBack) 1f else 0f)
        byteBuffer.putFloat(leaf.shading)
        byteBuffer.putFloat(0f)
        // Premultiplied, so the colour goes in scaled by its own alpha.
        byteBuffer.putFloat((((blank shr 16) and 0xFF) / 255f) * blankAlpha)
        byteBuffer.putFloat((((blank shr 8) and 0xFF) / 255f) * blankAlpha)
        byteBuffer.putFloat(((blank and 0xFF) / 255f) * blankAlpha)
        byteBuffer.putFloat(blankAlpha)
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(
                size = UNIFORM_SIZE.toLong(), usage = BufferUsage.Uniform or BufferUsage.CopyDst
            )
        )
        device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)
        return uniformBuffer
    }

    /** Set [pipeline] and its own bind group on [pass] - each pipeline's layout wants its own. */
    private fun attach(
        pass: GPURenderPassEncoder,
        pipeline: GPURenderPipeline,
        uniforms: GPUBuffer,
        front: GPUTextureView,
        back: GPUTextureView,
    ) {
        pass.setPipeline(pipeline)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.groupLayout(), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = uniforms),
                        GPUBindGroupEntry(1, textureView = front),
                        GPUBindGroupEntry(2, textureView = back),
                        GPUBindGroupEntry(3, sampler = flipSampler),
                    )
                )
            )
        )
    }

    /** Cut the blank face out, over the leaf's own half of the grid - see [punchPipeline]. */
    private fun cut(pass: GPURenderPassEncoder, format: Int, uniforms: GPUBuffer) {
        pass.setPipeline(punchPipelines[format])
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = punchBindGroupLayout,
                    entries = arrayOf(GPUBindGroupEntry(0, buffer = uniforms))
                )
            )
        )
        pass.draw(SHEET_VERTICES, 1, SHEET_VERTICES)
    }

    /**
     * The cross-section is a circular arc of radius `len / bend` hinged on the spine, its tangent
     * running `phi` to `phi + bend` - so the sheet swings and curls at once. Heights are width
     * fractions like x, keeping the arc round.
     */
    override val code = """
struct Uniforms {
    // The leaf's two faces within their cached surfaces: normalised (x1, y1, x2, y2).
    front_rect: vec4<f32>,
    back_rect: vec4<f32>,
    // spine x, direction (+1 resting right of the spine), turn angle, curl
    geom: vec4<f32>,
    // sheet length, top y, bottom y, surface aspect
    span: vec4<f32>,
    // front textured, back textured, shading strength, unused
    flags: vec4<f32>,
    // What a face with no page behind it is painted, premultiplied - alpha 0 to leave it unpainted.
    blank: vec4<f32>,
}

@group(0) @binding(0) var<uniform> flip: Uniforms;
@group(0) @binding(1) var front_tex: texture_2d<f32>;
@group(0) @binding(2) var back_tex: texture_2d<f32>;
@group(0) @binding(3) var flip_sampler: sampler;

const PI: f32 = 3.14159265;

// Eye distance, in the width fractions the geometry is measured in.
const EYE: f32 = 2.6;

// The key light: centred over the book, [LIGHT_DISTANCE] out from the page, [LIGHT_ABOVE] up, in
// [EYE]'s own width fractions. Nearer than the eye on purpose - a light beyond it magnifies its
// shadow less than the eye magnifies the sheet (1.09 against 1.15), so the shadow lands inside the
// silhouette casting it and never shows. Nearer, it escapes on every side. Height is no substitute:
// dropped straight down, a shadow hides behind a sheet that runs the height of the page.
const LIGHT_DISTANCE: f32 = 1.25;
const LIGHT_ABOVE: f32 = 0.0;

// How fast a shadow fades with the sheet's lift, per leaf length, and how dark it is at contact.
// Gentle: steeper, and it is spent before the curl lifts it clear of the leaf at all.
const SHADOW_FALLOFF: f32 = 1.0;
const SHADOW_DEPTH: f32 = 1.0;

// How far the curl's far edge falls below the page it left - see [leaf_shade].
const CURL_SHADE: f32 = 0.25;

// Penumbra width, as a fraction of the shadow's span and of its height, and how far it spreads per
// leaf length of lift - what softens a rising shadow, since its depth barely thins it.
const SOFT_ALONG: f32 = 0.08;
const SOFT_DOWN: f32 = 0.05;
const SOFT_SPREAD: f32 = 1.0;

fn leaf_radius() -> f32 { return flip.span.x / flip.geom.w; }

/// The tangent angle at arc fraction [s] along the leaf - 0 at the spine, 1 at the outer edge.
fn leaf_angle(s: f32) -> f32 { return flip.geom.z + flip.geom.w * s; }

/// A point on the leaf in centred width fractions, at tangent angle [b] and vertical fraction [v].
fn leaf_point(b: f32, v: f32) -> vec3<f32> {
    let r = leaf_radius();
    let phi = flip.geom.z;
    let y = mix(flip.span.y, flip.span.z, v);
    return vec3<f32>(
        (flip.geom.x - 0.5) + flip.geom.y * r * (sin(b) - sin(phi)),
        (y - 0.5) / flip.span.w,
        r * (cos(phi) - cos(b)),
    );
}

/// One point of the sheet, as far as casting a shadow cares: how far out from the spine, how high.
struct Cast {
    out: f32,
    z: f32,
}

fn cast_at(b: f32) -> Cast {
    let r = leaf_radius();
    let phi = flip.geom.z;
    var c: Cast;
    c.out = r * (sin(b) - sin(phi));
    c.z = r * (cos(phi) - cos(b));
    return c;
}

struct Span {
    lo: Cast,
    hi: Cast,
}

/// What the sheet shadows, as one span across the page - the two points that bound the rest.
///
/// Its strips stop running in footprint order once it leans past vertical: each one then retraces
/// ground the ones before it covered. Cast strip by strip that band takes shadow twice over - two
/// layers of paper block no more light than one - and creases where it turns back. Filling the span
/// its extremes bound covers it once: the hinge, the far edge, and the crest at vertical.
fn shadow_span() -> Span {
    let phi = flip.geom.z;
    let bend = flip.geom.w;

    var lo = cast_at(phi);
    var hi = cast_at(phi + bend);
    if (hi.out < lo.out) {
        let swap = lo;
        lo = hi;
        hi = swap;
    }
    if (phi < 0.5 * PI && phi + bend > 0.5 * PI) {
        let crest = cast_at(0.5 * PI);
        if (crest.out > hi.out) { hi = crest; }
        if (crest.out < lo.out) { lo = crest; }
    }

    var span: Span;
    span.lo = lo;
    span.hi = hi;
    return span;
}

/// Centred width fractions back to normalised surface coordinates, under perspective.
fn project(p: vec3<f32>) -> vec2<f32> {
    let s = EYE / (EYE - p.z);
    return vec2<f32>(0.5 + p.x * s, 0.5 + p.y * s * flip.span.w);
}

/// The lamp - see [LIGHT_DISTANCE]. Fixed in the surface: one carried by the leaf would hold its
/// shadow at the same offset all turn.
fn light_pos() -> vec3<f32> {
    return vec3<f32>(0.0, -LIGHT_ABOVE, LIGHT_DISTANCE);
}

/// Where a point on the leaf lays its shadow on the page. The clamp guards only a leaf risen as
/// high as its own light.
fn shadow_cast(p: vec3<f32>) -> vec2<f32> {
    let light = light_pos();
    let t = light.z / max(light.z - p.z, 0.25 * light.z);
    return light.xy + t * (p.xy - light.xy);
}

/// How dark the shadow is where the point casting it sits [z] above the page.
fn shadow_alpha(z: f32) -> f32 {
    return flip.flags.z * SHADOW_DEPTH * exp(-SHADOW_FALLOFF * max(z, 0.0) / flip.span.x);
}

/// How dark the sheet is at tangent angle [b] - the curl's form, not a light in the world.
///
/// The halves either side are blitted from their caches unshaded, so flat paper is 1.0 by
/// definition and the leaf has to meet that where it joins them. At the spine it is the same
/// unoccluded sheet, so it stays 1.0 - which no Lambert term can manage, the sheet standing
/// vertical there under a light that grazes it. This measures the turn from the hinge instead: 0
/// there, most at the far edge, and continuous across the fold since it never asks which face
/// shows. Eased off at both ends of the turn, so the leaf lands as lit as the half it becomes.
fn leaf_shade(b: f32) -> f32 {
    let turned = 1.0 - cos(b - flip.geom.z);
    return mix(1.0, 1.0 - CURL_SHADE * turned, flip.flags.z);
}

/// The face showing at a point of the sheet - whichever way the surface is turned.
struct Face {
    front: bool,
    /// False past this face's edges, and in the gutter before it starts.
    covers: bool,
    /// False for a side with no page: it draws blank rather than sampling.
    textured: bool,
    /// Surface coordinate, and so the texture coordinate - which lays the face on the sheet at its
    /// own size instead of stretching it over the whole sheet.
    uv: vec2<f32>,
    rect: vec4<f32>,
    /// Which way this face runs from the spine.
    side: f32,
}

fn face_at(s: f32, v: f32, b: f32) -> Face {
    var f: Face;
    f.front = cos(b) > 0.0;
    f.side = select(-flip.geom.y, flip.geom.y, f.front);
    f.uv = vec2<f32>(flip.geom.x + f.side * s * flip.span.x, mix(flip.span.y, flip.span.z, v));
    f.rect = select(flip.back_rect, flip.front_rect, f.front);
    f.textured = select(flip.flags.y, flip.flags.x, f.front) > 0.5;
    f.covers = f.uv.x >= f.rect.x && f.uv.x <= f.rect.z &&
        f.uv.y >= f.rect.y && f.uv.y <= f.rect.w;
    return f;
}

/// Fades the shadow towards every edge - a stand-in penumbra, the end at the spine included: there
/// the sheet meets the page as a fold, not a knife, and an unfeathered band reads as ink.
///
/// In the shadow's own [s] and [v], so it holds however the span was reached. Widens with the
/// caster's height [z] - a contact shadow is crisp, one thrown from a lifted curl broad.
fn shadow_softness(s: f32, v: f32, z: f32) -> f32 {
    let spread = 1.0 + SOFT_SPREAD * max(z, 0.0) / flip.span.x;
    let soft_s = min(SOFT_ALONG * spread, 0.5);
    let soft_v = min(SOFT_DOWN * spread, 0.5);
    return smoothstep(0.0, soft_s, s) * smoothstep(0.0, soft_s, 1.0 - s) *
        smoothstep(0.0, soft_v, v) * smoothstep(0.0, soft_v, 1.0 - v);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Arc fraction along the leaf and vertical fraction down it.
    @location(0) s: f32,
    @location(1) v: f32,
    // The sheet point, carried so the fragment stage need not redo the arc's trig.
    @location(2) world: vec3<f32>,
    // 1 on the shadow half; every vertex of a quad shares it. Not flat - compat mode rejects that.
    @location(3) shadow: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const COLS: u32 = ${COLS}u;
    const ROWS: u32 = ${ROWS}u;
    const SHEET: u32 = ${SHEET_VERTICES}u;

    // Shadow grid first, leaf over it. Split by index, not two draws - one uniform buffer, not two.
    let shadow = vertex_index < SHEET;
    let i = select(vertex_index - SHEET, vertex_index, shadow);

    let quad_index = i / 6u;
    let vert_in_quad = i % 6u;
    let col = quad_index % COLS;
    let row = quad_index / COLS;

    let s0 = f32(col) / f32(COLS);
    let s1 = f32(col + 1u) / f32(COLS);
    let v0 = f32(row) / f32(ROWS);
    let v1 = f32(row + 1u) / f32(ROWS);

    var sv: vec2<f32>;
    switch (vert_in_quad) {
        case 0u: { sv = vec2<f32>(s0, v0); }
        case 1u: { sv = vec2<f32>(s0, v1); }
        case 2u: { sv = vec2<f32>(s1, v0); }
        case 3u: { sv = vec2<f32>(s1, v0); }
        case 4u: { sv = vec2<f32>(s0, v1); }
        default: { sv = vec2<f32>(s1, v1); }
    }

    var world: vec3<f32>;
    var screen: vec2<f32>;
    if (shadow) {
        // Across the span the sheet shadows rather than along the sheet - see [shadow_span] - so
        // the footprint runs in order and is covered once. Dropped through the light onto the page,
        // landing at z = 0, so no perspective divide.
        let span = shadow_span();
        let y = mix(flip.span.y, flip.span.z, sv.y);
        world = vec3<f32>(
            (flip.geom.x - 0.5) + flip.geom.y * mix(span.lo.out, span.hi.out, sv.x),
            (y - 0.5) / flip.span.w,
            mix(span.lo.z, span.hi.z, sv.x),
        );
        let flat = shadow_cast(world);
        screen = vec2<f32>(0.5 + flat.x, 0.5 + flat.y * flip.span.w);
    } else {
        world = leaf_point(leaf_angle(sv.x), sv.y);
        screen = project(world);
    }

    var out: VertexOutput;
    out.position = vec4<f32>(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0, 0.0, 1.0);
    out.s = sv.x;
    out.v = sv.y;
    out.world = world;
    out.shadow = select(0.0, 1.0, shadow);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Premultiplied black, so the shadow multiplies what is under it down rather than tinting it.
    // Not clipped to the sheet's own rect the way a face is: a shadow falls where it falls, and its
    // span is already what bounds it.
    if (in.shadow > 0.5) {
        return vec4<f32>(
            0.0, 0.0, 0.0,
            shadow_alpha(in.world.z) * shadow_softness(in.s, in.v, in.world.z),
        );
    }

    let b = leaf_angle(in.s);
    let face = face_at(in.s, in.v, b);

    if (!face.covers) { discard; }

    // A blank face with nothing to paint it is not drawn at all, so a turn off the first or last
    // page shows what is behind the surface through the sheet - see [blankAlpha].
    if (!face.textured && flip.blank.a <= 0.0) { discard; }

    var texel = flip.blank;
    if (face.textured) {
        if (face.front) {
            texel = textureSampleLevel(front_tex, flip_sampler, face.uv, 0.0);
        } else {
            texel = textureSampleLevel(back_tex, flip_sampler, face.uv, 0.0);
        }
    }

    // Premultiplied throughout - see premultipliedOutput - so shading scales rgb alone.
    return vec4<f32>(texel.rgb * leaf_shade(b), texel.a);
}

/// The blank face alone, for the pipeline that cuts it out - see [blankAlpha]. What it returns is
/// discarded by that blend, which takes zero of both sides.
@fragment
fn fs_punch(in: VertexOutput) -> @location(0) vec4<f32> {
    let face = face_at(in.s, in.v, leaf_angle(in.s));
    if (!face.covers || face.textured) { discard; }
    return vec4<f32>(0.0);
}"""
}
