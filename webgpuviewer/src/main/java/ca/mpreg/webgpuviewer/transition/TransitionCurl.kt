package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.endAndRelease
import ca.mpreg.webgpuviewer.renderer.groupLayout
import ca.mpreg.webgpuviewer.renderer.setTransientBindGroup
import ca.mpreg.webgpuviewer.transition.Transition.Companion.beginClearedPass
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blendBackgroundColor
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A page curl in the manner of Google Play Books: the page is taken by its edge or corner and
 * rolls over after the finger, showing its back, casting a shadow on the page it uncovers.
 *
 * Worked in a canonical frame where the turning sheet is hinged on its left and turns leftward -
 * right-to-left reading mirrors x in and out of it. In that frame page A, the earlier page, is the
 * one that turns away and page B lies under it: forward A is page 1, backward A is page 2 turning
 * back into place, so a backward turn is a forward one played in reverse.
 *
 * The grabbed point of the outer edge goes where the finger has moved it, the fold between the
 * two, as a sheet folded flat would. The fold itself is rolled round a cylinder of radius R, set
 * back half its circumference so the corner still lands under the finger. Nothing on the hinge
 * moves: the corner is held within reach of both ends of it, as the paper would hold it.
 *
 * Distances are in surface widths, y scaled by height over width, so the roll stays round.
 */
object TransitionCurl : Transition() {
    override val premultipliedOutput = true

    private const val UNIFORM_SIZE = 96

    /** Curl radius at its largest, as a fraction of the sheet's width. */
    private const val RADIUS = 0.085f

    /** How close to the top or bottom a grab counts as taking the corner itself. */
    private const val CORNER_SNAP = 0.14f

    /** How far a tap-driven turn lifts its corner, as a fraction of the sheet's height. */
    private const val AUTO_LIFT = 0.16f

    /** Paper for a face with nothing printed on it, when the page names no colour of its own. */
    private const val PAPER = 0xFFFAFAF7.toInt()

    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(UNIFORM_SIZE).order(ByteOrder.nativeOrder())
    }

    private val curlSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
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
    ) = render(page1, page2, encoder, dst, frac, pos1, pos2, tiles, TurnGesture(false, true, 0f))

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
        gesture: TurnGesture,
    ) {
        val parent = page1.parent ?: page2.parent
        // A curl turns sideways; a vertical pager slides instead.
        if (parent?.isVertical == true) {
            TransitionBasic.Vertical.render(page1, page2, encoder, dst, frac, pos1, pos2, tiles)
            return
        }

        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)
        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        val mirror = parent?.isReversed == true
        // Page 1 is the earlier page when it leaves toward the hinge side.
        val aIsPage1 = (if (mirror) -frac else frac) > 0f
        val t = turnProgress(frac, aIsPage1)
        val pageA = if (aIsPage1) page1 else page2
        val pageB = if (aIsPage1) page2 else page1
        val cachedA = if (aIsPage1) cached1 else cached2
        val cachedB = if (aIsPage1) cached2 else cached1
        // A missing face is painted as paper, so the other view only stands in for binding.
        val viewA = cachedA ?: cachedB
        val viewB = cachedB ?: cachedA

        val aspect = dst.height.toFloat() / dst.width
        val geometry = sheet(pageA, pageB, dst, mirror)

        val background = blendBackgroundColor(
            page1.backgroundColor ?: 0xFF000000.toInt(),
            page2.backgroundColor ?: 0xFF000000.toInt(),
            abs(frac).coerceIn(0f, 1f),
        )

        val pass = beginClearedPass(encoder, dst)
        try {
            if (surfaceFill(page1, page2)) Draw.rect(pass, dst.format, 0f, 0f, 1f, 1f, background)
            if (viewA == null || viewB == null || geometry == null) return

            val curl = curl(geometry, aspect, t, frac, aIsPage1, pos1, pos2, dst, gesture)
            val paper = paperColor(pageA)
            val uniforms = uniforms(
                geometry, curl, aspect, t, mirror, cachedA != null, cachedB != null, paper
            )
            val pipeline = pipelines[dst.format]
            pass.setPipeline(pipeline)
            pass.setTransientBindGroup(
                0, device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.groupLayout(), entries = arrayOf(
                            GPUBindGroupEntry(0, buffer = uniforms),
                            GPUBindGroupEntry(1, textureView = viewA),
                            GPUBindGroupEntry(2, textureView = viewB),
                            GPUBindGroupEntry(3, sampler = curlSampler),
                        )
                    )
                )
            )
            pass.draw(6)
            uniforms.close()
        } finally {
            pass.endAndRelease()
        }
    }

    /** How far page A has turned away, 0 lying flat to 1 turned over, from a frame's [frac]. */
    private fun turnProgress(frac: Float, aIsPage1: Boolean): Float {
        val f = abs(frac).coerceIn(0f, 1f)
        return if (aIsPage1) f else 1f - f
    }

    /** The turning sheet, in canonical normalised coordinates. */
    private class Sheet(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        /** Where it is hinged - its own left edge. */
        val hinge: Float,
        /** A book spread: the back of the leaf is page B's other half, not A seen through. */
        val dual: Boolean,
    )

    /** The fold for one frame, in canonical metric coordinates. */
    private class Curl(
        val axisX: Float,
        val axisY: Float,
        /** Unit normal of the fold, pointing into the part that has lifted. */
        val nx: Float,
        val ny: Float,
        val radius: Float,
        /** How much the lifted sheet shades what is under it, 0 lying flat either way. */
        val lift: Float,
    )

    private fun canonical(rect: FloatArray, mirror: Boolean): FloatArray =
        if (mirror) floatArrayOf(1f - rect[2], rect[1], 1f - rect[0], rect[3]) else rect

    /**
     * The sheet page A turns: the leaf on the far side of the spine for a spread, the whole page
     * hinged on its inner edge for a single. Null with nothing placed to turn at all.
     */
    private fun sheet(pageA: ImagePage, pageB: ImagePage, dst: GPUTexture, mirror: Boolean): Sheet? {
        val dual = pageA is ImagePage.ImageSpread || pageB is ImagePage.ImageSpread
        if (dual) {
            val spine = pageA.spineX(dst) ?: pageB.spineX(dst) ?: 0.5f
            val hinge = if (mirror) 1f - spine else spine
            // Canonical right is the side the leaf lies on: the screen's left when mirrored.
            val leaf = (pageA.leafRect(dst, left = mirror) ?: pageB.leafRect(dst, left = mirror))
                ?.let { canonical(it, mirror) }
                ?: (pageA.leafRect(dst, left = !mirror) ?: pageB.leafRect(dst, left = !mirror))
                    ?.let { canonical(it, mirror) }
                    ?.let { floatArrayOf(2f * hinge - it[2], it[1], 2f * hinge - it[0], it[3]) }
                ?: return null
            val x2 = max(leaf[2], hinge + 1e-3f)
            return Sheet(hinge, leaf[1], x2, leaf[3], hinge, dual = true)
        }
        val rect = (pageA.pageRect(dst) ?: pageB.pageRect(dst))?.let { canonical(it, mirror) }
            ?: floatArrayOf(0f, 0f, 1f, 1f)
        if (rect[2] - rect[0] <= 1e-4f || rect[3] - rect[1] <= 1e-4f) return null
        return Sheet(rect[0], rect[1], rect[2], rect[3], rect[0], dual = false)
    }

    private fun smoothstep(x: Float): Float {
        val e = x.coerceIn(0f, 1f)
        return e * e * (3f - 2f * e)
    }

    /**
     * Where the fold lies. The grabbed point of the outer edge travels with the finger across -
     * which [frac] already tracks one to one - and up or down with it while it is held; let go,
     * the lift eases out over the settle so the page lands flat.
     */
    private fun curl(
        sheet: Sheet,
        aspect: Float,
        t: Float,
        frac: Float,
        aIsPage1: Boolean,
        pos1: Offset,
        pos2: Offset,
        dst: GPUTexture,
        gesture: TurnGesture,
    ): Curl {
        val sx1 = sheet.hinge
        val sx2 = sheet.x2
        val sy1 = sheet.y1 * aspect
        val sy2 = sheet.y2 * aspect
        val width = sx2 - sx1

        // Turned over is the outer edge mirrored across the hinge, two sheet widths off. A leaf of a
        // spread is about half the surface, so that is the finger's own travel; a single page
        // needs twice it, so the corner starts at the finger's pace and gains toward the end.
        val travel = 2f * width
        val pace = min(1f, 1f / travel)
        val eased = pace * t + (1f - pace) * t * t
        val px = sx2 - travel * eased

        val height = sheet.y2 - sheet.y1
        var grab = if (gesture.auto) sheet.y2 else (pos1.y / dst.height).coerceIn(sheet.y1, sheet.y2)
        if (grab - sheet.y1 < CORNER_SNAP * height) grab = sheet.y1
        if (sheet.y2 - grab < CORNER_SNAP * height) grab = sheet.y2
        val gy = grab * aspect

        val fingerDy = (pos2.y - pos1.y) / dst.height * aspect
        val dy = when {
            gesture.auto -> -AUTO_LIFT * (sy2 - sy1) * sin(PI * t).toFloat()
            // Eased in, so the curl never starts already lifted.
            gesture.held -> fingerDy * smoothstep(t / 0.06f)
            else -> {
                val release = gesture.releaseFrac
                if (release == 0f || sign(release) != sign(frac)) {
                    0f
                } else {
                    val tRelease = turnProgress(release, aIsPage1)
                    val envelope = when {
                        t <= tRelease -> if (tRelease > 1e-4f) t / tRelease else 0f
                        else -> if (tRelease < 1f - 1e-4f) (1f - t) / (1f - tRelease) else 0f
                    }
                    fingerDy * smoothstep(tRelease / 0.06f) * envelope
                }
            }
        }

        val p0x = sx2
        val p0y = gy
        var cx = px
        var cy = gy + dy
        // The hinge does not move: the corner stays within its own reach of both ends of it.
        repeat(3) {
            for (hy in floatArrayOf(sy1, sy2)) {
                val reach = hypot(p0x - sx1, p0y - hy)
                val vx = cx - sx1
                val vy = cy - hy
                val l = hypot(vx, vy)
                if (l > reach && l > 0f) {
                    cx = sx1 + vx * reach / l
                    cy = hy + vy * reach / l
                }
            }
        }

        val lift = smoothstep(t / 0.05f) * (1f - smoothstep((t - 0.9f) / 0.1f))
        val dx = p0x - cx
        val dyc = p0y - cy
        val dist = hypot(dx, dyc)
        if (dist < 1e-5f) {
            // Lying flat: a fold past the far edge leaves the whole sheet where it is.
            return Curl(sx2 + 10f, 0f, 1f, 0f, 1e-4f, 0f)
        }
        val nx = dx / dist
        val ny = dyc / dist
        // Rolled tightest never: large enough to read as paper, small enough to land flat.
        val radius = max(
            min(RADIUS * max(width, 0.3f), dist * 0.2f) * (1f - smoothstep((t - 0.8f) / 0.2f)),
            1e-4f,
        )
        val mx = (p0x + cx) * 0.5f
        val my = (p0y + cy) * 0.5f
        val back = PI.toFloat() * radius * 0.5f
        return Curl(mx - nx * back, my - ny * back, nx, ny, radius, lift)
    }

    private fun hypot(x: Float, y: Float) = sqrt(x * x + y * y)

    /** See [TransitionFlip] - a page asking for no background leaves the surface see-through. */
    private fun surfaceFill(page1: ImagePage, page2: ImagePage): Boolean {
        fun asks(page: ImagePage): Boolean {
            val color = page.backgroundColor ?: return false
            return (color ushr 24) != 0
        }
        return asks(page1) || asks(page2)
    }

    /** What the sheet is made of where nothing is printed: its own background if it has one. */
    private fun paperColor(page: ImagePage): Int {
        val color = page.backgroundColor ?: return PAPER
        return if ((color ushr 24) == 0xFF) color else PAPER
    }

    private fun uniforms(
        sheet: Sheet,
        curl: Curl,
        aspect: Float,
        t: Float,
        mirror: Boolean,
        hasA: Boolean,
        hasB: Boolean,
        paper: Int,
    ) = byteBufferLocal.get().let { byteBuffer ->
        byteBuffer.clear()
        // Sheet in metric coordinates.
        byteBuffer.putFloat(sheet.hinge)
        byteBuffer.putFloat(sheet.y1 * aspect)
        byteBuffer.putFloat(sheet.x2)
        byteBuffer.putFloat(sheet.y2 * aspect)
        byteBuffer.putFloat(curl.axisX)
        byteBuffer.putFloat(curl.axisY)
        byteBuffer.putFloat(curl.nx)
        byteBuffer.putFloat(curl.ny)
        byteBuffer.putFloat(curl.radius)
        byteBuffer.putFloat(aspect)
        byteBuffer.putFloat(if (mirror) 1f else 0f)
        byteBuffer.putFloat(if (sheet.dual) 1f else 0f)
        byteBuffer.putFloat(t)
        byteBuffer.putFloat(curl.lift)
        byteBuffer.putFloat(if (hasA) 1f else 0f)
        byteBuffer.putFloat(if (hasB) 1f else 0f)
        byteBuffer.putFloat(((paper shr 16) and 0xFF) / 255f)
        byteBuffer.putFloat(((paper shr 8) and 0xFF) / 255f)
        byteBuffer.putFloat((paper and 0xFF) / 255f)
        byteBuffer.putFloat(1f)
        byteBuffer.putFloat(sheet.hinge)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(
                size = UNIFORM_SIZE.toLong(), usage = BufferUsage.Uniform or BufferUsage.CopyDst
            )
        )
        device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)
        uniformBuffer
    }

    /**
     * One full-surface pass: for each pixel, the topmost layer of the rolled sheet over it - the
     * back lying flat, the back or front of the roll, the front still flat - else what is under
     * the sheet, with the shadows the lifted paper throws on it.
     */
    override val code = """
struct Uniforms {
    // The turning sheet, metric: hinge x, top, outer x, bottom.
    sheet: vec4<f32>,
    // A point on the roll's axis, and the unit normal into the lifted part.
    axis: vec4<f32>,
    // Roll radius, height over width, mirrored, book spread.
    geom: vec4<f32>,
    // Turn progress, shadow strength, A drawn, B drawn.
    state: vec4<f32>,
    // Paper, premultiplied.
    paper: vec4<f32>,
    // Hinge x, unused.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex_a: texture_2d<f32>;
@group(0) @binding(2) var tex_b: texture_2d<f32>;
@group(0) @binding(3) var curl_sampler: sampler;

const PI: f32 = 3.14159265;

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
    let pos = corners[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
    out.uv = pos;
    return out;
}

fn mirrored() -> bool { return u.geom.z > 0.5; }

/// A canonical metric point back to a texture coordinate on the surface.
fn tex_uv(m: vec2<f32>) -> vec2<f32> {
    let x = select(m.x, 1.0 - m.x, mirrored());
    return vec2<f32>(x, m.y / u.geom.y);
}

fn sample_a(m: vec2<f32>) -> vec4<f32> {
    if (u.state.z < 0.5) { return vec4<f32>(0.0); }
    return textureSampleLevel(tex_a, curl_sampler, tex_uv(m), 0.0);
}

fn sample_b(m: vec2<f32>) -> vec4<f32> {
    if (u.state.w < 0.5) { return vec4<f32>(0.0); }
    return textureSampleLevel(tex_b, curl_sampler, tex_uv(m), 0.0);
}

/// Premultiplied [c] over the paper - the page as printed on its sheet.
fn on_paper(c: vec4<f32>) -> vec4<f32> {
    return c + u.paper * (1.0 - c.a);
}

/// Signed distance from [p] to the sheet's edge, negative inside.
fn sheet_distance(p: vec2<f32>) -> f32 {
    let c = (u.sheet.xy + u.sheet.zw) * 0.5;
    let half_size = (u.sheet.zw - u.sheet.xy) * 0.5;
    let q = abs(p - c) - half_size;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn in_sheet(p: vec2<f32>) -> bool { return sheet_distance(p) <= 0.0; }

/// Black of strength [a] laid over premultiplied [c].
fn shade_over(c: vec4<f32>, a: f32) -> vec4<f32> {
    return vec4<f32>(c.rgb * (1.0 - a), c.a + a * (1.0 - c.a));
}

/// The front face at sheet point [src].
fn front_color(src: vec2<f32>) -> vec4<f32> {
    return on_paper(sample_a(src));
}

/// The back face at sheet point [src]: in a book, the next spread's other half landing there;
/// on a single page, its own print showing faintly through the paper.
fn back_color(src: vec2<f32>) -> vec4<f32> {
    if (u.geom.w > 0.5) {
        let landed = vec2<f32>(2.0 * u.extra.x - src.x, src.y);
        return on_paper(sample_b(landed));
    }
    let through = on_paper(sample_a(src));
    return mix(through, u.paper, 0.86);
}

/// What lies under the sheet: page B where A has lifted off it; around the sheet, A giving way
/// to B over the turn; for a spread, A's own other half until the leaf lands on it.
fn underlay(q: vec2<f32>, p: vec2<f32>) -> vec4<f32> {
    let t = u.state.x;
    if (u.geom.w > 0.5 && q.x < u.extra.x) {
        return mix(sample_a(p), sample_b(p), smoothstep(0.85, 1.0, t));
    }
    if (in_sheet(p)) { return sample_b(p); }
    return mix(sample_a(p), sample_b(p), t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let q = vec2<f32>(select(in.uv.x, 1.0 - in.uv.x, mirrored()), in.uv.y);
    let p = vec2<f32>(q.x, q.y * u.geom.y);

    let n = u.axis.zw;
    let r = u.geom.x;
    let lift = u.state.y;
    let d = dot(p - u.axis.xy, n);

    // Which layer of the sheet is on top here, and the point of the flat sheet it came from.
    // 0 nothing, 1 front lying flat, 2 front on the roll, 3 back on the roll, 4 back lying flat.
    var layer = 0;
    var src = p;
    var theta = 0.0;
    if (d < 0.0) {
        let flap = p + n * (PI * r - 2.0 * d);
        if (in_sheet(flap)) {
            layer = 4;
            src = flap;
            theta = PI;
        } else if (in_sheet(p)) {
            layer = 1;
        }
    } else if (d <= r) {
        let a1 = asin(clamp(d / r, 0.0, 1.0));
        let upper = p + n * (r * (PI - a1) - d);
        let lower = p + n * (r * a1 - d);
        if (in_sheet(upper)) {
            layer = 3;
            src = upper;
            theta = PI - a1;
        } else if (in_sheet(lower)) {
            layer = 2;
            src = lower;
            theta = a1;
        }
    }

    // Paper darkens into the fold, on both faces, easing off as the sheet lies flat again.
    let crease = 1.0 - 0.14 * lift * exp(min(d, 0.0) / (0.5 * r + 0.01));

    if (layer == 2) {
        let lit = 0.62 + 0.38 * cos(theta);
        let c = front_color(src);
        return vec4<f32>(c.rgb * lit * crease, c.a);
    }
    if (layer == 3 || layer == 4) {
        let lit = 0.62 + 0.38 * max(-cos(theta), 0.0);
        let g = (theta - 0.72 * PI) / 0.12;
        let gloss = 0.1 * lift * exp(-g * g);
        let c = back_color(src);
        return vec4<f32>(min(c.rgb * lit * crease + gloss * c.a, vec3<f32>(c.a)), c.a);
    }

    var base: vec4<f32>;
    if (layer == 1) {
        let c = front_color(p);
        base = vec4<f32>(c.rgb * crease, c.a);
    } else {
        base = underlay(q, p);
    }

    // The flap's shadow: soft, just past the edges of the part lying folded over.
    var shadow = 0.0;
    if (d < 0.0) {
        let edge = sheet_distance(p + n * (PI * r - 2.0 * d));
        shadow = 0.32 * lift * (1.0 - smoothstep(0.0, 0.035, edge));
    }
    // The roll's shadow on the page it uncovers, falling away from the curl and ending with it at
    // the sheet's top and bottom.
    if (layer == 0 && d > 0.0) {
        let foot = p - n * d + n * (0.5 * PI * r);
        let ends = 1.0 - smoothstep(0.0, 0.03, sheet_distance(foot));
        let thrown = 0.5 * lift * ends * exp(-max(d - 0.5 * r, 0.0) / (0.02 + 0.6 * r));
        shadow = 1.0 - (1.0 - shadow) * (1.0 - thrown);
    }
    if (shadow <= 0.0 && base.a <= 0.0) { discard; }
    return shade_over(base, shadow);
}"""
}
