package ca.mpreg.webgpuviewer.renderer

/**
 * [TileRenderer.downscaler]'s only implementation: every destination pixel is the average of the
 * source pixels its footprint covers, weighted by how much of each it covers, in linear light.
 *
 * Band-limiting the source's own detail is the right answer for shrinking, so unlike the
 * magnifying direction there is nothing here for a network to improve on. It is a [Downscaler]
 * anyway, so both directions are configured the same way.
 */
class DownscalerBox : Downscaler() {
    override val code: String get() = CODE

    internal companion object {
        /** Shared: identical per instance, and what keys [RenderPage]'s pipeline. */
        internal const val CODE = """
fn loop_over_tile(
    tex: texture_2d<f32>,
    start_i: vec2<i32>,
    end_i: vec2<i32>,
    src_start: vec2<f32>,
    src_end: vec2<f32>,
    local_offset: vec2<i32>
) -> vec4<f32> {
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let y_f = f32(y);

        var y_overlap = 1.0;
        if (y == start_i.y) {
            y_overlap = min(y_f + 1.0, src_end.y) - src_start.y;
        } else if (y == end_i.y - 1) {
            y_overlap = src_end.y - max(y_f, src_start.y);
        }
        y_overlap = max(0.0, y_overlap);

        let py = y + local_offset.y;

        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);

            var x_overlap = 1.0;
            if (x == start_i.x) {
                x_overlap = min(x_f + 1.0, src_end.x) - src_start.x;
            } else if (x == end_i.x - 1) {
                x_overlap = src_end.x - max(x_f, src_start.x);
            }
            x_overlap = max(0.0, x_overlap);

            let weight = x_overlap * y_overlap;
            let px = x + local_offset.x;

            let texel = to_linear_exact(textureLoad(tex, vec2<i32>(px, py), 0));
            color_sum += texel * weight;
            weight_sum += weight;
        }
    }
    return color_sum / max(weight_sum, 0.0001);
}

fn resolve_minify(src_start: vec2<f32>, scale: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions());
    let src_end = src_start + scale;

    let start_i = vec2<i32>(clamp(floor(src_start), vec2<f32>(0.0), src_size_f));
    let end_i   = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));

    let ts = i32(transform.tile_size);

    let tile_TL = start_i / ts;
    let tile_BR = (end_i - 1) / ts;

    let in_bounds = start_i.x >= 0 && start_i.y >= 0 && (end_i.x - 1) < ts * 2 && (end_i.y - 1) < ts * 2;
    let is_single_tile = all(tile_TL == tile_BR) && in_bounds;

    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;

        var avg_color = vec4<f32>(0.0);

        if (idx == 0) {
            avg_color = loop_over_tile(src_tex0, start_i, end_i, src_start, src_end, local_offset);
        } else if (idx == 1) {
            avg_color = loop_over_tile(src_tex1, start_i, end_i, src_start, src_end, local_offset);
        } else if (idx == 2) {
            avg_color = loop_over_tile(src_tex2, start_i, end_i, src_start, src_end, local_offset);
        } else {
            avg_color = loop_over_tile(src_tex3, start_i, end_i, src_start, src_end, local_offset);
        }

        return to_srgb_exact(avg_color);
    } else {
        for (var y: i32 = start_i.y; y < end_i.y; y++) {
            let y_f = f32(y);
            var y_overlap = 1.0;
            if (y == start_i.y) {
                y_overlap = min(y_f + 1.0, src_end.y) - src_start.y;
            } else if (y == end_i.y - 1) {
                y_overlap = src_end.y - max(y_f, src_start.y);
            }
            y_overlap = max(0.0, y_overlap);

            for (var x: i32 = start_i.x; x < end_i.x; x++) {
                let x_f = f32(x);
                var x_overlap = 1.0;
                if (x == start_i.x) {
                    x_overlap = min(x_f + 1.0, src_end.x) - src_start.x;
                } else if (x == end_i.x - 1) {
                    x_overlap = src_end.x - max(x_f, src_start.x);
                }
                x_overlap = max(0.0, x_overlap);

                let weight = x_overlap * y_overlap;
                let texel = to_linear_exact(totalLoad(vec2<i32>(x, y)));
                color_sum += texel * weight;
                weight_sum += weight;
            }
        }

        return to_srgb_exact(color_sum / max(weight_sum, 0.0001));
    }
}
"""
    }
}
