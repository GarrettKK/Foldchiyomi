/* HDR pixel passes over the extended-sRGB RGBA half-float layout the decoder
 * hands over: sRGB primaries and transfer, 1.0 at diffuse white, highlights
 * above, negatives out of gamut. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <jni.h>
#include <stdint.h>
#include <vector>

/* ------------------------------------------------------------ buffer guard */

/* Every entry point below takes dimensions from Java and indexes the buffers by
 * them, so a caller whose width, height and buffer disagree would write past
 * the end of the heap - a corruption that surfaces far from here, if it
 * surfaces at all. The capacity is free to ask for, so it is always asked for.
 */
static inline bool buffer_holds(JNIEnv *env, jobject buffer, size_t need) {
  if (!buffer)
    return false;
  const jlong capacity = env->GetDirectBufferCapacity(buffer);
  return capacity >= 0 && (size_t)capacity >= need;
}

/* Pixel count, or 0 when [width] x [height] x [bpp] cannot be addressed.
 *
 * Computed in 64 bits and bounded against SIZE_MAX because armeabi-v7a and x86
 * are 32-bit targets: there a large image's byte count does not fit a size_t,
 * and the wrap would make every bounds check below pass on a span far smaller
 * than the loops go on to walk. Callers can multiply the result by [bpp]
 * safely. */
static inline size_t pixel_count(jint width, jint height, size_t bpp) {
  if (width <= 0 || height <= 0 || bpp == 0)
    return 0;
  const uint64_t px = (uint64_t)(uint32_t)width * (uint64_t)(uint32_t)height;
  if (px == 0 || px > (uint64_t)SIZE_MAX / bpp)
    return 0;
  return (size_t)px;
}

/* ------------------------------------------------------------- half float */

static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      /* Subnormal: renormalise into a float's exponent range. */
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static inline uint16_t float_to_half(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent == 128)
    return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0u));

  /* Saturate rather than go infinite, so a later blend stays a number. */
  if (exponent > 15)
    return (uint16_t)(sign | 0x7bffu);

  if (exponent < -14) {
    if (exponent < -25)
      return (uint16_t)sign;
    uint32_t shift = (uint32_t)(-exponent - 14);
    uint32_t full = mantissa | 0x800000u;
    uint32_t sub = full >> (shift + 13);
    uint32_t round = (full >> (shift + 12)) & 1u;
    return (uint16_t)(sign | (sub + round));
  }

  uint32_t half = (uint32_t)((exponent + 15) << 10) | (mantissa >> 13);
  if (mantissa & 0x1000u)
    half++;

  return (uint16_t)(sign | half);
}

/* ------------------------------------------------------ transfer functions */

/* The sRGB transfer, mirrored through zero so it holds for the negatives an
 * out-of-gamut colour lands on. */
static inline float srgb_decode(float x) {
  float a = std::fabs(x);
  float y = a <= 0.04045f ? a / 12.92f : std::pow((a + 0.055f) / 1.055f, 2.4f);
  return x < 0.0f ? -y : y;
}

static inline float srgb_encode(float x) {
  float a = std::fabs(x);
  float y =
      a <= 0.0031308f ? a * 12.92f : 1.055f * std::pow(a, 1.0f / 2.4f) - 0.055f;
  return x < 0.0f ? -y : y;
}

static inline uint8_t srgb_encode_u8(float linear) {
  /* NaN fails every comparison, so min/max would pass it through to lround,
   * which is undefined for it. Checked rather than clamped. */
  if (!std::isfinite(linear))
    return 0;
  float s = srgb_encode(linear);
  return (uint8_t)std::lround(std::min(std::max(s, 0.0f), 1.0f) * 255.0f);
}

/* --------------------------------------------------------------- resize */

/* resize.cpp's box filter without the clamp to 1.0, which would throw away
 * every highlight. */
extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaNativeF16(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight) {
  const size_t src_px = pixel_count(srcWidth, srcHeight, 8);
  if (!src_px)
    return;

  const int dstWidth = srcWidth / 2;
  const int dstHeight = srcHeight / 2;
  const size_t dst_px = pixel_count(dstWidth, dstHeight, 8);
  if (!dst_px)
    return;

  if (!buffer_holds(env, src_buffer, src_px * 8) ||
      !buffer_holds(env, dst_buffer, dst_px * 8))
    return;

  const uint16_t *src =
      (const uint16_t *)env->GetDirectBufferAddress(src_buffer);
  uint16_t *dst = (uint16_t *)env->GetDirectBufferAddress(dst_buffer);
  if (!src || !dst)
    return;

  /* Exactly 2x2 per output pixel, so area weighting collapses to a plain
   * average. */
  for (int y = 0; y < dstHeight; ++y) {
    const uint16_t *row0 = src + (size_t)(y * 2) * srcWidth * 4;
    const uint16_t *row1 = src + (size_t)(y * 2 + 1) * srcWidth * 4;
    uint16_t *q = dst + (size_t)y * dstWidth * 4;

    for (int x = 0; x < dstWidth; ++x) {
      const uint16_t *p[4] = {
          row0 + (size_t)(x * 2) * 4,
          row0 + (size_t)(x * 2 + 1) * 4,
          row1 + (size_t)(x * 2) * 4,
          row1 + (size_t)(x * 2 + 1) * 4,
      };

      float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumA = 0.0f;

      for (int i = 0; i < 4; ++i) {
        float a = half_to_float(p[i][3]);
        sumR += srgb_decode(half_to_float(p[i][0])) * a;
        sumG += srgb_decode(half_to_float(p[i][1])) * a;
        sumB += srgb_decode(half_to_float(p[i][2])) * a;
        sumA += a;
      }

      float outR = 0.0f, outG = 0.0f, outB = 0.0f;
      float outA = sumA * 0.25f;

      if (sumA > 0.00001f) {
        float inv = 1.0f / sumA;
        outR = sumR * inv;
        outG = sumG * inv;
        outB = sumB * inv;
      }

      q[0] = float_to_half(srgb_encode(outR));
      q[1] = float_to_half(srgb_encode(outG));
      q[2] = float_to_half(srgb_encode(outB));
      q[3] = float_to_half(outA);
      q += 4;
    }
  }
}

/* Each output pixel's source span along one axis, as [first, first + count)
 * with a weight apiece, over one flat weight array. */
struct AreaSpans {
  std::vector<int> first, count, offset;
  std::vector<float> weight;
};

static AreaSpans area_spans(int src, int dst) {
  AreaSpans s;
  s.first.resize(dst);
  s.count.resize(dst);
  s.offset.resize(dst);
  const double scale = (double)src / dst;
  for (int i = 0; i < dst; ++i) {
    const double start = i * scale;
    const double end = start + scale;
    const int lo = std::min(src - 1, (int)start);
    const int hi = std::min(src - 1, std::max(lo, (int)std::ceil(end) - 1));
    s.first[i] = lo;
    s.count[i] = hi - lo + 1;
    s.offset[i] = (int)s.weight.size();
    for (int p = lo; p <= hi; ++p) {
      const double w = std::min((double)p + 1.0, end) - std::max((double)p, start);
      s.weight.push_back(w > 0.0 ? (float)w : 0.0f);
    }
  }
  return s;
}

/* As resizeLinearAreaNativeF16, to any size; a larger one repeats pixels. */
extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaToNativeF16(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight, jint dstWidth, jint dstHeight) {
  const size_t src_px = pixel_count(srcWidth, srcHeight, 8);
  const size_t dst_px = pixel_count(dstWidth, dstHeight, 8);
  if (!src_px || !dst_px)
    return;

  if (!buffer_holds(env, src_buffer, src_px * 8) ||
      !buffer_holds(env, dst_buffer, dst_px * 8))
    return;

  const uint16_t *src =
      (const uint16_t *)env->GetDirectBufferAddress(src_buffer);
  uint16_t *dst = (uint16_t *)env->GetDirectBufferAddress(dst_buffer);
  if (!src || !dst)
    return;

  const AreaSpans xs = area_spans(srcWidth, dstWidth);
  const AreaSpans ys = area_spans(srcHeight, dstHeight);

  for (int y = 0; y < dstHeight; ++y) {
    uint16_t *q = dst + (size_t)y * dstWidth * 4;
    for (int x = 0; x < dstWidth; ++x) {
      float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumA = 0.0f, sumW = 0.0f;

      for (int j = 0; j < ys.count[y]; ++j) {
        const float wy = ys.weight[ys.offset[y] + j];
        const uint16_t *row =
            src + ((size_t)(ys.first[y] + j) * srcWidth + xs.first[x]) * 4;
        for (int i = 0; i < xs.count[x]; ++i) {
          const float w = wy * xs.weight[xs.offset[x] + i];
          if (w <= 0.0f)
            continue;
          const uint16_t *p = row + (size_t)i * 4;
          const float a = half_to_float(p[3]) * w;
          sumR += srgb_decode(half_to_float(p[0])) * a;
          sumG += srgb_decode(half_to_float(p[1])) * a;
          sumB += srgb_decode(half_to_float(p[2])) * a;
          sumA += a;
          sumW += w;
        }
      }

      float outR = 0.0f, outG = 0.0f, outB = 0.0f, outA = 0.0f;
      if (sumW > 0.0f)
        outA = sumA / sumW;
      if (sumA > 0.00001f) {
        const float inv = 1.0f / sumA;
        outR = sumR * inv;
        outG = sumG * inv;
        outB = sumB * inv;
      }

      q[0] = float_to_half(srgb_encode(outR));
      q[1] = float_to_half(srgb_encode(outG));
      q[2] = float_to_half(srgb_encode(outB));
      q[3] = float_to_half(outA);
      q += 4;
    }
  }
}

/* -------------------------------------------------------------- tone map */

/* Below [knee] nothing moves; above it a rational roll-off landing [peak]
 * exactly on 1.0.
 *
 * The knee has to sit below 1.0: the output ceiling is 1.0, so the only room
 * for anything above SDR white is room taken from the SDR range. */
static inline float tonemap_knee(float x, float knee, float peak) {
  if (x <= knee)
    return x;

  const float room = 1.0f - knee;  /* output left above the knee */
  const float range = peak - knee; /* input to fit into it */
  if (range <= 0.0f)
    return std::min(x, 1.0f);

  const float t = x - knee;
  /* t/(t+s) only approaches 1, so it's divided by its own value at t == range
   * to land there exactly. s == range/2 gives a moderate curve. */
  const float s = range * 0.5f;
  const float shape = (t / (t + s)) / (range / (range + s));

  return knee + room * std::min(shape, 1.0f);
}

/* Proportional to the headroom there actually is: the SDR range given up is the
 * price of fitting the highlights, and a dim HDR frame should not pay a bright
 * one's price. */
static inline float shoulder_knee(float peak) {
  if (!(peak > 1.05f))
    return 1.0f; /* nothing above white: a plain re-encode */
  const float stops = std::log2(peak);
  return 1.0f - 0.2f * std::min(1.0f, stops / 2.0f);
}

/* The curve takes max(r, g, b) and scales the channels by the result, so hue is
 * untouched and no channel escapes still above 1.0; per-channel would
 * desaturate every highlight.
 *
 * The peak is measured, not taken from the file: PQ declares 10000 nits and HLG
 * 1000, and fitting to that put every real highlight between 203 and 1000 nits
 * into six levels under white. */
extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_toneMapToSdrNative(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint width, jint height) {
  const size_t count = pixel_count(width, height, 8);
  if (!count)
    return;
  if (!buffer_holds(env, src_buffer, count * 8) ||
      !buffer_holds(env, dst_buffer, count * 4))
    return;

  const uint16_t *src =
      (const uint16_t *)env->GetDirectBufferAddress(src_buffer);
  /* Byte at a time, not a packed uint32: a word write would need the
   * endian-dependent shuffle that makes resize.cpp's channel names read
   * backwards. */
  uint8_t *dst = (uint8_t *)env->GetDirectBufferAddress(dst_buffer);
  if (!src || !dst)
    return;

  /* Non-finite skipped - see scaleHdrPeakNative. */
  float peak = 1.0f;
  for (size_t i = 0; i < count; ++i) {
    const uint16_t *p = src + i * 4;
    for (int c = 0; c < 3; ++c) {
      const float v = srgb_decode(half_to_float(p[c]));
      if (std::isfinite(v) && v > peak)
        peak = v;
    }
  }

  const float knee = shoulder_knee(peak);

  for (size_t i = 0; i < count; ++i) {
    const uint16_t *p = src + i * 4;

    float r = srgb_decode(half_to_float(p[0]));
    float g = srgb_decode(half_to_float(p[1]));
    float b = srgb_decode(half_to_float(p[2]));
    float a = half_to_float(p[3]);

    float m = std::max(std::max(r, g), b);
    if (m > knee) {
      float scale = tonemap_knee(m, knee, std::max(peak, m)) / m;
      r *= scale;
      g *= scale;
      b *= scale;
    }

    uint8_t *q = dst + i * 4;
    q[0] = srgb_encode_u8(r);
    q[1] = srgb_encode_u8(g);
    q[2] = srgb_encode_u8(b);
    q[3] =
        std::isfinite(a)
            ? (uint8_t)std::lround(std::min(std::max(a, 0.0f), 1.0f) * 255.0f)
            : (uint8_t)255;
  }
}

/* ------------------------------------------------------------- gain map */

/* Apply an unapplied gain map to its SDR base.
 *
 * libultrahdr's applyGain, not libvips's uhdr2scRGB: the map value is used
 * *raw*, with no transfer function undone. libvips runs three-channel maps
 * through the sRGB EOTF first, which casts colour across the whole SDR part of
 * the picture - the reason this is done here.
 *
 * Gain comes from a per-channel LUT because the map has only 256 values and the
 * pow/log2/exp2 per pixel per channel is most of the cost. [weight] scales it
 * in log space for a display with less headroom than the content carries. */
extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_applyGainmapNative(
    JNIEnv *env, jobject thiz, jobject base_buffer, jobject gain_buffer,
    jobject dst_buffer, jint width, jint height, jint gainWidth,
    jint gainHeight, jint gainChannels, jfloatArray jgamma,
    jfloatArray jminBoost, jfloatArray jmaxBoost, jfloatArray joffsetSdr,
    jfloatArray joffsetHdr, jfloat weight) {
  if (gainChannels != 1 && gainChannels != 3)
    return;

  const size_t count = pixel_count(width, height, 8);
  const size_t gain_px =
      pixel_count(gainWidth, gainHeight, (size_t)gainChannels);
  if (!count || !gain_px)
    return;

  if (!buffer_holds(env, base_buffer, count * 4) ||
      !buffer_holds(env, gain_buffer, gain_px * (size_t)gainChannels) ||
      !buffer_holds(env, dst_buffer, count * 8))
    return;

  const uint8_t *base =
      (const uint8_t *)env->GetDirectBufferAddress(base_buffer);
  const uint8_t *gain =
      (const uint8_t *)env->GetDirectBufferAddress(gain_buffer);
  uint16_t *dst = (uint16_t *)env->GetDirectBufferAddress(dst_buffer);
  if (!base || !gain || !dst)
    return;

  /* Seeded with the identity, so a metadata array that cannot be read leaves
   * the base image untouched rather than multiplied by whatever the stack held.
   */
  float gamma[3] = {1.0f, 1.0f, 1.0f};
  float minBoost[3] = {1.0f, 1.0f, 1.0f};
  float maxBoost[3] = {1.0f, 1.0f, 1.0f};
  float offsetSdr[3] = {0.0f, 0.0f, 0.0f};
  float offsetHdr[3] = {0.0f, 0.0f, 0.0f};

  /* GainmapInput is public, so these arrays are whatever the app passed. A
   * short or null one makes GetFloatArrayRegion throw - and a JNI exception
   * does not unwind C++, so without this the loop below would run on
   * uninitialised metadata and only fail later, somewhere else. */
  {
    jfloatArray arrays[5] = {jgamma, jminBoost, jmaxBoost, joffsetSdr,
                             joffsetHdr};
    float *out[5] = {gamma, minBoost, maxBoost, offsetSdr, offsetHdr};

    for (int i = 0; i < 5; ++i) {
      if (!arrays[i] || env->GetArrayLength(arrays[i]) < 3)
        return;
      env->GetFloatArrayRegion(arrays[i], 0, 3, out[i]);
      if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return;
      }
    }
  }

  /* Log-space weight, so anything outside [0, 1] is either no gain or more than
   * the file carries. NaN fails the comparisons and lands on 1.0. */
  const float w = (std::isfinite(weight) && weight >= 0.0f && weight <= 1.0f)
                      ? weight
                      : 1.0f;

  static const int LUT = 256;
  /* Metadata is file-supplied, so the boost is bounded before it reaches exp2.
   * 16 stops is already four times any real gain map and keeps every LUT entry
   * a finite float. */
  static const float MAX_STOPS = 16.0f;
  float gainLut[3][256];
  for (int c = 0; c < 3; ++c) {
    /* A boost of zero or less is meaningless and log2 would return -inf; treat
     * it as no change, which makes malformed metadata fall back to the base
     * image. */
    const float lo_boost =
        std::isfinite(minBoost[c]) && minBoost[c] > 0.0f ? minBoost[c] : 1.0f;
    const float hi_boost =
        std::isfinite(maxBoost[c]) && maxBoost[c] > 0.0f ? maxBoost[c] : 1.0f;
    const float lo =
        std::min(std::max(std::log2(lo_boost), -MAX_STOPS), MAX_STOPS);
    const float hi =
        std::min(std::max(std::log2(hi_boost), -MAX_STOPS), MAX_STOPS);

    const float gam = gamma[c];
    const bool use_gamma = std::isfinite(gam) && gam > 0.0f && gam != 1.0f;

    for (int v = 0; v < LUT; ++v) {
      float g = (float)v / (LUT - 1);
      if (use_gamma)
        g = std::pow(g, 1.0f / gam);
      gainLut[c][v] = std::exp2((lo * (1.0f - g) + hi * g) * w);
    }

    /* An offset that is not a number would carry into every pixel. */
    if (!std::isfinite(offsetSdr[c]))
      offsetSdr[c] = 0.0f;
    if (!std::isfinite(offsetHdr[c]))
      offsetHdr[c] = 0.0f;
  }

  /* Map pixel centres onto base pixel centres, so the map isn't shifted half a
   * texel. */
  const float sx = (float)gainWidth / (float)width;
  const float sy = (float)gainHeight / (float)height;

  for (int y = 0; y < height; ++y) {
    float gy = ((float)y + 0.5f) * sy - 0.5f;
    int y0 = (int)std::floor(gy);
    float fy = gy - (float)y0;
    int y0c = std::min(std::max(y0, 0), gainHeight - 1);
    int y1c = std::min(std::max(y0 + 1, 0), gainHeight - 1);

    const uint8_t *grow0 = gain + (size_t)y0c * gainWidth * gainChannels;
    const uint8_t *grow1 = gain + (size_t)y1c * gainWidth * gainChannels;
    const uint8_t *p = base + (size_t)y * width * 4;
    uint16_t *q = dst + (size_t)y * width * 4;

    for (int x = 0; x < width; ++x, p += 4, q += 4) {
      float gx = ((float)x + 0.5f) * sx - 0.5f;
      int x0 = (int)std::floor(gx);
      float fx = gx - (float)x0;
      int x0c = std::min(std::max(x0, 0), gainWidth - 1) * gainChannels;
      int x1c = std::min(std::max(x0 + 1, 0), gainWidth - 1) * gainChannels;

      for (int c = 0; c < 3; ++c) {
        const int mc = gainChannels == 1 ? 0 : c;

        /* Map sampled first, gain applied after - libultrahdr's order. */
        float v = (grow0[x0c + mc] * (1.0f - fx) + grow0[x1c + mc] * fx) *
                      (1.0f - fy) +
                  (grow1[x0c + mc] * (1.0f - fx) + grow1[x1c + mc] * fx) * fy;

        int vi = (int)v;
        vi = std::min(std::max(vi, 0), LUT - 1);
        const int vi1 = std::min(vi + 1, LUT - 1);
        const float vf = v - (float)vi;
        const float g = gainLut[c][vi] * (1.0f - vf) + gainLut[c][vi1] * vf;

        const float linear = srgb_decode(p[c] / 255.0f);
        const float out = (linear + offsetSdr[c]) * g - offsetHdr[c];
        q[c] = float_to_half(srgb_encode(out));
      }

      /* Alpha is untouched by the gain - it isn't light. */
      q[3] = float_to_half(p[3] / 255.0f);
    }
  }
}

/* --------------------------------------------------------- peak rescale */

/* Scale in place so the image's own brightest pixel lands on [targetPeak];
 * returns the peak it ends up with. Measured, not taken from the format, which
 * crushed a 400-nit frame as though it reached PQ's 10000.
 *
 * `v^weight` with `weight = log2(target)/log2(peak)`, the same shape the gain
 * map path gets by weighting the map as it is applied.
 *
 * Values at or under SDR white are left alone: the power would *brighten* them,
 * since log2 below 1 is negative, and washing out the SDR range is worse than a
 * compressed highlight. An image already under the target is untouched rather
 * than brightened to fill it. */
extern "C" JNIEXPORT jfloat JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_scaleHdrPeakNative(JNIEnv *env,
                                                        jobject thiz,
                                                        jobject buffer,
                                                        jint width, jint height,
                                                        jfloat targetPeak) {
  const size_t count = pixel_count(width, height, 8);
  if (!count)
    return 1.0f;
  if (!buffer_holds(env, buffer, count * 8))
    return 1.0f;

  uint16_t *px = (uint16_t *)env->GetDirectBufferAddress(buffer);
  if (!px)
    return 1.0f;

  /* Non-finite samples are skipped rather than measured: one infinity in a
   * corrupt float source would become the peak, and every real highlight would
   * compress against it. */
  float peak = 1.0f;
  for (size_t i = 0; i < count; ++i) {
    const uint16_t *q = px + i * 4;
    for (int c = 0; c < 3; ++c) {
      const float v = srgb_decode(half_to_float(q[c]));
      if (std::isfinite(v) && v > peak)
        peak = v;
    }
  }

  if (!std::isfinite(targetPeak) || !(targetPeak > 1.0f) || peak <= targetPeak)
    return peak;

  const float weight = std::log2(targetPeak) / std::log2(peak);
  if (!(weight > 0.0f) || weight >= 1.0f)
    return peak;

  /* Scaled from max(r, g, b) so hue is untouched and no channel escapes the
   * peak the others were fitted to. */
  for (size_t i = 0; i < count; ++i) {
    uint16_t *q = px + i * 4;

    const float r = srgb_decode(half_to_float(q[0]));
    const float g = srgb_decode(half_to_float(q[1]));
    const float b = srgb_decode(half_to_float(q[2]));

    const float m = std::max(std::max(r, g), b);
    /* NaN fails this too, which is what leaves a non-finite pixel exactly as it
     * was. */
    if (!(m > 1.0f))
      continue;

    const float scale = std::pow(m, weight) / m;

    q[0] = float_to_half(srgb_encode(r * scale));
    q[1] = float_to_half(srgb_encode(g * scale));
    q[2] = float_to_half(srgb_encode(b * scale));
    /* Alpha untouched - it isn't light. */
  }

  return targetPeak;
}
