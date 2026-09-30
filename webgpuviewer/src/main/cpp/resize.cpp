#include <algorithm>
#include <cmath>
#include <jni.h>
#include <mutex>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

float srgbToLinearLUT[256];
std::once_flag lutsOnceFlag;

/* A bare bool flag races: two mipmap levels can resize concurrently on first
 * use. */
void initLUTs() {
  std::call_once(lutsOnceFlag, [] {
    for (int i = 0; i < 256; i++) {
      float val01 = i / 255.0f;
      srgbToLinearLUT[i] = (val01 <= 0.04045f)
                               ? (val01 / 12.92f)
                               : std::pow((val01 + 0.055f) / 1.055f, 2.4f);
    }
  });
}

inline uint8_t linearToSRGBExact(float linearVal) {
  if (linearVal <= 0.0f)
    return 0;
  if (linearVal >= 1.0f)
    return 255;

  float srgb01 = (linearVal <= 0.0031308f)
                     ? (linearVal * 12.92f)
                     : (1.055f * std::pow(linearVal, 1.0f / 2.4f) - 0.055f);

  return static_cast<uint8_t>(std::roundf(srgb01 * 255.0f));
}

inline uint8_t exactLinearToAlpha(float linearAlpha) {
  if (linearAlpha <= 0.0f)
    return 0;
  if (linearAlpha >= 1.0f)
    return 255;
  return static_cast<uint8_t>(std::roundf(linearAlpha * 255.0f));
}

/* Box filter in linear light to any size; a larger one repeats pixels. A
 * mismatched size/buffer indexes past its end rather than throwing, so both are
 * checked. */
static void resizeArea(JNIEnv *env, jobject src_buffer, jobject dst_buffer,
                       jint srcWidth, jint srcHeight, jint dstWidth,
                       jint dstHeight) {
  initLUTs();

  if (!env || !src_buffer || !dst_buffer)
    return;
  if (srcWidth <= 0 || srcHeight <= 0)
    return;

  uint32_t *src = (uint32_t *)env->GetDirectBufferAddress(src_buffer);
  uint32_t *dst = (uint32_t *)env->GetDirectBufferAddress(dst_buffer);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return;
  }
  if (!src || !dst)
    return;

  /* 64-bit: a 32-bit product wraps for a large enough image. */
  const jlong srcCapacity = env->GetDirectBufferCapacity(src_buffer);
  const jlong srcNeeded =
      static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight) * 4LL;
  if (srcCapacity < 0 || srcCapacity < srcNeeded)
    return;

  if (dstWidth <= 0 || dstHeight <= 0)
    return;

  const jlong dstCapacity = env->GetDirectBufferCapacity(dst_buffer);
  const jlong dstNeeded =
      static_cast<jlong>(dstWidth) * static_cast<jlong>(dstHeight) * 4LL;
  if (dstCapacity < 0 || dstCapacity < dstNeeded)
    return;

  double scaleX = (double)srcWidth / dstWidth;
  double scaleY = (double)srcHeight / dstHeight;

  for (int y = 0; y < dstHeight; ++y) {
    double srcYStart = y * scaleY;
    double srcYEnd = srcYStart + scaleY;
    int yMin = std::max(0, (int)srcYStart);
    int yMax = std::min(srcHeight - 1, (int)srcYEnd);

    for (int x = 0; x < dstWidth; ++x) {
      double srcXStart = x * scaleX;
      double srcXEnd = srcXStart + scaleX;
      int xMin = std::max(0, (int)srcXStart);
      int xMax = std::min(srcWidth - 1, (int)srcXEnd);

      uint32_t finalA = 0, finalR = 0, finalG = 0, finalB = 0;

#ifdef __ARM_NEON
      float32x4_t v_sum_a = vdupq_n_f32(0.0f);
      float32x4_t v_sum_r = vdupq_n_f32(0.0f);
      float32x4_t v_sum_g = vdupq_n_f32(0.0f);
      float32x4_t v_sum_b = vdupq_n_f32(0.0f);
      float totalWeight = 0.0f;

      float sumA_scalar = 0.0f;
      float sumR_scalar = 0.0f;
      float sumG_scalar = 0.0f;
      float sumB_scalar = 0.0f;

      for (int sy = yMin; sy <= yMax; ++sy) {
        double yWeight = std::min((double)sy + 1.0, srcYEnd) -
                         std::max((double)sy, srcYStart);
        if (yWeight <= 0)
          continue;

        int sx = xMin;

        for (; sx <= xMax - 3; sx += 4) {
          float weights[4];
          float linearA[4], linearR[4], linearG[4], linearB[4];

          for (int i = 0; i < 4; ++i) {
            double xWeight = std::min((double)(sx + i) + 1.0, srcXEnd) -
                             std::max((double)(sx + i), srcXStart);
            weights[i] = (xWeight > 0) ? (float)(xWeight * yWeight) : 0.0f;
            totalWeight += weights[i];

            uint32_t pixel = src[(size_t)sy * srcWidth + (sx + i)];
            float aVal = ((pixel >> 24) & 0xFF) / 255.0f;

            linearA[i] = aVal;
            linearR[i] = srgbToLinearLUT[(pixel >> 16) & 0xFF] * aVal;
            linearG[i] = srgbToLinearLUT[(pixel >> 8) & 0xFF] * aVal;
            linearB[i] = srgbToLinearLUT[pixel & 0xFF] * aVal;
          }

          float32x4_t v_w = vld1q_f32(weights);
          float32x4_t v_a = vld1q_f32(linearA);
          float32x4_t v_r = vld1q_f32(linearR);
          float32x4_t v_g = vld1q_f32(linearG);
          float32x4_t v_b = vld1q_f32(linearB);

          v_sum_a = vmlaq_f32(v_sum_a, v_a, v_w);
          v_sum_r = vmlaq_f32(v_sum_r, v_r, v_w);
          v_sum_g = vmlaq_f32(v_sum_g, v_g, v_w);
          v_sum_b = vmlaq_f32(v_sum_b, v_b, v_w);
        }

        for (; sx <= xMax; ++sx) {
          double xWeight = std::min((double)sx + 1.0, srcXEnd) -
                           std::max((double)sx, srcXStart);
          if (xWeight <= 0)
            continue;

          float pWeight = (float)(xWeight * yWeight);
          totalWeight += pWeight;

          uint32_t pixel = src[(size_t)sy * srcWidth + sx];
          float aVal = ((pixel >> 24) & 0xFF) / 255.0f;
          float rVal = srgbToLinearLUT[(pixel >> 16) & 0xFF] * aVal;
          float gVal = srgbToLinearLUT[(pixel >> 8) & 0xFF] * aVal;
          float bVal = srgbToLinearLUT[pixel & 0xFF] * aVal;

          sumA_scalar += aVal * pWeight;
          sumR_scalar += rVal * pWeight;
          sumG_scalar += gVal * pWeight;
          sumB_scalar += bVal * pWeight;
        }
      }

      if (totalWeight > 0.0f) {
        float sumA = vgetq_lane_f32(v_sum_a, 0) + vgetq_lane_f32(v_sum_a, 1) +
                     vgetq_lane_f32(v_sum_a, 2) + vgetq_lane_f32(v_sum_a, 3) +
                     sumA_scalar;
        float sumR = vgetq_lane_f32(v_sum_r, 0) + vgetq_lane_f32(v_sum_r, 1) +
                     vgetq_lane_f32(v_sum_r, 2) + vgetq_lane_f32(v_sum_r, 3) +
                     sumR_scalar;
        float sumG = vgetq_lane_f32(v_sum_g, 0) + vgetq_lane_f32(v_sum_g, 1) +
                     vgetq_lane_f32(v_sum_g, 2) + vgetq_lane_f32(v_sum_g, 3) +
                     sumG_scalar;
        float sumB = vgetq_lane_f32(v_sum_b, 0) + vgetq_lane_f32(v_sum_b, 1) +
                     vgetq_lane_f32(v_sum_b, 2) + vgetq_lane_f32(v_sum_b, 3) +
                     sumB_scalar;

        float invWeight = 1.0f / totalWeight;
        float finalLinearA = sumA * invWeight;
        float finalLinearR = sumR * invWeight;
        float finalLinearG = sumG * invWeight;
        float finalLinearB = sumB * invWeight;

        if (finalLinearA > 0.00001f) {
          float invAlpha = 1.0f / finalLinearA;
          finalLinearR = std::min(1.0f, finalLinearR * invAlpha);
          finalLinearG = std::min(1.0f, finalLinearG * invAlpha);
          finalLinearB = std::min(1.0f, finalLinearB * invAlpha);
        } else {
          finalLinearR = 0.0f;
          finalLinearG = 0.0f;
          finalLinearB = 0.0f;
        }

        finalA = exactLinearToAlpha(finalLinearA);
        finalR = linearToSRGBExact(finalLinearR);
        finalG = linearToSRGBExact(finalLinearG);
        finalB = linearToSRGBExact(finalLinearB);
      }
#else
      float sumA = 0.0f, sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
      float totalWeight = 0.0f;

      int numXPixels = xMax - xMin + 1;

      /* A 256 cap silently truncated (and skewed) a downscale wider than that
       * per pixel. */
      std::vector<float> xWeights(numXPixels);

      for (int i = 0; i < numXPixels; ++i) {
        int sx = xMin + i;
        double xWeight = std::min((double)sx + 1.0, srcXEnd) -
                         std::max((double)sx, srcXStart);
        xWeights[i] = (xWeight > 0) ? (float)xWeight : 0.0f;
      }

      for (int sy = yMin; sy <= yMax; ++sy) {
        double yWeight = std::min((double)sy + 1.0, srcYEnd) -
                         std::max((double)sy, srcYStart);
        if (yWeight <= 0)
          continue;

        float yWeightF = (float)yWeight;
        size_t srcRowOffset = (size_t)sy * srcWidth;

        for (int sx = xMin; sx <= xMax; ++sx) {
          int cacheIdx = sx - xMin;

          float pWeight = xWeights[cacheIdx] * yWeightF;
          if (pWeight <= 0.0f)
            continue;

          uint32_t pixel = src[srcRowOffset + sx];

          uint8_t a = (pixel >> 24) & 0xFF;
          uint8_t r = (pixel >> 16) & 0xFF;
          uint8_t g = (pixel >> 8) & 0xFF;
          uint8_t b = pixel & 0xFF;

          float aVal = a / 255.0f;

          sumA += aVal * pWeight;
          sumR += srgbToLinearLUT[r] * aVal * pWeight;
          sumG += srgbToLinearLUT[g] * aVal * pWeight;
          sumB += srgbToLinearLUT[b] * aVal * pWeight;

          totalWeight += pWeight;
        }
      }

      if (totalWeight > 0.0f) {
        float invWeight = 1.0f / totalWeight;
        float finalLinearA = sumA * invWeight;
        float finalLinearR = sumR * invWeight;
        float finalLinearG = sumG * invWeight;
        float finalLinearB = sumB * invWeight;

        if (finalLinearA > 0.00001f) {
          float invAlpha = 1.0f / finalLinearA;
          finalLinearR = std::min(1.0f, finalLinearR * invAlpha);
          finalLinearG = std::min(1.0f, finalLinearG * invAlpha);
          finalLinearB = std::min(1.0f, finalLinearB * invAlpha);
        } else {
          finalLinearR = 0.0f;
          finalLinearG = 0.0f;
          finalLinearB = 0.0f;
        }

        finalA = exactLinearToAlpha(finalLinearA);
        finalR = linearToSRGBExact(finalLinearR);
        finalG = linearToSRGBExact(finalLinearG);
        finalB = linearToSRGBExact(finalLinearB);
      }
#endif

      dst[(size_t)y * dstWidth + x] =
          (finalA << 24) | (finalR << 16) | (finalG << 8) | finalB;
    }
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaNative(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight) {
  resizeArea(env, src_buffer, dst_buffer, srcWidth, srcHeight, srcWidth / 2,
             srcHeight / 2);
}

extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaToNative(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight, jint dstWidth, jint dstHeight) {
  resizeArea(env, src_buffer, dst_buffer, srcWidth, srcHeight, dstWidth,
             dstHeight);
}
