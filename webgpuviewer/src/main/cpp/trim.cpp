// CPU implementations of the trim and background-detection
//
// Pixel layout is RGBA8, row-major, tightly packed: byte 0 is red, byte 3 is
// alpha. That matches both TextureFormat.RGBA8Unorm and Android's ARGB_8888
// bitmaps, which are RGBA in memory order despite the name. Bytes are read
// individually rather than through a uint32_t so the channel identity does not
// depend on endianness.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <jni.h>
#include <thread>
#include <vector>

namespace {

constexpr int kChannels = 4;

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

/**
 * Foreground test for one background colour, matching is_foreground in the WGSL
 * trim shaders.
 *
 * The shader composites the pixel over the background before comparing:
 *   diff = |rgb * a + bg * (1 - a) - bg| = a * |rgb - bg|
 * so the blend collapses to a single multiply by alpha. Everything is kept in
 * 0..255 units to avoid normalising every pixel: the shader's
 *   a01 * |c01 - bg01| > threshold
 * scales to
 *   a * |c - bg255| > threshold * 255 * 255
 */
struct ColorTest {
  float bg255[3];
  float thresholdScaled;
  // Fully opaque pixels dominate real pages, and for those the alpha multiply
  // drops out, leaving a comparison that depends only on the byte value.
  uint8_t opaqueForeground[3][256];
};

ColorTest makeColorTest(float r, float g, float b, float threshold) {
  ColorTest test{};
  test.bg255[0] = r * 255.0f;
  test.bg255[1] = g * 255.0f;
  test.bg255[2] = b * 255.0f;
  test.thresholdScaled = threshold * 255.0f * 255.0f;

  const float opaqueThreshold = threshold * 255.0f;
  for (int ch = 0; ch < 3; ++ch) {
    for (int c = 0; c < 256; ++c) {
      test.opaqueForeground[ch][c] =
          std::fabs(static_cast<float>(c) - test.bg255[ch]) > opaqueThreshold;
    }
  }
  return test;
}

inline bool isForeground(const ColorTest &test, const uint8_t *px) {
  const uint8_t a = px[3];
  if (a == 255) {
    return (test.opaqueForeground[0][px[0]] | test.opaqueForeground[1][px[1]] |
            test.opaqueForeground[2][px[2]]) != 0;
  }
  const float af = static_cast<float>(a);
  return af * std::fabs(static_cast<float>(px[0]) - test.bg255[0]) >
             test.thresholdScaled ||
         af * std::fabs(static_cast<float>(px[1]) - test.bg255[1]) >
             test.thresholdScaled ||
         af * std::fabs(static_cast<float>(px[2]) - test.bg255[2]) >
             test.thresholdScaled;
}

/**
 * Bounding box of the foreground pixels.
 *
 * Seeded the way the shader seeds its result buffer - min at the image extent,
 * max at zero - so an image with no foreground at all produces the same
 * (width, height, 0, 0) the GPU path produces, and the callers that already
 * handle that degenerate result keep working.
 */
struct Bounds {
  int32_t minX, minY, maxX, maxY;

  void reset(int width, int height) {
    minX = width;
    minY = height;
    maxX = 0;
    maxY = 0;
  }

  void merge(const Bounds &other) {
    minX = std::min(minX, other.minX);
    minY = std::min(minY, other.minY);
    maxX = std::max(maxX, other.maxX);
    maxY = std::max(maxY, other.maxY);
  }
};

/**
 * Accumulate bounds for every colour over rows [y0, y1).
 *
 * Colours are the inner loop so a row is walked while it is still in cache.
 * Within a row, the scan runs inward from both ends and stops at the first hit:
 * a page with margins costs two short scans, and only a row that is entirely
 * background has to be read end to end (there is no way to prove it empty
 * otherwise).
 */
void scanBand(const uint8_t *pixels, int width, int y0, int y1,
              const ColorTest *tests, int colorCount, Bounds *bounds) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t *row = pixels + static_cast<size_t>(y) * width * kChannels;

    for (int ci = 0; ci < colorCount; ++ci) {
      const ColorTest &test = tests[ci];
      Bounds &bb = bounds[ci];

      int first = -1;
      for (int x = 0; x < width; ++x) {
        if (isForeground(test, row + static_cast<size_t>(x) * kChannels)) {
          first = x;
          break;
        }
      }
      if (first < 0) {
        continue; // Row is entirely background for this colour.
      }

      int last = first;
      for (int x = width - 1; x > first; --x) {
        if (isForeground(test, row + static_cast<size_t>(x) * kChannels)) {
          last = x;
          break;
        }
      }

      if (first < bb.minX)
        bb.minX = first;
      if (last > bb.maxX)
        bb.maxX = last;
      if (y < bb.minY)
        bb.minY = y;
      if (y > bb.maxY)
        bb.maxY = y;
    }
  }
}

int chooseThreadCount(int width, int height) {
  // Below roughly a quarter-megapixel the scan is short enough that spawning
  // threads costs more than it saves.
  if (static_cast<int64_t>(width) * height < 256 * 1024) {
    return 1;
  }
  unsigned hardware = std::thread::hardware_concurrency();
  int count = static_cast<int>(std::min(hardware ? hardware : 1u, 8u));
  // Keep bands big enough to be worth a thread.
  count = std::min(count, height / 64);
  return std::max(1, count);
}

// ---------------------------------------------------------------------------
// Background detection
// ---------------------------------------------------------------------------

inline double linearToSrgb(double c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

struct SrgbToLinearLut {
  double value[256];
  SrgbToLinearLut() {
    for (int i = 0; i < 256; ++i) {
      const double c = i / 255.0;
      value[i] = c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }
  }
};

inline double srgbByteToLinear(uint8_t b) {
  static const SrgbToLinearLut lut;
  return lut.value[b];
}

constexpr double kMinCoverage = 0.9;

constexpr double kColorMatchTolerance = 0.05;

struct EdgeLine {
  const uint8_t *base;
  size_t strideBytes;
  int64_t count;
};

inline const uint8_t *edgePixel(const EdgeLine &line, int64_t i) {
  return line.base + static_cast<size_t>(i) * line.strideBytes;
}

struct EdgeResult {
  bool solid;
  bool isWhite;
  double linearMean[3]; // meaningful only if solid and !isWhite
  double coverage;      // meaningful only if solid - higher is more confident
};

EdgeResult classifyEdge(const EdgeLine &line) {
  if (line.count == 0) {
    return {false, false, {0.0, 0.0, 0.0}, 0.0};
  }

  double sum[3] = {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < line.count; ++i) {
    const uint8_t *px = edgePixel(line, i);
    for (int ch = 0; ch < 3; ++ch) {
      sum[ch] += srgbByteToLinear(px[ch]);
    }
  }
  const double inv = 1.0 / static_cast<double>(line.count);
  const double mean[3] = {sum[0] * inv, sum[1] * inv, sum[2] * inv};

  const double tol = kColorMatchTolerance;
  int64_t closeToMean = 0;
  int64_t closeToWhite = 0;
  double inlierSum[3] = {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < line.count; ++i) {
    const uint8_t *px = edgePixel(line, i);
    double lin[3];
    bool nearMean = true;
    bool nearWhite = true;
    for (int ch = 0; ch < 3; ++ch) {
      lin[ch] = srgbByteToLinear(px[ch]);
      if (std::fabs(lin[ch] - mean[ch]) > tol) {
        nearMean = false;
      }
      if (std::fabs(lin[ch] - 1.0) > tol) {
        nearWhite = false;
      }
    }
    if (nearMean) {
      ++closeToMean;
      for (int ch = 0; ch < 3; ++ch) {
        inlierSum[ch] += lin[ch];
      }
    }
    if (nearWhite) {
      ++closeToWhite;
    }
  }

  const double meanCoverage = static_cast<double>(closeToMean) * inv;
  const double whiteCoverage = static_cast<double>(closeToWhite) * inv;

  if (whiteCoverage >= kMinCoverage) {
    return {true, true, {0.0, 0.0, 0.0}, whiteCoverage};
  }
  if (meanCoverage >= kMinCoverage) {
    const double inlierInv = 1.0 / static_cast<double>(closeToMean);
    const double refinedMean[3] = {inlierSum[0] * inlierInv,
                                   inlierSum[1] * inlierInv,
                                   inlierSum[2] * inlierInv};
    return {true,
            false,
            {refinedMean[0], refinedMean[1], refinedMean[2]},
            meanCoverage};
  }
  return {false, false, {0.0, 0.0, 0.0}, 0.0};
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_ca_mpreg_webgpuviewer_TrimNative_findTrim(JNIEnv *env, jobject thiz,
                                               jobject pixelBuffer, jint width,
                                               jint height, jfloatArray colors,
                                               jfloat threshold,
                                               jintArray outBounds) {
  if (!env || !pixelBuffer || !colors || !outBounds) {
    return JNI_FALSE;
  }
  if (width <= 0 || height <= 0) {
    return JNI_FALSE;
  }
  if (!std::isfinite(threshold)) {
    return JNI_FALSE;
  }

  const uint8_t *pixels =
      static_cast<const uint8_t *>(env->GetDirectBufferAddress(pixelBuffer));
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return JNI_FALSE;
  }
  if (pixels == nullptr) {
    return JNI_FALSE;
  }
  const jlong capacity = env->GetDirectBufferCapacity(pixelBuffer);
  if (capacity < 0 ||
      capacity < static_cast<jlong>(width) * height * kChannels) {
    return JNI_FALSE;
  }

  const jsize colorFloats = env->GetArrayLength(colors);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return JNI_FALSE;
  }
  const int colorCount = colorFloats / 3;
  const jsize outLen = env->GetArrayLength(outBounds);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return JNI_FALSE;
  }
  /* 16 is well past any real caller's trim-color count. */
  if (colorCount <= 0 || colorCount > 16 || outLen < colorCount * 4) {
    return JNI_FALSE;
  }

  jfloat *colorData = env->GetFloatArrayElements(colors, nullptr);
  if (colorData == nullptr) {
    return JNI_FALSE;
  }

  std::vector<ColorTest> tests;
  tests.reserve(colorCount);
  for (int i = 0; i < colorCount; ++i) {
    tests.push_back(makeColorTest(colorData[i * 3], colorData[i * 3 + 1],
                                  colorData[i * 3 + 2], threshold));
  }
  env->ReleaseFloatArrayElements(colors, colorData, JNI_ABORT);

  std::vector<Bounds> merged(colorCount);
  for (int i = 0; i < colorCount; ++i) {
    merged[i].reset(width, height);
  }

  const int threadCount = chooseThreadCount(width, height);

  if (threadCount == 1) {
    scanBand(pixels, width, 0, height, tests.data(), colorCount, merged.data());
  } else {
    // Worker threads only read the direct buffer and their own band results, so
    // they never touch JNI and need no JNIEnv of their own.
    std::vector<std::vector<Bounds>> bandResults(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount - 1);

    const int rowsPerBand = (height + threadCount - 1) / threadCount;

    for (int t = 0; t < threadCount; ++t) {
      bandResults[t].resize(colorCount);
      for (int i = 0; i < colorCount; ++i) {
        bandResults[t][i].reset(width, height);
      }
    }

    for (int t = 1; t < threadCount; ++t) {
      const int y0 = std::min(t * rowsPerBand, height);
      const int y1 = std::min(y0 + rowsPerBand, height);
      if (y0 >= y1) {
        continue;
      }
      workers.emplace_back([=, &tests, &bandResults] {
        scanBand(pixels, width, y0, y1, tests.data(), colorCount,
                 bandResults[t].data());
      });
    }

    scanBand(pixels, width, 0, std::min(rowsPerBand, height), tests.data(),
             colorCount, bandResults[0].data());

    for (auto &worker : workers) {
      worker.join();
    }

    for (int t = 0; t < threadCount; ++t) {
      for (int i = 0; i < colorCount; ++i) {
        merged[i].merge(bandResults[t][i]);
      }
    }
  }

  std::vector<jint> out(colorCount * 4);
  for (int i = 0; i < colorCount; ++i) {
    out[i * 4 + 0] = merged[i].minX;
    out[i * 4 + 1] = merged[i].minY;
    out[i * 4 + 2] = merged[i].maxX;
    out[i * 4 + 3] = merged[i].maxY;
  }
  env->SetIntArrayRegion(outBounds, 0, colorCount * 4, out.data());

  return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_ca_mpreg_webgpuviewer_TrimNative_detectBackground(JNIEnv *env,
                                                       jobject thiz,
                                                       jobject pixelBuffer,
                                                       jint width, jint height,
                                                       jfloat threshold) {
  (void)threshold;
  const jint kWhite = static_cast<jint>(0xFFFFFFFFu);

  if (!env || !pixelBuffer) {
    return kWhite;
  }
  if (width <= 0 || height <= 0) {
    return kWhite;
  }

  const uint8_t *pixels =
      static_cast<const uint8_t *>(env->GetDirectBufferAddress(pixelBuffer));
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return kWhite;
  }
  if (pixels == nullptr) {
    return kWhite;
  }
  const jlong capacity = env->GetDirectBufferCapacity(pixelBuffer);
  if (capacity < 0 ||
      capacity < static_cast<jlong>(width) * height * kChannels) {
    return kWhite;
  }

  const size_t stride = static_cast<size_t>(width) * kChannels;
  const EdgeLine left{pixels, stride, height};
  const EdgeLine right{pixels + static_cast<size_t>(width - 1) * kChannels,
                       stride, height};
  const EdgeLine top{pixels, kChannels, width};
  const EdgeLine bottom{pixels + static_cast<size_t>(height - 1) * stride,
                        kChannels, width};

  const EdgeLine *edges[4] = {&left, &right, &top, &bottom};

  int nonWhiteCount = 0;
  double linearSum[3] = {0.0, 0.0, 0.0};
  for (auto &edge : edges) {
    const EdgeResult result = classifyEdge(*edge);
    if (!result.solid) {
      continue;
    }
    if (result.isWhite) {
      continue;
    }
    nonWhiteCount++;
    for (int ch = 0; ch < 3; ++ch) {
      linearSum[ch] += result.linearMean[ch];
    }
  }

  if (nonWhiteCount > 0) {
    const double inv = 1.0 / static_cast<double>(nonWhiteCount);
    const int r = std::clamp(
        static_cast<int>(linearToSrgb(linearSum[0] * inv) * 255.0 + 0.5), 0,
        255);
    const int g = std::clamp(
        static_cast<int>(linearToSrgb(linearSum[1] * inv) * 255.0 + 0.5), 0,
        255);
    const int b = std::clamp(
        static_cast<int>(linearToSrgb(linearSum[2] * inv) * 255.0 + 0.5), 0,
        255);
    const int rgb = (r << 16) | (g << 8) | b;
    return static_cast<jint>(0xFF000000u | static_cast<uint32_t>(rgb));
  }

  /* No solid edge that isn't white, so white is the answer either way. */
  return kWhite;
}
