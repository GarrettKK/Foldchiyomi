// Quarter turns of a packed image buffer.
//
// A page wider than the screen reads better turned on its side than shrunk to fit, and a reader
// that offers that has to rotate the decoded pixels: there is no rotation anywhere between the
// image and the screen. A quarter turn is a transpose, which is pure memory movement, so the work
// here is entirely about the order the memory is touched in - see rotate_quarter.

#include <cstdint>
#include <cstring>
#include <jni.h>

namespace {

/**
 * Rows of the source become columns of the destination, so one of the two walks the memory in
 * strides. Both are walked in tiles instead, which keeps a run of each inside the cache rather
 * than taking a miss per pixel down a tall image.
 */
constexpr int kTile = 32;

/**
 * Turns a `width` x `height` image of `bpp`-byte pixels a quarter turn into `dst`, which is
 * `height` x `width`. Clockwise puts the source's first column down the destination's last
 * row; anticlockwise is the mirror of that.
 */
void rotate_quarter(const uint8_t *src, uint8_t *dst, int width, int height, int bpp,
                    bool clockwise) {
  const size_t srcStride = static_cast<size_t>(width) * bpp;
  const size_t dstStride = static_cast<size_t>(height) * bpp;

  for (int y0 = 0; y0 < height; y0 += kTile) {
    const int yEnd = (y0 + kTile < height) ? y0 + kTile : height;
    for (int x0 = 0; x0 < width; x0 += kTile) {
      const int xEnd = (x0 + kTile < width) ? x0 + kTile : width;

      for (int y = y0; y < yEnd; ++y) {
        const uint8_t *srcRow = src + static_cast<size_t>(y) * srcStride;
        for (int x = x0; x < xEnd; ++x) {
          // Clockwise: (x, y) -> (height - 1 - y, x). Anticlockwise: (x, y) -> (y, width - 1 - x).
          const int dstX = clockwise ? (height - 1 - y) : y;
          const int dstY = clockwise ? x : (width - 1 - x);
          uint8_t *dstPixel = dst + static_cast<size_t>(dstY) * dstStride +
                              static_cast<size_t>(dstX) * bpp;
          std::memcpy(dstPixel, srcRow + static_cast<size_t>(x) * bpp, bpp);
        }
      }
    }
  }
}

} // namespace

extern "C" JNIEXPORT void JNICALL Java_ca_mpreg_webgpuviewer_ImageUtil_rotateQuarterNative(
    JNIEnv *env, jobject /* thiz */, jobject srcBuffer, jobject dstBuffer, jint width, jint height,
    jint bytesPerPixel, jboolean clockwise) {
  if (width <= 0 || height <= 0 || bytesPerPixel <= 0) return;

  auto *src = static_cast<const uint8_t *>(env->GetDirectBufferAddress(srcBuffer));
  auto *dst = static_cast<uint8_t *>(env->GetDirectBufferAddress(dstBuffer));
  if (src == nullptr || dst == nullptr) return;

  // The caller sized both buffers; a short one here would be a write past the end.
  const jlong need = static_cast<jlong>(width) * height * bytesPerPixel;
  if (env->GetDirectBufferCapacity(srcBuffer) < need ||
      env->GetDirectBufferCapacity(dstBuffer) < need) {
    return;
  }

  rotate_quarter(src, dst, width, height, bytesPerPixel, clockwise == JNI_TRUE);
}
