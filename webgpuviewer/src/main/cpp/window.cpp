// androidx.webgpu's windowFromSurface returns ANativeWindow_fromSurface's pointer, which carries a
// reference the caller has to release, and offers nothing to release it with.

#include <android/native_window.h>
#include <jni.h>

extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_renderer_NativeWindow_release(JNIEnv *, jobject, jlong window) {
  if (window != 0) ANativeWindow_release(reinterpret_cast<ANativeWindow *>(window));
}
