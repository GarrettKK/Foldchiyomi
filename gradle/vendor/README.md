# Vendored dependencies

A Maven repository checked into the tree, for artifacts that no public repository serves in a form the build can rely on. `settings.gradle.kts` points Gradle here for the groups below and nowhere else.

## androidx.webgpu:webgpu 1.0.0-dev06

The WebGPU bindings (Dawn, with a JNI layer) that the page viewer in `:webgpuviewer` renders with. Google doesn't publish `androidx.webgpu` to its Maven repository yet; the viewer's upstream builds it and serves it from a GitHub-hosted repository at `https://raw.githubusercontent.com/mpreg-ca/androidx-webgpu-repo/main`.

On 2026-10-27 that repository republished `1.0.0-dev06` in place. The new AAR's `classes.jar` still declares the native method `GPU.createInstance`, but its `libwebgpu_c_bundled.so` exports `Java_androidx_webgpu_GPU_createGPUInstance` instead, so the reader failed to start on every device with an `UnsatisfiedLinkError` (Foldchiyomi 1.8.0). Builds before it had the AAR as first published, from Gradle's cache.

The AAR here is the artifact as it should be: the republished `classes.jar`, `annotations.zip`, manifest and metadata, with the four `libwebgpu_c_bundled.so` taken from Foldchiyomi's 1.7.1 release APKs, which were built from the original publication. All 181 native methods the classes declare resolve against them. The POM is the upstream one without its "prefer Gradle module metadata" marker, since the `.module` file carries checksums for the upstream AAR.

| ABI | SHA-256 of `libwebgpu_c_bundled.so` |
|---|---|
| arm64-v8a | `cbc5020bec3ab9dc3bc073363587590cd538b419da827e8d0c8f60f8e2a67ca7` |
| armeabi-v7a | `8ecd1a26e15f1350071f34cef259f4442d11f6c55091e4f629ed8b4b1a308a59` |
| x86 | `3952b2bd27965907fa1f508d3818e1f1d62c03bdf3d83973f49590c1813a2d39` |
| x86_64 | `342d49e0bd1e5c616bbe3f86503761ec5ab0558142963a68e9056d7d7e17c45c` |

To move to a newer `androidx.webgpu`, drop the new AAR and POM in beside this one under its own version directory, and check that its classes and native libraries agree before bumping the version in `gradle/libs.versions.toml`.
