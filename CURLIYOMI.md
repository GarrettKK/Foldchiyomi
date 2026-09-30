# Curliyomi

Curliyomi is a personal Android reader fork based on [Mihon](https://github.com/mihonapp/mihon), itself part of the Tachiyomi reader ecosystem.

## Reader defaults

- The WebGPU reader uses an interactive page curl in the style of Google Play Books (`TransitionCurl`). Take a page by its edge or corner and it rolls over after your finger, showing its back and casting a shadow on the page underneath; release to complete or cancel the turn. Taps and volume keys play the same curl from the bottom corner.
- Two-page spreads are enabled by default. The existing per-series reading settings can still change this.
- Existing Mihon/Tachiyomi-compatible extension repositories and extension packages are supported by the inherited repository manager.

To add a repository in the app, open **Settings → Browse → Extension repos** and paste its repository index URL. For example, Keiyoushi documents its Mihon-compatible repository at [keiyoushi/extensions](https://github.com/keiyoushi/extensions).

## Build

Use JDK 21, Android SDK, and the Android SDK components required by the project. Then run:

```sh
./gradlew assembleFoss
```

The APK is written under `app/build/outputs/apk/foss/`.

## WebGPU viewer module

The page viewer (`ca.mpreg:webgpuviewer`, MIT) is carried in-tree as the `:webgpuviewer` module, taken from upstream tag 49, so the reader can have its own transition. Its prebuilt native library `libresize.so` is extracted at build time from the published 49 AAR; `androidx.webgpu` comes from the upstream's Maven repository.

## Upstream and license

This project retains the upstream source and its notices. See `LICENSE` and the upstream repository for the complete license and third-party notices. Changes in this fork are limited to app identity and reader defaults/labels.
