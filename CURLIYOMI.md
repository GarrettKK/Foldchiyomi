# Curliyomi

Curliyomi is a personal Android reader fork based on [Mihon](https://github.com/mihonapp/mihon), itself part of the Tachiyomi reader ecosystem.

## Reader defaults

- The WebGPU reader uses an interactive, touch-tracked page curl (`TransitionFlip`). Drag a page edge to follow the turn, then release to complete or cancel it.
- Two-page spreads are enabled by default. The existing per-series reading settings can still change this.
- Existing Mihon/Tachiyomi-compatible extension repositories and extension packages are supported by the inherited repository manager.

To add a repository in the app, open **Settings → Browse → Extension repos** and paste its repository index URL. For example, Keiyoushi documents its Mihon-compatible repository at [keiyoushi/extensions](https://github.com/keiyoushi/extensions).

## Build

Use JDK 21, Android SDK, and the Android SDK components required by the project. Then run:

```sh
./gradlew assembleFoss
```

The APK is written under `app/build/outputs/apk/foss/`.

## Upstream and license

This project retains the upstream source and its notices. See `LICENSE` and the upstream repository for the complete license and third-party notices. The app vendors the MIT-licensed WebGPU viewer in `webgpuviewer/` so the page-turn mesh can respond to the drag position. The curl uses more vertical mesh slices and tightens around the finger, while retaining the upstream renderer for image decoding, page spreads, and shadows.
