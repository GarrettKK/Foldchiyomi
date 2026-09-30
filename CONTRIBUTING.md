# Contributing

Thanks for your interest in Foldchiyomi!

- **Bugs and ideas:** open an [issue](https://github.com/GarrettKK/Foldchiyomi/issues/new/choose). For bugs, please include your device and whether it was folded or unfolded.
- **Pull requests** are welcome. Keep them focused, and describe how you tested on a real device.
- **Extensions and sources** are not part of this project, and problems with them can't be fixed here.

Foldchiyomi is a fork of [Mihon](https://github.com/mihonapp/mihon). Please don't report Foldchiyomi issues to the Mihon team. If a bug also happens in Mihon itself, report it there.

## Building

Use JDK 21 and the Android SDK, then run:

```sh
./gradlew assembleFoss -Pdist=foss
```

## Releasing

1. Bump `versionCode` and `versionName` in `app/build.gradle.kts`, add a section to `CHANGELOG.md`, and merge.
2. Publish a GitHub release on `main` with a new tag `v<versionName>` (e.g. `v1.0.1`). `release.yml` builds, signs and attaches the APKs.
