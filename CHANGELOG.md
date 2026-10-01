# Changelog

All notable changes to Foldchiyomi are documented here. For the changes it inherits from Mihon, see [Mihon's changelog](https://github.com/mihonapp/mihon/blob/main/CHANGELOG.md).

## [v1.3.0] - Unreleased

### Added
- Dual page view has a new **When unfolded** mode, now the default: two pages on a foldable's inner screen, one page on its cover screen, switching by itself as you fold and unfold. Pick it under the reader's settings, **Dual page view**.
- The first time you open a series, the reader asks how it reads: right to left for manga, left to right for comics, vertical or webtoon. The answer is saved for that series. Turn it off with **Don't ask again**, or under **Settings → Reader → Ask for each new series**.

## [v1.2.0] - 2026-10-01

### Improved
- Bubble zoom is sharper and cleaner. The bubble is enlarged with ArtCNN, a small neural network made for upscaling line art that runs on the phone's GPU, so lettering gets real edges instead of blur. The paper is cleaned to white and the ink to black, which clears up the grey haze and JPEG noise of low-quality scans. Colour is left alone.
- Bubble zoom can be turned off: **Settings → Reader → Paged → Bubble zoom**, or in the reader's own settings sheet.

## [v1.1.0] - 2026-10-01

### Added
- Bubble zoom: double-tap a speech bubble and it lifts off the page, enlarged and sharp, the way Google Play Books does it. Tap anywhere to put it back. Where the tap isn't on a bubble, double-tap zooms in as before.

## [v1.0.0] - 2026-09-30

First release. Based on Mihon 0.20.4 with its unreleased changes as of late September 2026.

### Added
- Interactive page curl, the default page transition. Grab a page by its edge or corner and it rolls over after your finger, showing the back of the sheet and casting shadows. It works in single-page and two-page spreads, backward, and right-to-left.
- Foldchiyomi identity: name, `app.foldchiyomi` package, the 折 icon and a monochrome themed icon.
- Signed releases on GitHub, with in-app update checks against this repository.

### Changed
- Two-page spreads are on by default.

### Removed
- Firebase analytics and crash reporting: never included in any build.

[v1.2.0]: https://github.com/GarrettKK/Foldchiyomi/releases/tag/v1.2.0
[v1.1.0]: https://github.com/GarrettKK/Foldchiyomi/releases/tag/v1.1.0
[v1.0.0]: https://github.com/GarrettKK/Foldchiyomi/releases/tag/v1.0.0
