# Changelog

All notable changes to Foldchiyomi are documented here. For the changes it inherits from Mihon, see [Mihon's changelog](https://github.com/mihonapp/mihon/blob/main/CHANGELOG.md).

## [v1.6.0] - Unreleased

### Added
- **Zoom to panel.** Settings → Reader → Paged → Zoom to panel, or the switch in the reader's sheet. The bubble gesture on artwork then zooms the page to frame the panel under it; another one zooms back out. It finds panels by the white gutters between them, so it skips borderless panels and dark pages, where it zooms in as usual.

### Improved
- **Tap another bubble while one is open** and it opens in its place. A tap on the open bubble, or anywhere else, closes it as before.

## [v1.5.0] - 2026-10-08

### Added
- **A new icon**, designed by a reader from the community: an open book of panels, with a speech bubble saying 折.
- **Shift spread pages.** A button in the reader's bottom bar, shown in two-page view, pairs the chapter's pages the other way round, for a chapter whose first page is half of a spread rather than a lone cover.
- **Fit height and original size apply to two-page spreads.** Pick them under scale type, and a spread grows past the screen's width, to pan.
- **Bubble zoom by long press.** Settings → Reader → Paged → Bubble zoom gesture. A long press that finds no bubble opens the page menu as before.

### Improved
- Bubble zoom opens faster, and follows the double tap animation speed setting: Fast and No animation apply to it too.
- In two-page view, a bubble cut by the seam between the pages now opens; before, touching the page's inner edge disqualified it.

## [v1.4.0] - 2026-10-02

### Added
- An empty library now shows how to read comics already on your phone, in three steps with a picture of the folder layout, and buttons to open Local source and the storage folder setting.

### Changed
- Reading modes have names that say what they're for: **Right to left (Manga)**, **Left to right (Comics)** and **Long strip (Webtoon)**.

## [v1.3.2] - 2026-10-01

### Fixed
- Reader settings that did nothing now work, or are hidden where they can't. Some were left over from Mihon's older readers and never reached Foldchiyomi's.
  - **Double tap to zoom** can be turned off in the paged reader too, and the webtoon switch now works. Bubble zoom still opens bubbles with it off.
  - **Double tap animation speed** applies.
  - **Animate page transitions** off turns page turns into a cut.
  - In the webtoon modes, tap zones, inverted tapping and crop borders follow the **Webtoon** section's settings, as the settings screen says they do.
  - Hidden while the high-quality renderer is on, since it has no use for them: splitting and rotating wide pages, webtoon side padding (the reader's own page width slider replaces it) and the menu-hide sensitivity. Scale type lists only the types it supports, and notes that a two-page spread always fits the screen.
  - **Dual page view** is in Settings → Reader → Paged as well as in the reader's sheet.

## [v1.3.1] - 2026-10-01

### Improved
- Bubble zoom works on Western comics. Their lettering is denser and runs into the bubble's outline, bubbles get joined together, and some have spiky outlines; most of those were skipped. On a page of *Batman: Year One*, 25 of 26 bubbles and narration boxes now open, up from 11, while lettering laid straight over artwork still doesn't.

### Fixed
- Turning a single page with a black reader background showed the back of the sheet almost black. It's paper-white again, with the print showing faintly through.

## [v1.3.0] - 2026-10-01

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
