---
title: Pitch Helper
emoji: ⚾
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
---

# Pitch Helper

Personal pitching mechanics aid. Upload side-on clips (one pitch each, or a whole session that it splits automatically) and get leaks, metrics vs. Driveline OBP ranges, and you-vs-target guide images.

Everything runs in your browser: pose tracking uses Google MediaPipe (loaded from a CDN), and your videos never leave your device. History is saved in your browser on that device.

## Files
- `src/analysis.js`: the analysis (a direct port of the Python version). `src/app.js`: pose tracking, guide images and UI.
- `build.py`: inlines both into `index.html`. Run `python build.py` after editing `src/`.
- `index.html`: the whole app in one file, ready to deploy (pose tracking, analysis, guide images, results, replay, history). Kept as a single file so it also works when the Space is private.
- `logo.png` (optional): your team logo, shown in the header. In a private Space the logo may not load; everything else works.

## Filming
Side-on, camera facing your chest at leg lift (RHP: 3B side, LHP: 1B side), hip height, tripod, no panning or zooming. Slow-mo (120–240 fps). If a slow-mo clip was exported at 30 fps, enter the real capture fps.

## Targets
From Driveline's [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics), 411 pitches from 100 pitchers, 3D motion capture. CC BY-NC-SA 4.0: non-commercial use only, with attribution. Phone estimates vs. lab data, so track your own trend over time.
