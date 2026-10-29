# Pitch Helper ⚾

A pitching mechanics analyser that works from a phone video. Film a pitch side-on and it finds leg lift, foot strike and release, measures your mechanics, flags energy leaks, and draws "you vs target" guide images. Targets come from Driveline's OpenBiomechanics motion-capture data.

Built as a personal training aid for newer pitchers who don't have access to a biomechanics lab.

## What it checks
| Metric | Leak it flags |
|---|---|
| Stride length (% of height) | Short stride |
| Front knee, foot strike → release | Front knee collapsing instead of bracing |
| Shoulder turn by foot strike | Flying open early |
| Max hip–shoulder separation | Limited separation |
| Peak timing pelvis → trunk → arm | Out-of-order sequence |
| Trunk tilt at release | Upright finish |

Each metric is compared to the 10th–90th percentile range of 411 pitches from 100 pitchers (70–94 mph, mostly college) in the [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics).

## How it works
- **Pose tracking:** Google MediaPipe Pose Landmarker (heavy model), 2D image points plus 3D world points for every frame.
- **Left/right fixing:** pose models often swap sides mid-pitch. Frames are untangled in 3D, then kept consistent frame to frame, and the throwing arm is picked by hand speed.
- **Events:** foot strike from the lead foot's first contact, release from peak hand speed after foot strike, leg lift from the highest knee before it.
- **Rotation:** hip and shoulder angles are measured in 3D against the stride line, so the camera doesn't have to be perfectly side-on. The app detects the camera angle and marks metrics it can't measure as n/a instead of guessing.
- **Auto-split:** a whole bullpen can go in one video. A quick low-detail pass finds each pitch from spikes in hand speed during a stride, then each one is analysed on its own.

## Two versions
| | [`web/`](web) | [`python/`](python) |
|---|---|---|
| Runs | Entirely in the browser (nothing uploaded) | Locally, Flask UI + command line |
| Stack | JavaScript, MediaPipe Tasks Vision, Canvas | Python, MediaPipe, OpenCV, NumPy, Flask |
| Output | Scorecard, guide images, frame replay, history | Same, plus annotated video and reports |
| Hosting | Free static site (e.g. Hugging Face Static Space) | Your own computer |

The analysis code is a line-for-line port between the two and gives identical results on the same tracking data.

## Filming tips
- Side-on, facing your chest at leg lift (RHP: 3B side, LHP: 1B side), hip height, tripod, no panning or zooming.
- Slow-mo (120–240 fps). If slow-mo is saved as a 30 fps file, enter the real capture fps.
- Whole body in frame, plain background, good light.

## Credits and license
- Targets: [Driveline OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics), CC BY-NC-SA 4.0. The target numbers in this repo are derived from that data, so they're **non-commercial use only**, with attribution.
- Pose tracking: [Google MediaPipe](https://developers.google.com/mediapipe).
- Phone video is 2D and lower frame rate than a lab, so treat results as estimates. Your own trend over time matters more than hitting the median exactly.
- Code: MIT (see `LICENSE`). The OBP-derived targets stay under CC BY-NC-SA 4.0.
