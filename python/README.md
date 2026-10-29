# Pitch Helper (Python)

Personal pitching mechanics aid. Give it a side-on clip of one pitch; it marks leg lift, foot strike and release, measures key mechanics, and flags likely energy leaks.

## Setup
```
pip install -r requirements.txt
```
The pose model (~30 MB) downloads automatically on first run to `~/.pitch_helper/`.

## Filming
- **Side-on, facing your chest at leg lift** (RHP: 3B side, LHP: 1B side), so your front leg is nearest the camera. Hip height, whole body in frame, **no panning or zooming**.
- **Slow-mo (120–240 fps)**, tripod, same spot every session.
- **One pitch per clip** (trim to set position → follow-through), **or** one long video of the session with "Several pitches" / `--split`. Warm-up throws can get picked up too; delete those results.
- Plain background and good light help tracking.

## Web UI (easiest)
```
python app.py
```
Open http://localhost:5000. To upload straight from your phone, open the "phone" URL it prints (same Wi-Fi). Drop in one or more clips, set throwing hand / velo, and hit Analyze. The History tab charts any metric across sessions.

If auto-detection can't find the pitch (clip too short, camera moves), you still get the replay. Step through it frame by frame, tap **Leg lift / Foot strike / Release**, and re-analyze. You can also fix wrong events on any result under "Events look wrong?".

Only run it on a network you trust; there's no login.

## Command line
```
python pitch_helper.py pitch1.mp4 --throws R
python pitch_helper.py pitch1.mp4 --throws R --velo 74 --fps 240
```
- `--fps`: set this if your phone exports slow-mo at 30 fps playback (check the report header; it should show the real capture rate).
- `--velo`: radar reading, logged to history.
- `--events 79,137,183`: set leg lift, foot strike and release frames by hand (frame numbers are shown in `annotated.mp4`).
- `--model lite|full|heavy`: heavy is most accurate, lite is fastest.
- `--split`: the video has several pitches (e.g. a whole bullpen). It finds each pitch, saves it as its own clip in `pitch_output/<video>_clips/`, and analyzes each one. Set `--fps` for slow-mo saved at 30 fps, or it won't find them.

## Output (`pitch_output/<clip>/`)
- `annotated.mp4`: skeleton overlay with events and live knee/separation readouts
- `keyframes.png`: leg lift / foot strike / release
- `guide_*.png`: you vs. target images (trunk, stride, front knee, separation, hips & shoulders at foot strike)

## Camera view
The app checks where the camera is relative to your stride line:
- **Side-on:** everything is measured.
- **At an angle:** rotation (separation, hips/shoulders) is still measured against your stride line but is less precise. Trunk tilt and arm angle need a side view, so they show as n/a.
- **Behind or in front:** rotation, trunk tilt and timing sequence can't be measured and show as n/a. Film side-on instead.

Metrics marked n/a are left blank in `history.csv`, so they don't skew your trend charts.
- `report.txt` / `report.json`: metrics and flags
- `pitch_output/history.csv`: one row per clip, for tracking trends vs. velo

## What it checks
| Metric | Leak it flags |
|---|---|
| Stride length (% height) | Short stride |
| Lead knee, foot strike → release | Front knee collapsing instead of bracing |
| Shoulder rotation by foot strike | Flying open early |
| Max hip–shoulder separation | Limited separation |
| Peak timing pelvis → trunk → arm | Out-of-order sequence |
| Trunk tilt at release | Upright finish |

## Targets (Driveline OBP)
Flags and the green "target" guides come from Driveline's [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics): 411 pitches from 100 pitchers (70–94 mph, mostly college), 3D motion capture.
- **Green target** = OBP median.
- **Flag** = outside the range of 80% of OBP pitchers (10th/90th percentile).
- Numbers live in `OBP` at the top of `pitch_helper.py`. To rebuild them from a newer OBP release:
  ```
  git clone https://github.com/drivelineresearch/openbiomechanics
  python build_obp_targets.py openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv
  ```
- OBP data is **CC BY-NC-SA 4.0: non-commercial use only**, with attribution. Fine for personal training; check the license before using this commercially.
- OBP is 3D lab data and your numbers are phone estimates (rotation especially), so your own trend over time matters more than hitting the median exactly.

## Calibrating
Record 10–20 of your best pitches as a baseline and look at your typical numbers in `history.csv`. If a phone-measured metric sits consistently off from OBP even on good pitches (rotation metrics are most likely), you can nudge its numbers in `OBP` at the top of `pitch_helper.py`. Trust trends over absolute values.

## Deploy on Hugging Face Spaces
Docker Spaces need a paid Hugging Face plan. For free hosting, use the [browser version](../web) instead.

1. Create a new Space: SDK **Docker**, template **Blank**, hardware **CPU basic (free)**.
2. Upload `app.py`, `pitch_helper.py`, `requirements.txt`, `Dockerfile` and a `README.md` that starts with this block (Hugging Face reads it):
   ```
   ---
   title: Pitch Helper
   sdk: docker
   app_port: 7860
   ---
   ```
3. In the Space's **Settings → Variables and secrets**, add secrets:
   - `APP_PASSWORD`: the password you and teammates type to get in
   - `SECRET_KEY`: any long random text (keeps you logged in across restarts)
4. Wait for the build (a few minutes), then open the app.

Notes: free CPU is slower (a minute or two per clip). Without paid persistent storage, uploaded clips and history reset whenever the Space restarts or goes to sleep after ~48 h unused. OBP data is non-commercial, so don't charge for access.
