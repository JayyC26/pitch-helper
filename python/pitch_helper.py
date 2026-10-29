#!/usr/bin/env python3
"""
Pitch Helper — personal pitching mechanics aid.

Analyzes ONE pitch per clip, filmed side-on with the camera facing your chest at
leg lift (RHP: 3B side; LHP: 1B side), on a tripod, ideally in slow-mo.

Usage:
    python pitch_helper.py my_pitch.mp4 --throws R
    python pitch_helper.py my_pitch.mp4 --throws L --velo 72 --fps 240

Outputs (in ./pitch_output/<clip name>/):
    annotated.mp4   skeleton overlay with key events marked
    keyframes.png   leg lift / foot strike / release side by side
    report.txt      metrics + plain-language flags
    report.json     same data, machine readable
    guide_*.png     you vs. target images
and adds a row to ./pitch_output/history.csv for session tracking.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Targets from Driveline's OpenBiomechanics Project (OBP): 411 pitches from 100
# pitchers, 70–94 mph, mostly college. Each list is the [10th, 25th, 50th, 75th,
# 90th] percentile, converted to Pitch Helper's conventions by build_obp_targets.py.
# Data © Driveline Baseball, CC BY-NC-SA 4.0 (non-commercial use only):
# https://github.com/drivelineresearch/openbiomechanics
# OBP is 3D lab motion capture; your numbers are phone estimates, so use these as
# a guide and track your own trend over time.
# ---------------------------------------------------------------------------
OBP = {
    "stride_pct":        [76.0, 79.9, 83.7, 87.7, 90.4],    # stride length, % of height
    "trunk_tilt_deg":    [20.8, 27.2, 34.7, 44.1, 48.2],    # forward trunk tilt at release
    "separation_deg":    [23.4, 27.6, 32.3, 36.7, 40.8],    # peak hip-shoulder separation
    "shoulder_open_deg": [26.5, 33.7, 41.7, 52.1, 62.8],    # shoulder turn from most-closed to foot strike
    "knee_change_deg":   [-25.4, -17.5, -10.0, -1.9, 2.2],  # lead knee bend change FS->release (- = straightens)
    "hips_at_fs_deg":    [20.5, 27.1, 34.3, 42.1, 49.6],    # hips turned toward plate at foot strike (0 = closed)
    "shoulders_at_fs_deg": [-11.3, -2.7, 5.5, 12.3, 19.2],  # shoulders turned toward plate at foot strike
}
P10, P25, P50, P75, P90 = range(5)
OBP_CREDIT = "Targets: Driveline OpenBiomechanics Project (CC BY-NC-SA 4.0)"


def obp_mid(key):
    """'27–44' style middle-half range for a metric."""
    return f"{OBP[key][P25]:.0f}–{OBP[key][P75]:.0f}"


TARGETS = {
    "min_fps_for_timing": 100,   # below this, sequencing/timing results are low-confidence
    "knee_noise_deg": 5.0,       # phone tracking jitter: don't flag knee changes smaller than this
}

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# MediaPipe landmark indices (person's own left/right)
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28

SKELETON = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
]


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------
def get_model(kind: str) -> str:
    cache = Path.home() / ".pitch_helper"
    cache.mkdir(exist_ok=True)
    path = cache / f"pose_landmarker_{kind}.task"
    if not path.exists():
        print(f"Downloading pose model ({kind}) — first run only...")
        urllib.request.urlretrieve(MODEL_URLS[kind], path)
    return str(path)


class Cancelled(Exception):
    """Raised when the user stops an analysis."""


def extract_pose(video_path: str, fps: float, model_kind: str, on_frame=None):
    """Returns pts2d (N,33,2) pixels, world (N,33,3) metres, vis (N,33). NaN where undetected.
    on_frame(done, total) is called after each frame; it may raise Cancelled to stop early."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_model(model_kind)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pts, world, vis = [], [], []
    i = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if on_frame:
                try:
                    on_frame(i, total)
                except Cancelled:
                    cap.release()
                    raise
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int(round(i * 1000.0 / fps)))
            if res.pose_landmarks:
                lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
                pts.append([[p.x * w, p.y * h] for p in lm])
                world.append([[p.x, p.y, p.z] for p in wl])
                vis.append([p.visibility for p in lm])
            else:
                pts.append(np.full((33, 2), np.nan))
                world.append(np.full((33, 3), np.nan))
                vis.append(np.zeros(33))
            i += 1
            if i % 50 == 0:
                print(f"  pose: {i} frames", end="\r")
    cap.release()
    print(f"  pose: {i} frames done      ")
    return np.array(pts, float), np.array(world, float), np.array(vis, float)


# ---------------------------------------------------------------------------
# Splitting a long video into pitches
# ---------------------------------------------------------------------------
# A quick pass with the light model (~12 looks per real second) finds pitches: a big spike in
# hand speed (relative to the hips, so camera shake and walking don't count) while the feet are
# spread in a stride. Each pitch is then cut out as its own clip and analyzed normally.
SPLIT = dict(
    pre=1.8, post=0.7,   # seconds kept before / after each release (real time)
    min_gap=3.0,         # two pitches can't be closer than this (s)
    min_speed=6.0,       # hand speed, torso lengths per second (a hard throw is ~20-40)
    rel_speed=0.3,       # ...and at least this share of the fastest throw in the video
    min_spread=1.5,      # feet apart, torso lengths (standing ~0.6, full stride ~2.5)
)
SCAN_HZ = 12


def find_pitches(times, pts) -> list:
    """times: (N,) real seconds; pts: (N,33,2) pixels. Returns [{t, speed, stride}] in time order."""
    times, pts = np.asarray(times, float), np.asarray(pts, float)
    n = len(times)
    need = [L_SH, R_SH, L_HIP, R_HIP, L_WR, R_WR, L_ANK, R_ANK]
    ok = np.array([np.isfinite(pts[i, need]).all() for i in range(n)], bool) if n else np.zeros(0, bool)
    if not ok.any():
        return []
    hip = (pts[:, L_HIP] + pts[:, R_HIP]) / 2
    sh = (pts[:, L_SH] + pts[:, R_SH]) / 2
    T = float(np.median(np.linalg.norm(sh[ok] - hip[ok], axis=1)))
    if not T > 0:
        return []
    speed, spread = np.full(n, np.nan), np.full(n, np.nan)
    prev = -1
    for i in range(n):
        if not ok[i]:
            continue
        spread[i] = np.linalg.norm(pts[i, L_ANK] - pts[i, R_ANK]) / T
        dt = times[i] - times[prev] if prev >= 0 else 0.0
        if 0 < dt < 0.25:
            a, b = pts[prev, L_WR] - hip[prev], pts[prev, R_WR] - hip[prev]
            c, d = pts[i, L_WR] - hip[i], pts[i, R_WR] - hip[i]
            same = [np.linalg.norm(a - c), np.linalg.norm(b - d)]
            crossed = [np.linalg.norm(a - d), np.linalg.norm(b - c)]  # left/right can swap
            m = same if sum(same) <= sum(crossed) else crossed
            speed[i] = max(m) / T / dt
        prev = i
    fs = speed[np.isfinite(speed)]
    thr = max(SPLIT["min_speed"], SPLIT["rel_speed"] * (fs.max() if len(fs) else 0))
    cands = []
    for i in range(n):
        if not speed[i] >= thr:
            continue
        dt = times - times[i]
        near = np.abs(dt) <= 0.15
        if np.nanmax(np.where(near, speed, -np.inf)) > speed[i]:
            continue
        win = (dt >= -0.8) & (dt <= 0.4) & np.isfinite(spread)
        wide = spread[win].max() if win.any() else 0.0
        if wide >= SPLIT["min_spread"]:
            cands.append({"t": float(times[i]), "speed": float(speed[i]), "stride": float(wide)})
    kept = []
    for c in sorted(cands, key=lambda c: -c["speed"]):
        if all(abs(k["t"] - c["t"]) >= SPLIT["min_gap"] for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c["t"])


def scan_video(video_path: str, capture_fps: float, on_progress=None):
    """Light-model pass over the whole video, ~SCAN_HZ looks per real second.
    Returns (times in real seconds, pts (N,33,2)). on_progress(done, total) may raise Cancelled."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    every = max(1, int(round(capture_fps / SCAN_HZ)))  # frames between looks
    scale = min(1.0, 480 / max(w, h))
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_model("lite")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    times, pts = [], []
    i = 0
    with vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            if i % every:
                if not cap.grab():
                    break
                i += 1
                continue
            if on_progress:
                try:
                    on_progress(i, total)
                except Cancelled:
                    cap.release()
                    raise
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(round(i * 1000.0 / capture_fps)))
            times.append(i / capture_fps)
            pts.append([[p.x * w, p.y * h] for p in res.pose_landmarks[0]] if res.pose_landmarks
                       else np.full((33, 2), np.nan))
            i += 1
    cap.release()
    return np.array(times, float), np.array(pts, float).reshape(-1, 33, 2)


def split_video(video_path: str, capture_fps: float, out_dir: Path, on_progress=None) -> list:
    """Find the pitches in a long video and save each one as its own clip in out_dir.
    Returns [(clip_path, pitch_info)]. Frames are copied exactly, at the original frame rate."""
    times, pts = scan_video(video_path, capture_fps, on_progress)
    found = find_pitches(times, pts)
    if not found:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    spans = [(max(0, int((p["t"] - SPLIT["pre"]) * capture_fps)),
              min(total, int((p["t"] + SPLIT["post"]) * capture_fps) + 1)) for p in found]
    stem = Path(video_path).stem
    paths = [out_dir / f"{stem}_pitch{k + 1}.mp4" for k in range(len(found))]
    writers = [None] * len(found)
    i = 0
    while i < spans[-1][1]:
        ok, frame = cap.read()
        if not ok:
            break
        for k, (a, b) in enumerate(spans):
            if a <= i < b:
                if writers[k] is None:
                    writers[k] = cv2.VideoWriter(str(paths[k]), cv2.VideoWriter.fourcc(*"mp4v"), file_fps, size)
                writers[k].write(frame)
            elif i == b and writers[k] is not None:
                writers[k].release()
        i += 1
    cap.release()
    for wr in writers:
        if wr is not None:
            wr.release()
    return [(paths[k], found[k]) for k in range(len(found)) if writers[k] is not None]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def fill_and_smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Linear-interpolate NaNs along time, then centered moving average."""
    out = arr.copy()
    flat = out.reshape(out.shape[0], -1)
    t = np.arange(flat.shape[0])
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = ~np.isnan(col)
        if good.sum() >= 2:
            col[~good] = np.interp(t[~good], t[good], col[good])
        elif good.sum() == 1:
            col[:] = col[good][0]
        if win > 1:
            pad = win // 2
            padded = np.pad(col, pad, mode="edge")
            col[:] = np.convolve(padded, np.ones(win) / win, mode="valid")
        flat[:, c] = col
    return flat.reshape(arr.shape)


def angle_at(a, b, c) -> np.ndarray:
    """Angle ABC in degrees, vectorised over frames."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rotations_vs_stride(W, fs, rel, rhp):
    """Hip and shoulder rotation (deg) every frame, measured in 3D against the throwing line
    (back ankle -> front ankle at foot strike). 0 = closed (lead side pointing at the plate),
    90 = square to the plate, >90 = rotated past square. Returns (hips, shoulders, axis_deg)
    where axis_deg is the stride line's angle vs. the camera (0 = camera side-on)."""
    lead_side, throw_side = (0, 1) if rhp else (1, 0)   # index into (left, right) pairs
    hips, shs, anks = (L_HIP, R_HIP), (L_SH, R_SH), (L_ANK, R_ANK)
    ang = lambda v: np.degrees(np.arctan2(v[..., 2], v[..., 0]))
    axis = ang(W[fs, anks[lead_side]] - W[fs, anks[throw_side]])

    def turn(pair):
        r = ang(W[:, pair[lead_side]] - W[:, pair[throw_side]]) - axis
        r = np.degrees(np.unwrap(np.radians(r)))
        return r - 360 * np.round(r[fs] / 360)

    hip, sh = turn(hips), turn(shs)
    if sh[rel] - sh[fs] < 0:   # make "toward the plate" positive (shoulders open after foot strike)
        hip, sh = -hip, -sh
    return hip, sh, float(axis)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
class EventError(ValueError):
    """Pose was tracked, but leg lift / foot strike / release couldn't be found reliably."""


def prepare(pts, world, fps):
    win = max(3, int(round(fps * 0.025)) | 1)
    return fill_and_smooth(pts, win), fill_and_smooth(world, win)


def body_height_px(P):
    feet = np.maximum(P[:, L_ANK, 1], P[:, R_ANK, 1])
    return float(np.nanpercentile(feet - P[:, NOSE, 1], 90) * 1.1)


def find_camera_cut(pts, height_px):
    """Frame index of a sudden jump in body position (camera cut / zoom), else None."""
    hip = (pts[:, L_HIP] + pts[:, R_HIP]) / 2
    jump = np.linalg.norm(np.diff(hip, axis=0), axis=1)
    jump[np.isnan(jump)] = 0
    i = int(np.argmax(jump))
    return i + 1 if jump[i] > 0.25 * height_px else None


# Left/right joint pairs, grouped so a whole limb set swaps together
ARM_PAIRS = [(L_SH, R_SH), (L_EL, R_EL), (L_WR, R_WR), (17, 18), (19, 20), (21, 22)]
LEG_PAIRS = [(L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANK, R_ANK), (29, 30), (31, 32)]


def throw_direction(P, throw_wr):
    """+1 if the pitch goes toward image right, else -1.
    Main cue: at peak hand speed (around release) the throwing hand is out in front of the
    hips, toward the plate. That's relative to the body, so a camera panning to follow the
    pitcher (broadcast video) can't fool it. Backed up by the hand's direction of travel and,
    when the camera is still, how far the hips travel."""
    hip_x = (P[:, L_HIP, 0] + P[:, R_HIP, 0]) / 2
    H = body_height_px(P)
    sp = np.linalg.norm(np.diff(P[:, throw_wr], axis=0), axis=1)
    k = int(np.nanargmax(sp))
    win = slice(max(0, k - 2), min(len(P), k + 3))
    votes = 0.0
    ahead = np.nanmedian(P[win, throw_wr, 0] - hip_x[win])
    if abs(ahead) > 0.08 * H:
        votes += 2 * np.sign(ahead)
    votes += np.sign(P[min(k + 1, len(P) - 1), throw_wr, 0] - P[k, throw_wr, 0])
    n = max(2, len(hip_x) // 4)
    travel = np.nanmedian(hip_x[-n:]) - np.nanmedian(hip_x[:n])
    if abs(travel) > 0.3 * H:
        votes += np.sign(travel)
    return float(np.sign(votes) or 1.0)


def fix_sides(pts, world, vis, rhp, swap_legs=False, flip_dir=False):
    """MediaPipe often mixes up left/right when a pitcher turns from facing the camera to facing
    away. The whole body is always relabelled together, so the skeleton can't end up crossed
    (left shoulder joined to right hip):
    1) per frame, if the torso is crossed in 3D, fix the arms to match the legs;
    2) frame to frame, keep left/right consistent with the previous frame;
    3) overall, make the throwing arm (fastest wrist) the right arm for a RHP (left for a LHP).
    swap_legs: manual "sides are mixed up" override (flips the whole body's left/right)."""
    pts, world, vis = pts.copy(), world.copy(), vis.copy()
    ALL = ARM_PAIRS + LEG_PAIRS

    def swap(t, pairs):
        for a, b in pairs:
            for arr in (pts, world, vis):
                arr[t, [a, b]] = arr[t, [b, a]]

    # Uncross the torso frame by frame (in 3D, where left and right are clearly apart)
    W = world
    d_ok = (np.linalg.norm(W[:, L_SH] - W[:, L_HIP], axis=-1) + np.linalg.norm(W[:, R_SH] - W[:, R_HIP], axis=-1))
    d_x = (np.linalg.norm(W[:, L_SH] - W[:, R_HIP], axis=-1) + np.linalg.norm(W[:, R_SH] - W[:, L_HIP], axis=-1))
    for t in np.where(d_x < d_ok * 0.9)[0]:
        swap(t, ARM_PAIRS)

    main = [(L_SH, R_SH), (L_HIP, R_HIP), (L_WR, R_WR), (L_KNEE, R_KNEE), (L_ANK, R_ANK)]
    prev = None
    for t in range(len(pts)):
        cur = np.array([[pts[t, a], pts[t, b]] for a, b in main])  # (5, 2 sides, 2)
        if np.isnan(cur).any():
            continue
        if prev is not None:
            keep = np.linalg.norm(cur - prev, axis=-1).sum()
            flip = np.linalg.norm(cur[:, ::-1] - prev, axis=-1).sum()
            if flip < keep * 0.8:
                swap(t, ALL)
                cur = cur[:, ::-1]
        prev = cur

    P = fill_and_smooth(pts, 3)
    sp = lambda j: np.nanpercentile(np.linalg.norm(np.diff(P[:, j], axis=0), axis=1), 97)
    right_throws = sp(R_WR) >= sp(L_WR)
    if (right_throws != rhp) != bool(swap_legs):
        swap(slice(None), ALL)
        P = fill_and_smooth(pts, 3)
    d = throw_direction(P, R_WR if rhp else L_WR)
    if flip_dir:  # manual override from the UI
        d = -d
    return pts, world, vis, d


def body_height_world(W):
    """Standing height in metres from segment lengths (pose-independent):
    shin + thigh + trunk + shoulder→nose ≈ 0.897 × height (standard anthropometry)."""
    seg = lambda a, b: np.linalg.norm(W[:, a] - W[:, b], axis=-1)
    mid_h = (W[:, L_HIP] + W[:, R_HIP]) / 2
    mid_s = (W[:, L_SH] + W[:, R_SH]) / 2
    shin = (seg(L_KNEE, L_ANK) + seg(R_KNEE, R_ANK)) / 2
    thigh = (seg(L_HIP, L_KNEE) + seg(R_HIP, R_KNEE)) / 2
    trunk = np.linalg.norm(mid_s - mid_h, axis=-1)
    head = np.linalg.norm(W[:, NOSE] - mid_s, axis=-1)
    return float(np.nanmedian(shin + thigh + trunk + head) / 0.897)


def detect_events(P, fps, height_px, rhp, direction):
    n = len(P)
    lead_ank, back_ank = (L_ANK, R_ANK) if rhp else (R_ANK, L_ANK)
    throw_wr = R_WR if rhp else L_WR

    # Front foot measured relative to the back foot, so a camera that pans to follow the
    # pitcher doesn't hide the stride.
    fwd = direction * (P[:, lead_ank, 0] - P[:, back_ank, 0])
    ext = int(np.argmax(fwd))
    start = float(np.min(fwd[:ext + 1]))
    travel = fwd[ext] - start
    if travel < 0.3 * height_px:
        raise EventError(
            "Couldn't find a stride: the front foot never got far ahead of the back foot. The clip "
            "probably stops before foot strike, or the camera is behind/in front of the pitcher.")

    reach = int(np.argmax(fwd >= start + 0.85 * travel))
    # Foot strike = first contact: the lowest point of the front foot (heel, toe or ankle) reaches
    # the level it settles at once planted. (The ankle alone reads "landed" a few frames early.)
    heel, toe = (29, 31) if rhp else (30, 32)
    low = np.max(P[:, [lead_ank, heel, toe], 1], axis=1)   # image y grows downward
    planted = low[ext:min(n, ext + max(3, int(0.1 * fps)))]
    ground = float(np.median(planted))
    fs = next((i for i in range(reach, n) if low[i] >= ground - 0.012 * height_px), ext)

    # Leg lift = highest knee before foot strike (either knee, in case sides blur mid-lift)
    lift = int(np.argmin(np.minimum(P[:fs + 1, L_KNEE, 1], P[:fs + 1, R_KNEE, 1])))
    if lift >= fs or (fs - lift) < max(2, 0.1 * fps):
        raise EventError("Couldn't separate leg lift from foot strike. Make sure the clip starts "
                         "before the leg lift.")

    if n - fs < max(3, int(0.08 * fps)):
        raise EventError("Clip ends too soon after foot strike. Include the follow-through.")
    wr_speed = np.linalg.norm(np.gradient(P[:, throw_wr, :], axis=0), axis=1)
    end = min(n, fs + int(0.35 * fps) + 1)
    rel = fs + 1 + int(np.argmax(wr_speed[fs + 1:end]))
    return lift, fs, rel


def analyze(pts, world, vis, fps: float, throws: str, events=None, swap_legs=False,
            flip_dir=False) -> dict:
    """events: optional (leg_lift, foot_strike, release) frame indices to skip auto-detection.
    swap_legs: flip which leg is treated as the lead leg (manual fix).
    flip_dir: flip which side the plate is on (manual fix)."""
    n = len(pts)
    if n < 10:
        raise ValueError("Clip too short — need at least 10 frames.")
    detected = ~np.isnan(pts[:, 0, 0])
    if detected.mean() < 0.5:
        raise ValueError("Pitcher detected in under half the frames. Check lighting/framing.")

    rhp = throws.upper() == "R"
    pts, world, vis, direction = fix_sides(pts, world, vis, rhp, swap_legs, flip_dir)
    P, W = prepare(pts, world, fps)
    lead_hip, lead_knee, lead_ank = (L_HIP, L_KNEE, L_ANK) if rhp else (R_HIP, R_KNEE, R_ANK)
    back_ank = R_ANK if rhp else L_ANK
    throw_sh, throw_wr = (R_SH, R_WR) if rhp else (L_SH, L_WR)
    height_px = body_height_px(P)

    manual = events is not None
    if manual:
        lift, fs, rel = (int(e) for e in events)
        if not (0 <= lift < fs < rel < n):
            raise EventError("Events must be in order: leg lift < foot strike < release.")
    else:
        cut = find_camera_cut(pts, height_px)
        if cut is not None:
            raise EventError(f"Camera cut or jump around frame {cut}. Use a fixed camera, "
                             "or mark the events by hand.")
        lift, fs, rel = detect_events(P, fps, height_px, rhp, direction)

    wr_speed = np.linalg.norm(np.gradient(P[:, throw_wr, :], axis=0), axis=1) * fps  # px/s

    # --- Lower body ------------------------------------------------------
    # Stride from 3D world landmarks (x/z ground plane), so a camera that isn't perfectly
    # side-on doesn't shrink it. Falls back to the 2D image distance if 3D looks off.
    gap = W[fs, lead_ank] - W[fs, back_ank]
    h_m = body_height_world(W)
    stride_3d = np.hypot(gap[0], gap[2]) / h_m * 100 if h_m > 0.5 else np.nan
    stride_2d = np.linalg.norm(P[fs, lead_ank] - P[fs, back_ank]) / height_px * 100
    stride_pct = stride_3d if np.isfinite(stride_3d) and 20 < stride_3d < 140 else stride_2d
    knee_flex = 180 - angle_at(P[:, lead_hip], P[:, lead_knee], P[:, lead_ank])
    knee_fs, knee_rel = float(knee_flex[fs]), float(knee_flex[rel])

    # --- Trunk -----------------------------------------------------------
    mid_hip = (P[:, L_HIP] + P[:, R_HIP]) / 2
    mid_sh = (P[:, L_SH] + P[:, R_SH]) / 2
    tv = mid_sh - mid_hip
    trunk_tilt = np.degrees(np.arctan2(direction * tv[:, 0], -tv[:, 1]))  # + = toward target

    # Rotation is measured against your actual throwing line (back foot -> front foot at foot
    # strike, in 3D), like OBP measures it relative to home plate. Measuring against the camera
    # instead breaks whenever the camera isn't perfectly side-on.
    hip_rot, sh_rot, axis_deg = rotations_vs_stride(W, fs, rel, rhp)
    sep = hip_rot - sh_rot
    off = abs(((axis_deg + 90) % 180) - 90)  # stride line vs camera: 0 = side-on, 90 = behind/in front
    view = "side" if off < 30 else ("angled" if off < 60 else "end-on")
    side_on = view == "side"            # trunk tilt is measured in 2D, so it needs a side view
    rot_reliable = view != "end-on"     # rotation can't be seen from straight behind/in front
    # Separation peaks around foot strike, so only look there (a noisy estimate searched over
    # the whole delivery can "peak" in the follow-through).
    a = max(lift, fs - int(0.05 * fps))
    b = max(a, min(rel - 1, fs + int(0.12 * fps)))
    sep_max = float(np.max(sep[a:b + 1]))
    sep_max_frame = a + int(np.argmax(sep[a:b + 1]))
    # Shoulder turn toward the plate from their most-closed point (after leg lift) to foot strike
    sh_open_at_fs = float(sh_rot[fs] - np.min(sh_rot[lift:fs + 1]))

    # --- Kinematic sequence ---------------------------------------------
    pel_vel = np.gradient(hip_rot) * fps
    trk_vel = np.gradient(sh_rot) * fps
    seq_end = min(n, rel + int(0.1 * fps) + 1)
    t_pel = lift + int(np.argmax(pel_vel[lift:seq_end]))
    t_trk = lift + int(np.argmax(trk_vel[lift:seq_end]))
    t_arm = lift + int(np.argmax(wr_speed[lift:seq_end]))
    seq_ok = t_pel <= t_trk + 1 and t_trk <= t_arm + 1

    # --- Arm -------------------------------------------------------------
    arm_v = P[rel, throw_wr] - P[rel, throw_sh]
    arm_angle = float(np.degrees(np.arctan2(-arm_v[1], abs(arm_v[0]))))

    key_joints = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, throw_wr]
    confidence = float(np.mean(vis[:, key_joints]))
    ms = lambda frames: round(frames * 1000.0 / fps)

    metrics = {
        "stride_pct_height": round(float(stride_pct), 1),
        "lead_knee_flex_at_fs_deg": round(knee_fs, 1),
        "lead_knee_flex_at_release_deg": round(knee_rel, 1),
        "lead_knee_change_deg": round(knee_rel - knee_fs, 1),
        "hip_shoulder_sep_max_deg": round(sep_max, 1),
        "shoulder_rotation_by_fs_deg": round(sh_open_at_fs, 1),
        "trunk_tilt_at_release_deg": round(float(trunk_tilt[rel]), 1),
        "arm_angle_at_release_deg": round(arm_angle, 1),
        "fs_to_release_ms": ms(rel - fs),
        "sequence_pelvis_trunk_arm_ms": [ms(t_pel - fs), ms(t_trk - fs), ms(t_arm - fs)],
        "sequence_in_order": bool(seq_ok),
        "rotation_reliable": rot_reliable,
        "camera_view": view,
    }
    events = {"leg_lift": lift, "foot_strike": fs, "release": rel, "max_separation": sep_max_frame}

    flags = build_flags(metrics, fps, confidence)
    series = {"knee_flex": knee_flex, "sep": sep, "trunk_tilt": trunk_tilt,
              "sep_window": (lift, b) if rot_reliable else None,
              "hip_rot": hip_rot, "sh_rot": sh_rot, "direction": direction,
              "height_px": height_px, "rhp": rhp}
    return {"metrics": metrics, "events": events, "flags": flags, "fps": fps,
            "confidence": round(confidence, 2), "throws": throws.upper(),
            "events_source": "manual" if manual else "auto",
            "_P": P, "_series": series}


def build_flags(m: dict, fps: float, conf: float) -> list:
    rot_ok = m.get("rotation_reliable", True)
    T, flags = TARGETS, []

    def add(level, title, detail, tip):
        flags.append({"level": level, "title": title, "detail": detail, "tip": tip})

    if fps < T["min_fps_for_timing"]:
        add("note", "Low frame rate",
            f"Clip is {fps:.0f} fps, so arm timing and sequencing are rough.",
            "Film in slow-mo (120–240 fps) for reliable timing.")
    if conf < 0.6:
        add("note", "Low tracking confidence",
            f"Average joint visibility {conf:.2f}.",
            "Use better lighting, a plain background, and keep the whole body in frame.")
    if not 60 <= m["fs_to_release_ms"] <= 350:
        add("note", "Event timing looks off",
            f"Foot strike → release took {m['fs_to_release_ms']} ms (typical is ~100–200 ms).",
            "Check the marked frames in the replay and fix them if they're wrong.")

    # Leak flags fire when you're outside the range of 80% of OBP pitchers (10th/90th percentile)
    if m["stride_pct_height"] < OBP["stride_pct"][P10]:
        add("leak", "Short stride",
            f"Stride is {m['stride_pct_height']:.0f}% of your height. OBP pitchers: middle half "
            f"{obp_mid('stride_pct')}%, and 9 in 10 stride at least {OBP['stride_pct'][P10]:.0f}%.",
            "Mark a landing spot further out, and work on hip mobility and drive off the back leg.")
    knee_limit = max(OBP["knee_change_deg"][P90], T["knee_noise_deg"])
    if m["lead_knee_change_deg"] > knee_limit:
        add("leak", "Front knee collapsing",
            f"Front knee bent {m['lead_knee_change_deg']:.0f}° more between foot strike and release. "
            f"Most OBP pitchers straighten it (typically ~{-OBP['knee_change_deg'][P50]:.0f}° straighter).",
            "Focus on a firm front-side block: step-behind or walk-in drills, plus quad/glute strength.")
    if m.get("camera_view") == "end-on":
        add("note", "Camera is behind or in front of the pitcher",
            "From this view the stride and the turn happen toward/away from the camera, so rotation, "
            "trunk tilt and lead-leg detection aren't reliable. Rotation and trunk aren't flagged.",
            "Film side-on: camera facing your chest at leg lift, hip height, on a tripod.")
    elif m.get("camera_view") == "angled":
        add("note", "Camera at an angle",
            "Rotation is measured against your stride line so it still works, but it's less precise. "
            "Trunk tilt needs a side view, so it isn't measured or flagged.",
            "For the best numbers, film side-on: camera facing your chest at leg lift, hip height, tripod.")
    if rot_ok and m["shoulder_rotation_by_fs_deg"] > OBP["shoulder_open_deg"][P90]:
        add("leak", "Shoulders opening early",
            f"Shoulders turned ~{m['shoulder_rotation_by_fs_deg']:.0f}° toward the plate by foot strike. "
            f"OBP pitchers: middle half {obp_mid('shoulder_open_deg')}°, 9 in 10 under "
            f"{OBP['shoulder_open_deg'][P90]:.0f}°.",
            "Keep the glove side closed longer; hip-lead and rocker drills help delay trunk rotation.")
    if rot_ok and m["hip_shoulder_sep_max_deg"] < OBP["separation_deg"][P10]:
        add("leak", "Limited hip–shoulder separation",
            f"Max separation ~{max(0.0, m['hip_shoulder_sep_max_deg']):.0f}° (phone estimate, often reads low). "
            f"OBP pitchers: middle half {obp_mid('separation_deg')}°.",
            "Work thoracic mobility and let the hips lead before the shoulders turn.")
    if rot_ok and not m["sequence_in_order"] and fps >= T["min_fps_for_timing"]:
        add("leak", "Out-of-order sequence",
            "Peaks should go pelvis → trunk → arm; this clip's order was off "
            f"(ms from foot strike: {m['sequence_pelvis_trunk_arm_ms']}).",
            "Drills that separate hip and trunk rotation (e.g., hip-lead walk-ins) can help.")
    if m.get("camera_view", "side") == "side" and m["trunk_tilt_at_release_deg"] < OBP["trunk_tilt_deg"][P10]:
        add("watch", "Upright at release",
            f"Forward trunk tilt at release ~{m['trunk_tilt_at_release_deg']:.0f}°. "
            f"OBP pitchers: middle half {obp_mid('trunk_tilt_deg')}°.",
            "Finish over a firm front leg. Some of this is style, so compare against your best pitches.")
    if not [f for f in flags if f["level"] == "leak"]:
        add("good", "No major leaks flagged",
            "All checked metrics are within the range of OBP pitchers.",
            "Track these numbers over sessions alongside velo.")
    return flags


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def draw_pose(frame, p, color=(80, 220, 255)):
    for a, b in SKELETON:
        cv2.line(frame, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), color, 3, cv2.LINE_AA)
    for j in {j for pair in SKELETON for j in pair}:
        cv2.circle(frame, tuple(int(v) for v in p[j]), 5, (255, 255, 255), -1, cv2.LINE_AA)


def put_label(frame, text, y, scale=0.8, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (10, y - th - 8), (20 + tw, y + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render(video_path, P, fps, out_dir: Path, events=None, series=None) -> float:
    """Writes annotated.mp4 (+ keyframes.png when events are known). Returns playback fps."""
    ev = events or {}
    names = {ev[k]: label for k, label in
             (("leg_lift", "LEG LIFT"), ("foot_strike", "FOOT STRIKE"), ("release", "RELEASE")) if k in ev}
    hold = max(1, int(fps * 0.12))  # keep event label up briefly
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = min(fps, 60)  # slower playback for high-fps clips
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), out_fps, (w, h))
    keyframes = {}
    # Per-frame JPEGs power the frame picker in the web UI (phones can't scrub <video> reliably)
    frames_dir = out_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    fscale = min(1.0, 540 / max(1, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(P):
            break
        draw_pose(frame, P[i])
        if series:
            put_label(frame, f"Lead knee bend {series['knee_flex'][i]:5.0f} deg", 35)
            win = series.get("sep_window")
            if win and win[0] <= i <= win[1]:  # only from leg lift to just after foot strike
                put_label(frame, f"Hip-shoulder sep {series['sep'][i]:5.0f} deg", 70)
        txt = f"frame {i}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (w - tw - 22, h - 42), (w - 6, h - 12), (0, 0, 0), -1)
        cv2.putText(frame, txt, (w - tw - 14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        for f, name in names.items():
            if f <= i < f + hold:
                put_label(frame, name, h - 25, 1.1, (0, 255, 120))
            if i == f:
                keyframes[name] = frame.copy()
        writer.write(frame)
        small = cv2.resize(frame, None, fx=fscale, fy=fscale) if fscale < 1 else frame
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        i += 1
    cap.release()
    writer.release()

    tiles = [keyframes[k] for k in ("LEG LIFT", "FOOT STRIKE", "RELEASE") if k in keyframes]
    kf = out_dir / "keyframes.png"
    if tiles:
        th = 480
        tiles = [cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th)) for t in tiles]
        cv2.imwrite(str(kf), np.hstack(tiles))
    elif kf.exists():
        kf.unlink()
    return out_fps


# ---------------------------------------------------------------------------
# Target guides: your keyframe + a green "ghost" of the target position
# ---------------------------------------------------------------------------
# The green "target" in each guide is the OBP median (see OBP at the top).
GUIDE = {
    "trunk_tilt_deg": OBP["trunk_tilt_deg"][P50],
    "stride_pct": OBP["stride_pct"][P50],
    "separation_deg": OBP["separation_deg"][P50],
    "knee_change_deg": OBP["knee_change_deg"][P50],
}
# Team colours (BGR): navy backgrounds, red/gold accents
NAVY, NAVY2 = (44, 26, 18), (56, 33, 24)
GOOD, BAD, GHOST, INK = (152, 213, 62), (102, 90, 255), (120, 255, 120), (255, 255, 255)
HIP_C, SH_C = (255, 190, 90), (80, 170, 255)  # BGR: blue hips, orange shoulders (photo + top view)


def _dashed(img, p, q, color, th, dash=14):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(1, int(np.linalg.norm(q - p) / dash))
    for k in range(0, n, 2):
        a, b = p + (q - p) * k / n, p + (q - p) * min(k + 1, n) / n
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, th, cv2.LINE_AA)


def _pt(p):
    return tuple(int(v) for v in p)


def _caption(img, lines, colors):
    """Dark band under the image; text wraps to the image width."""
    font, scale, maxw = cv2.FONT_HERSHEY_SIMPLEX, 0.82, img.shape[1] - 28
    rows = []
    for t, c in zip(lines, colors):
        cur = ""
        for word in t.split():
            trial = (cur + " " + word).strip()
            if cur and cv2.getTextSize(trial, font, scale, 2)[0][0] > maxw:
                rows.append((cur, c))
                cur = word
            else:
                cur = trial
        rows.append((cur, c))
    band = np.full((38 * len(rows) + 18, img.shape[1], 3), NAVY, np.uint8)
    band[:4] = (53, 179, 245)  # gold stripe
    for k, (t, c) in enumerate(rows):
        cv2.putText(band, t, (14, 36 + 38 * k), font, scale, c, 2, cv2.LINE_AA)
    return np.vstack([img, band])


def _faint_pose(img, p, th):
    layer = img.copy()
    for a, b in SKELETON:
        cv2.line(layer, _pt(p[a]), _pt(p[b]), (230, 230, 230), max(1, th - 1), cv2.LINE_AA)
    cv2.addWeighted(layer, 0.45, img, 0.55, 0, dst=img)


def _label(img, text, at, color, scale=0.6, placed=None):
    scale *= 1.3
    """Text with a dark backing box, kept inside the image and nudged off earlier labels."""
    (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(min(max(6, at[0]), img.shape[1] - tw - 8))
    y = int(min(max(th_ + 8, at[1]), img.shape[0] - 8))
    if placed is not None:
        for _ in range(8):
            box = (x - 4, y - th_ - 6, x + tw + 4, y + 6)
            hit = next((b for b in placed if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])), None)
            if hit is None:
                break
            y = hit[3] + th_ + 10 if hit[3] + th_ + 16 < img.shape[0] else hit[1] - 8
        placed.append((x - 4, y - th_ - 6, x + tw + 4, y + 6))
    cv2.rectangle(img, (x - 4, y - th_ - 6), (x + tw + 4, y + 6), NAVY, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _rot_panel(height, you, target, you_txt, tgt_txt, you_ok):
    """Side panel: hips + shoulders seen from above, YOU on top and TYPICAL below, plate at the top.
    you/target = (hip_deg, shoulder_deg); 0 = closed (side-on to the plate), 90 = square.
    Each line gets an arrow showing which way it FACES (belt buckle for the hips, chest for the
    shoulders): closed = arrow points sideways, square = arrow points at the plate. Separation
    shows up as the hips arrow turned further toward the plate than the chest arrow."""
    W = int(height * 0.85)
    panel = np.full((height, W, 3), NAVY2, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.arrowedLine(panel, (W // 2, 50), (W // 2, 12), (230, 230, 230), 3, tipLength=0.3)
    cv2.putText(panel, "plate", (W // 2 + 12, 40), font, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(panel, "seen from above", (12, height - 14), font, 0.65, (160, 160, 160), 2, cv2.LINE_AA)
    r = min(W, height / 2) * 0.27
    # facing direction: rot 0 (closed) -> points right (side-on), rot 90 (square) -> points up at the plate
    face = lambda rot: np.array([np.cos(np.radians(rot)), -np.sin(np.radians(rot))])
    along = lambda rot: np.array([np.sin(np.radians(rot)), np.cos(np.radians(rot))])  # the line itself
    for k, ((hd, sd), txt, col) in enumerate(((you, you_txt, GOOD if you_ok else BAD),
                                              (target, tgt_txt, GHOST))):
        cy = int(height * (0.36 if k == 0 else 0.78))
        c = np.array([W / 2 - r * 0.35, cy + 4])
        cv2.putText(panel, txt, (12, cy - int(r) - 30), font, 0.85, col, 2, cv2.LINE_AA)
        # gap between where the hips and the chest face
        a0, a1 = sorted((-hd, -sd))
        if a1 - a0 > 2:
            cv2.ellipse(panel, _pt(c), (int(r * 0.55),) * 2, 0, a0, a1, (235, 235, 235), 2, cv2.LINE_AA)
        for rot, colr, half, alen, name in ((sd, SH_C, r, r * 1.25, "chest"), (hd, HIP_C, r * 0.75, r * 1.0, "hips")):
            cv2.line(panel, _pt(c - along(rot) * half), _pt(c + along(rot) * half), colr, 6, cv2.LINE_AA)
            tip = c + face(rot) * alen
            cv2.arrowedLine(panel, _pt(c), _pt(tip), colr, 4, cv2.LINE_AA, tipLength=0.22)
            lab = tip + face(rot) * 10 + (4, 8)
            if name == "hips" and abs(((hd - sd) + 180) % 360 - 180) < 30:
                lab = lab + (0, 26)  # arrows nearly overlap: drop the hips label below the chest label
            cv2.putText(panel, name, _pt(np.clip(lab, (4, 20), (W - 70, height - 40))), font, 0.7, colr, 2, cv2.LINE_AA)
        cv2.circle(panel, _pt(c), 9, (200, 200, 200), 2, cv2.LINE_AA)
    return panel


def _hip_shoulder_lines(img, p, th):
    """Thick blue hip line and orange shoulder line on the body (same colours as the top view).
    Returns their midpoints/ends so labels can be placed after cropping."""
    out = {}
    for name, (a, b), col in (("hips", (L_HIP, R_HIP), HIP_C), ("shoulders", (L_SH, R_SH), SH_C)):
        pa, pb = p[a], p[b]
        mid = (pa + pb) / 2
        half = (pb - pa) / 2
        # extend a little past the joints so a short (foreshortened) line is still visible
        ext = half * 1.25 if np.linalg.norm(half) > th * 3 else half + np.array([th * 3, 0])
        cv2.line(img, _pt(mid - ext), _pt(mid + ext), (20, 20, 20), th * 2 + 4, cv2.LINE_AA)
        cv2.line(img, _pt(mid - ext), _pt(mid + ext), col, th * 2, cv2.LINE_AA)
        for q in (pa, pb):
            cv2.circle(img, _pt(q), th + 2, INK, -1, cv2.LINE_AA)
        out[name] = (mid, mid + ext, col)
    return out


def _crop_map(img, pts, pad=0.3, out_h=520):
    """Crop around the given points (resized to out_h tall). Also returns a function
    mapping frame coords -> cropped coords, for placing labels."""
    pts = np.asarray([p for p in pts if np.all(np.isfinite(p))])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = img.shape[:2]
    ax0, ay0 = int(max(0, cx - side * 0.6)), int(max(0, cy - side / 2))
    ax1, ay1 = int(min(w, cx + side * 0.6)), int(min(h, cy + side / 2))
    c = img[ay0:ay1, ax0:ax1]
    s = out_h / c.shape[0]
    return cv2.resize(c, (int(c.shape[1] * s), out_h)), (lambda q: (np.asarray(q, float) - (ax0, ay0)) * s)


def _label_hip_sh(crop, f, lines_):
    """Label each line just past its outer end, so the text never covers the line itself."""
    placed = []
    for name in ("shoulders", "hips"):
        mid, end, col = lines_[name]
        m_, e_ = f(mid), f(end)
        tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * 1.3, 2)[0][0]
        if e_[0] >= m_[0]:
            at = e_ + (18, 10)            # line ends on the right: text to its right
        else:
            at = e_ + (-tw - 22, 10)      # line ends on the left: text to its left
        _label(crop, name, at, col, 0.55, placed)


def make_guides(video_path, result, out_dir: Path) -> list:
    """Writes guide_*.png and returns [{key, title, ok, file, lines}] (flagged first)."""
    P, S, ev, m = result["_P"], result["_series"], result["events"], result["metrics"]
    rhp, d, H = S["rhp"], S["direction"], S["height_px"]
    lead_hip, lead_knee, lead_ank = (L_HIP, L_KNEE, L_ANK) if rhp else (R_HIP, R_KNEE, R_ANK)
    back_ank = R_ANK if rhp else L_ANK
    fs, rel, sepf, lift = ev["foot_strike"], ev["release"], ev["max_separation"], ev["leg_lift"]
    flagged = {f["title"] for f in result["flags"] if f["level"] in ("leak", "watch")}
    th = max(3, int(H / 90))
    for old in out_dir.glob("guide_*.png"):
        old.unlink()

    want, frames, cap, i = {fs, rel, sepf}, {}, cv2.VideoCapture(str(video_path)), 0
    while len(frames) < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = fr
        i += 1
    cap.release()

    guides = []
    body = [0, 11, 12, 23, 24, 25, 26, 27, 28]

    def save(key, title, ok, img, lines):
        img = _caption(img, lines, [GOOD if ok else BAD, (200, 200, 200)])
        cv2.imwrite(str(out_dir / f"guide_{key}.png"), img)
        guides.append({"key": key, "title": title, "ok": ok, "file": f"guide_{key}.png", "lines": lines})

    rot_ok = m.get("rotation_reliable", True)  # side-on camera? (rotation + trunk need it)

    # 1) Trunk tilt at release (2D measurement, side view only)
    if m.get("camera_view", "side") == "side" and rel in frames:
        img, p = frames[rel].copy(), P[rel]
        _faint_pose(img, p, th)
        hip = (p[L_HIP] + p[R_HIP]) / 2
        sh = (p[L_SH] + p[R_SH]) / 2
        L = np.linalg.norm(sh - hip) * 1.15
        t = np.radians(GUIDE["trunk_tilt_deg"])
        ghost = hip + L * np.array([d * np.sin(t), -np.cos(t)])
        you_end = hip + (sh - hip) * 1.15
        ok = "Upright at release" not in flagged
        col = GOOD if ok else BAD
        _dashed(img, hip, hip + [0, -L], (180, 180, 180), max(1, th // 2))
        _dashed(img, hip, ghost, GHOST, th)
        cv2.line(img, _pt(hip), _pt(you_end), col, th + 1, cv2.LINE_AA)
        cv2.circle(img, _pt(hip), th + 2, INK, -1, cv2.LINE_AA)
        crop, f = _crop_map(img, list(p[body + [15, 16]]) + [ghost])
        placed = []
        _label(crop, f"target {GUIDE['trunk_tilt_deg']:.0f}", f(ghost) + (14 * d - (0 if d > 0 else 110), -10), GHOST, 0.6, placed)
        _label(crop, f"you {m['trunk_tilt_at_release_deg']:.0f}", f(you_end) + (-30, -14), col, 0.6, placed)
        _label(crop, "upright", f(hip + [0, -L]) + (-14 * d - (60 if d > 0 else 0), 6), (180, 180, 180), 0.45, placed)
        save("trunk", "Trunk tilt at release", ok, crop,
             [f"Forward trunk tilt: you {m['trunk_tilt_at_release_deg']:.0f} deg | OBP median {GUIDE['trunk_tilt_deg']:.0f} "
              f"(middle half {obp_mid('trunk_tilt_deg')})",
              "Lean your chest toward the plate over a firm front leg."])

    # 2) Stride length at foot strike (target drawn along your actual stride line)
    if fs in frames:
        img, p = frames[fs].copy(), P[fs]
        _faint_pose(img, p, th)
        back, lead = p[back_ank], p[lead_ank]
        ratio = GUIDE["stride_pct"] / max(1e-6, m["stride_pct_height"])
        target = back + (lead - back) * ratio
        ok = "Short stride" not in flagged
        col = GOOD if ok else BAD
        u = (lead - back) / (np.linalg.norm(lead - back) + 1e-9)
        nrm = np.array([-u[1], u[0]]) * th * 4
        cv2.line(img, _pt(back), _pt(lead), col, th + 1, cv2.LINE_AA)
        _dashed(img, lead, target, GHOST, th)
        for q, c_ in ((back, INK), (lead, col), (target, GHOST)):
            cv2.line(img, _pt(q - nrm), _pt(q + nrm), c_, th, cv2.LINE_AA)
        crop, f = _crop_map(img, list(p[body]) + [target])
        placed = []
        _label(crop, "you", f(lead) + (-14, -22), col, 0.55, placed)
        _label(crop, "target", f(target) + (-24, 34), GHOST, 0.55, placed)
        _label(crop, "back foot", f(back) + (-40, 34), INK, 0.5, placed)
        save("stride", "Stride length at foot strike", ok, crop,
             [f"Stride: you {m['stride_pct_height']:.0f}% of your height | OBP median {GUIDE['stride_pct']:.0f}% "
              f"(middle half {obp_mid('stride_pct')}%)",
              "Green tick = where your front foot would land at target length."])

    # 3) Front knee block: knee should hold or straighten from foot strike to release
    if rel in frames:
        img, p = frames[rel].copy(), P[rel]
        _faint_pose(img, p, th)
        hip, knee, ank = p[lead_hip], p[lead_knee], p[lead_ank]
        # OBP median: the knee straightens ~10 deg from foot strike to release
        tgt_flex = max(0.0, m["lead_knee_flex_at_fs_deg"] + GUIDE["knee_change_deg"])
        thigh = np.linalg.norm(hip - knee)
        shin = (ank - knee) / (np.linalg.norm(ank - knee) + 1e-9)
        best = None
        for sgn in (1, -1):  # rotate thigh about the knee to the target bend; pick the sensible side
            a = np.radians(180 - tgt_flex) * sgn
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            g = knee + rot @ shin * thigh
            if best is None or np.linalg.norm(g - hip) < np.linalg.norm(best - hip):
                best = g
        ok = "Front knee collapsing" not in flagged
        col = GOOD if ok else BAD
        _dashed(img, knee, best, GHOST, th)
        cv2.line(img, _pt(hip), _pt(knee), col, th + 1, cv2.LINE_AA)
        cv2.line(img, _pt(knee), _pt(ank), col, th + 1, cv2.LINE_AA)
        for q in (hip, knee, ank):
            cv2.circle(img, _pt(q), th + 1, INK, -1, cv2.LINE_AA)
        crop, f = _crop_map(img, list(p[body]) + [best])
        _label(crop, "you", f(hip) + (12, 0), col, 0.55)
        if np.linalg.norm(best - hip) > th * 4:
            _label(crop, "firm", f(best) + (12, 0), GHOST, 0.55)
        chg = m["lead_knee_change_deg"]
        save("knee", "Front knee block", ok, crop,
             [f"Front knee bend: {m['lead_knee_flex_at_fs_deg']:.0f} deg at foot strike -> "
              f"{m['lead_knee_flex_at_release_deg']:.0f} deg at release ({chg:+.0f})",
              f"Green = OBP median: the front leg straightens ~{-GUIDE['knee_change_deg']:.0f} deg after landing."])


    # 4) Hip-shoulder separation at its max (around foot strike)
    if rot_ok and sepf in frames:
        img, p = frames[sepf].copy(), P[sepf]
        _faint_pose(img, p, th)
        lines_ = _hip_shoulder_lines(img, p, th)
        crop, f = _crop_map(img, list(p[body]))
        _label_hip_sh(crop, f, lines_)
        hr, sr = float(S["hip_rot"][sepf]), float(S["sh_rot"][sepf])
        sep = max(0.0, m["hip_shoulder_sep_max_deg"])
        ok = "Limited hip–shoulder separation" not in flagged
        panel = _rot_panel(crop.shape[0], (hr, sr), (hr, hr - GUIDE["separation_deg"]),
                           f"YOU: {sep:.0f} deg apart", f"TYPICAL: ~{GUIDE['separation_deg']:.0f} deg apart", ok)
        save("separation", "Hip–shoulder separation", ok, np.hstack([crop, panel]),
             [f"Separation: you {sep:.0f} deg | OBP median {GUIDE['separation_deg']:.0f} "
              f"(middle half {obp_mid('separation_deg')}; phone estimate)",
              "Arrows = which way your belt buckle (blue) and chest (orange) face. Hips turn to the plate first."])

    # 5) Hips vs shoulders at foot strike: hips should already be opening, shoulders still closed
    if rot_ok and fs in frames:
        img, p = frames[fs].copy(), P[fs]
        _faint_pose(img, p, th)
        lines_ = _hip_shoulder_lines(img, p, th)
        crop, f = _crop_map(img, list(p[body]))
        _label_hip_sh(crop, f, lines_)
        hr, sr = float(S["hip_rot"][fs]), float(S["sh_rot"][fs])
        oh, osh = OBP["hips_at_fs_deg"][P50], OBP["shoulders_at_fs_deg"][P50]
        ok = "Shoulders opening early" not in flagged
        panel = _rot_panel(crop.shape[0], (hr, sr), (oh, osh),
                           "YOU at foot strike", "TYPICAL (OBP median)", ok)
        save("shoulders", "Hips & shoulders at foot strike", ok, np.hstack([crop, panel]),
             [f"At foot strike: your hips {hr:.0f} deg open, shoulders {sr:.0f} deg | "
              f"OBP median: hips {oh:.0f}, shoulders {osh:.0f} (0 = fully closed; phone estimate)",
              "Arrows = which way your belt buckle (blue) and chest (orange) face. Hips open first, chest stays closed."])

    guides.sort(key=lambda g: g["ok"])  # flagged first
    return guides


UNMEASURED_ROTATION = ("hip_shoulder_sep_max_deg", "shoulder_rotation_by_fs_deg", "sequence_in_order")
UNMEASURED_SIDE_ONLY = ("trunk_tilt_at_release_deg", "arm_angle_at_release_deg")


def unmeasured(m: dict) -> set:
    """Metrics this clip's camera view can't measure (shown as n/a, kept out of history)."""
    out = set()
    if not m.get("rotation_reliable", True):
        out |= set(UNMEASURED_ROTATION)
    if m.get("camera_view", "side") != "side":
        out |= set(UNMEASURED_SIDE_ONLY)
    return out


def write_reports(result, video_path, out_dir: Path, velo):
    m, flags = result["metrics"], result["flags"]
    na = unmeasured(m)
    v = lambda k, unit="°": "n/a (camera view)" if k in na else f"{m[k]}{unit}"
    icons = {"leak": "[LEAK]", "watch": "[WATCH]", "note": "[NOTE]", "good": "[OK]"}
    lines = [
        f"Pitch Helper report — {Path(video_path).name}",
        f"{datetime.now():%Y-%m-%d %H:%M}  |  throws {result['throws']}  |  {result['fps']:.0f} fps"
        + (f"  |  velo {velo} mph" if velo else ""),
        "",
        f"METRICS (phone estimates — compare against your own baseline)   camera view: {m.get('camera_view', 'side')}",
        f"  Stride length ............... {m['stride_pct_height']}% of height",
        f"  Lead knee bend @ foot strike  {m['lead_knee_flex_at_fs_deg']}°",
        f"  Lead knee bend @ release .... {m['lead_knee_flex_at_release_deg']}°  "
        f"(change {m['lead_knee_change_deg']:+}°, negative = bracing)",
        f"  Max hip–shoulder separation . {v('hip_shoulder_sep_max_deg')}",
        f"  Shoulder turn by FS ......... {v('shoulder_rotation_by_fs_deg')}",
        f"  Trunk tilt @ release ........ {v('trunk_tilt_at_release_deg')}",
        f"  Arm slot @ release .......... {v('arm_angle_at_release_deg')}",
        f"  Foot strike → release ....... {m['fs_to_release_ms']} ms",
        f"  Peak timing (pelvis/trunk/arm, ms after FS): "
        + ("n/a (camera view)" if "sequence_in_order" in na else
           f"{m['sequence_pelvis_trunk_arm_ms']}  {'in order' if m['sequence_in_order'] else 'OUT OF ORDER'}"),
        "",
        "FLAGS",
    ]
    for f in flags:
        lines += [f"  {icons[f['level']]} {f['title']}: {f['detail']}", f"      → {f['tip']}"]
    lines += ["", OBP_CREDIT]
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    public = {k: v for k, v in result.items() if not k.startswith("_")}
    public.update({"video": str(video_path), "velo_mph": velo, "analyzed_at": datetime.now().isoformat()})
    (out_dir / "report.json").write_text(json.dumps(public, indent=2), encoding="utf-8")

    hist = out_dir.parent / "history.csv"
    row = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "clip": Path(video_path).name,
           "velo_mph": velo or ""}
    row.update({k: ("" if k in na else val) for k, val in m.items() if not isinstance(val, list)})
    row["leaks"] = "; ".join(f["title"] for f in flags if f["level"] == "leak")
    rows = []
    if hist.exists():  # re-analyzing a clip replaces its old row
        with open(hist, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("clip") != row["clip"]]
    rows.append(row)
    with open(hist, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return "\n".join(lines)


def run_one(video, throws, fps, model, out_root: Path, velo=None, events=None):
    out_dir = out_root / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {video} at {fps:.0f} fps...")
    pts, world, vis = load_or_extract(str(video), fps, model, out_dir)
    try:
        result = analyze(pts, world, vis, fps, throws, events)
    except EventError as e:
        render(str(video), prepare(pts, world, fps)[0], fps, out_dir)
        print(f"Could not find the pitch events: {e}\n"
              f"Open {out_dir / 'annotated.mp4'}, note the frame numbers, then re-run on that clip with\n"
              f"  --events LEG_LIFT,FOOT_STRIKE,RELEASE")
        return False
    except ValueError as e:
        print(f"Could not analyze: {e}")
        return False
    render(str(video), result["_P"], fps, out_dir, result["events"], result["_series"])
    make_guides(str(video), result, out_dir)
    print(write_reports(result, str(video), out_dir, velo))
    print(f"\nSaved to {out_dir}/")
    return True


def main():
    ap = argparse.ArgumentParser(description="Analyze pitches filmed side-on.")
    ap.add_argument("video")
    ap.add_argument("--throws", choices=["R", "L", "r", "l"], required=True, help="Throwing hand")
    ap.add_argument("--fps", type=float, help="Override capture fps (e.g. 240 for slow-mo exports)")
    ap.add_argument("--velo", type=float, help="Radar velocity in mph, logged to history")
    ap.add_argument("--model", choices=list(MODEL_URLS), default="heavy", help="Pose model size")
    ap.add_argument("--out", default="pitch_output", help="Output folder")
    ap.add_argument("--events", help="Manual frames 'leg_lift,foot_strike,release' (see frame numbers "
                                     "in annotated.mp4). Reuses saved pose data, so it's fast.")
    ap.add_argument("--split", action="store_true",
                    help="The video has several pitches: find them, save each as its own clip, analyze each")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"File not found: {args.video}")
    cap = cv2.VideoCapture(args.video)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    out_root = Path(args.out)
    throws = args.throws.upper()

    if not args.split:
        events = [int(x) for x in args.events.split(",")] if args.events else None
        if not run_one(args.video, throws, fps, args.model, out_root, args.velo, events):
            sys.exit(1)
        return
    if args.events:
        sys.exit("--events works on one clip. Run it on one of the split clips instead.")
    clip_dir = out_root / (Path(args.video).stem + "_clips")
    print(f"Finding pitches in {args.video}...")
    last = [0]
    def prog(done, total):
        if total and done - last[0] >= total / 20:
            last[0] = done
            print(f"  scan: {100 * done // total}%", end="\r")
    clips = split_video(args.video, fps, clip_dir, prog)
    if not clips:
        sys.exit("No pitches found. Check the whole body is in frame and side-on"
                 + ("" if args.fps else ", and set --fps if this is slow-mo saved at 30 fps") + ".")
    print(f"Found {len(clips)} pitch(es): " + ", ".join(f"{p['t']:.1f}s" for _, p in clips)
          + f"\nClips saved to {clip_dir}/ (delete any that aren't real pitches and their output folders).")
    for path, _ in clips:
        run_one(path, throws, fps, args.model, out_root, args.velo)


def load_or_extract(video, fps, model, out_dir: Path, on_frame=None):
    """Pose data is cached as pose.npz so re-analysis (e.g. fixing events) is instant."""
    cache = out_dir / "pose.npz"
    if cache.exists() and cache.stat().st_mtime >= os.path.getmtime(video):
        d = np.load(cache)
        return d["pts"], d["world"], d["vis"]
    pts, world, vis = extract_pose(video, fps, model, on_frame)
    np.savez_compressed(cache, pts=pts, world=world, vis=vis)
    return pts, world, vis


if __name__ == "__main__":
    main()
