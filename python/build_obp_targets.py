#!/usr/bin/env python3
"""
Rebuild the OBP-based targets used by pitch_helper.py.

Data: Driveline OpenBiomechanics Project, baseball_pitching/data/poi/poi_metrics.csv
      https://github.com/drivelineresearch/openbiomechanics  (CC BY-NC-SA 4.0, non-commercial)

    git clone https://github.com/drivelineresearch/openbiomechanics
    python build_obp_targets.py openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv

Prints a dict to paste over OBP in pitch_helper.py. Each metric maps to its
10th / 25th / 50th / 75th / 90th percentiles, converted to Pitch Helper's conventions.
"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1])
conv = {
    # pitch_helper metric      : OBP column(s), converted
    "stride_pct":       d["stride_length"] * 100,                         # fraction -> % height
    "trunk_tilt_deg":   d["torso_anterior_tilt_br"],                      # trunk flexion at release
    "separation_deg":   d["max_rotation_hip_shoulder_separation"],        # peak hip-shoulder sep
    "shoulder_open_deg": d["torso_rotation_fp"] - d["torso_rotation_min"],# turn from most-closed to foot plant
    "knee_change_deg":  -d["lead_knee_extension_from_fp_to_br"],          # + = knee bends more (OBP: + = extends)
    "hips_at_fs_deg":   d["pelvis_rotation_fp"],                          # 0 = closed, 90 = square to plate
    "shoulders_at_fs_deg": d["torso_rotation_fp"],
}
out = {k: [round(float(v.quantile(q)), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] for k, v in conv.items()}
print(f"# {len(d)} pitches, {d['session'].nunique()} sessions, "
      f"{d['pitch_speed_mph'].min():.0f}-{d['pitch_speed_mph'].max():.0f} mph")
print("OBP = {")
for k, v in out.items():
    print(f'    "{k}": {v},')
print("}")
