// Pitch Helper — pitching mechanics analysis (browser version).
// A direct port of pitch_helper.py: same side-fixing, event detection, rotation, metrics and flags.
// Pose arrays: pts[t][j] = [x, y] pixels, world[t][j] = [x, y, z] metres, vis[t][j] = 0..1.
// Missing detections are NaN.

// ---------------------------------------------------------------------------
// Targets from Driveline's OpenBiomechanics Project (OBP): 411 pitches from 100 pitchers,
// 70–94 mph, mostly college. [10th, 25th, 50th, 75th, 90th] percentiles.
// Data © Driveline Baseball, CC BY-NC-SA 4.0 (non-commercial):
// https://github.com/drivelineresearch/openbiomechanics
// ---------------------------------------------------------------------------
export const OBP = {
  stride_pct: [76.0, 79.9, 83.7, 87.7, 90.4],
  trunk_tilt_deg: [20.8, 27.2, 34.7, 44.1, 48.2],
  separation_deg: [23.4, 27.6, 32.3, 36.7, 40.8],
  shoulder_open_deg: [26.5, 33.7, 41.7, 52.1, 62.8],
  knee_change_deg: [-25.4, -17.5, -10.0, -1.9, 2.2],
  hips_at_fs_deg: [20.5, 27.1, 34.3, 42.1, 49.6],
  shoulders_at_fs_deg: [-11.3, -2.7, 5.5, 12.3, 19.2],
};
const P10 = 0, P25 = 1, P50 = 2, P75 = 3, P90 = 4;
export const OBP_CREDIT = "Targets: Driveline OpenBiomechanics Project (CC BY-NC-SA 4.0)";
const obpMid = k => `${OBP[k][P25].toFixed(0)}–${OBP[k][P75].toFixed(0)}`;
const TARGETS = { min_fps_for_timing: 100, knee_noise_deg: 5.0 };

// MediaPipe landmark indices (person's own left/right)
export const NOSE = 0;
export const L_SH = 11, R_SH = 12, L_EL = 13, R_EL = 14, L_WR = 15, R_WR = 16;
export const L_HIP = 23, R_HIP = 24, L_KNEE = 25, R_KNEE = 26, L_ANK = 27, R_ANK = 28;
export const SKELETON = [
  [L_SH, R_SH], [L_SH, L_EL], [L_EL, L_WR], [R_SH, R_EL], [R_EL, R_WR],
  [L_SH, L_HIP], [R_SH, R_HIP], [L_HIP, R_HIP],
  [L_HIP, L_KNEE], [L_KNEE, L_ANK], [R_HIP, R_KNEE], [R_KNEE, R_ANK],
];
const ARM_PAIRS = [[L_SH, R_SH], [L_EL, R_EL], [L_WR, R_WR], [17, 18], [19, 20], [21, 22]];
const LEG_PAIRS = [[L_HIP, R_HIP], [L_KNEE, R_KNEE], [L_ANK, R_ANK], [29, 30], [31, 32]];
const ALL_PAIRS = ARM_PAIRS.concat(LEG_PAIRS);

export class EventError extends Error { constructor(m) { super(m); this.name = "EventError"; } }
export class InputError extends Error { constructor(m) { super(m); this.name = "InputError"; } }

// ---------------------------------------------------------------------------
// Small numeric helpers (numpy equivalents)
// ---------------------------------------------------------------------------
const fin = Number.isFinite;
const deg = r => r * 180 / Math.PI, rad = d => d * Math.PI / 180;
const sign = x => (x > 0 ? 1 : x < 0 ? -1 : 0);
const pmod = (a, b) => ((a % b) + b) % b;                // Python-style modulo
const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const mul = (a, s) => a.map(v => v * s);
const norm = a => Math.hypot(...a);
const dist = (a, b) => norm(sub(a, b));
const mid = (a, b) => a.map((v, i) => (v + b[i]) / 2);
const clone = arr => arr.map(f => f.map(p => p.slice()));

export function percentile(values, q) {
  const s = values.filter(fin).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const pos = (s.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
const median = v => percentile(v, 50);
function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k] || (!fin(a[k]) && fin(a[i]))) k = i; return k; }
function argmin(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[k] || (!fin(a[k]) && fin(a[i]))) k = i; return k; }
function gradient(a) {
  const n = a.length, g = new Array(n);
  if (n < 2) return a.map(() => 0);
  g[0] = a[1] - a[0]; g[n - 1] = a[n - 1] - a[n - 2];
  for (let i = 1; i < n - 1; i++) g[i] = (a[i + 1] - a[i - 1]) / 2;
  return g;
}
function unwrapDeg(d) {           // numpy.unwrap on degrees
  const out = d.slice();
  let corr = 0;
  for (let i = 1; i < d.length; i++) {
    const dd = d[i] - d[i - 1];
    let m = pmod(dd + 180, 360) - 180;
    if (m === -180 && dd > 0) m = 180;
    if (Math.abs(dd) >= 180) corr += m - dd;
    out[i] = d[i] + corr;
  }
  return out;
}
const series = (A, j, k) => A.map(f => f[j][k]);
const speed = (A, j) => {           // |gradient| of a joint's track, per frame
  const K = A[0][j].length, g = range(0, K).map(k => gradient(series(A, j, k)));
  return A.map((_, t) => Math.hypot(...g.map(c => c[t])));
};

/** Linear-interpolate NaNs along time, then centered moving average (like pitch_helper.fill_and_smooth). */
export function fillSmooth(arr, win) {
  const out = clone(arr), n = out.length, J = out[0].length, K = out[0][0].length;
  for (let j = 0; j < J; j++) for (let k = 0; k < K; k++) {
    const col = out.map(f => f[j][k]);
    const good = range(0, n).filter(t => fin(col[t]));
    if (good.length >= 2) {
      let gi = 0;
      for (let t = 0; t < n; t++) {
        if (fin(col[t])) continue;
        if (t < good[0]) col[t] = col[good[0]];
        else if (t > good[good.length - 1]) col[t] = col[good[good.length - 1]];
        else {
          while (good[gi + 1] < t) gi++;
          const a = good[gi], b = good[gi + 1];
          col[t] = col[a] + (col[b] - col[a]) * (t - a) / (b - a);
        }
      }
    } else if (good.length === 1) col.fill(col[good[0]]);
    if (win > 1 && good.length) {
      const pad = Math.floor(win / 2), padded = [];
      for (let t = -pad; t < n + pad; t++) padded.push(col[Math.min(n - 1, Math.max(0, t))]);
      for (let t = 0; t < n; t++) { let s = 0; for (let w = 0; w < win; w++) s += padded[t + w]; col[t] = s / win; }
    }
    for (let t = 0; t < n; t++) out[t][j][k] = col[t];
  }
  return out;
}

function angleAt(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm(v1) * norm(v2) + 1e-9);
  return deg(Math.acos(Math.max(-1, Math.min(1, cos))));
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
export function prepare(pts, world, fps) {
  const win = Math.max(3, Math.round(fps * 0.025) | 1);
  return [fillSmooth(pts, win), fillSmooth(world, win)];
}

export function bodyHeightPx(P) {
  return percentile(P.map(f => Math.max(f[L_ANK][1], f[R_ANK][1]) - f[NOSE][1]), 90) * 1.1;
}

function findCameraCut(pts, H) {
  const hip = pts.map(f => mid(f[L_HIP], f[R_HIP]));
  let best = 0, bi = 0;
  for (let t = 0; t < hip.length - 1; t++) {
    const j = dist(hip[t + 1], hip[t]);
    if (fin(j) && j > best) { best = j; bi = t; }
  }
  return best > 0.25 * H ? bi + 1 : null;
}

/** +1 if the pitch goes toward image right, else -1 (hand in front of hips at peak hand speed). */
function throwDirection(P, wr) {
  const n = P.length, hipX = P.map(f => (f[L_HIP][0] + f[R_HIP][0]) / 2), H = bodyHeightPx(P);
  const sp = range(0, n - 1).map(t => dist(P[t + 1][wr], P[t][wr]));
  const k = argmax(sp);
  const win = range(Math.max(0, k - 2), Math.min(n, k + 3));
  let votes = 0;
  const ahead = median(win.map(t => P[t][wr][0] - hipX[t]));
  if (Math.abs(ahead) > 0.08 * H) votes += 2 * sign(ahead);
  votes += sign(P[Math.min(k + 1, n - 1)][wr][0] - P[k][wr][0]);
  const q = Math.max(2, Math.floor(n / 4));
  const travel = median(hipX.slice(-q)) - median(hipX.slice(0, q));
  if (Math.abs(travel) > 0.3 * H) votes += sign(travel);
  return sign(votes) || 1;
}

/** Fix MediaPipe left/right mix-ups; whole body relabelled together so the skeleton never crosses. */
function fixSides(pts, world, vis, rhp, swapSides, flipDir) {
  pts = clone(pts); world = clone(world); vis = vis.map(v => v.slice());
  const swap = (t, pairs) => {
    for (const [a, b] of pairs) {
      [pts[t][a], pts[t][b]] = [pts[t][b], pts[t][a]];
      [world[t][a], world[t][b]] = [world[t][b], world[t][a]];
      [vis[t][a], vis[t][b]] = [vis[t][b], vis[t][a]];
    }
  };
  const n = pts.length;
  // 1) uncross the torso frame by frame, in 3D
  for (let t = 0; t < n; t++) {
    const W = world[t];
    const ok = dist(W[L_SH], W[L_HIP]) + dist(W[R_SH], W[R_HIP]);
    const x = dist(W[L_SH], W[R_HIP]) + dist(W[R_SH], W[L_HIP]);
    if (x < ok * 0.9) swap(t, ARM_PAIRS);
  }
  // 2) frame-to-frame continuity
  const main = [[L_SH, R_SH], [L_HIP, R_HIP], [L_WR, R_WR], [L_KNEE, R_KNEE], [L_ANK, R_ANK]];
  let prev = null;
  for (let t = 0; t < n; t++) {
    let cur = main.map(([a, b]) => [pts[t][a].slice(), pts[t][b].slice()]);
    if (cur.some(p => p.some(q => q.some(v => !fin(v))))) continue;
    if (prev) {
      let keep = 0, flip = 0;
      cur.forEach((p, i) => {
        keep += dist(p[0], prev[i][0]) + dist(p[1], prev[i][1]);
        flip += dist(p[1], prev[i][0]) + dist(p[0], prev[i][1]);
      });
      if (flip < keep * 0.8) { swap(t, ALL_PAIRS); cur = cur.map(p => [p[1], p[0]]); }
    }
    prev = cur;
  }
  // 3) throwing arm = fastest wrist
  let P = fillSmooth(pts, 3);
  const sp = j => percentile(range(0, n - 1).map(t => dist(P[t + 1][j], P[t][j])), 97);
  const rightThrows = sp(R_WR) >= sp(L_WR);
  if ((rightThrows !== rhp) !== !!swapSides) {
    for (let t = 0; t < n; t++) swap(t, ALL_PAIRS);
    P = fillSmooth(pts, 3);
  }
  let d = throwDirection(P, rhp ? R_WR : L_WR);
  if (flipDir) d = -d;
  return { pts, world, vis, d };
}

function bodyHeightWorld(W) {
  const vals = W.map(f => {
    const shin = (dist(f[L_KNEE], f[L_ANK]) + dist(f[R_KNEE], f[R_ANK])) / 2;
    const thigh = (dist(f[L_HIP], f[L_KNEE]) + dist(f[R_HIP], f[R_KNEE])) / 2;
    const mh = mid(f[L_HIP], f[R_HIP]), ms = mid(f[L_SH], f[R_SH]);
    return shin + thigh + dist(ms, mh) + dist(f[NOSE], ms);
  });
  return median(vals) / 0.897;
}

function detectEvents(P, fps, H, rhp, d) {
  const n = P.length;
  const [leadAnk, backAnk] = rhp ? [L_ANK, R_ANK] : [R_ANK, L_ANK];
  const throwWr = rhp ? R_WR : L_WR;
  // front foot relative to the back foot, so a panning camera can't hide the stride
  const fwd = P.map(f => d * (f[leadAnk][0] - f[backAnk][0]));
  const ext = argmax(fwd);
  const start = Math.min(...fwd.slice(0, ext + 1));
  const travel = fwd[ext] - start;
  if (travel < 0.3 * H) throw new EventError(
    "Couldn't find a stride: the front foot never got far ahead of the back foot. The clip probably " +
    "stops before foot strike, or the camera is behind/in front of the pitcher.");
  const reach = fwd.findIndex(v => v >= start + 0.85 * travel);
  // foot strike = first contact: lowest point of heel/toe/ankle reaches its planted level
  const [heel, toe] = rhp ? [29, 31] : [30, 32];
  const low = P.map(f => Math.max(f[leadAnk][1], f[heel][1], f[toe][1]));
  const ground = median(low.slice(ext, Math.min(n, ext + Math.max(3, Math.floor(0.1 * fps)))));
  let fs = ext;
  for (let i = reach; i < n; i++) if (low[i] >= ground - 0.012 * H) { fs = i; break; }
  const knees = P.slice(0, fs + 1).map(f => Math.min(f[L_KNEE][1], f[R_KNEE][1]));
  const lift = argmin(knees);
  if (lift >= fs || (fs - lift) < Math.max(2, 0.1 * fps))
    throw new EventError("Couldn't separate leg lift from foot strike. Make sure the clip starts before the leg lift.");
  if (n - fs < Math.max(3, Math.floor(0.08 * fps)))
    throw new EventError("Clip ends too soon after foot strike. Include the follow-through.");
  const wr = speed(P, throwWr);
  const end = Math.min(n, fs + Math.floor(0.35 * fps) + 1);
  const rel = fs + 1 + argmax(wr.slice(fs + 1, end));
  return [lift, fs, rel];
}

/** Hip & shoulder rotation vs the throwing line (back ankle -> front ankle at foot strike), in 3D. */
function rotationsVsStride(W, fs, rel, rhp) {
  const [lead, thr] = rhp ? [0, 1] : [1, 0];
  const ang = v => deg(Math.atan2(v[2], v[0]));
  const anks = [L_ANK, R_ANK], hips = [L_HIP, R_HIP], shs = [L_SH, R_SH];
  const axis = ang(sub(W[fs][anks[lead]], W[fs][anks[thr]]));
  const turn = pair => {
    let r = unwrapDeg(W.map(f => ang(sub(f[pair[lead]], f[pair[thr]])) - axis));
    const off = 360 * Math.round(r[fs] / 360);
    return r.map(v => v - off);
  };
  let hip = turn(hips), sh = turn(shs);
  if (sh[rel] - sh[fs] < 0) { hip = hip.map(v => -v); sh = sh.map(v => -v); }
  return { hip, sh, axis };
}

export function analyze(pts, world, vis, fps, throws, opts = {}) {
  const { events = null, swapSides = false, flipDir = false } = opts;
  const n = pts.length;
  if (n < 10) throw new InputError("Clip too short — need at least 10 frames.");
  const detected = pts.filter(f => fin(f[0][0])).length / n;
  if (detected < 0.5) throw new InputError("Pitcher detected in under half the frames. Check lighting/framing.");

  const rhp = throws.toUpperCase() === "R";
  const fx = fixSides(pts, world, vis, rhp, swapSides, flipDir);
  const direction = fx.d;
  const [P, W] = prepare(fx.pts, fx.world, fps);
  const [leadHip, leadKnee, leadAnk] = rhp ? [L_HIP, L_KNEE, L_ANK] : [R_HIP, R_KNEE, R_ANK];
  const backAnk = rhp ? R_ANK : L_ANK;
  const [throwSh, throwWr] = rhp ? [R_SH, R_WR] : [L_SH, L_WR];
  const H = bodyHeightPx(P);

  let lift, fs, rel;
  const manual = !!events;
  if (manual) {
    [lift, fs, rel] = events.map(Number);
    if (!(0 <= lift && lift < fs && fs < rel && rel < n))
      throw new EventError("Events must be in order: leg lift < foot strike < release.");
  } else {
    const cut = findCameraCut(fx.pts, H);
    if (cut !== null) throw new EventError(`Camera cut or jump around frame ${cut}. Use a fixed camera, or mark the events by hand.`);
    [lift, fs, rel] = detectEvents(P, fps, H, rhp, direction);
  }

  const wrSpeed = speed(P, throwWr).map(v => v * fps);

  // lower body
  const gap = sub(W[fs][leadAnk], W[fs][backAnk]);
  const hM = bodyHeightWorld(W);
  const stride3d = hM > 0.5 ? Math.hypot(gap[0], gap[2]) / hM * 100 : NaN;
  const stride2d = dist(P[fs][leadAnk], P[fs][backAnk]) / H * 100;
  const stridePct = fin(stride3d) && stride3d > 20 && stride3d < 140 ? stride3d : stride2d;
  const kneeFlex = P.map(f => 180 - angleAt(f[leadHip], f[leadKnee], f[leadAnk]));
  const kneeFs = kneeFlex[fs], kneeRel = kneeFlex[rel];

  // trunk (2D)
  const trunkTilt = P.map(f => {
    const tv = sub(mid(f[L_SH], f[R_SH]), mid(f[L_HIP], f[R_HIP]));
    return deg(Math.atan2(direction * tv[0], -tv[1]));
  });

  // rotation vs stride line
  const { hip: hipRot, sh: shRot, axis } = rotationsVsStride(W, fs, rel, rhp);
  const sep = hipRot.map((v, t) => v - shRot[t]);
  const off = Math.abs(pmod(axis + 90, 180) - 90);
  const view = off < 30 ? "side" : off < 60 ? "angled" : "end-on";
  const rotReliable = view !== "end-on";
  const a = Math.max(lift, fs - Math.floor(0.05 * fps));
  const b = Math.max(a, Math.min(rel - 1, fs + Math.floor(0.12 * fps)));
  const sepWin = sep.slice(a, b + 1);
  const sepMax = Math.max(...sepWin), sepMaxFrame = a + argmax(sepWin);
  const shOpen = shRot[fs] - Math.min(...shRot.slice(lift, fs + 1));

  // kinematic sequence
  const pelVel = gradient(hipRot).map(v => v * fps), trkVel = gradient(shRot).map(v => v * fps);
  const seqEnd = Math.min(n, rel + Math.floor(0.1 * fps) + 1);
  const tPel = lift + argmax(pelVel.slice(lift, seqEnd));
  const tTrk = lift + argmax(trkVel.slice(lift, seqEnd));
  const tArm = lift + argmax(wrSpeed.slice(lift, seqEnd));
  const seqOk = tPel <= tTrk + 1 && tTrk <= tArm + 1;

  // arm slot (2D)
  const armV = sub(P[rel][throwWr], P[rel][throwSh]);
  const armAngle = deg(Math.atan2(-armV[1], Math.abs(armV[0])));

  const key = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, throwWr];
  let cs = 0, cn = 0;
  for (const f of fx.vis) for (const j of key) { cs += f[j]; cn++; }
  const confidence = cs / cn;
  const ms = fr => Math.round(fr * 1000 / fps);
  const r1 = v => Math.round(v * 10) / 10;

  const metrics = {
    stride_pct_height: r1(stridePct),
    lead_knee_flex_at_fs_deg: r1(kneeFs),
    lead_knee_flex_at_release_deg: r1(kneeRel),
    lead_knee_change_deg: r1(kneeRel - kneeFs),
    hip_shoulder_sep_max_deg: r1(sepMax),
    shoulder_rotation_by_fs_deg: r1(shOpen),
    trunk_tilt_at_release_deg: r1(trunkTilt[rel]),
    arm_angle_at_release_deg: r1(armAngle),
    fs_to_release_ms: ms(rel - fs),
    sequence_pelvis_trunk_arm_ms: [ms(tPel - fs), ms(tTrk - fs), ms(tArm - fs)],
    sequence_in_order: seqOk,
    rotation_reliable: rotReliable,
    camera_view: view,
  };
  const ev = { leg_lift: lift, foot_strike: fs, release: rel, max_separation: sepMaxFrame };
  return {
    metrics, events: ev, flags: buildFlags(metrics, fps, confidence), fps,
    confidence: Math.round(confidence * 100) / 100, throws: throws.toUpperCase(),
    events_source: manual ? "manual" : "auto",
    P, series: { kneeFlex, sep, trunkTilt, hipRot, shRot, direction, H, rhp,
                 sepWindow: rotReliable ? [lift, b] : null },
  };
}

export function buildFlags(m, fps, conf) {
  const rotOk = m.rotation_reliable !== false, flags = [], T = TARGETS;
  const add = (level, title, detail, tip) => flags.push({ level, title, detail, tip });
  const f0 = v => Number(v).toFixed(0);
  if (fps < T.min_fps_for_timing)
    add("note", "Low frame rate", `Clip is ${f0(fps)} fps, so arm timing and sequencing are rough.`,
      "Film in slow-mo (120–240 fps) for reliable timing.");
  if (conf < 0.6)
    add("note", "Low tracking confidence", `Average joint visibility ${conf.toFixed(2)}.`,
      "Use better lighting, a plain background, and keep the whole body in frame.");
  if (!(m.fs_to_release_ms >= 60 && m.fs_to_release_ms <= 350))
    add("note", "Event timing looks off", `Foot strike → release took ${m.fs_to_release_ms} ms (typical is ~100–200 ms).`,
      "Check the marked frames in the replay and fix them if they're wrong.");
  if (m.stride_pct_height < OBP.stride_pct[P10])
    add("leak", "Short stride",
      `Stride is ${f0(m.stride_pct_height)}% of your height. OBP pitchers: middle half ${obpMid("stride_pct")}%, ` +
      `and 9 in 10 stride at least ${f0(OBP.stride_pct[P10])}%.`,
      "Mark a landing spot further out, and work on hip mobility and drive off the back leg.");
  const kneeLimit = Math.max(OBP.knee_change_deg[P90], T.knee_noise_deg);
  if (m.lead_knee_change_deg > kneeLimit)
    add("leak", "Front knee collapsing",
      `Front knee bent ${f0(m.lead_knee_change_deg)}° more between foot strike and release. ` +
      `Most OBP pitchers straighten it (typically ~${f0(-OBP.knee_change_deg[P50])}° straighter).`,
      "Focus on a firm front-side block: step-behind or walk-in drills, plus quad/glute strength.");
  if (m.camera_view === "end-on")
    add("note", "Camera is behind or in front of the pitcher",
      "From this view the stride and the turn happen toward/away from the camera, so rotation, trunk tilt " +
      "and lead-leg detection aren't reliable. Rotation and trunk aren't flagged.",
      "Film side-on: camera facing your chest at leg lift, hip height, on a tripod.");
  else if (m.camera_view === "angled")
    add("note", "Camera at an angle",
      "Rotation is measured against your stride line so it still works, but it's less precise. " +
      "Trunk tilt needs a side view, so it isn't measured or flagged.",
      "For the best numbers, film side-on: camera facing your chest at leg lift, hip height, tripod.");
  if (rotOk && m.shoulder_rotation_by_fs_deg > OBP.shoulder_open_deg[P90])
    add("leak", "Shoulders opening early",
      `Shoulders turned ~${f0(m.shoulder_rotation_by_fs_deg)}° toward the plate by foot strike. ` +
      `OBP pitchers: middle half ${obpMid("shoulder_open_deg")}°, 9 in 10 under ${f0(OBP.shoulder_open_deg[P90])}°.`,
      "Keep the glove side closed longer; hip-lead and rocker drills help delay trunk rotation.");
  if (rotOk && m.hip_shoulder_sep_max_deg < OBP.separation_deg[P10])
    add("leak", "Limited hip–shoulder separation",
      `Max separation ~${f0(Math.max(0, m.hip_shoulder_sep_max_deg))}° (phone estimate, often reads low). ` +
      `OBP pitchers: middle half ${obpMid("separation_deg")}°.`,
      "Work thoracic mobility and let the hips lead before the shoulders turn.");
  if (rotOk && !m.sequence_in_order && fps >= T.min_fps_for_timing)
    add("leak", "Out-of-order sequence",
      `Peaks should go pelvis → trunk → arm; this clip's order was off (ms from foot strike: ${m.sequence_pelvis_trunk_arm_ms.join(", ")}).`,
      "Drills that separate hip and trunk rotation (e.g., hip-lead walk-ins) can help.");
  if ((m.camera_view || "side") === "side" && m.trunk_tilt_at_release_deg < OBP.trunk_tilt_deg[P10])
    add("watch", "Upright at release",
      `Forward trunk tilt at release ~${f0(m.trunk_tilt_at_release_deg)}°. OBP pitchers: middle half ${obpMid("trunk_tilt_deg")}°.`,
      "Finish over a firm front leg. Some of this is style, so compare against your best pitches.");
  if (!flags.some(f => f.level === "leak"))
    add("good", "No major leaks flagged", "All checked metrics are within the range of OBP pitchers.",
      "Track these numbers over sessions alongside velo.");
  return flags;
}

/** Metrics this clip's camera view can't measure (shown as n/a, kept out of history). */
export function unmeasured(m) {
  const out = new Set();
  if (m.rotation_reliable === false)
    ["hip_shoulder_sep_max_deg", "shoulder_rotation_by_fs_deg", "sequence_in_order"].forEach(k => out.add(k));
  if ((m.camera_view || "side") !== "side")
    ["trunk_tilt_at_release_deg", "arm_angle_at_release_deg"].forEach(k => out.add(k));
  return out;
}

// ---------------------------------------------------------------------------
// Splitting a long video into pitches. Input: a quick, low-rate scan of the whole video
// (times in real seconds, 2D points). A pitch = a big spike in hand speed (relative to the hips,
// so camera shake and walking don't count) while the feet are spread in a stride.
// ---------------------------------------------------------------------------
export const SPLIT = {
  pre: 1.8, post: 0.7,       // window kept around each pitch's release (real seconds)
  minGap: 3.0,               // two pitches can't be closer than this (s)
  minSpeed: 6,               // hand speed, torso lengths per second (a hard throw is ~20-40)
  relSpeed: 0.3,             // ...and at least this share of the fastest throw in the video
  minSpread: 1.5,            // feet apart, torso lengths (standing ~0.6, full stride ~2.5)
};
export function findPitches(times, pts) {
  const n = times.length, need = [L_SH, R_SH, L_HIP, R_HIP, L_WR, R_WR, L_ANK, R_ANK];
  const ok = i => need.every(j => fin(pts[i][j][0]) && fin(pts[i][j][1]));
  const hipMid = i => mid(pts[i][L_HIP], pts[i][R_HIP]);
  const T = median(range(0, n).filter(ok).map(i => dist(mid(pts[i][L_SH], pts[i][R_SH]), hipMid(i))));
  if (!(T > 0)) return [];
  const speed = new Array(n).fill(NaN), spread = new Array(n).fill(NaN);
  let prev = -1;
  for (let i = 0; i < n; i++) {
    if (!ok(i)) continue;
    spread[i] = dist(pts[i][L_ANK], pts[i][R_ANK]) / T;
    const dt = prev >= 0 ? times[i] - times[prev] : 0;
    if (dt > 0 && dt < 0.25) {
      const rel = (k, j) => sub(pts[k][j], hipMid(k));
      const a = rel(prev, L_WR), b = rel(prev, R_WR), c = rel(i, L_WR), d = rel(i, R_WR);
      const same = [dist(a, c), dist(b, d)], crossed = [dist(a, d), dist(b, c)];   // left/right can swap
      const m = same[0] + same[1] <= crossed[0] + crossed[1] ? same : crossed;
      speed[i] = Math.max(...m) / T / dt;
    }
    prev = i;
  }
  const top = Math.max(0, ...speed.filter(fin)), thr = Math.max(SPLIT.minSpeed, SPLIT.relSpeed * top);
  const cands = [];
  for (let i = 0; i < n; i++) {
    if (!(speed[i] >= thr)) continue;
    let peak = true, wide = 0;
    for (let k = 0; k < n; k++) {
      const dt = times[k] - times[i];
      if (Math.abs(dt) <= 0.15 && speed[k] > speed[i]) peak = false;
      if (dt >= -0.8 && dt <= 0.4 && spread[k] > wide) wide = spread[k];
    }
    if (peak && wide >= SPLIT.minSpread) cands.push({ t: times[i], speed: speed[i], stride: wide });
  }
  const kept = [];
  for (const c of cands.sort((a, b) => b.speed - a.speed))
    if (kept.every(k => Math.abs(k.t - c.t) >= SPLIT.minGap)) kept.push(c);
  return kept.sort((a, b) => a.t - b.t);
}
