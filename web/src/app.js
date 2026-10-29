// Pitch Helper — browser app. Everything runs on this device: pose tracking (MediaPipe),
// analysis (analysis.js), guide images (canvas) and history (localStorage).
import * as A from "./analysis.js";

const MP_VER = "0.10.14";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/+esm`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`;
const MODEL_URL = k => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${k}/float16/1/pose_landmarker_${k}.task`;
const FRAME_H = 540;          // stored replay frames are this tall (keeps memory low)
const DETECT_MAX = 960;       // longest side sent to the pose model

const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode etc. */ } },
};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const int0 = v => { const r = Math.round(v); return r === 0 ? 0 : r; };   // no "-0"

// ---------------------------------------------------------------------------
// Pose tracking
// ---------------------------------------------------------------------------
const landmarkers = {};
async function getLandmarker(kind, status, delegate = "GPU") {
  const key = kind + "/" + delegate;
  if (landmarkers[key]) return landmarkers[key];
  if (window.__PH_FAKE_LANDMARKER__) return (landmarkers[key] = Object.assign(window.__PH_FAKE_LANDMARKER__, { __clock: 0, __delegate: "test" }));  // test hook
  status("Loading pose model (first time can take a moment)…");
  const { FilesetResolver, PoseLandmarker } = await import(MP_URL);
  const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
  const make = delegate => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL(kind), delegate },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  let lm;
  if (delegate === "GPU") {
    try { lm = await make("GPU"); lm.__delegate = "GPU"; } catch { lm = await make("CPU"); lm.__delegate = "CPU"; }
  } else { lm = await make("CPU"); lm.__delegate = "CPU"; }
  lm.__clock = 0;  // VIDEO mode needs ever-increasing timestamps, across clips too
  return (landmarkers[key] = lm);
}

function loadVideo(file) {
  return new Promise((res, rej) => {
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto"; v.crossOrigin = "anonymous";
    v.src = URL.createObjectURL(file);
    let done = false;
    const ok = () => { if (!done && v.videoWidth) { done = true; res(v); } };
    v.onloadeddata = ok; v.oncanplay = ok;
    v.onloadedmetadata = () => setTimeout(ok, 300);
    v.onerror = () => { if (!done) { done = true; rej(new Error("This browser can't open that video. Try exporting it as .mp4 (H.264).")); } };
    v.load();
    setTimeout(() => { if (!done) { done = true; v.videoWidth ? res(v) : rej(new Error("Video took too long to load.")); } }, 20000);
  });
}

/** Frame rate of the file itself (not the slow-mo capture rate). Uses the exact timestamps of
 *  decoded frames: the smallest gap between two frames is one frame. High-fps files are re-checked
 *  at 1/10 speed so the screen can show every frame. */
function sampleFrameTimes(v, rate, maxMs, maxFrames) {
  return new Promise(res => {
    const times = new Set(); let finished = false;
    const finish = () => { if (finished) return; finished = true; v.pause(); res([...times].sort((a, b) => a - b)); };
    const cb = (_now, md) => { times.add(Math.round(md.mediaTime * 1e6) / 1e6); times.size < maxFrames ? v.requestVideoFrameCallback(cb) : finish(); };
    v.currentTime = 0; v.playbackRate = rate;
    v.requestVideoFrameCallback(cb);
    v.play().catch(finish);
    setTimeout(finish, maxMs);
  });
}
const minGap = t => { let d = Infinity; for (let i = 1; i < t.length; i++) { const x = t[i] - t[i - 1]; if (x > 1e-4 && x < d) d = x; } return d < Infinity ? d : null; };
async function estimateFileFps(v) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return null;
  const d1 = minGap(await sampleFrameTimes(v, 1, 1500, 45));
  if (!d1) return null;
  let fps = 1 / d1;
  if (fps > 40) {                                   // could be limited by the screen's refresh rate
    const d2 = minGap(await sampleFrameTimes(v, 0.1, 3000, 45));
    if (d2 && 1 / d2 > fps) fps = 1 / d2;
  }
  v.playbackRate = 1;
  const std = [24, 25, 30, 48, 50, 60, 90, 100, 120, 240, 480];
  return std.find(s => Math.abs(s - fps) / s < 0.06) || Math.round(fps);
}

function seekTo(v, t) {
  return new Promise(res => {
    let done = false, mt = null;
    const fin = () => { if (done) return; done = true; v.removeEventListener("seeked", onSeek); res(mt); };
    const onSeek = () => {
      if (v.requestVideoFrameCallback) { v.requestVideoFrameCallback((_n, md) => { mt = md.mediaTime; fin(); }); setTimeout(fin, 40); } else fin();
    };
    v.addEventListener("seeked", onSeek);
    v.currentTime = t;
    setTimeout(fin, 4000);
  });
}

/** Track every frame of the clip, or of [t0, t1] (file seconds) when splitting a long video. */
async function extractPose(file, kind, delegate, o, status, isCancelled) {
  if (window.__PH_TEST_POSE__) return window.__PH_TEST_POSE__(file, status, isCancelled, o);  // test hook
  const v = o.video || await loadVideo(file);
  const W = v.videoWidth, H = v.videoHeight;
  if (!o.fileFps) status("Checking frame rate…");
  const fileFps = o.fileFps || (await estimateFileFps(v)) || 30;
  const lm = await getLandmarker(kind, status, delegate);
  const nAll = Math.max(1, Math.floor(v.duration * fileFps + 1e-6));
  const f0 = o.t0 != null ? Math.max(0, Math.floor(o.t0 * fileFps)) : 0;
  const n = (o.t1 != null ? Math.min(nAll, Math.ceil(o.t1 * fileFps)) : nAll) - f0;
  lm.__clock += 1000;    // new clip: let the tracker start fresh
  const ds = Math.min(1, DETECT_MAX / Math.max(W, H));
  const det = Object.assign(document.createElement("canvas"), { width: Math.round(W * ds), height: Math.round(H * ds) });
  const dctx = det.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(1, FRAME_H / H);
  const small = Object.assign(document.createElement("canvas"), { width: Math.round(W * scale), height: Math.round(H * scale) });
  const sctx = small.getContext("2d");
  const pts = [], world = [], vis = [], frames = [];
  let lastMt = null, dup = 0;
  const jpegs = [];                       // replay frames are encoded in the background
  const t0 = performance.now();
  const nanF = k => Array.from({ length: 33 }, () => new Array(k).fill(NaN));
  for (let i = 0; i < n; i++) {
    if (isCancelled()) throw new Error("Stopped.");
    const mt = await seekTo(v, Math.min(v.duration - 1e-3, (f0 + i + 0.5) / fileFps));
    if (mt != null && lastMt != null && Math.abs(mt - lastMt) < 1e-5) { dup++; continue; }   // same frame again: skip
    if (mt != null) lastMt = mt;
    dctx.drawImage(v, 0, 0, det.width, det.height);
    sctx.drawImage(v, 0, 0, small.width, small.height);
    lm.__clock += Math.max(1, Math.round(1000 / fileFps));
    const res = lm.detectForVideo(det, lm.__clock);
    if (res.landmarks && res.landmarks.length) {
      const L = res.landmarks[0], WL = res.worldLandmarks[0];
      pts.push(L.map(p => [p.x * W, p.y * H]));
      world.push(WL.map(p => [p.x, p.y, p.z]));
      vis.push(L.map(p => p.visibility ?? 1));
    } else { pts.push(nanF(2)); world.push(nanF(3)); vis.push(new Array(33).fill(0)); }
    jpegs.push(new Promise(r => small.toBlob(b => r(URL.createObjectURL(b)), "image/jpeg", 0.78)));   // snapshot now, encode later
    if (i % 3 === 0) {
      const per = (performance.now() - t0) / (i + 1), left = Math.round(per * (n - i - 1) / 1000);
      status(`Tracking pose: frame ${i + 1}/${n} (${Math.round(100 * (i + 1) / n)}%)${i > 5 ? ` · about ${left > 90 ? Math.round(left / 60) + " min" : left + " s"} left` : ""}`);
    }
  }
  frames.push(...(await Promise.all(jpegs)));
  if (!o.video) URL.revokeObjectURL(v.src);
  // if duplicates were dropped, the real frame rate was lower than estimated
  const realFps = dup > n * 0.2 ? Math.round(fileFps * pts.length / n) : fileFps;
  window.__PH_LAST__ = { fileFps: realFps, frames: pts.length, dropped: dup };   // for debugging
  return { pts, world, vis, frames, scale, W, H, fileFps: realFps, model: kind, delegate: lm.__delegate };
}

/** Quick pass over a long video with the light model, ~12 looks per real second, to find pitches.
 *  Plays the video (much faster than seeking) and tracks whichever frames come by, adjusting
 *  the playback speed so detection keeps up. `k` = file seconds per real second (8 for 240 fps
 *  slow-mo saved as a 30 fps file). Returns times in real seconds. */
const SCAN_HZ = 12, SCAN_MAX = 480;
async function scanVideo(v, k, delegate, status, isCancelled) {
  if (window.__PH_TEST_SCAN__) return window.__PH_TEST_SCAN__(v, k, status);   // test hook
  const lm = await getLandmarker("lite", status, delegate);
  const W = v.videoWidth, H = v.videoHeight, ds = Math.min(1, SCAN_MAX / Math.max(W, H));
  const det = Object.assign(document.createElement("canvas"), { width: Math.round(W * ds), height: Math.round(H * ds) });
  const dctx = det.getContext("2d", { willReadFrequently: true });
  const times = [], pts = [], step = k / SCAN_HZ;            // file seconds between looks
  const nanP = () => Array.from({ length: 33 }, () => [NaN, NaN]);
  let last = -Infinity;
  const look = mt => {
    dctx.drawImage(v, 0, 0, det.width, det.height);
    lm.__clock += Math.max(1, Math.round((mt - Math.max(0, last)) / k * 1000)); last = mt;
    const res = lm.detectForVideo(det, lm.__clock), L = res.landmarks?.[0];
    times.push(mt / k); pts.push(L ? L.map(p => [p.x * W, p.y * H]) : nanP());
    status(`Finding pitches: ${Math.round(100 * mt / v.duration)}%`);
  };
  lm.__clock += 1000;
  const canPlay = "requestVideoFrameCallback" in HTMLVideoElement.prototype && await (async () => {
    v.currentTime = 0; v.playbackRate = Math.min(16, Math.max(0.5, 2 * k));
    try { await v.play(); v.pause(); return true; } catch { return false; }
  })();
  if (canPlay) {
    await new Promise((res, rej) => {
      let done = false, stall;
      const end = err => { if (done) return; done = true; clearTimeout(stall); v.pause(); v.onended = null; err ? rej(err) : res(); };
      const cb = (_now, md) => {
        if (done) return;
        if (isCancelled()) return end(new Error("Stopped."));
        clearTimeout(stall); stall = setTimeout(() => end(), 8000);   // playback stalled: use what we have
        const gap = md.mediaTime - last;
        if (gap >= step * 0.95) {
          look(md.mediaTime);
          // keep ~SCAN_HZ looks per real second: slow down if we're falling behind, speed up if not
          if (gap > step * 1.4) v.playbackRate = Math.max(0.25, v.playbackRate * 0.8);
          else if (gap < step * 1.15) v.playbackRate = Math.min(16, v.playbackRate * 1.1);
        }
        v.requestVideoFrameCallback(cb);
      };
      v.onended = () => end();
      v.currentTime = 0;
      v.requestVideoFrameCallback(cb);
      v.play().catch(e => end(e));
    });
    v.playbackRate = 1;
  } else {                                                   // fallback: step through by seeking
    for (let t = 0; t < v.duration; t += step) {
      if (isCancelled()) throw new Error("Stopped.");
      look((await seekTo(v, t)) ?? t);
    }
  }
  return { times, pts };
}
async function thumbAt(v, t) {
  await seekTo(v, Math.max(0, Math.min(v.duration - 1e-3, t)));
  const h = 150, c = canvasOf(v.videoWidth * h / v.videoHeight, h);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.7);
}

// ---------------------------------------------------------------------------
// Canvas drawing for guides and replay
// ---------------------------------------------------------------------------
const C = {
  good: "rgb(62,213,152)", bad: "rgb(255,90,102)", ghost: "rgb(120,255,120)", ink: "#fff",
  hip: "rgb(90,190,255)", sh: "rgb(255,170,80)", navy: "rgb(18,26,44)", navy2: "rgb(24,33,56)", skel: "rgb(255,220,80)",
};
const FONT = '"Oswald","Arial Narrow",Impact,sans-serif';
const loadImg = url => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });
function canvasOf(w, h) { const c = document.createElement("canvas"); c.width = Math.round(w); c.height = Math.round(h); return c; }
function line(ctx, a, b, color, w, dash) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = "round";
  if (dash) ctx.setLineDash([w * 3, w * 3]);
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.restore();
}
function dot(ctx, p, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.fill(); }
function arrow(ctx, a, b, color, w) {
  line(ctx, a, b, color, w);
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), L = w * 4;
  ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - L * Math.cos(ang - 0.45), b[1] - L * Math.sin(ang - 0.45));
  ctx.lineTo(b[0] - L * Math.cos(ang + 0.45), b[1] - L * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fill(); ctx.restore();
}
function label(ctx, text, at, color, size, placed) {
  ctx.save(); ctx.font = `600 ${size}px ${FONT}`;
  const tw = ctx.measureText(text).width, th = size;
  let x = Math.min(Math.max(6, at[0]), ctx.canvas.width - tw - 8), y = Math.min(Math.max(th + 8, at[1]), ctx.canvas.height - 8);
  if (placed) {
    for (let k = 0; k < 8; k++) {
      const box = [x - 4, y - th - 4, x + tw + 4, y + 6];
      const hit = placed.find(b => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]));
      if (!hit) break;
      y = hit[3] + th + 10 < ctx.canvas.height ? hit[3] + th + 10 : hit[1] - 8;
    }
    placed.push([x - 4, y - th - 4, x + tw + 4, y + 6]);
  }
  ctx.fillStyle = "rgba(12,18,32,.92)"; ctx.fillRect(x - 4, y - th - 4, tw + 8, th + 10);
  ctx.fillStyle = color; ctx.fillText(text, x, y); ctx.restore();
}
function faintPose(ctx, p, th) {
  ctx.save(); ctx.globalAlpha = 0.45;
  for (const [a, b] of A.SKELETON) line(ctx, p[a], p[b], "#e6e6e6", Math.max(1, th - 1));
  ctx.restore();
}
/** Crop around points to `outH` tall. Returns {canvas, f} where f maps frame -> crop coords. */
function cropAround(src, pts, outH = 520, pad = 0.3) {
  const P = pts.filter(p => p.every(Number.isFinite));
  const xs = P.map(p => p[0]), ys = P.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const side = Math.max(x1 - x0, y1 - y0) * (1 + 2 * pad), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const ax0 = Math.max(0, cx - side * 0.6), ay0 = Math.max(0, cy - side / 2);
  const ax1 = Math.min(src.width, cx + side * 0.6), ay1 = Math.min(src.height, cy + side / 2);
  const s = outH / (ay1 - ay0), out = canvasOf((ax1 - ax0) * s, outH);
  out.getContext("2d").drawImage(src, ax0, ay0, ax1 - ax0, ay1 - ay0, 0, 0, out.width, out.height);
  return { canvas: out, f: q => [(q[0] - ax0) * s, (q[1] - ay0) * s] };
}
function caption(img, lines, colors) {
  const tmp = canvasOf(10, 10).getContext("2d"), size = 19, maxw = img.width - 28, rows = [];
  tmp.font = `600 ${size}px ${FONT}`;
  lines.forEach((t, i) => {
    let cur = "";
    for (const w of t.split(" ")) {
      const trial = (cur + " " + w).trim();
      if (cur && tmp.measureText(trial).width > maxw) { rows.push([cur, colors[i]]); cur = w; } else cur = trial;
    }
    rows.push([cur, colors[i]]);
  });
  const out = canvasOf(img.width, img.height + rows.length * 28 + 18), ctx = out.getContext("2d");
  ctx.drawImage(img, 0, 0); ctx.fillStyle = C.navy; ctx.fillRect(0, img.height, out.width, out.height - img.height);
  ctx.fillStyle = "rgb(245,179,53)"; ctx.fillRect(0, img.height, out.width, 3);
  ctx.font = `600 ${size}px ${FONT}`;
  rows.forEach(([t, c], k) => { ctx.fillStyle = c; ctx.fillText(t, 14, img.height + 30 + k * 28); });
  return out;
}
function hstack(a, b) {
  const out = canvasOf(a.width + b.width, Math.max(a.height, b.height)), ctx = out.getContext("2d");
  ctx.drawImage(a, 0, 0); ctx.drawImage(b, a.width, 0); return out;
}
/** Top view: hips & shoulders from above, YOU on top, TYPICAL below. Arrows = which way they face. */
function rotPanel(h, you, target, youTxt, tgtTxt, youOk) {
  const W = Math.round(h * 0.85), cv = canvasOf(W, h), ctx = cv.getContext("2d");
  ctx.fillStyle = C.navy2; ctx.fillRect(0, 0, W, h);
  arrow(ctx, [W / 2, 52], [W / 2, 14], "#e6e6e6", 3);
  ctx.font = `600 20px ${FONT}`; ctx.fillStyle = "#e6e6e6"; ctx.fillText("plate", W / 2 + 12, 38);
  ctx.font = `500 16px ${FONT}`; ctx.fillStyle = "#a0a0a0"; ctx.fillText("seen from above", 12, h - 14);
  const r = Math.min(W, h / 2) * 0.27;
  const face = rot => [Math.cos(rot * Math.PI / 180), -Math.sin(rot * Math.PI / 180)];
  const along = rot => [Math.sin(rot * Math.PI / 180), Math.cos(rot * Math.PI / 180)];
  [[you, youTxt, youOk ? C.good : C.bad], [target, tgtTxt, C.ghost]].forEach(([[hd, sd], txt, col], k) => {
    const cy = h * (k === 0 ? 0.36 : 0.78), c = [W / 2 - r * 0.35, cy + 4];
    ctx.font = `700 22px ${FONT}`; ctx.fillStyle = col; ctx.fillText(txt, 12, cy - r - 30);
    const a0 = Math.min(-hd, -sd), a1 = Math.max(-hd, -sd);
    if (a1 - a0 > 2) {
      ctx.save(); ctx.strokeStyle = "#ebebeb"; ctx.lineWidth = 2; ctx.beginPath();
      ctx.arc(c[0], c[1], r * 0.55, a0 * Math.PI / 180, a1 * Math.PI / 180); ctx.stroke(); ctx.restore();
    }
    [[sd, C.sh, r, r * 1.25, "chest"], [hd, C.hip, r * 0.75, r, "hips"]].forEach(([rot, colr, half, alen, name]) => {
      const al = along(rot), fc = face(rot);
      line(ctx, [c[0] - al[0] * half, c[1] - al[1] * half], [c[0] + al[0] * half, c[1] + al[1] * half], colr, 6);
      const tip = [c[0] + fc[0] * alen, c[1] + fc[1] * alen];
      arrow(ctx, c, tip, colr, 4);
      let lab = [tip[0] + fc[0] * 10 + 4, tip[1] + fc[1] * 10 + 8];
      if (name === "hips" && Math.abs((((hd - sd) % 360) + 540) % 360 - 180) < 30) lab[1] += 26;  // arrows overlap
      ctx.font = `600 19px ${FONT}`; ctx.fillStyle = colr;
      ctx.fillText(name, Math.min(Math.max(4, lab[0]), W - 70), Math.min(Math.max(20, lab[1]), h - 40));
    });
    ctx.strokeStyle = "#c8c8c8"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(c[0], c[1], 9, 0, 7); ctx.stroke();
  });
  return cv;
}

async function makeGuides(clip, r) {
  const { metrics: m, events: ev, flags } = r, S = r.series, s = clip.scale;
  const P = r.P.map(f => f.map(p => [p[0] * s, p[1] * s]));
  const rhp = S.rhp, d = S.direction, th = Math.max(3, Math.round(S.H * s / 90));
  const [leadHip, leadKnee, leadAnk] = rhp ? [A.L_HIP, A.L_KNEE, A.L_ANK] : [A.R_HIP, A.R_KNEE, A.R_ANK];
  const backAnk = rhp ? A.R_ANK : A.L_ANK;
  const flagged = new Set(flags.filter(f => f.level === "leak" || f.level === "watch").map(f => f.title));
  const G = { trunk: A.OBP.trunk_tilt_deg[2], stride: A.OBP.stride_pct[2], sep: A.OBP.separation_deg[2], knee: A.OBP.knee_change_deg[2] };
  const mid = (k, a, b) => [(P[k][a][0] + P[k][b][0]) / 2, (P[k][a][1] + P[k][b][1]) / 2];
  const body = [0, 11, 12, 23, 24, 25, 26, 27, 28];
  const frame = async k => { const img = await loadImg(clip.frames[k]); const cv = canvasOf(img.width, img.height); cv.getContext("2d").drawImage(img, 0, 0); return cv; };
  const out = [];
  const save = (key, title, ok, cv, lines) => {
    const fin = caption(cv, lines, [ok ? C.good : C.bad, "#c8c8c8"]);
    out.push({ key, title, ok, lines, url: fin.toDataURL("image/jpeg", 0.86) });
  };
  const sideView = (m.camera_view || "side") === "side", rotOk = m.rotation_reliable !== false;
  const { foot_strike: fs, release: rel, max_separation: sepf } = ev;

  if (sideView) {   // 1) trunk tilt at release
    const cv = await frame(rel), ctx = cv.getContext("2d"), p = P[rel];
    faintPose(ctx, p, th);
    const hip = mid(rel, A.L_HIP, A.R_HIP), sh = mid(rel, A.L_SH, A.R_SH);
    const L = Math.hypot(sh[0] - hip[0], sh[1] - hip[1]) * 1.15, t = G.trunk * Math.PI / 180;
    const ghost = [hip[0] + L * d * Math.sin(t), hip[1] - L * Math.cos(t)];
    const youEnd = [hip[0] + (sh[0] - hip[0]) * 1.15, hip[1] + (sh[1] - hip[1]) * 1.15];
    const ok = !flagged.has("Upright at release"), col = ok ? C.good : C.bad;
    line(ctx, hip, [hip[0], hip[1] - L], "#b4b4b4", Math.max(1, th / 2), true);
    line(ctx, hip, ghost, C.ghost, th, true);
    line(ctx, hip, youEnd, col, th + 1); dot(ctx, hip, th + 2, C.ink);
    const { canvas, f } = cropAround(cv, body.concat([15, 16]).map(j => p[j]).concat([ghost]));
    const c2 = canvas.getContext("2d"), placed = [];
    const g = f(ghost), y = f(youEnd), up = f([hip[0], hip[1] - L]);
    label(c2, `target ${G.trunk.toFixed(0)}`, [g[0] + 14 * d - (d > 0 ? 0 : 120), g[1] - 10], C.ghost, 20, placed);
    label(c2, `you ${m.trunk_tilt_at_release_deg.toFixed(0)}`, [y[0] - 30, y[1] - 14], col, 20, placed);
    label(c2, "upright", [up[0] - 14 * d - (d > 0 ? 70 : 0), up[1] + 6], "#b4b4b4", 15, placed);
    save("trunk", "Trunk tilt at release", ok, canvas,
      [`Forward trunk tilt: you ${m.trunk_tilt_at_release_deg.toFixed(0)} deg | OBP median ${G.trunk.toFixed(0)}`,
       "Lean your chest toward the plate over a firm front leg."]);
  }
  {   // 2) stride
    const cv = await frame(fs), ctx = cv.getContext("2d"), p = P[fs];
    faintPose(ctx, p, th);
    const back = p[backAnk], lead = p[leadAnk], ratio = G.stride / Math.max(1e-6, m.stride_pct_height);
    const target = [back[0] + (lead[0] - back[0]) * ratio, back[1] + (lead[1] - back[1]) * ratio];
    const ok = !flagged.has("Short stride"), col = ok ? C.good : C.bad;
    const len = Math.hypot(lead[0] - back[0], lead[1] - back[1]) + 1e-9, u = [(lead[0] - back[0]) / len, (lead[1] - back[1]) / len];
    const nrm = [-u[1] * th * 4, u[0] * th * 4];
    line(ctx, back, lead, col, th + 1); line(ctx, lead, target, C.ghost, th, true);
    for (const [q, c] of [[back, C.ink], [lead, col], [target, C.ghost]]) line(ctx, [q[0] - nrm[0], q[1] - nrm[1]], [q[0] + nrm[0], q[1] + nrm[1]], c, th);
    const { canvas, f } = cropAround(cv, body.map(j => p[j]).concat([target]));
    const c2 = canvas.getContext("2d"), placed = [];
    const L1 = f(lead), T1 = f(target), B1 = f(back);
    label(c2, "you", [L1[0] - 14, L1[1] - 22], col, 19, placed);
    label(c2, "target", [T1[0] - 24, T1[1] + 34], C.ghost, 19, placed);
    label(c2, "back foot", [B1[0] - 40, B1[1] + 34], C.ink, 17, placed);
    save("stride", "Stride length at foot strike", ok, canvas,
      [`Stride: you ${m.stride_pct_height.toFixed(0)}% of your height | OBP median ${G.stride.toFixed(0)}%`,
       "Green tick = where your front foot would land at target length."]);
  }
  {   // 3) front knee block
    const cv = await frame(rel), ctx = cv.getContext("2d"), p = P[rel];
    faintPose(ctx, p, th);
    const hip = p[leadHip], knee = p[leadKnee], ank = p[leadAnk];
    const tgt = Math.max(0, m.lead_knee_flex_at_fs_deg + G.knee), thigh = Math.hypot(hip[0] - knee[0], hip[1] - knee[1]);
    const sl = Math.hypot(ank[0] - knee[0], ank[1] - knee[1]) + 1e-9, shin = [(ank[0] - knee[0]) / sl, (ank[1] - knee[1]) / sl];
    let best = null;
    for (const sg of [1, -1]) {
      const a = (180 - tgt) * Math.PI / 180 * sg, g = [knee[0] + (Math.cos(a) * shin[0] - Math.sin(a) * shin[1]) * thigh, knee[1] + (Math.sin(a) * shin[0] + Math.cos(a) * shin[1]) * thigh];
      if (!best || Math.hypot(g[0] - hip[0], g[1] - hip[1]) < Math.hypot(best[0] - hip[0], best[1] - hip[1])) best = g;
    }
    const ok = !flagged.has("Front knee collapsing"), col = ok ? C.good : C.bad;
    line(ctx, knee, best, C.ghost, th, true); line(ctx, hip, knee, col, th + 1); line(ctx, knee, ank, col, th + 1);
    for (const q of [hip, knee, ank]) dot(ctx, q, th + 1, C.ink);
    const { canvas, f } = cropAround(cv, body.map(j => p[j]).concat([best]));
    const c2 = canvas.getContext("2d"), h1 = f(hip), b1 = f(best);
    label(c2, "you", [h1[0] + 12, h1[1]], col, 19);
    if (Math.hypot(best[0] - hip[0], best[1] - hip[1]) > th * 4) label(c2, "firm", [b1[0] + 12, b1[1]], C.ghost, 19);
    const chg = int0(m.lead_knee_change_deg);
    save("knee", "Front knee block", ok, canvas,
      [`Front knee bend: ${m.lead_knee_flex_at_fs_deg.toFixed(0)} deg at foot strike -> ${m.lead_knee_flex_at_release_deg.toFixed(0)} deg at release (${chg > 0 ? "+" : ""}${chg})`,
       `Green = OBP median: the front leg straightens ~${(-G.knee).toFixed(0)} deg after landing.`]);
  }
  const hipShoulderGuide = async (k, key, title, ok, you, target, youTxt, tgtTxt, lines) => {
    const cv = await frame(k), ctx = cv.getContext("2d"), p = P[k];
    faintPose(ctx, p, th);
    const ends = {};
    for (const [name, a, b, col] of [["hips", A.L_HIP, A.R_HIP, C.hip], ["shoulders", A.L_SH, A.R_SH, C.sh]]) {
      const m0 = [(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2], half = [(p[b][0] - p[a][0]) / 2, (p[b][1] - p[a][1]) / 2];
      const ext = Math.hypot(...half) > th * 3 ? [half[0] * 1.25, half[1] * 1.25] : [half[0] + th * 3, half[1]];
      const e1 = [m0[0] - ext[0], m0[1] - ext[1]], e2 = [m0[0] + ext[0], m0[1] + ext[1]];
      line(ctx, e1, e2, "rgb(20,20,20)", th * 2 + 4); line(ctx, e1, e2, col, th * 2);
      dot(ctx, p[a], th + 2, C.ink); dot(ctx, p[b], th + 2, C.ink);
      ends[name] = [m0, e2, col];
    }
    const { canvas, f } = cropAround(cv, body.map(j => p[j]));
    const c2 = canvas.getContext("2d"), placed = [];
    for (const name of ["shoulders", "hips"]) {
      const [m0, e, col] = ends[name], M = f(m0), E = f(e);
      c2.font = `600 19px ${FONT}`; const tw = c2.measureText(name).width;
      label(c2, name, E[0] >= M[0] ? [E[0] + 18, E[1] + 10] : [E[0] - tw - 22, E[1] + 10], col, 19, placed);
    }
    save(key, title, ok, hstack(canvas, rotPanel(canvas.height, you, target, youTxt, tgtTxt, ok)), lines);
  };
  if (rotOk) {
    const hr = S.hipRot[sepf], sr = S.shRot[sepf], sep = Math.max(0, m.hip_shoulder_sep_max_deg);
    await hipShoulderGuide(sepf, "separation", "Hip–shoulder separation", !flagged.has("Limited hip–shoulder separation"),
      [hr, sr], [hr, hr - G.sep], `YOU: ${sep.toFixed(0)} deg apart`, `TYPICAL: ~${G.sep.toFixed(0)} deg apart`,
      [`Separation: you ${sep.toFixed(0)} deg | OBP median ${G.sep.toFixed(0)} (phone estimate)`,
       "Arrows = which way your belt buckle (blue) and chest (orange) face. Hips turn to the plate first."]);
    const oh = A.OBP.hips_at_fs_deg[2], osh = A.OBP.shoulders_at_fs_deg[2], hf = S.hipRot[fs], sf = S.shRot[fs];
    await hipShoulderGuide(fs, "shoulders", "Hips & shoulders at foot strike", !flagged.has("Shoulders opening early"),
      [hf, sf], [oh, osh], "YOU at foot strike", "TYPICAL (OBP median)",
      [`At foot strike: your hips ${hf.toFixed(0)} deg open, shoulders ${sf.toFixed(0)} deg | OBP median: hips ${oh.toFixed(0)}, shoulders ${osh.toFixed(0)}`,
       "Arrows = which way your belt buckle (blue) and chest (orange) face. Hips open first, chest stays closed."]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// History (this device only)
// ---------------------------------------------------------------------------
const HKEY = "ph_history_v1";
const histLoad = () => { try { return JSON.parse(store.get(HKEY, "[]")); } catch { return []; } };
function histSave(clip, r) {
  const rows = histLoad().filter(x => x.id !== clip.id), na = A.unmeasured(r.metrics);
  const row = { id: clip.id, date: new Date().toISOString().slice(0, 16).replace("T", " "), clip: clip.name, velo_mph: clip.velo ?? "" };
  for (const [k, v] of Object.entries(r.metrics)) if (!Array.isArray(v)) row[k] = na.has(k) ? "" : v;
  row.leaks = r.flags.filter(f => f.level === "leak").map(f => f.title).join("; ");
  rows.push(row); store.set(HKEY, JSON.stringify(rows));
}

// ---------------------------------------------------------------------------
// UI: upload
// ---------------------------------------------------------------------------
let picked = [], throws = store.get("throws", "R"), running = null;
const coarse = matchMedia("(pointer:coarse)").matches;
document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === b));
  $("#analyze").hidden = b.dataset.t !== "analyze"; $("#history").hidden = b.dataset.t !== "history";
  if (b.dataset.t === "history") loadHistory();
});
const setThrows = v => { throws = v; store.set("throws", v); document.querySelectorAll("#throws button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); };
document.querySelectorAll("#throws button").forEach(b => b.onclick = e => { e.preventDefault(); setThrows(b.dataset.v); });
setThrows(throws);
$("#fps").value = store.get("fps", ""); $("#fps").onchange = e => store.set("fps", e.target.value);
$("#delegate").value = store.get("delegate", "CPU"); $("#delegate").onchange = e => store.set("delegate", e.target.value);
$("#split").value = store.get("split", "one"); $("#split").onchange = e => store.set("split", e.target.value);

const drop = $("#drop"), fileIn = $("#file");
drop.onclick = () => fileIn.click();
fileIn.onchange = () => pick(fileIn.files);
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => pick(e.dataTransfer.files));
function pick(fl) {
  picked = [...fl].filter(f => f.type.startsWith("video/") || /\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name));
  $("#files").textContent = picked.length ? picked.map(f => `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`).join(" · ") : "";
  $("#go").disabled = !picked.length;
}
$("#stop").onclick = () => { if (running) { running.stop = true; $("#stop").disabled = true; $("#stop").textContent = "Stopping…"; } };

$("#go").onclick = async () => {
  const queue = picked; picked = []; fileIn.value = ""; $("#files").textContent = "";
  const stop = $("#stop"); $("#go").hidden = true; stop.hidden = false; stop.disabled = false; stop.textContent = "Stop";
  running = { stop: false };
  const velo = parseFloat($("#velo").value) || null, capFps = parseFloat($("#fps").value) || null, kind = "heavy", delegate = $("#delegate").value;
  const split = $("#split").value === "many";
  // Always the "heavy" model: the lighter ones track position fine but their 3D depth is too coarse for hip rotation.
  const isCancelled = () => running.stop;
  const newCard = () => { document.querySelector(".placeholder")?.remove(); const c = document.createElement("div"); c.className = "card"; $("#results").prepend(c); return c; };
  const runOne = async (card, f, label, o, v0) => {
    const st = t => { card.innerHTML = `<div class="status"><div class="spin"></div>${label} — ${esc(t)}</div>`; };
    if (running.stop) { card.innerHTML = `<div class="status">Skipped: ${label}</div>`; return; }
    st("opening video…");
    try {
      const pose = await extractPose(f, kind, delegate, o, st, isCancelled);
      const clip = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name: o.name || f.name, velo: v0, throws,
        ...pose, fps: capFps || pose.fileFps, opts: {} };
      await analyzeAndShow(card, clip);
    } catch (e) {
      card.innerHTML = running.stop ? `<div class="status">Stopped: ${label}</div>` : `<div class="err"><b>${label}</b>: ${esc(e.message)}</div>`;
      if (!running.stop) console.error(e);
    }
  };
  for (const [n, f] of queue.entries()) {
    const card = newCard();
    const name = esc(f.name) + (queue.length > 1 ? ` (${n + 1}/${queue.length})` : "");
    if (!split) { await runOne(card, f, name, {}, velo); continue; }
    // Long video: find the pitches, let the user pick, then analyze each one as its own clip.
    let v = null;
    try {
      if (running.stop) { card.innerHTML = `<div class="status">Skipped: ${name}</div>`; continue; }
      const st = t => { card.innerHTML = `<div class="status"><div class="spin"></div>${name} — ${esc(t)}</div>`; };
      st("opening video…");
      v = await loadVideo(f);
      st("Checking frame rate…");
      const fileFps = (await estimateFileFps(v)) || 30, k = (capFps || fileFps) / fileFps;
      const scan = await scanVideo(v, k, delegate, st, isCancelled);
      const found = A.findPitches(scan.times, scan.pts);
      if (!found.length) {
        card.innerHTML = `<div class="err"><b>${name}</b>: No pitches found. Check the whole body is in frame and side-on.` +
          `${capFps ? "" : " If this is slow-mo saved at 30 fps, enter the capture fps (e.g. 240) and try again."}</div>`;
        continue;
      }
      const span = p => [Math.max(0, (p.t - A.SPLIT.pre) * k), Math.min(v.duration, (p.t + A.SPLIT.post) * k)];
      for (const p of found) p.thumb = await thumbAt(v, p.t * k);
      const chosen = await pickPitches(card, f.name, found, velo, k);
      card.remove();
      const base = f.name.replace(/\.[^.]+$/, "");
      for (const p of chosen) {
        const [t0, t1] = span(p), lab = `${esc(f.name)} · pitch ${p.n}`;
        await runOne(newCard(), f, lab, { video: v, fileFps, t0, t1, name: `${base} · pitch ${p.n}` }, p.velo);
      }
    } catch (e) {
      card.innerHTML = running.stop ? `<div class="status">Stopped: ${name}</div>` : `<div class="err"><b>${name}</b>: ${esc(e.message)}</div>`;
      if (!running.stop) console.error(e);
    } finally { if (v) URL.revokeObjectURL(v.src); }
  }
  running = null; stop.hidden = true; $("#go").hidden = false; $("#go").disabled = !picked.length; $("#velo").value = "";
};

/** Show the pitches found in a long video; resolves with the ticked ones (and their velos). */
function pickPitches(card, fname, found, velo, k) {
  const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  card.innerHTML = `<div class="rhead"><h2>${esc(fname)}</h2><span>${found.length} pitch${found.length > 1 ? "es" : ""} found</span></div>
    <p class="small">Untick anything that isn't a real pitch (warm-up tosses, throws back to the catcher). Add velo for each one if you have it.</p>
    <div class="picks">${found.map((p, i) => `<label class="pick"><img src="${p.thumb}" alt="Pitch ${i + 1}">
      <span class="pl"><input type="checkbox" checked data-i="${i}"> <b>Pitch ${i + 1}</b><small>${mmss(p.t * k)} in video</small></span>
      <input type="number" step="0.1" placeholder="velo" value="${velo ?? ""}" data-v="${i}" aria-label="Velo for pitch ${i + 1}"></label>`).join("")}</div>
    <button class="go" id="pickgo">Analyze ${found.length} pitch${found.length > 1 ? "es" : ""}</button>`;
  const btn = card.querySelector("#pickgo"); btn.removeAttribute("id");
  const boxes = [...card.querySelectorAll("[data-i]")];
  const paint = () => { const c = boxes.filter(b => b.checked).length; btn.disabled = !c; btn.textContent = `Analyze ${c} pitch${c === 1 ? "" : "es"}`; };
  boxes.forEach(b => b.onchange = paint);
  return new Promise((res, rej) => {
    const poll = setInterval(() => { if (running?.stop) { clearInterval(poll); rej(new Error("Stopped.")); } }, 300);
    btn.onclick = () => {
      clearInterval(poll);
      res(found.map((p, i) => ({ ...p, n: i + 1, velo: parseFloat(card.querySelector(`[data-v="${i}"]`).value) || null }))
        .filter((_, i) => boxes[i].checked));
    };
  });
}

async function analyzeAndShow(card, clip) {
  let r;
  try { r = A.analyze(clip.pts, clip.world, clip.vis, clip.fps, clip.throws, clip.opts); }
  catch (e) {
    if (e.name !== "EventError") throw e;
    const [P] = A.prepare(clip.pts, clip.world, clip.fps);
    return showNeedsEvents(card, clip, P, e.message);
  }
  card.innerHTML = `<div class="status"><div class="spin"></div>Drawing guides…</div>`;
  r.guides = await makeGuides(clip, r);
  histSave(clip, r);
  showResult(card, clip, r);
}

// ---------------------------------------------------------------------------
// UI: results
// ---------------------------------------------------------------------------
const CHECKS = [
  { k: "stride_pct_height", o: "stride_pct", g: "stride", f: "Short stride", n: "Stride length", u: "% of height" },
  { k: "hip_shoulder_sep_max_deg", o: "separation_deg", g: "separation", f: "Limited hip–shoulder separation", n: "Hip–shoulder separation", u: "°" },
  { k: "shoulder_rotation_by_fs_deg", o: "shoulder_open_deg", g: "shoulders", f: "Shoulders opening early", n: "Shoulder turn by foot strike", u: "°" },
  { k: "lead_knee_change_deg", o: "knee_change_deg", g: "knee", f: "Front knee collapsing", n: "Front knee block", u: "° bend after landing", sign: true },
  { k: "trunk_tilt_at_release_deg", o: "trunk_tilt_deg", g: "trunk", f: "Upright at release", n: "Trunk tilt at release", u: "°" },
  { k: "sequence_in_order", f: "Out-of-order sequence", n: "Hips → trunk → arm sequence", seq: true },
];
function rangeBar(v, o, st) {
  const lo = Math.min(o[0], v), hi = Math.max(o[4], v), pad = Math.max((hi - lo) * 0.1, 1), a = lo - pad, b = hi + pad, x = t => (t - a) / (b - a) * 100;
  return `<div class="rbar"><span class="r80" style="left:${x(o[0])}%;width:${x(o[4]) - x(o[0])}%"></span>
  <span class="r50" style="left:${x(o[1])}%;width:${x(o[3]) - x(o[1])}%"></span><span class="rmed" style="left:${x(o[2])}%"></span>
  <span class="rme ${st}" style="left:${x(v)}%"></span></div>
  <div class="rlab">OBP pitchers: middle half ${o[1].toFixed(0)}–${o[3].toFixed(0)}, median ${o[2].toFixed(0)}</div>`;
}
function headerHtml(clip, r) {
  return `<div class="rhead"><h2>${esc(clip.name)}</h2><span>${clip.throws}HP · ${Math.round(clip.fps)} fps${clip.velo ? ` · ${clip.velo} mph` : ""}` +
    `${r && r.confidence != null ? ` · tracking ${Math.round(r.confidence * 100)}%` : ""}${r && r.events_source === "manual" ? " · events set by hand" : ""}` +
    `${clip.model ? ` · ${clip.model}/${clip.delegate}` : ""}</span></div>`;
}
function showResult(card, clip, r) {
  const m = r.metrics, na = A.unmeasured(m);
  const flagBy = Object.fromEntries(r.flags.map(f => [f.title, f])), guideBy = Object.fromEntries(r.guides.map(g => [g.key, g]));
  let opened = false;
  const rows = CHECKS.map(c => {
    const fl = flagBy[c.f], g = c.g && guideBy[c.g];
    const st = na.has(c.k) ? "na" : fl ? fl.level : "good";
    let val, bar = "";
    if (st === "na") val = `<b>n/a</b><small>camera view</small>`;
    else if (c.seq) val = `<b>${m.sequence_in_order ? "In order" : "Off"}</b><small>${m.sequence_pelvis_trunk_arm_ms.join(" / ")} ms after FS</small>`;
    else { const v = +m[c.k], r0 = int0(v); val = `<b>${c.sign && r0 > 0 ? "+" : ""}${r0}</b><small>${c.u}</small>`; bar = rangeBar(v, A.OBP[c.o], st); }
    const text = fl ? `<p>${esc(fl.detail)}</p><p class="tip">→ ${esc(fl.tip)}</p>`
      : st === "na" ? `<p>This camera angle can't measure this. Film side-on to get it.</p>` : `<p>In the normal range for OBP pitchers. Keep it there.</p>`;
    const img = g ? `<img src="${g.url}" alt="${esc(g.title)}: ${esc(g.lines[0])}" data-zoom>` : "";
    const open = (st === "leak" || st === "watch") && !opened; if (open) opened = true;
    return `<details class="chk ${st}" ${open ? "open" : ""}><summary><span class="cs">${{ leak: "Leak", watch: "Watch", good: "Good", na: "n/a" }[st]}</span>
      <span class="cn">${c.n}</span><span class="cv">${val}</span><span class="cb">${bar}</span><span class="chev"></span></summary>
      <div class="cbody${img ? " withimg" : ""}"><div class="ctext">${text}</div>${img}</div></details>`;
  }).join("");
  const leaks = CHECKS.filter(c => !na.has(c.k) && flagBy[c.f]?.level === "leak").length;
  const verdict = leaks ? `<div class="verdict bad"><b>${leaks} leak${leaks > 1 ? "s" : ""} found</b><span>Start with the first one below.</span></div>`
    : `<div class="verdict ok"><b>No major leaks</b><span>Everything checked is within the range of OBP pitchers.</span></div>`;
  const notes = r.flags.filter(f => f.level === "note").map(f => `<div class="notebar"><b>${esc(f.title)}:</b> ${esc(f.detail)}</div>`).join("");
  const other = [["Knee bend at foot strike", m.lead_knee_flex_at_fs_deg, "°"], ["Knee bend at release", m.lead_knee_flex_at_release_deg, "°"],
    ["Arm slot", na.has("arm_angle_at_release_deg") ? null : m.arm_angle_at_release_deg, "°"], ["Foot strike → release", m.fs_to_release_ms, " ms"]]
    .map(([n, v, u]) => `<span><small>${n}</small><b>${v == null ? "n/a" : esc(v) + u}</b></span>`).join("");
  card.innerHTML = headerHtml(clip, r) + `
   <div class="rtabs"><button class="rtab on" data-p="b">Breakdown</button><button class="rtab" data-p="r">Replay</button><button class="rtab" data-p="f">Fix events</button></div>
   <div class="rpanel" data-p="b">${verdict}${notes}<div class="chks">${rows}</div><div class="other">${other}</div>
    <p class="small">Arm slot = shoulder-to-wrist angle at release (~90° over the top, ~45° three-quarter, ~0° sidearm). It's a style, not a flaw. Just keep it consistent.</p></div>
   <div class="rpanel" data-p="r" hidden>${viewerHtml(false)}<div class="hactions"><button type="button" class="dl-pose">Download tracking data</button></div>
    <p class="small">Send this file if the results look wrong. It has the body points only, no video.</p></div>
   <div class="rpanel" data-p="f" hidden><p class="small">Events wrong, or the skeleton on the wrong side? Fix it here and re-analyze. It's instant.</p>${viewerHtml(true)}</div>`;
  card.querySelectorAll(".rtab").forEach(t => t.onclick = () => {
    card.querySelectorAll(".rtab").forEach(x => x.classList.toggle("on", x === t));
    card.querySelectorAll(".rpanel").forEach(p => p.hidden = p.dataset.p !== t.dataset.p);
  });
  card.querySelectorAll(".rpanel[data-p=r],.rpanel[data-p=f]").forEach(p => bindViewer(p, card, clip, r.P, r));
  card.querySelector(".dl-pose").onclick = () => downloadPose(clip, r);
}
function showNeedsEvents(card, clip, P, message) {
  card.innerHTML = headerHtml(clip, null) + `<div class="flag note"><b>Couldn't find the pitch automatically</b><p>${esc(message)}</p>
    <p class="tip">→ Slide to each moment, tap the matching button, then re-analyze.</p></div>${viewerHtml(true)}`;
  bindViewer(card, card, clip, P, null);
}

// Frame viewer: replay with skeleton overlay; with setters for fixing events.
function viewerHtml(setters) {
  return `<div class="ev"><canvas class="fcv"></canvas>
   <input type="range" class="fslider" min="0" value="0" step="1" aria-label="Frame">
   <div class="evbar"><button data-step="-5">«</button><button data-step="-1">‹</button>
    <span>frame <b class="fno">0</b></span><button data-step="1">›</button><button data-step="5">»</button>
    <button class="play" type="button">▶</button></div>
   ${setters ? `<div class="evset">${["Leg lift", "Foot strike", "Release"].map((n, i) => `<button data-set="${i}">Set ${n.toLowerCase()} <b data-jump="${i}">—</b></button>`).join("")}</div>
   <button class="go redo" disabled>Re-analyze with these frames</button>
   <button class="swapleg" type="button" data-fix="swapSides">Left/right mixed up (wrong lead leg or arm)? Swap sides</button>
   <button class="swapleg" type="button" data-fix="flipDir">Plate on the wrong side? Flip direction</button>
   <div class="evmsg"></div>` : `<div class="evjump"></div>`}</div>`;
}
function bindViewer(root, card, clip, P, r) {
  const cv = root.querySelector(".fcv"), ctx = cv.getContext("2d"), sl = root.querySelector(".fslider"), fno = root.querySelector(".fno");
  const n = clip.frames.length, last = n - 1, s = clip.scale, S = r?.series, E = r?.events;
  sl.max = last;
  const names = E ? { [E.leg_lift]: "LEG LIFT", [E.foot_strike]: "FOOT STRIKE", [E.release]: "RELEASE" } : {};
  const cache = new Map();
  const getImg = async i => { if (!cache.has(i)) cache.set(i, loadImg(clip.frames[i])); return cache.get(i); };
  let cur = 0, playing = false, token = 0;
  const draw = async i => {
    const my = ++token, img = await getImg(i);
    if (my !== token) return;
    if (cv.width !== img.width) { cv.width = img.width; cv.height = img.height; }
    ctx.drawImage(img, 0, 0);
    const p = P[i].map(q => [q[0] * s, q[1] * s]), th = Math.max(2, Math.round(cv.height / 200));
    for (const [a, b] of A.SKELETON) if ([p[a], p[b]].every(q => q.every(Number.isFinite))) line(ctx, p[a], p[b], C.skel, th);
    for (const j of new Set(A.SKELETON.flat())) if (p[j].every(Number.isFinite)) dot(ctx, p[j], th + 1, "#fff");
    const sz = Math.max(13, Math.round(cv.height / 30));
    if (S) {
      label(ctx, `Lead knee bend ${S.kneeFlex[i].toFixed(0)}°`, [8, sz + 8], "#fff", sz);
      if (S.sepWindow && i >= S.sepWindow[0] && i <= S.sepWindow[1]) label(ctx, `Hip-shoulder sep ${S.sep[i].toFixed(0)}°`, [8, 2 * sz + 22], "#fff", sz);
    }
    if (names[i]) label(ctx, names[i], [8, cv.height - 10], "rgb(62,213,152)", sz + 4);
  };
  const show = i => { cur = Math.max(0, Math.min(last, i | 0)); sl.value = cur; fno.textContent = cur; draw(cur); for (const d of [1, 2, 3]) if (cur + d <= last) getImg(cur + d); };
  sl.oninput = () => show(+sl.value);
  root.querySelectorAll("[data-step]").forEach(b => b.onclick = () => { playing = false; show(cur + +b.dataset.step); });
  const playBtn = root.querySelector(".play");
  playBtn.onclick = async () => {
    playing = !playing; playBtn.textContent = playing ? "❚❚" : "▶";
    if (cur >= last) show(0);
    while (playing && cur < last) { show(cur + 1); await sleep(1000 / 30); }
    playing = false; playBtn.textContent = "▶";
  };
  const jump = root.querySelector(".evjump");
  if (jump && E) jump.innerHTML = [["Leg lift", E.leg_lift], ["Foot strike", E.foot_strike], ["Release", E.release]]
    .map(([n, f]) => `<button type="button" data-goto="${f}">${n} · ${f}</button>`).join("");
  root.querySelectorAll("[data-goto]").forEach(b => b.onclick = () => show(+b.dataset.goto));

  const redo = root.querySelector(".redo");
  if (redo) {
    const msg = root.querySelector(".evmsg");
    const ev = E ? [E.leg_lift, E.foot_strike, E.release] : [null, null, null];
    const paint = () => {
      root.querySelectorAll("[data-set]").forEach(b => { const i = +b.dataset.set; b.querySelector("b").textContent = ev[i] ?? "—"; b.classList.toggle("on", ev[i] != null); });
      const ok = ev.every(x => x != null) && ev[0] < ev[1] && ev[1] < ev[2];
      redo.disabled = !ok; msg.textContent = ev.every(x => x != null) && !ok ? "Frames must go leg lift < foot strike < release." : "";
    };
    root.querySelectorAll("[data-set]").forEach(b => b.onclick = e => {
      const i = +b.dataset.set;
      if (e.target.dataset.jump != null && ev[i] != null) { show(ev[i]); return; }
      ev[i] = cur; paint();
    });
    const rerun = async patch => {
      clip.opts = { ...clip.opts, ...patch };
      msg.innerHTML = '<span class="status"><span class="spin"></span>Re-analyzing…</span>';
      try { await analyzeAndShow(card, clip); } catch (e) { msg.textContent = e.message; }
    };
    redo.onclick = () => rerun({ events: ev.slice() });
    root.querySelectorAll(".swapleg").forEach(b => b.onclick = () => {
      const k = b.dataset.fix, all = ev.every(x => x != null);
      rerun({ [k]: !clip.opts[k], events: all ? ev.slice() : clip.opts.events || null });
    });
    paint();
  }
  show(E ? E.foot_strike : Math.round(last / 2));
}

/** Tracking data as JSON (body points only, no video), for debugging. */
function downloadPose(clip, r) {
  const rd = (a, k) => a.map(f => f.map(p => p.map(v => Number.isFinite(v) ? +v.toFixed(k) : null)));
  const data = { app: "pitch-helper-web", mp: MP_VER, model: clip.model, delegate: clip.delegate, clip: clip.name,
    fileFps: clip.fileFps, fps: clip.fps, W: clip.W, H: clip.H, throws: clip.throws, events: r?.events || null, opts: clip.opts,
    pts: rd(clip.pts, 2), world: rd(clip.world, 5), vis: clip.vis.map(v => v.map(x => +(+x).toFixed(3))) };
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" })),
    download: clip.name.replace(/\.[^.]+$/, "") + "_tracking.json" });
  a.click();
}

// Tap any guide image to view it full-size.
document.addEventListener("click", e => {
  const z = document.querySelector(".zoom"); if (z) { z.remove(); return; }
  const im = e.target.closest("[data-zoom]"); if (!im) return;
  const d = document.createElement("div"); d.className = "zoom"; d.innerHTML = `<img src="${im.src}" alt="${esc(im.alt)}">`; document.body.append(d);
});

// ---------------------------------------------------------------------------
// UI: history
// ---------------------------------------------------------------------------
const LABELS = { stride_pct_height: "Stride", lead_knee_flex_at_fs_deg: "Knee @ foot strike", lead_knee_flex_at_release_deg: "Knee @ release",
  lead_knee_change_deg: "Knee change", hip_shoulder_sep_max_deg: "Max separation", shoulder_rotation_by_fs_deg: "Shoulder turn by FS",
  trunk_tilt_at_release_deg: "Trunk tilt @ release", arm_angle_at_release_deg: "Arm slot", fs_to_release_ms: "Foot strike → release", velo_mph: "Velo (mph)" };
let hist = [];
function loadHistory() {
  hist = histLoad();
  if (!hist.length) { $("#chart").innerHTML = '<div class="empty">No pitches yet. Analyzed clips are saved here, on this device.</div>'; $("#htable").innerHTML = ""; return; }
  const keys = Object.keys(LABELS).filter(k => hist.some(r => r[k] !== "" && r[k] != null && !isNaN(parseFloat(r[k]))));
  const sel = $("#hmetric"), prev = sel.value || "stride_pct_height";
  sel.innerHTML = keys.map(k => `<option value="${k}">${LABELS[k]}</option>`).join("");
  sel.value = keys.includes(prev) ? prev : keys[0]; sel.onchange = drawChart; drawChart();
  const cols = ["date", "clip", "velo_mph", ...Object.keys(LABELS).filter(k => k !== "velo_mph"), "leaks"];
  $("#htable").innerHTML = `<table><tr>${cols.map(c => `<th>${LABELS[c] || c.replace(/_/g, " ")}</th>`).join("")}</tr>` +
    hist.slice().reverse().map(r => `<tr>${cols.map(c => `<td>${esc(r[c] === "" || r[c] == null ? (c === "velo_mph" || c === "leaks" ? "—" : "n/a") : r[c])}</td>`).join("")}</tr>`).join("") + "</table>" +
    `<div class="hactions"><button type="button" id="hcsv">Download CSV</button><button type="button" id="hclear">Clear history</button></div>`;
  $("#hcsv").onclick = () => {
    const cols2 = Object.keys(hist[0]);
    const csv = [cols2.join(","), ...hist.map(r => cols2.map(c => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(","))].join("\n");
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "pitch_history.csv" });
    a.click();
  };
  $("#hclear").onclick = () => { if (confirm("Delete all saved pitches on this device?")) { store.set(HKEY, "[]"); loadHistory(); } };
}
function drawChart() {
  const k = $("#hmetric").value, pts = hist.map((r, i) => [i, parseFloat(r[k])]).filter(p => !isNaN(p[1]));
  if (!pts.length) { $("#chart").innerHTML = '<div class="empty">No data for this metric.</div>'; return; }
  const W = 900, H = 240, P = 40, ys = pts.map(p => p[1]); let lo = Math.min(...ys), hi = Math.max(...ys); if (lo === hi) { lo -= 1; hi += 1; }
  const x = i => P + (pts.length === 1 ? (W - 2 * P) / 2 : (i / (pts.length - 1)) * (W - 2 * P)), y = v => H - P - (v - lo) / (hi - lo) * (H - 2 * P);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  $("#chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${LABELS[k]} over sessions">
   <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" stroke="var(--line)"/>
   <text x="4" y="${y(hi) + 4}">${hi.toFixed(1)}</text><text x="4" y="${y(lo) + 4}">${lo.toFixed(1)}</text>
   <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
   ${pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p[1])}" r="4" fill="var(--accent)"><title>${esc(hist[p[0]].date)}: ${p[1]}</title></circle>`).join("")}
   <text x="${P}" y="${H - 12}">oldest</text><text x="${W - P}" y="${H - 12}" text-anchor="end">latest</text></svg>`;
}
