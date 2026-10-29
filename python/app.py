#!/usr/bin/env python3
"""
Pitch Helper — web UI.

    python app.py

Then open http://localhost:5000 on this computer, or the "phone" URL it prints
to upload clips straight from your phone (same Wi-Fi).
"""

import csv
import hmac
import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory, session
from werkzeug.utils import secure_filename

import cv2
import pitch_helper as ph

# Where clips, results and history are kept. On Hugging Face, /data is the persistent disk
# (if you add one); otherwise files live inside the container and reset on restart.
OUT = Path(os.environ.get("PITCH_OUTPUT")
           or ("/data/pitch_output" if os.access("/data", os.W_OK) else "pitch_output")).resolve()
UPLOADS = OUT / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

# --- Optional password ------------------------------------------------------------
# Set APP_PASSWORD (a Hugging Face "secret") to require a login. Unset = no login (local use).
PASSWORD = os.environ.get("APP_PASSWORD") or None
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30  # stay logged in for 30 days
if os.environ.get("COOKIE_SECURE"):
    # HTTPS hosting (Hugging Face shows the app inside an iframe, which needs these)
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_SAMESITE="None")

OPEN_ENDPOINTS = {"login", "logo"}


@app.before_request
def require_login():
    if not PASSWORD or request.endpoint in OPEN_ENDPOINTS or session.get("ok"):
        return None
    if request.path == "/":
        return redirect("/login")
    return jsonify(error="Not logged in. Refresh the page."), 401


@app.route("/login", methods=["GET", "POST"])
def login():
    if not PASSWORD:
        return redirect("/")
    err = ""
    if request.method == "POST":
        if hmac.compare_digest(request.form.get("password", "").encode(), PASSWORD.encode()):
            session.permanent = True
            session["ok"] = True
            return redirect("/")
        time.sleep(1)  # slow down guessing
        err = "<p class=e>Wrong password.</p>"
    return LOGIN_PAGE.replace("{{ERR}}", err)


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login" if PASSWORD else "/")


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pitch Helper · Login</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0c1220;color:#f3f5fa;
 font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
form{width:min(340px,90vw);background:#18213a;border:1px solid #2b3757;border-radius:16px;padding:26px;
 box-shadow:0 10px 30px rgba(0,0,0,.3);position:relative;overflow:hidden}
form::after{content:"";position:absolute;left:0;right:0;bottom:0;height:4px;background:linear-gradient(90deg,#d7263d 70%,#f5b335 70%)}
h1{margin:0;font:700 30px "Arial Narrow",Impact,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#d7263d;
 -webkit-text-stroke:1px #f5b335;paint-order:stroke fill}
.s{margin:0 0 18px;color:#9aa5bf;letter-spacing:3px;text-transform:uppercase;font-size:12px}
input{width:100%;box-sizing:border-box;background:#0c1220;border:1px solid #2b3757;color:#f3f5fa;border-radius:9px;padding:11px;font:inherit}
input:focus{outline:2px solid #f5b335}
button{margin-top:12px;width:100%;background:#f5b335;color:#1a1405;border:0;border-radius:10px;padding:12px;
 font:700 15px "Arial Narrow",Impact,sans-serif;letter-spacing:2px;text-transform:uppercase;cursor:pointer}
.e{color:#ff5a66;margin:10px 0 0}
</style></head><body><form method="post"><h1>Pitch Helper</h1>
<input type="password" name="password" placeholder="Team password" autofocus autocomplete="current-password">
<button>Enter</button>{{ERR}}</form></body></html>"""
lock = threading.Lock()  # one analysis at a time
jobs = {}  # job id -> {"done", "total", "stage", "cancel"}


def job_hook(job_id, stage="Tracking pose"):
    """Progress callback for pose extraction; raises Cancelled if the user hit Stop."""
    if not job_id:
        return None
    def hook(done, total):
        j = jobs.setdefault(job_id, {"cancel": False})
        if j["cancel"]:
            raise ph.Cancelled()
        j.update(done=done, total=total, stage=stage)
    return hook


def to_web_video(src: Path) -> Path:
    """Re-encode to H.264 so browsers can play it. Falls back to the original."""
    try:
        import imageio_ffmpeg
        dst = src.with_name("annotated_web.mp4")
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
            check=True)
        return dst
    except Exception as e:  # noqa: BLE001
        print("Video re-encode skipped:", e)
        return src


def ensure_readable(video: Path) -> Path:
    """If OpenCV can't decode the upload (e.g. some HEVC .mov), convert it to H.264 .mp4,
    keeping every frame and the original frame rate."""
    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if ok:
        return video
    import imageio_ffmpeg
    dst = video.with_name(video.stem + "_conv.mp4")
    r = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
         "-map", "0:v:0", "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"Couldn't read this video file ({video.suffix}). "
                         "Try exporting it as .mp4 from your phone or editor.")
    return dst


def run_clip(video: Path, throws: str, fps_override, velo, model: str, events=None,
             job_id=None, swap_legs=False, flip_dir=False, stage="Tracking pose") -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    out_dir = OUT / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    hook = job_hook(job_id, stage)
    pts, world, vis = ph.load_or_extract(str(video), fps, model, out_dir, hook)
    if job_id:
        jobs.setdefault(job_id, {"cancel": False})["stage"] = "Building replay"
    stamp = f"?v={int(time.time() * 1000)}"  # bust browser cache after re-analysis
    rel = lambda p: "/files/" + p.relative_to(OUT).as_posix() + stamp
    base = {"clip": video.name, "fps": fps, "throws": throws, "velo": velo, "model": model,
            "n_frames": len(pts), "v": stamp, "swap_legs": bool(swap_legs), "flip_dir": bool(flip_dir),
            "frames_url": "/files/" + (out_dir / "frames").relative_to(OUT).as_posix() + "/"}
    try:
        result = ph.analyze(pts, world, vis, fps, throws, events, swap_legs, flip_dir)
    except ph.EventError as e:
        out_fps = ph.render(str(video), ph.prepare(pts, world, fps)[0], fps, out_dir)
        return {**base, "needs_events": True, "message": str(e), "events": None,
                "out_fps": out_fps, "video_url": rel(to_web_video(out_dir / "annotated.mp4"))}

    out_fps = ph.render(str(video), result["_P"], fps, out_dir, result["events"], result["_series"])
    ph.write_reports(result, str(video), out_dir, velo)
    try:
        guides = [{**g, "url": rel(out_dir / g["file"])} for g in ph.make_guides(str(video), result, out_dir)]
    except Exception as e:  # noqa: BLE001 — guides are a bonus; never fail the analysis over them
        app.logger.exception("guides failed: %s", e)
        guides = []
    web = to_web_video(out_dir / "annotated.mp4")
    return {
        "guides": guides,
        **base, "metrics": result["metrics"], "flags": result["flags"],
        "unmeasured": sorted(ph.unmeasured(result["metrics"])),
        "obp": ph.OBP,
        "confidence": result["confidence"], "events": result["events"],
        "events_source": result["events_source"], "out_fps": out_fps,
        "video_url": rel(web),
        "keyframes_url": rel(out_dir / "keyframes.png") if (out_dir / "keyframes.png").exists() else None,
        "report_url": rel(out_dir / "report.txt"),
    }


@app.post("/reanalyze")
def reanalyze():
    d = request.get_json(force=True)
    video = UPLOADS / secure_filename(d.get("clip", ""))
    if not d.get("clip") or not video.exists():
        return jsonify(error="Clip not found. Upload it again."), 404
    try:
        ev = [int(x) for x in d["events"]] if d.get("events") else None
        with lock:
            return jsonify(run_clip(video, d.get("throws", "R"), float(d["fps"]),
                                    d.get("velo"), d.get("model", "heavy"), ev,
                                    swap_legs=bool(d.get("swap_legs")),
                                    flip_dir=bool(d.get("flip_dir"))))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=str(e)), 422
    except Exception as e:  # noqa: BLE001
        app.logger.exception("re-analysis failed")
        return jsonify(error=f"Re-analysis failed: {e}"), 500


@app.post("/analyze")
def analyze():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video uploaded."), 400
    throws = request.form.get("throws", "R").upper()
    num = lambda k: float(request.form[k]) if request.form.get(k) else None
    job_id = request.form.get("job") or None
    if job_id:
        jobs.setdefault(job_id, {"cancel": False}).update(stage="Preparing video", done=0, total=0)
    stem = Path(secure_filename(f.filename) or "clip.mp4")
    dest = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem.stem}{stem.suffix or '.mp4'}"
    f.save(dest)
    try:
        dest = ensure_readable(dest)
        with lock:
            if job_id and jobs.get(job_id, {}).get("cancel"):
                raise ph.Cancelled()
            model = request.form.get("model", "heavy")
            if request.form.get("split") != "1":
                return jsonify(run_clip(dest, throws, num("fps"), num("velo"), model, job_id=job_id))
            # Long video: find each pitch, save it as its own clip, analyze each one.
            cap = cv2.VideoCapture(str(dest))
            fps = num("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            clips = ph.split_video(str(dest), fps, UPLOADS, job_hook(job_id, "Finding pitches"))
            if not clips:
                raise ValueError("No pitches found. Check the whole body is in frame and side-on"
                                 + ("" if num("fps") else ". If this is slow-mo saved at 30 fps, enter the capture fps (e.g. 240)") + ".")
            results = []
            for k, (path, _) in enumerate(clips):
                try:
                    results.append(run_clip(path, throws, fps, num("velo"), model, job_id=job_id,
                                            stage=f"Pitch {k + 1}/{len(clips)}: tracking pose"))
                except ph.Cancelled:
                    raise
                except ValueError as e:
                    results.append({"clip": path.name, "error": str(e)})
            return jsonify(results=results)
    except ph.Cancelled:
        return jsonify(cancelled=True, error="Stopped."), 409
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:  # noqa: BLE001
        app.logger.exception("analysis failed")
        return jsonify(error=f"Analysis failed: {e}"), 500
    finally:
        jobs.pop(job_id, None)


@app.get("/progress/<job_id>")
def progress(job_id):
    return jsonify(jobs.get(job_id) or {})


@app.post("/cancel/<job_id>")
def cancel(job_id):
    if job_id in jobs:  # still uploading? the browser aborts that side itself
        jobs[job_id]["cancel"] = True
    return jsonify(ok=True)


@app.get("/history")
def history():
    path = OUT / "history.csv"
    if not path.exists():
        return jsonify([])
    with open(path, newline="", encoding="utf-8") as fh:
        return jsonify(list(csv.DictReader(fh)))


@app.get("/files/<path:p>")
def files(p):
    return send_from_directory(OUT, p)


@app.get("/logo")
def logo():
    """Your team logo, if you drop logo.png / logo.jpg / logo.svg next to app.py."""
    here = Path(__file__).resolve().parent
    for name in ("logo.png", "logo.jpg", "logo.jpeg", "logo.svg", "logo.webp"):
        if (here / name).exists():
            return send_from_directory(here, name)
    return ("", 404)


@app.get("/")
def index():
    return PAGE


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pitch Helper</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* Team colours: navy, red, gold */
:root{--bg:#0c1220;--bg2:#121a2c;--card:#18213a;--card2:#1e2843;--line:#2b3757;--text:#f3f5fa;--muted:#9aa5bf;
 --navy:#1f2a47;--red:#d7263d;--red2:#ff4d5e;--gold:#f5b335;--gold2:#ffd27a;
 --accent:var(--gold);--leak:#ff5a66;--watch:#f5b335;--note:#7fb6ff;--good:#3ed598;
 --display:"Oswald","Arial Narrow","Roboto Condensed",Impact,sans-serif;
 --body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}
body{margin:0;color:var(--text);font:15px/1.55 var(--body);
 background:radial-gradient(1200px 500px at 50% -200px,#22305a 0%,transparent 70%),
  repeating-linear-gradient(90deg,rgba(255,255,255,.012) 0 2px,transparent 2px 6px),var(--bg);min-height:100vh}
.wrap{max-width:1800px;margin:0 auto;padding:22px 28px 60px;display:grid;grid-template-columns:370px minmax(0,1fr);gap:26px;align-items:start}
.side{position:sticky;top:18px;max-height:calc(100vh - 36px);overflow-y:auto;scrollbar-width:thin}
main{min-width:0}
.side .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.side .row label input,.side .row label select{width:100%}
.side .row .seg{width:100%}.side .row .seg button{flex:1}
.side .row .go{grid-column:1/-1;margin-left:0;width:100%}
.side .row label.full{grid-column:1/-1}
.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;min-height:420px;color:var(--muted);
 border-style:dashed}
.placeholder b{font:700 26px var(--display);letter-spacing:2px;text-transform:uppercase;color:var(--text);margin:10px 0 6px}
.placeholder .ball{width:64px;height:64px;border-radius:50%;background:#f3f5fa;position:relative;box-shadow:0 6px 18px rgba(0,0,0,.35)}
.placeholder .ball::before,.placeholder .ball::after{content:"";position:absolute;top:8px;bottom:8px;width:22px;border:3px dashed var(--red);border-radius:50%}
.placeholder .ball::before{left:-8px;border-left:0;border-top:0;border-bottom:0}.placeholder .ball::after{right:-8px;border-right:0;border-top:0;border-bottom:0}
.res{display:grid;gap:22px}
.res .gh{margin-top:0}
@media (min-width:1300px){.res{grid-template-columns:minmax(0,1fr) minmax(0,1.3fr)}}
.rtabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:-2px 0 16px}
.rtab{background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);padding:8px 14px;cursor:pointer;font:600 13px var(--display);letter-spacing:1.4px;text-transform:uppercase}
.rtab.on{color:var(--text);border-bottom-color:var(--red)}
.verdict{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;padding:12px 16px;border-radius:12px;margin-bottom:12px}
.verdict b{font:700 22px var(--display);letter-spacing:1px;text-transform:uppercase}.verdict span{color:var(--muted)}
.verdict.bad{background:rgba(255,90,102,.08);border:1px solid rgba(255,90,102,.35)}.verdict.bad b{color:var(--leak)}
.verdict.ok{background:rgba(62,213,152,.08);border:1px solid rgba(62,213,152,.35)}.verdict.ok b{color:var(--good)}
.notebar{font-size:13px;color:var(--muted);padding:8px 12px;border-left:3px solid var(--note);background:rgba(127,182,255,.06);border-radius:0 8px 8px 0;margin-bottom:8px}
.notebar b{color:var(--note)}
.chks{display:grid;gap:8px;margin-top:6px}
.chk{background:var(--bg);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.chk>summary{list-style:none;display:grid;grid-template-columns:64px minmax(150px,1.1fr) 150px minmax(180px,2fr) 18px;gap:14px;align-items:center;padding:12px 16px;cursor:pointer}
.chk>summary::-webkit-details-marker{display:none}
.chk>summary:hover{background:rgba(255,255,255,.025)}
.cs{font:700 11px var(--display);letter-spacing:1.2px;text-transform:uppercase;text-align:center;padding:4px 0;border-radius:6px}
.chk.leak .cs{background:rgba(255,90,102,.15);color:var(--leak)}.chk.watch .cs{background:rgba(245,179,53,.15);color:var(--watch)}
.chk.good .cs{background:rgba(62,213,152,.13);color:var(--good)}.chk.na .cs{background:rgba(154,165,191,.12);color:var(--muted)}
.chk.leak{border-color:rgba(255,90,102,.45)}.chk.watch{border-color:rgba(245,179,53,.45)}
.cn{font:600 16px var(--display);letter-spacing:.6px;text-transform:uppercase}
.cv b{font:700 26px var(--display);display:block;line-height:1}.cv small{color:var(--muted);font-size:12px}
.chk.na .cv b{color:var(--muted)}
.chev{width:10px;height:10px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:.15s;margin-top:-4px}
.chk[open] .chev{transform:rotate(-135deg);margin-top:4px}
.rbar{position:relative;height:12px;border-radius:6px;background:rgba(255,255,255,.04)}
.rbar span{position:absolute;top:0;bottom:0}
.r80{background:rgba(62,213,152,.12);border-radius:6px}.r50{background:rgba(62,213,152,.28);border-radius:6px}
.rmed{width:2px;background:var(--good);margin-left:-1px}
.rme{width:16px;height:16px;top:-2px!important;margin-left:-8px;border-radius:50%;border:2px solid var(--bg);box-shadow:0 0 0 2px rgba(0,0,0,.3)}
.rme.good{background:var(--good)}.rme.leak{background:var(--leak)}.rme.watch{background:var(--watch)}
.rlab{font-size:11px;color:var(--muted);margin-top:5px}
.cbody{padding:4px 16px 16px;display:grid;gap:14px}
.cbody p{margin:0 0 6px;color:var(--muted)}.cbody .tip{color:var(--text)}
.cbody img{width:100%;max-width:680px;border-radius:10px;cursor:zoom-in;border:1px solid var(--line)}
@media (min-width:1100px){.cbody.withimg{grid-template-columns:minmax(220px,1fr) minmax(0,2fr);align-items:start}}
.other{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.other small{display:block;color:var(--muted);font:600 10px var(--display);letter-spacing:1px;text-transform:uppercase}.other b{font:600 17px var(--display)}
.small{font-size:13px;color:var(--muted)}
@media (max-width:760px){.chk>summary{grid-template-columns:58px 1fr auto 14px;row-gap:8px}.cb{grid-column:1/-1}}
@media (max-width:950px){.wrap{grid-template-columns:1fr;padding:18px 14px 50px}.side{position:static;max-height:none;overflow:visible}}
/* header: jersey-style wordmark */
.brand{display:flex;align-items:center;gap:16px;padding:18px 20px;margin-bottom:18px;border-radius:16px;
 background:linear-gradient(135deg,var(--navy),#16203a);border:1px solid var(--line);position:relative;overflow:hidden}
.brand::after{content:"";position:absolute;inset:auto 0 0 0;height:4px;background:linear-gradient(90deg,var(--red) 0 70%,var(--gold) 70%)}
.brand img{width:64px;height:64px;object-fit:contain;flex:none}
.brand .wm{font-family:var(--display);font-weight:700;line-height:1;letter-spacing:.5px}
.brand .team{font-size:34px;color:var(--red);-webkit-text-stroke:1.5px var(--gold);paint-order:stroke fill;text-transform:uppercase}
.brand .app{font-size:15px;letter-spacing:4px;color:var(--text);text-transform:uppercase;margin-top:6px;display:flex;align-items:center;gap:8px}
.brand .app::before,.brand .app::after{content:"";height:2px;width:18px;background:var(--text);border-radius:2px}
.sub{color:var(--muted);margin:-6px 0 16px;font-size:14px}
.tabs{display:flex;gap:6px;margin-bottom:16px}
.tab{background:var(--bg2);border:1px solid var(--line);color:var(--muted);padding:8px 18px;border-radius:99px;cursor:pointer;
 font:600 13px var(--display);letter-spacing:1.5px;text-transform:uppercase}
.tab.on{background:var(--gold);border-color:var(--gold);color:#1a1405}
.card{background:linear-gradient(180deg,var(--card),var(--bg2));border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:16px;
 box-shadow:0 10px 30px rgba(0,0,0,.25)}
.drop{border:2px dashed #3a4870;border-radius:14px;padding:34px 16px;text-align:center;cursor:pointer;transition:.15s;background:rgba(255,255,255,.015)}
.drop.over,.drop:hover{border-color:var(--gold);background:rgba(245,179,53,.06)}
.drop b{display:block;font:600 19px var(--display);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}.drop small{color:var(--muted)}
.files{margin:10px 0 0;color:var(--muted);font-size:14px}
.row{display:flex;flex-wrap:wrap;gap:14px;margin-top:16px;align-items:end}
label{display:flex;flex-direction:column;gap:5px;font:600 11px var(--display);letter-spacing:1.2px;text-transform:uppercase;color:var(--muted)}
input,select{background:var(--bg);border:1px solid var(--line);color:var(--text);border-radius:9px;padding:8px 10px;font:15px var(--body);width:130px}
input:focus,select:focus{outline:2px solid var(--gold);outline-offset:1px}
.seg{display:flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--bg)}
.seg button{background:none;border:0;color:var(--muted);padding:8px 16px;cursor:pointer;font:15px var(--body)}
.seg button.on{background:var(--red);color:#fff;font-weight:600}
.go{margin-left:auto;background:var(--gold);color:#1a1405;border:0;border-radius:11px;padding:12px 28px;cursor:pointer;
 font:700 16px var(--display);letter-spacing:1.5px;text-transform:uppercase;box-shadow:0 4px 0 #b77d12}
.go:active{transform:translateY(2px);box-shadow:0 2px 0 #b77d12}
.go:disabled{opacity:.35;cursor:default;box-shadow:none}
.go.stop{background:var(--red);color:#fff;box-shadow:0 4px 0 #8f1626}
[hidden]{display:none!important}
details{color:var(--muted);font-size:14px;margin-top:14px}summary{cursor:pointer}summary:hover{color:var(--text)}
details ul{margin:8px 0 0;padding-left:20px}
.status{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:14px;margin:6px 0}
.spin{width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--gold);border-radius:50%;animation:s 1s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.rhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)}
.rhead h2{font:600 22px var(--display);letter-spacing:.5px;margin:0;word-break:break-all;text-transform:uppercase}.rhead span{color:var(--muted);font-size:13px}
.flag{border-left:4px solid;padding:10px 14px;border-radius:0 10px 10px 0;background:rgba(0,0,0,.18);margin-bottom:8px}
.flag b{display:block;font:600 16px var(--display);letter-spacing:.6px;text-transform:uppercase}.flag p{margin:2px 0 0;color:var(--muted);font-size:14px}
.flag .tip{color:var(--text)}
.leak{border-color:var(--leak)}.leak b{color:var(--leak)}
.watch{border-color:var(--watch)}.watch b{color:var(--watch)}
.note{border-color:var(--note)}.note b{color:var(--note)}
.good{border-color:var(--good)}.good b{color:var(--good)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:14px 0}
.m{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:10px 12px;position:relative;overflow:hidden}
.m::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--red)}
.m small{color:var(--muted);font:600 11px var(--display);letter-spacing:1px;text-transform:uppercase;display:block}
.m b{font:700 28px var(--display);letter-spacing:.5px;color:var(--text)}
.media{display:grid;gap:12px}
video,.media img{width:100%;border-radius:12px;background:#000}
a{color:var(--gold)}a:hover{color:var(--gold2)}
.err{color:var(--leak)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font:600 11px var(--display);letter-spacing:1px;text-transform:uppercase}
tr:hover td{background:rgba(255,255,255,.02)}
.tscroll{overflow-x:auto}
svg text{fill:var(--muted);font-size:11px}
.empty{color:var(--muted);text-align:center;padding:30px}
.credit{color:var(--muted);font-size:12px;margin-top:20px}
.ev{margin-top:12px;display:grid;gap:10px}
.gh{font:700 18px var(--display);letter-spacing:1.5px;text-transform:uppercase;margin:20px 0 10px;display:flex;align-items:center;gap:10px}
.gh::after{content:"";flex:1;height:2px;background:linear-gradient(90deg,var(--red),transparent)}
.guides{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:12px;margin-bottom:14px}
.guides figure{margin:0;border:2px solid var(--line);border-radius:12px;overflow:hidden;background:var(--bg)}
.guides figure.bad{border-color:var(--leak)}.guides figure.ok{border-color:#2a6b55}
.guides img{width:100%;display:block;cursor:zoom-in}
.guides figure.wide{grid-column:1/-1}
.zoom{position:fixed;inset:0;background:rgba(6,10,20,.94);display:flex;align-items:center;justify-content:center;z-index:50;cursor:zoom-out;padding:10px}
.zoom img{max-width:100%;max-height:100%;object-fit:contain}
.guides figcaption{padding:8px 12px;font:600 14px var(--display);letter-spacing:.8px;text-transform:uppercase}
.guides .bad figcaption{color:var(--leak)}.guides .ok figcaption{color:var(--good)}
.fimg{width:100%;border-radius:12px;background:#000;display:block;min-height:120px;object-fit:contain;max-height:70vh}
.fslider{width:100%;accent-color:var(--gold);height:32px}
.evset b[data-jump]{text-decoration:underline dotted;padding:0 4px}
.evbar{display:flex;align-items:center;gap:6px;justify-content:center}
.evbar span{min-width:130px;text-align:center;color:var(--muted)}.evbar b{color:var(--text);font-family:var(--display);font-size:18px}
.ev button:not(.go){background:var(--bg);border:1px solid var(--line);color:var(--text);border-radius:9px;padding:8px 12px;cursor:pointer;font:15px var(--body)}
.evset{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.evset button.on{border-color:var(--gold)}.evset b{color:var(--gold)}
.ev .go{margin:0;width:100%}
.ev .swapleg{background:none!important;border:1px dashed var(--line)!important;color:var(--muted)!important}
.evmsg{color:var(--watch);font-size:13px;min-height:1em}
.fix{margin-top:12px}
@media (max-width:600px){.evset{grid-template-columns:1fr}.go{margin-left:0;width:100%}
 .brand{padding:14px}.brand .team{font-size:26px}.brand img{width:48px;height:48px}}
</style></head><body><div class="wrap">
<aside class="side">
<header class="brand">
 <img src="/logo" alt="" onerror="this.remove()">
 <div class="wm"><div class="team">Pitch Helper</div></div>
</header>
<div class="sub">Upload side-on clips of your pitches. Get leaks, metrics and an annotated replay.</div>
<div class="tabs"><button class="tab on" data-t="analyze">Analyze</button><button class="tab" data-t="history">History</button></div>

<div class="card" id="upload">
  <div class="drop" id="drop"><b>Drop clips here or tap to choose</b><small>One pitch per clip, or a whole session · slow-mo recommended</small>
   <input type="file" id="file" accept="video/*" multiple hidden></div>
  <div class="files" id="files"></div>
  <div class="row">
   <label>Throws<div class="seg" id="throws"><button data-v="R">Right</button><button data-v="L">Left</button></div></label>
   <label>Velo (mph)<input id="velo" type="number" step="0.1" placeholder="optional"></label>
   <label>Capture fps<input id="fps" type="number" placeholder="auto"></label>
   <label class="full">Each clip has<select id="split"><option value="one">One pitch</option><option value="many">Several pitches (auto-split)</option></select></label>
   <label>Model<select id="model"><option value="heavy">Accurate</option><option value="full">Balanced</option><option value="lite">Fast</option></select></label>
   <button class="go" id="go" disabled>Analyze</button>
   <button class="go stop" id="stop" hidden>Stop</button>
  </div>
  <details><summary>Filming tips</summary><ul>
   <li>Camera faces your chest at leg lift (RHP: 3B side, LHP: 1B side), at hip height on a tripod. Don't pan or zoom.</li>
   <li>Whole body in frame, plain background, good light.</li>
   <li>Use slow-mo (120–240 fps). If a slow-mo clip shows as 30 fps in the result, enter the real capture fps (e.g. 240).</li>
   <li>Trim to set position → follow-through, or record the whole session in one video and choose "Several pitches". Same spot every session.</li></ul></details>
 </div>
 <p class="credit">Targets: <a href="https://github.com/drivelineresearch/openbiomechanics" target="_blank" rel="noopener">Driveline OpenBiomechanics Project</a> (CC BY-NC-SA 4.0, non-commercial). Phone estimates vs. 3D lab data, so use as a guide.</p>
 <p class="credit"><a href="/logout">Log out</a></p>
</aside>

<main>
<section id="analyze">
 <div id="results"><div class="card placeholder"><div class="ball"></div><b>Ready when you are</b>
  Drop a pitching clip on the left. Your leaks, metrics and you-vs-target guides show up here.</div></div>
</section>

<section id="history" hidden>
 <div class="card"><div class="row" style="margin-top:0">
  <label>Chart metric<select id="hmetric" style="width:240px"></select></label></div>
  <div id="chart"></div></div>
 <div class="card tscroll" id="htable"></div>
</section>
</main>
</div>

<script>
const $=s=>document.querySelector(s);
const store={get(k,d){try{return localStorage.getItem(k)??d}catch{return d}},set(k,v){try{localStorage.setItem(k,v)}catch{}}};
let picked=[], throws=store.get("throws","R");
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{
 document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===b));
 $("#analyze").hidden=b.dataset.t!=="analyze";$("#history").hidden=b.dataset.t!=="history";
 if(b.dataset.t==="history")loadHistory();});

const setThrows=v=>{throws=v;store.set("throws",v);document.querySelectorAll("#throws button").forEach(b=>b.classList.toggle("on",b.dataset.v===v))};
document.querySelectorAll("#throws button").forEach(b=>b.onclick=e=>{e.preventDefault();setThrows(b.dataset.v)});setThrows(throws);
$("#model").value=store.get("model","heavy");$("#model").onchange=e=>store.set("model",e.target.value);
$("#fps").value=store.get("fps","");$("#fps").onchange=e=>store.set("fps",e.target.value);
$("#split").value=store.get("split","one");$("#split").onchange=e=>store.set("split",e.target.value);

const drop=$("#drop"),fileIn=$("#file");
drop.onclick=()=>fileIn.click();
fileIn.onchange=()=>pick(fileIn.files);
["dragover","dragenter"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",e=>pick(e.dataTransfer.files));
function pick(fl){picked=[...fl].filter(f=>f.type.startsWith("video/")||/\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(f.name));
 $("#files").textContent=picked.length?picked.map(f=>`${f.name} (${(f.size/1e6).toFixed(1)} MB)`).join(" · "):"";
 $("#go").disabled=!picked.length;}

let running=null; // {stop, xhr, job} while a batch is in progress
const newId=()=>Date.now().toString(36)+Math.random().toString(36).slice(2);
function post(fd,onUpload){return new Promise((res,rej)=>{
 const x=new XMLHttpRequest();running.xhr=x;x.open("POST","/analyze");x.responseType="json";
 x.upload.onprogress=e=>{if(e.lengthComputable)onUpload(e.loaded/e.total)};
 x.onload=()=>res({ok:x.status<300,j:x.response||{error:"Server error "+x.status}});
 x.onerror=()=>rej(new Error("Network error — is app.py still running?"));
 x.onabort=()=>rej(new Error("Stopped."));x.send(fd);})}

$("#stop").onclick=()=>{if(!running)return;running.stop=true;
 if(running.job)fetch("/cancel/"+running.job,{method:"POST"}).catch(()=>{});
 if(running.xhr)running.xhr.abort();
 $("#stop").disabled=true;$("#stop").textContent="Stopping…";};

$("#go").onclick=async()=>{
 const queue=picked;picked=[];fileIn.value="";$("#files").textContent="";
 const stop=$("#stop");$("#go").hidden=true;stop.hidden=false;stop.disabled=false;stop.textContent="Stop";
 running={stop:false};
 for(const [n,f] of queue.entries()){
  document.querySelector(".placeholder")?.remove();
  const card=document.createElement("div");card.className="card";$("#results").prepend(card);
  const name=esc(f.name)+(queue.length>1?` (${n+1}/${queue.length})`:"");
  if(running.stop){card.innerHTML=`<div class="status">Skipped: ${name}</div>`;continue}
  const st=t=>{card.innerHTML=`<div class="status"><div class="spin"></div>${name} — ${t}</div>`};
  st("uploading…");
  const job=running.job=newId();
  const fd=new FormData();fd.append("video",f);fd.append("throws",throws);fd.append("job",job);
  ["velo","fps","model"].forEach(k=>{const v=$("#"+k).value;if(v)fd.append(k,v)});
  if($("#split").value==="many")fd.append("split","1");
  const poll=setInterval(async()=>{try{const p=await (await fetch("/progress/"+job)).json();
   if(p.stage&&!running.stop)st(p.total?`${p.stage}: frame ${p.done}/${p.total} (${Math.round(100*p.done/p.total)}%)`:p.stage+"…")}catch{}},700);
  try{const {ok,j}=await post(fd,p=>{if(p<1)st(`uploading ${Math.round(p*100)}%`)});
   if(ok&&j.results){card.remove();
    for(const r of j.results){const c=document.createElement("div");c.className="card";$("#results").prepend(c);
     if(r.error)c.innerHTML=`<div class="err"><b>${esc(r.clip)}</b>: ${esc(r.error)}</div>`;else showResult(c,r);}}
   else if(ok)showResult(card,j);
   else card.innerHTML=j.cancelled?`<div class="status">Stopped: ${name}</div>`:`<div class="err"><b>${name}</b>: ${esc(j.error)}</div>`;}
  catch(e){card.innerHTML=running.stop?`<div class="status">Stopped: ${name}</div>`:`<div class="err"><b>${name}</b>: ${esc(e.message)}</div>`;}
  finally{clearInterval(poll)}
 }
 running=null;stop.hidden=true;$("#go").hidden=false;$("#go").disabled=!picked.length;$("#velo").value="";
};

const LABELS={stride_pct_height:["Stride","% height"],lead_knee_flex_at_fs_deg:["Knee @ foot strike","°"],
 lead_knee_flex_at_release_deg:["Knee @ release","°"],lead_knee_change_deg:["Knee change","° (− = brace)"],
 hip_shoulder_sep_max_deg:["Max separation","°"],shoulder_rotation_by_fs_deg:["Shoulder turn by FS","°"],
 trunk_tilt_at_release_deg:["Trunk tilt @ release","°"],arm_angle_at_release_deg:["Arm slot","°"],
 fs_to_release_ms:["Foot strike → release","ms"]};

const EVN=["Leg lift","Foot strike","Release"];
function showResult(card,j){
 const title=`<div class="rhead"><h2>${esc(j.clip.replace(/^\d{8}-\d{6}_/,""))}</h2>
  <span>${j.throws}HP · ${Math.round(j.fps)} fps${j.velo?` · ${j.velo} mph`:""}${j.confidence!=null?` · tracking ${Math.round(j.confidence*100)}%`:""}${j.events_source==="manual"?" · events set by hand":""}</span></div>`;
 const editor=`<div class="ev">
   <img class="fimg" alt="Selected frame">
   <input type="range" class="fslider" min="0" max="${j.n_frames-1}" step="1" value="0" aria-label="Frame">
   <div class="evbar"><button data-step="-5">«</button><button data-step="-1">‹</button>
    <span>frame <b class="fno">0</b> / ${j.n_frames-1}</span><button data-step="1">›</button><button data-step="5">»</button></div>
   <div class="evset">${EVN.map((n,i)=>`<button data-set="${i}">Set ${n.toLowerCase()} <b data-jump="${i}" title="Jump to this frame">—</b></button>`).join("")}</div>
   <button class="go redo" disabled>Re-analyze with these frames</button>
   <button class="swapleg" type="button" data-fix="swap_legs">Left/right mixed up (wrong lead leg or arm)? Swap sides</button>
   <button class="swapleg" type="button" data-fix="flip_dir">Plate on the wrong side? Flip direction</button>
   <div class="evmsg"></div></div>`;
 const video=`<video src="${j.video_url}" controls playsinline muted preload="metadata"></video>`;
 if(j.needs_events){
  card.innerHTML=title+`<div class="flag note"><b>Couldn't find the pitch automatically</b><p>${esc(j.message)}</p>
   <p class="tip">→ Slide to each moment, tap the matching button, then re-analyze.</p></div>
   ${editor}<details class="fix"><summary>Play replay</summary><div class="media">${video}</div></details>`;
 } else {
  card.innerHTML=title+renderBreakdown(j,editor,video);
  card.querySelectorAll(".rtab").forEach(t=>t.onclick=()=>{
   card.querySelectorAll(".rtab").forEach(x=>x.classList.toggle("on",x===t));
   card.querySelectorAll(".rpanel").forEach(p=>p.hidden=p.dataset.p!==t.dataset.p);});
 }
 bindEditor(card,j);
}

// One row per check: your number, where it sits in the OBP range, and (expanded) the tip + guide image.
const CHECKS=[
 {k:"stride_pct_height",o:"stride_pct",g:"stride",f:"Short stride",n:"Stride length",u:"% of height"},
 {k:"hip_shoulder_sep_max_deg",o:"separation_deg",g:"separation",f:"Limited hip–shoulder separation",n:"Hip–shoulder separation",u:"°"},
 {k:"shoulder_rotation_by_fs_deg",o:"shoulder_open_deg",g:"shoulders",f:"Shoulders opening early",n:"Shoulder turn by foot strike",u:"°"},
 {k:"lead_knee_change_deg",o:"knee_change_deg",g:"knee",f:"Front knee collapsing",n:"Front knee block",u:"° bend after landing",sign:true},
 {k:"trunk_tilt_at_release_deg",o:"trunk_tilt_deg",g:"trunk",f:"Upright at release",n:"Trunk tilt at release",u:"°"},
 {k:"sequence_in_order",f:"Out-of-order sequence",n:"Hips → trunk → arm sequence",seq:true},
];
function rangeBar(v,o,st){
 const lo=Math.min(o[0],v),hi=Math.max(o[4],v),pad=Math.max((hi-lo)*0.1,1),a=lo-pad,b=hi+pad,x=t=>(t-a)/(b-a)*100;
 return `<div class="rbar"><span class="r80" style="left:${x(o[0])}%;width:${x(o[4])-x(o[0])}%"></span>
  <span class="r50" style="left:${x(o[1])}%;width:${x(o[3])-x(o[1])}%"></span><span class="rmed" style="left:${x(o[2])}%"></span>
  <span class="rme ${st}" style="left:${x(v)}%"></span></div>
  <div class="rlab">OBP pitchers: middle half ${o[1].toFixed(0)}–${o[3].toFixed(0)}, median ${o[2].toFixed(0)}</div>`;
}
function renderBreakdown(j,editor,video){
 const m=j.metrics,na=new Set(j.unmeasured||[]),obp=j.obp||{};
 const flagBy=Object.fromEntries(j.flags.map(f=>[f.title,f])),guideBy=Object.fromEntries((j.guides||[]).map(g=>[g.key,g]));
 const notes=j.flags.filter(f=>f.level==="note");
 let opened=false; // only the first problem starts expanded
 const rows=CHECKS.map(c=>{
  const fl=flagBy[c.f],g=c.g&&guideBy[c.g];
  const st=na.has(c.k)?"na":fl?fl.level:"good";
  const label={leak:"Leak",watch:"Watch",good:"Good",na:"n/a"}[st];
  let val,bar="";
  if(st==="na"){val=`<b>n/a</b><small>camera view</small>`}
  else if(c.seq){val=`<b>${m.sequence_in_order?"In order":"Off"}</b><small>${m.sequence_pelvis_trunk_arm_ms.join(" / ")} ms after FS</small>`}
  else{const v=+m[c.k];val=`<b>${c.sign&&v>0?"+":""}${v.toFixed(0)}</b><small>${c.u}</small>`;if(obp[c.o])bar=rangeBar(v,obp[c.o],st)}
  const text=fl?`<p>${esc(fl.detail)}</p><p class="tip">→ ${esc(fl.tip)}</p>`:
   st==="na"?`<p>This camera angle can't measure this. Film side-on to get it.</p>`:`<p>In the normal range for OBP pitchers. Keep it there.</p>`;
  const img=g?`<img src="${g.url}" alt="${esc(g.title)}: ${esc(g.lines[0])}" loading="lazy" data-zoom>`:"";
  const open=(st==="leak"||st==="watch")&&!opened;if(open)opened=true;
  return `<details class="chk ${st}" ${open?"open":""}><summary>
   <span class="cs">${label}</span><span class="cn">${c.n}</span><span class="cv">${val}</span><span class="cb">${bar}</span><span class="chev"></span></summary>
   <div class="cbody${img?" withimg":""}"><div class="ctext">${text}</div>${img}</div></details>`;}).join("");
 const leaks=CHECKS.filter(c=>!na.has(c.k)&&flagBy[c.f]?.level==="leak").length;
 const verdict=leaks?`<div class="verdict bad"><b>${leaks} leak${leaks>1?"s":""} found</b><span>Start with the first one below.</span></div>`:
  `<div class="verdict ok"><b>No major leaks</b><span>Everything checked is within the range of OBP pitchers.</span></div>`;
 const other=[["Knee bend at foot strike",m.lead_knee_flex_at_fs_deg,"°"],["Knee bend at release",m.lead_knee_flex_at_release_deg,"°"],
  ["Arm slot",na.has("arm_angle_at_release_deg")?null:m.arm_angle_at_release_deg,"°","Shoulder-to-wrist angle at release: ~90° over the top, ~45° three-quarter, ~0° sidearm. Not good or bad — check it stays consistent."],["Foot strike → release",m.fs_to_release_ms," ms"]]
  .map(([n,v,u,tip])=>`<span${tip?` title="${esc(tip)}"`:""}><small>${n}${tip?" ⓘ":""}</small><b>${v==null?"n/a":esc(v)+u}</b></span>`).join("");
 return `<div class="rtabs"><button class="rtab on" data-p="b">Breakdown</button><button class="rtab" data-p="r">Replay</button><button class="rtab" data-p="f">Fix events</button></div>
  <div class="rpanel" data-p="b">${verdict}
   ${notes.map(f=>`<div class="notebar"><b>${esc(f.title)}:</b> ${esc(f.detail)}</div>`).join("")}
   <div class="chks">${rows}</div><div class="other">${other}</div>
   <p class="small">Arm slot = shoulder-to-wrist angle at release (~90° over the top, ~45° three-quarter, ~0° sidearm). It's a style, not a flaw. Just keep it consistent.</p></div>
  <div class="rpanel" data-p="r" hidden><div class="media">${j.keyframes_url?`<img src="${j.keyframes_url}" alt="Leg lift, foot strike and release keyframes" data-zoom>`:""}${video}</div>
   <p class="small"><a href="${j.report_url}" target="_blank">Open text report</a></p></div>
  <div class="rpanel" data-p="f" hidden><p class="small">Events wrong, or the skeleton on the wrong side? Fix it here and re-analyze.</p>${editor}</div>`;
}

function bindEditor(card,j){
 // Frame picker uses per-frame JPEGs from the server, not the <video>, so it works on iPhone too.
 const img=card.querySelector(".fimg"),sl=card.querySelector(".fslider"),fno=card.querySelector(".fno"),
  redo=card.querySelector(".redo"),msg=card.querySelector(".evmsg");
 const ev=j.events?[j.events.leg_lift,j.events.foot_strike,j.events.release]:[null,null,null];
 const last=j.n_frames-1,url=i=>`${j.frames_url}${String(i).padStart(5,"0")}.jpg${j.v}`;
 let cur=0;
 const show=f=>{cur=Math.max(0,Math.min(last,f|0));img.src=url(cur);sl.value=cur;fno.textContent=cur;
  for(const d of [1,2,3,-1])if(cur+d>=0&&cur+d<=last)new Image().src=url(cur+d)}; // prefetch neighbours
 const paint=()=>{card.querySelectorAll("[data-set]").forEach(b=>{const i=+b.dataset.set;b.querySelector("b").textContent=ev[i]??"—";b.classList.toggle("on",ev[i]!=null)});
  const ok=ev.every(x=>x!=null)&&ev[0]<ev[1]&&ev[1]<ev[2];redo.disabled=!ok;
  msg.textContent=ev.every(x=>x!=null)&&!ok?"Frames must go leg lift < foot strike < release.":""};
 sl.oninput=()=>show(+sl.value);
 card.querySelectorAll("[data-step]").forEach(b=>b.onclick=()=>show(cur+(+b.dataset.step)));
 card.querySelectorAll("[data-set]").forEach(b=>b.onclick=e=>{
  const i=+b.dataset.set;
  if(e.target.dataset.jump!=null&&ev[i]!=null){show(ev[i]);return} // tap the number to jump back
  ev[i]=cur;paint()});
 show(ev[1]??Math.round(last/2));
 redo.onclick=async()=>{redo.disabled=true;msg.innerHTML='<span class="status"><span class="spin"></span>Re-analyzing…</span>';
  try{const r=await fetch("/reanalyze",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({clip:j.clip,throws:j.throws,fps:j.fps,velo:j.velo,model:j.model,events:ev,swap_legs:j.swap_legs,flip_dir:j.flip_dir})});
   const k=await r.json();if(r.ok)showResult(card,k);else{msg.textContent=k.error;redo.disabled=false}}
  catch(e){msg.textContent=e.message;redo.disabled=false}};
 card.querySelectorAll(".swapleg").forEach(btn=>btn.onclick=async e=>{const b=e.currentTarget,key=b.dataset.fix,label=b.textContent;
  b.disabled=true;b.textContent="Re-analyzing…";
  const all=ev.every(x=>x!=null);
  const body={clip:j.clip,throws:j.throws,fps:j.fps,velo:j.velo,model:j.model,events:all?ev:null,
   swap_legs:j.swap_legs,flip_dir:j.flip_dir};body[key]=!j[key];
  try{const r=await fetch("/reanalyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
   const k=await r.json();if(r.ok)showResult(card,k);else{msg.textContent=k.error;b.disabled=false;b.textContent=label}}
  catch(err){msg.textContent=err.message;b.disabled=false;b.textContent=label}});
 paint();
}

// Tap any guide image to view it full-size; tap again to close.
document.addEventListener("click",e=>{
 const z=document.querySelector(".zoom");if(z){z.remove();return}
 const im=e.target.closest("[data-zoom]");if(!im)return;
 const d=document.createElement("div");d.className="zoom";d.innerHTML=`<img src="${im.src}" alt="${esc(im.alt)}">`;document.body.append(d);});

let hist=[];
async function loadHistory(){
 hist=await (await fetch("/history")).json();
 if(!hist.length){$("#chart").innerHTML='<div class="empty">No sessions yet.</div>';$("#htable").innerHTML="";return}
 const keys=Object.keys(hist[0]).filter(k=>!["date","clip","leaks"].includes(k)&&hist.some(r=>r[k]!==""&&!isNaN(parseFloat(r[k]))));
 const sel=$("#hmetric");const prev=sel.value||"stride_pct_height";
 sel.innerHTML=keys.map(k=>`<option value="${k}">${LABELS[k]?LABELS[k][0]:k.replace(/_/g," ")}</option>`).join("");
 sel.value=keys.includes(prev)?prev:keys[0];sel.onchange=drawChart;drawChart();
 const cols=["date","clip","velo_mph",...Object.keys(LABELS),"leaks"];
 $("#htable").innerHTML=`<table><tr>${cols.map(c=>`<th>${LABELS[c]?LABELS[c][0]:c.replace(/_/g," ")}</th>`).join("")}</tr>`+
  hist.slice().reverse().map(r=>`<tr>${cols.map(c=>`<td>${esc(c==="clip"?(r[c]||"").replace(/^\d{8}-\d{6}_/,""):(r[c]===""?(c==="velo_mph"?"—":"n/a"):r[c]))}</td>`).join("")}</tr>`).join("")+"</table>";
}
function drawChart(){
 const k=$("#hmetric").value,pts=hist.map((r,i)=>[i,parseFloat(r[k])]).filter(p=>!isNaN(p[1]));
 if(pts.length<1){$("#chart").innerHTML='<div class="empty">No data for this metric.</div>';return}
 const W=900,H=240,P=40,ys=pts.map(p=>p[1]);let lo=Math.min(...ys),hi=Math.max(...ys);if(lo===hi){lo-=1;hi+=1}
 const x=i=>P+(pts.length===1?(W-2*P)/2:(i/(pts.length-1))*(W-2*P)),y=v=>H-P+( -(v-lo)/(hi-lo))*(H-2*P);
 const path=pts.map((p,i)=>`${i?"L":"M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
 $("#chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${k} over sessions">
  <line x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}" stroke="var(--line)"/>
  <text x="4" y="${y(hi)+4}">${hi.toFixed(1)}</text><text x="4" y="${y(lo)+4}">${lo.toFixed(1)}</text>
  <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
  ${pts.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p[1])}" r="4" fill="var(--accent)"><title>${esc(hist[p[0]].date)}: ${p[1]}</title></circle>`).join("")}
  <text x="${P}" y="${H-12}">oldest</text><text x="${W-P}" y="${H-12}" text-anchor="end">latest</text></svg>`;
}
</script></body></html>"""


if __name__ == "__main__":
    ip = lan_ip()
    print("\nPitch Helper running:")
    port = int(os.environ.get("PORT", 5000))
    print(f"  this computer:  http://localhost:{port}")
    if ip:
        print(f"  phone (same Wi-Fi): http://{ip}:{port}")
    print("Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, threaded=True)
