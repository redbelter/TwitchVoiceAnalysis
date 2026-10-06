#!/usr/bin/env python3
"""
vodpipe.py — orchestrator: Twitch URL (or local video) -> full voice analysis,
unattended, resumable.

  python vodpipe.py https://clips.twitch.tv/... or https://twitch.tv/videos/123
  python vodpipe.py C:\\Videos\\somevideo.mp4            # local file, no download
  python vodpipe.py <same> --streamer 00:15:00 --seed 02:11:37=alice
  python vodpipe.py --jobs | --status JOB | --stage NAME JOB

Jobs live under --workdir (default: ~/Downloads/vodpipe/<job_id>/).
job_id is ASCII ("v<digits>" / "c<clip>" / hash for local files) — VOD titles
contain Unicode look-alikes and '!' that break shell/ffprobe quoting.

Stages (idempotent; skipped when the artifact exists):
 download   twitch_dl.py                 -> *.mp4
 audio      ffmpeg 16k mono              -> audio.mp3
 transcribe twitch_transcribe.py         -> transcript.json/.txt/.srt
 diarize    chunk_diar.py      [DIAR py] -> rttm.json   (speech-activity VAD)
 voiceprint vp_cluster.py     [DIAR py]  -> vp_emb2.npy + vp_lab2.npy
 label      label_voices.py              -> labeled.txt + people.json
 solos      solo_track.py per lane>=20   -> solo_N_solo.wav (+timeline.json)
 reasr      twitch_transcribe.py each    -> solo_N_solo.json/.txt
 clean      map_clean.py per lane        -> <label>_clean.txt (orig timestamps)
 scan       flirt_scan.py                -> scan_flirting.txt

Cancel any time:  python vodpipe.py --cancel <job>   (also a button in the web
UI). Artifacts survive; re-running the same command resumes from cache.

--diar-python (or env VODPIPE_DIAR_PY) = a venv python that has nemo_toolkit
installed. Download/transcribe/label stages run under the current interpreter.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DIAR = os.environ.get("VODPIPE_DIAR_PY", "")
STAGES = ["download", "audio", "transcribe", "diarize", "voiceprint",
          "label", "solos", "reasr", "clean", "scan"]


def job_id_for(s):
    m = re.search(r"twitch\.tv/(?:videos/|p/)?(\d{6,})", s)
    if m:
        return "v" + m.group(1)
    m = re.search(r"clips\.twitch\.tv/([\w-]+)|twitch\.tv/\w+/clip/([\w-]+)", s)
    if m:
        return "c" + (m.group(1) or m.group(2))[:24].replace("-", "_")
    # local files: same file must hash the same however it was typed/dropped
    s = str(Path(s).resolve()).lower().replace("/", "\\")
    return "f" + hashlib.sha1(s.encode()).hexdigest()[:12]


def ffprobe_dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


class Job:
    def __init__(self, workdir: Path, jid, create=False, **kw):
        self.dir = workdir / jid
        self.path = self.dir / "state.json"
        self.id = jid
        if self.path.exists():
            self.st = json.loads(self.path.read_text(encoding="utf-8"))
        elif create:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "logs").mkdir(exist_ok=True)
            self.st = {"id": jid, "created": time.time(), **kw}
            self.save()
        else:
            sys.exit(f"[vodpipe] no job '{jid}' in {workdir}")

    def save(self):
        self.path.write_text(json.dumps(self.st, indent=1), encoding="utf-8")

    def stage(self, name):
        return self.st.setdefault("stages", {}).setdefault(name, {"status": "pending"})

    def set(self, name, status, **kw):
        s = self.stage(name)
        s["status"] = status
        s.update(kw, updated=time.time())
        self.save()

    def source(self):
        """Resolve current source video path (None until download lands)."""
        if self.st.get("input"):
            return Path(self.st["input"])
        return next(self.dir.glob("*.mp4"), None)


_pct_re = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_eta_re = re.compile(r"ETA[:\s]+([0-9]{1,2}:?[0-9:]{2,7}|\?+)")
_rate_re = re.compile(r"([\d.]+\s*[KMG]?i?B/s)|([\d.]+)\s*x\s*RT")


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"],
                             capture_output=True, text=True).stdout
        return str(int(pid)) in out
    except Exception:
        return False


def run(cmd, log, stage, job):
    print(f"[vodpipe] $ {' '.join(str(c) for c in cmd)}")
    prog = job.stage(stage).setdefault("prog", {})
    with open(log, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} :: {stage} =====\n")
        lf.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", cwd=str(job.dir))
        for line in p.stdout:
            lf.write(line)
            if line.startswith("[") and "] " in line[:15]:
                print("    " + line.rstrip()[:120])
                st = job.stage(stage)
                st["last"] = line.split("]", 1)[1].strip()[:160]
                m = _pct_re.search(line)
                if m:
                    prog["pct"] = float(m.group(1))
                m = _eta_re.search(line)
                if m:
                    prog["eta"] = m.group(1)
                m = _rate_re.search(line)
                if m:
                    prog["rate"] = (m.group(1) or m.group(2) + "x RT").strip()
                st["updated"] = time.time()
                job.save()
        rc = p.wait()
        lf.write(f"----- exit {rc}\n")
    return rc


def find_runner_pids(job):
    """Orchestrator pids running this job: recorded pid first, then a cmdline
    scan fallback for jobs started before runner_pid existed."""
    pids = []
    p = job.st.get("runner_pid")
    if p and pid_alive(p):
        return [int(p)]
    target = job.st.get("url") or ""
    if not target and job.st.get("input"):
        target = Path(job.st["input"]).name
    if not target:
        return []
    try:
        out = subprocess.run(["wmic", "process", "where", "name like '%python%'",
                              "get", "processid,commandline"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "vodpipe.py" in line and target in line:
                tok = line.rsplit(None, 1)[-1]
                if tok.isdigit():
                    pids.append(int(tok))
    except FileNotFoundError:
        pass
    return pids


def cancel_job(workdir: Path, jid: str):
    """Kill the orchestrator process tree and mark running stages cancelled.
    Artifacts on disk stay; re-running the same command resumes from cache.
    (State fields lost to a stale-save race are cosmetic only — every stage's
    done-check is artifact-based, so a mis-marked 'cancelled' self-heals.)"""
    job = Job(workdir, jid)
    pids = find_runner_pids(job)
    killed = []
    for p in pids:
        r = subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            killed.append(p)
    if killed:
        time.sleep(1.0)  # let the OS reclaim the tree before we rewrite state
    job = Job(workdir, jid)          # reload: runner may have saved while dying
    changed = []
    for k, v in job.st.get("stages", {}).items():
        if v.get("status") == "running":
            v["status"] = "cancelled"
            changed.append(k)
    job.st["cancelled"] = True
    job.st["runner_pid"] = None
    job.save()
    return {"killed_pids": killed, "cancelled_stages": changed}


def rttm_covers(job):
    p = job.dir / "rttm.json"
    if not p.exists() or not job.st.get("duration"):
        return False
    try:
        regs = json.loads(p.read_text(encoding="utf-8"))
        return bool(regs) and max(r[1] for r in regs) >= job.st["duration"] - 60
    except Exception:
        return False


def stage_plan(job, name):
    """Fresh plan per call. Returns list of (subtag, cmd, done_fn) — 0 = skip,
    1 = normal, N = dynamic fan-out (solos/reasr/clean)."""
    d, py, diar = job.dir, sys.executable, job.st.get("diar_py") or DEFAULT_DIAR
    audio, tj = d / "audio.mp3", d / "transcript.json"

    def J(p):
        return p.exists() and p.stat().st_size > 0

    if name == "download":
        if not job.st.get("url"):
            return []
        if next(d.glob("*.mp4"), None):
            return [("download", None, lambda: True)]
        return [("download",
                 [py, HERE / "twitch_dl.py", job.st["url"], "-o", d,
                  "-f", job.st.get("height", 720)],
                 lambda: next(d.glob("*.mp4"), None) is not None)]
    if name == "audio":
        src = job.source()
        return [("audio",
                 ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                  "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                  "-q:a", "4", str(audio.parent / "audio.part.mp3")],
                 lambda: J(audio))]
    if name == "transcribe":
        return [("transcribe",
                 [py, HERE / "twitch_transcribe.py", audio, "-m", "large-v3",
                  "--lang", job.st.get("lang", "en"), "--stem", "transcript"],
                 lambda: tj.exists())]
    if name == "diarize":
        return [("diarize",
                 [diar, HERE / "chunk_diar.py", audio,
                  str(job.st.get("duration")), str(d / "rttm.json")],
                 lambda: rttm_covers(job))]
    if name == "voiceprint":
        return [("voiceprint",
                 [diar, HERE / "vp_cluster.py", audio, tj, d / "rttm.json",
                  d / "voices_raw.txt", "0.40", str(d)],
                 lambda: J(d / "vp_emb2.npy") and J(d / "vp_lab2.npy"))]
    if name == "label":
        cmd = [py, HERE / "label_voices.py", "--json", tj, "--emb", d / "vp_emb2.npy",
               "--idx", d / "vp_idx2.json", "--lab", d / "vp_lab2.npy",
               "--out", d / "labeled.txt", "--people", d / "people.json"]
        if job.st.get("streamer"):
            cmd += ["--streamer", job.st["streamer"]]
        for s in job.st.get("seeds", []):
            cmd += ["--seed", s]
        return [("label", cmd,
                 lambda: J(d / "labeled.txt") and J(d / "people.json"))]
    if name == "solos":
        people = json.loads((d / "people.json").read_text(encoding="utf-8"))
        lanes = [nm for nm, p in sorted(people.items(), key=lambda x: -x[1]["talk_seconds"])
                 if nm != "STREAMER" and p["n_segments"] >= job.st.get("min_segs", 20)]
        job.st["lanes"] = lanes
        job.save()
        return [(f"solos:{i}",
                 [py, HERE / "solo_track.py", audio, tj, d / "labeled.txt",
                  d / f"solo_{i}", f"--cluster={nm}"],
                 lambda i=i: J(d / f"solo_{i}_solo.wav"))
                for i, nm in enumerate(lanes)]
    if name == "reasr":
        return [(f"reasr:{i}",
                 [py, HERE / "twitch_transcribe.py", d / f"solo_{i}_solo.wav",
                  "-m", "large-v3", "--lang", job.st.get("lang", "en")],
                 lambda i=i: J(d / f"solo_{i}_solo.json"))
                for i in range(len(job.st.get("lanes", [])))]
    if name == "clean":
        return [(f"clean:{i}",
                 [py, HERE / "map_clean.py", d / f"solo_{i}"],
                 lambda i=i: J(d / f"solo_{i}_clean.txt"))
                for i in range(len(job.st.get("lanes", [])))]
    if name == "scan":
        return [("scan",
                 [py, HERE / "flirt_scan.py", d / "labeled.txt", d / "scan_flirting.txt"],
                 lambda: J(d / "scan_flirting.txt"))]
    raise ValueError(name)


def acquire_lock(workdir: Path, jid: str):
    lk = workdir / "gpu.lock"
    if lk.exists():
        try:
            pid = int(lk.read_text(encoding="utf-8").split()[0])
            if pid != os.getpid():
                out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                     capture_output=True, text=True).stdout
                if str(pid) in out:
                    sys.exit(f"[vodpipe] GPU busy — another vodpipe holds the lock "
                             f"(pid {pid}): {lk.read_text(encoding='utf-8')}")
        except (ValueError, FileNotFoundError, IndexError):
            pass
    lk.write_text(f"{os.getpid()} {jid}", encoding="utf-8")
    return lk


def main():
    ap = argparse.ArgumentParser(prog="vodpipe", description="Twitch VOD voice-analysis pipeline")
    ap.add_argument("target", nargs="?", help="twitch url or local video file")
    ap.add_argument("--workdir", default=str(Path.home() / "Downloads" / "vodpipe"))
    ap.add_argument("--diar-python", default=DEFAULT_DIAR,
                    help="python exe for NeMo stages (or env VODPIPE_DIAR_PY)")
    ap.add_argument("-f", "--height", type=int, default=720)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--streamer", default=None, metavar="TS",
                    help="timestamp inside a host monologue; anchors the STREAMER lane")
    ap.add_argument("--seed", action="append", default=[], metavar="TS[=NAME]",
                    help="anchor the cluster speaking at TS as a named person")
    ap.add_argument("--min-segs", type=int, default=20,
                    help="minimum segments for a lane to get a solo track")
    ap.add_argument("--jobs", action="store_true")
    ap.add_argument("--status", metavar="JOB")
    ap.add_argument("--cancel", metavar="JOB")
    ap.add_argument("--stage", nargs=2, metavar=("NAME", "JOB"))
    ns = ap.parse_args()
    workdir = Path(ns.workdir)

    if ns.cancel:
        print(json.dumps(cancel_job(workdir, ns.cancel), indent=1))
        return

    if ns.jobs:
        for sp in sorted(workdir.glob("*/state.json")):
            st = json.loads(sp.read_text(encoding="utf-8"))
            done = sum(1 for v in st.get("stages", {}).values() if v["status"] == "done")
            print(f"{st['id']:26s} {done:>2}/{len(STAGES)}  "
                  f"{st.get('url') or st.get('input') or ''}")
        return
    if ns.status:
        print(json.dumps(Job(workdir, ns.status).st, indent=1))
        return
    if not ns.target:
        ap.error("need a twitch url or video file (or --jobs/--status)")

    is_url = ns.target.startswith("http") or "twitch.tv" in ns.target
    jid = job_id_for(ns.target)
    job = Job(workdir, jid, create=True,
              **({"url": ns.target} if is_url else {"input": str(Path(ns.target).resolve())}))
    job.st.update(diar_py=ns.diar_python, height=ns.height, lang=ns.lang,
                  streamer=ns.streamer, seed=ns.seed, seeds=ns.seed,
                  min_segs=ns.min_segs)
    job.save()

    if not is_url and not Path(job.st["input"]).exists():
        sys.exit(f"[vodpipe] no such file: {job.st['input']}")

    if ns.stage:  # single-stage mode
        name = ns.stage[0]
        for tag, cmd, done_fn in stage_plan(job, name):
            if done_fn():
                print(f"[vodpipe] {tag}: already done")
                continue
            rc = run(cmd, job.dir / "logs" / f"{name}.log", tag, job)
            job.set(tag, "done" if rc == 0 else "failed", rc=rc)
        return

    lk = acquire_lock(workdir, jid)
    try:
        job.st["cancelled"] = False
        job.st["runner_pid"] = os.getpid()
        job.save()
        for name in STAGES:
            # keep duration fresh (needs source, appears after download)
            src = job.source()
            if src and src.exists() and not job.st.get("duration"):
                job.st["duration"] = ffprobe_dur(src)
                job.save()
            if name in ("diarize", "voiceprint") and not job.st.get("diar_py"):
                sys.exit("[vodpipe] --diar-python (or VODPIPE_DIAR_PY) required for the "
                         "NeMo stages: a venv python with nemo_toolkit installed")
            if name == "diarize" and not job.st.get("duration"):
                sys.exit("[vodpipe] could not probe source duration — diarize needs it")
            plan = stage_plan(job, name)
            for tag, cmd, done_fn in plan:
                if done_fn():
                    job.set(tag, "done", note="cached")
                    continue
                t0 = time.time()
                job.set(tag, "running", started=t0, prog={})
                rc = run(cmd, job.dir / "logs" / f"{name}.log", tag, job)
                if rc != 0:
                    job.set(tag, "failed", rc=rc)
                    sys.exit(f"[vodpipe] {tag} failed (rc {rc}) — "
                             f"log: {job.dir/'logs'/(name+'.log')}")
                if name == "audio":  # atomic publish
                    part = job.dir / "audio.part.mp3"
                    if part.exists():
                        part.replace(audio_path(job))
                if not done_fn():
                    job.set(tag, "failed", rc=0, note="exit 0 but artifact missing")
                    sys.exit(f"[vodpipe] {tag} produced no artifact — see logs/")
                job.set(tag, "done", secs=round(time.time() - t0))
            job.set(name, "done", secs=None)
            print(f"[vodpipe] ✓ {name}")
        print(f"[vodpipe] JOB COMPLETE: {job.dir}")
    finally:
        lk = workdir / "gpu.lock"
        try:
            if str(os.getpid()) in lk.read_text(encoding="utf-8"):
                lk.unlink()
        except FileNotFoundError:
            pass


def audio_path(job):
    return job.dir / "audio.mp3"


if __name__ == "__main__":
    main()
