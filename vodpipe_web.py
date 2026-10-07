#!/usr/bin/env python3
"""
vodpipe_web.py — local dashboard for vodpipe jobs.

  python vodpipe_web.py [--port 5001] [--workdir ~/Downloads/vodpipe]

  http://127.0.0.1:5001          paste url -> job list + live stage progress
  #job/<id>                      LIVE view: progress + log tail while running;
                                 people, solo players, transcripts, scans appear
                                 as each stage finishes

Localhost-only tool; artifacts on disk are the database (no DB layer).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent
WORKDIR = Path.home() / "Downloads" / "vodpipe"


def _load_local_env():
    """The .bat launchers set VODPIPE_DIAR_PY via vodpipe.local.bat, but jobs
    spawned from THIS server never saw it — dashboard jobs then died at the
    NeMo stages. Parse the same file once at import."""
    import os
    if os.environ.get("VODPIPE_DIAR_PY"):
        return
    loc = HERE / "vodpipe.local.bat"
    if not loc.exists():
        return
    import re as _re
    for m in _re.finditer(r'set\s+"?(\w+)=([^"\r\n]+)"?', loc.read_text(encoding="utf-8", errors="replace")):
        os.environ.setdefault(m.group(1), m.group(2).strip())


_load_local_env()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

_ts_re = re.compile(r"\[(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*(?:(\S+):\s*)?(.*)")


def load_jobs():
    jobs = []
    for sp in sorted(WORKDIR.glob("*/state.json")):
        j = _read_json_retry(sp) or {"id": sp.parent.name, "stages": {},
                                     "url": "", "input": "(saving…)"}
        j.setdefault("id", sp.parent.name)
        j["_size_mb"] = dir_size_mb(sp.parent)
        j["_dir"] = str(sp.parent)
        mp = sp.parent / "meta.json"
        if mp.exists():
            m = _read_json_retry(mp)
            if m:
                j["_meta"] = m
        jobs.append(j)
    return jobs


def _read_json_retry(p, tries=12):
    """Windows: os.replace can leave a file briefly un-openable by a second
    reader (share error). State files are small; a 20 ms backoff is invisible
    at dashboard poll rates and turns 'job vanished for a moment' into nothing."""
    for i in range(tries):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError):
            time.sleep(0.02)
        except json.JSONDecodeError:
            if i < tries - 1:
                time.sleep(0.02)
                continue
            return None
    return None


def _read_text_retry(p, tries=12):
    for i in range(tries):
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, FileNotFoundError):
            time.sleep(0.02)
    return ""


_size_cache = {}


def dir_size_mb(d):
    """Cached per dir (20 s TTL) — job dirs hold 1000s of small files
    (vp_wavs2 slices); walking every one on every 3 s tick would be the
    slowest part of the page."""
    key = str(d)
    hit = _size_cache.get(key)
    now = time.time()
    if hit and now - hit[1] < 20:
        return hit[0]
    try:
        mb = round(sum(f.stat().st_size for f in d.rglob("*")
                       if f.is_file() and "_stale_parts" not in str(f)) / 1048576)
    except OSError:
        mb = hit[0] if hit else 0
    _size_cache[key] = (mb, now)
    return mb


def job_dir(jid):
    d = WORKDIR / jid
    if not (d / "state.json").exists():
        raise HTTPException(404, f"no job {jid}")
    return d


@app.get("/api/jobs")
def api_jobs():
    return load_jobs()


@app.post("/api/jobs")
async def api_new(payload: dict):
    target = (payload.get("url") or payload.get("file") or "").strip()
    if not target:
        raise HTTPException(400, "need url or file")
    cmd = [sys.executable, str(HERE / "vodpipe.py"), target,
           "--workdir", str(WORKDIR)]
    if payload.get("diar_python"):
        cmd += ["--diar-python", payload["diar_python"]]
    if payload.get("seed"):
        cmd += ["--seed", payload["seed"]]
    if payload.get("streamer"):
        cmd += ["--streamer", payload["streamer"]]
    import vodpipe as vp
    jid = vp.job_id_for(target)
    jdir = WORKDIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "logs").mkdir(exist_ok=True)
    # create state NOW, synchronously: the child takes a few seconds to boot,
    # and a job that only exists after spawn disappears from the list until then
    if not (jdir / "state.json").exists():
        vp.Job(jdir.parent, jid, create=True,
               **({"url": target} if target.startswith("http") or "twitch.tv" in target
                  else {"input": target}))
    spawn_log = open(jdir / "logs" / "spawn.log", "a", encoding="utf-8")
    # never DEVNULL: tracebacks would vanish and the job would die silently
    p = subprocess.Popen(cmd, stdout=spawn_log, stderr=spawn_log)
    return {"job": jid, "started_pid": p.pid}


@app.post("/api/job/{jid}/cancel")
def api_cancel(jid: str):
    job_dir(jid)
    import vodpipe as vp
    return vp.cancel_job(WORKDIR, jid)


@app.post("/api/job/{jid}/rerun")
def api_rerun(jid: str):
    """Re-spawn a stopped/failed/cancelled job. Idempotent stages skip via
    their cached artifacts, so this resumes cheaply — and redoes exactly what
    was dropped (e.g. solos re-cut after fragment consolidation)."""
    d = job_dir(jid)
    st = _read_json_retry(d / "state.json") or {}
    import vodpipe as vp
    target = st.get("url") or st.get("input") or ""
    if not target:
        raise HTTPException(400, "job has no url/input to rerun from")
    live = (st.get("stages", {}))
    if any(v.get("status") == "running" for v in live.values()) and vp.pid_alive(st.get("runner_pid")):
        raise HTTPException(409, "already running")
    cmd = [sys.executable, str(HERE / "vodpipe.py"), target, "--workdir", str(WORKDIR)]
    if st.get("diar_py"):
        cmd += ["--diar-python", st["diar_py"]]
    for s in st.get("seeds", []):
        cmd += ["--seed", s]
    if st.get("streamer"):
        cmd += ["--streamer", st["streamer"]]
    (d / "logs").mkdir(exist_ok=True)
    with open(d / "logs" / "spawn.log", "a", encoding="utf-8") as lg:
        p = subprocess.Popen(cmd, stdout=lg, stderr=lg)
    return {"job": jid, "started_pid": p.pid}


@app.post("/api/job/{jid}/open")
def api_open(jid: str):
    d = job_dir(jid)
    subprocess.Popen(["explorer", str(d)])   # explorer returns nonzero even on success
    return {"dir": str(d)}


@app.get("/api/queues")
def api_queues():
    out = []
    for p in sorted(WORKDIR.glob("queue_*.json")):
        try:
            q = _read_json_retry(p) or {}
            q["counts"] = {s: sum(1 for it in q["items"] if it["status"] == s)
                           for s in ("pending", "running", "done", "cached",
                                     "offline", "failed", "partial")}
            out.append(q)
        except Exception:
            pass
    return out


@app.post("/api/queue")
async def api_queue(payload: dict):
    login = (payload.get("login") or "").strip()
    if not login:
        raise HTTPException(400, "need login")
    cmd = [sys.executable, str(HERE / "vodpipe_queue.py"), login,
           "--workdir", str(WORKDIR)]
    if payload.get("clips"):
        cmd.append("--clips")
    if payload.get("max_gb"):
        cmd += ["--max-gb", str(payload["max_gb"])]
    log = open(WORKDIR / f"queue_{login.lower()}.log", "a", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=log, stderr=log)
    return {"queue": login, "started_pid": p.pid}


@app.post("/api/queue/{login}/cancel")
def api_queue_cancel(login: str):
    """Kill the queue runner; /T takes its vodpipe child tree with it."""
    out = subprocess.run(["wmic", "process", "where", "name like '%python%'",
                          "get", "processid,commandline"],
                         capture_output=True, text=True).stdout
    killed = []
    for ln in out.splitlines():
        if f"vodpipe_queue.py {login}" in ln or f"vodpipe_queue.py \"{login}\"" in ln:
            m = re.search(r"(\d+)\s*$", ln.strip())
            if m:
                r = subprocess.run(["taskkill", "/PID", m.group(1), "/T", "/F"],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    killed.append(int(m.group(1)))
    return {"killed": killed}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    d = job_dir(jid)
    st = _read_json_retry(d / "state.json") or {}
    st["id"] = st.get("id") or jid
    st["_artifacts"] = {n: (d / n).exists() for n in
                        ("labeled.txt", "people.json", "scan_flirting.txt",
                         "chat.json", "voice_stats.json", "names_suggested.json")}
    meta_p = d / "meta.json"
    if meta_p.exists():
        st["_meta"] = _read_json_retry(meta_p)
    # live download size while the mp4 is still a .part (largest wins —
    # stale parts from earlier runs can coexist)
    parts = list(d.glob("*.part"))
    st["_partial_mb"] = round(max((p.stat().st_size for p in parts), default=0) / 1048576)
    # playable media for the watch view (source video if finalized, else audio)
    vids = [p for p in d.glob("*.mp4") if p.stat().st_size > 1_000_000]
    st["_video"] = max(vids, key=lambda p: p.stat().st_size).name if vids else None
    st["_audio"] = "audio.mp3" if (d / "audio.mp3").exists() else None
    fm = d / "fragmerged.json"
    st["_frag"] = round(fm.stat().st_mtime) if fm.exists() else None
    st["_size_mb"] = dir_size_mb(d)
    st["_dir"] = str(d)
    return st


@app.get("/api/job/{jid}/log")
def api_log(jid: str, stage: str, tail: int = 14):
    d = job_dir(jid)
    p = d / "logs" / f"{stage}.log"
    if not p.exists():
        return {"lines": []}
    return {"lines": _read_text_retry(p).splitlines()[-tail:]}


@app.get("/api/job/{jid}/people")
def api_people(jid: str):
    p = job_dir(jid) / "people.json"
    return _read_json_retry(p) if p.exists() else {}


@app.get("/api/job/{jid}/voices")
def api_voices(jid: str):
    """Per-person voice profile (pitch/tempo/style + one-line description).
    Computed lazily and cached in the job dir as voice_stats.json — but only
    once the pipeline has settled, so we never profile half-written solo wavs."""
    d = job_dir(jid)
    cached = d / "voice_stats.json"
    if cached.exists():
        return _read_json_retry(cached) or {}
    if not (d / "people.json").exists():
        return {}
    try:
        st = _read_json_retry(d / "state.json") or {}
        vals = list(st.get("stages", {}).values())
        running = any(v.get("status") == "running" for v in vals)
        settled = bool(vals) and not running
        # profile once the job has settled — whether it finished cleanly,
        # was cancelled, or skipped identity stages on a tiny source
        if not settled or not any(v.get("status") == "done" for v in vals):
            return {}
    except Exception:
        return {}
    try:
        import voice_stats
        return voice_stats.run(d)
    except Exception as e:
        raise HTTPException(500, f"voice stats failed: {str(e)[:200]}")


@app.get("/api/job/{jid}/names")
def api_names(jid: str):
    """Probable display names per lane, inferred from self-intros, chat
    greetings + reply latency, and turn pairs. Side file only — people.json
    is never rewritten; --seed stays the source of truth."""
    d = job_dir(jid)
    cached = d / "names_suggested.json"
    res = None
    if cached.exists():
        res = _read_json_retry(cached) or {}
    elif (d / "labeled.txt").exists():
        try:
            import name_suggest
            res = name_suggest.run(d)
        except Exception as e:
            raise HTTPException(500, f"name suggestion failed: {str(e)[:200]}")
    if res is None:
        res = {}
    ov = _read_json_retry(d / "name_overrides.json") or {}
    res["overrides"] = ov
    return res


@app.post("/api/job/{jid}/confirm_name")
async def api_confirm_name(jid: str, body: dict):
    """Human confirms a suggested name: stored in name_overrides.json (a
    side file the dashboard merges over labels). Pipeline artifacts and
    labeled.txt are never rewritten by this."""
    lane = str(body.get("lane") or "").strip()
    name = str(body.get("name") or "").strip()
    if not lane:
        raise HTTPException(400, "lane required")
    d = job_dir(jid)
    p = d / "name_overrides.json"
    ov = _read_json_retry(p) or {}
    if name:
        if len(name) > 40:
            raise HTTPException(400, "name too long")
        ov[lane] = name
    else:
        ov.pop(lane, None)          # empty name = undo the confirmation
    from vodpipe import atomic_write
    atomic_write(p, json.dumps(ov, indent=1, ensure_ascii=False))
    return {"ok": True, "overrides": ov}


@app.get("/api/job/{jid}/transcript")
def api_transcript(jid: str, q: str = "", lane: str = "", limit: int = 400):
    src = job_dir(jid) / "labeled.txt"
    if not src.exists():
        return {"lines": []}
    ql = q.lower()
    out = []
    for ln in _read_text_retry(src).splitlines():
        m = _ts_re.match(ln)
        if not m:
            continue
        lab, txt = m[4] or "", m[5]
        if lane and lane != "ALL" and lab != lane:
            continue
        if ql and ql not in txt.lower():
            continue
        out.append({"t": hms_to_s(*m.groups()[:3]), "label": lab, "text": txt})
        if len(out) >= limit:
            break
    return {"lines": out}


def hms_to_s(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


@app.get("/api/job/{jid}/scan")
def api_scan(jid: str):
    p = job_dir(jid) / "scan_flirting.txt"
    return {"text": _read_text_retry(p) if p.exists() else ""}


@app.get("/api/job/{jid}/chat")
def api_chat(jid: str, q: str = "", user: str = "", around: float = -1,
             window: float = 120, limit: int = 300):
    """VOD chat replay, optionally filtered by keyword / chatter / time window.
    `around` (seconds) + window returns chat around that moment; used by the
    transcript 'chat here' buttons."""
    p = job_dir(jid) / "chat.json"
    if not p.exists():
        return {"lines": [], "total": 0}
    msgs = _read_json_retry(p) or []
    ql = q.lower()
    out = []
    for m in msgs:
        if around >= 0 and abs(m["t"] - around) > window:
            continue
        if user and m["user"].lower() != user.lower():
            continue
        if ql and ql not in m["text"].lower():
            continue
        out.append(m)
        if len(out) >= limit:
            break
    return {"lines": out, "total": len(msgs)}


@app.get("/media/{jid}/{name}")
def media(jid: str, name: str):
    d = job_dir(jid)
    p = (d / name).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


INDEX = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VODPipe</title><style>
:root{font-family:system-ui,sans-serif}
body{margin:0;padding:24px;background:var(--card,#101014);color:var(--foreground,#e8e8ea)}
*{box-sizing:border-box}
h1{font-size:17px;margin:0 0 14px}h2{font-size:14px;margin:18px 0 8px;color:var(--muted-foreground,#9a9aa2)}
input,button,select{background:var(--foreground,#1c1c22);color:var(--foreground,#e8e8ea);
 border:1px solid var(--border,#2c2c34);border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer;background:var(--accent,#5b5bd6);border-color:transparent}
.job{border:1px solid var(--border,#2c2c34);border-radius:8px;padding:10px 12px;margin:8px 0}
.bar{height:6px;border-radius:3px;background:var(--border,#2c2c34);overflow:hidden;margin:6px 0}
.bar i{display:block;height:100%;background:var(--accent,#5b5bd6)}
.st{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;margin:2px 3px 0 0;
 border:1px solid var(--border,#2c2c34)}
.done{color:#7ee08a;border-color:#2f5e39}.failed{color:#ff7b72;border-color:#6e2b2b}
.running{color:#f0c674;border-color:#6e5b2b}
.pending{opacity:.55}.cached{color:#7ee08a;border-color:#2f5e39;opacity:.7}
.offline,.waiting{color:#9a9aa2;border-style:dashed}.partial{color:#f0c674}
table{border-collapse:collapse;width:100%;font-size:13px}
td{padding:3px 8px;border-bottom:1px solid var(--border,#222);vertical-align:top}
/* People stays compact: ~10 visible rows, scroll for the rest — a 200-speaker
   job must not push the watch view off-screen. */
.pplscroll{max-height:186px;overflow-y:auto;border:1px solid var(--border,#222);
 border-radius:8px;background:var(--card,#141414)}
.pplscroll table{font-size:12px}
.pplscroll td{padding:2px 8px}
td.t{white-space:nowrap;color:var(--muted-foreground,#9a9aa2)}
a.t{color:var(--muted-foreground,#9a9aa2);text-decoration:none;cursor:pointer}
.lane{font-weight:600}
audio{width:100%;margin:4px 0 10px}
#log{font:11px/1.5 ui-monospace,Consolas,monospace;color:var(--muted-foreground,#9a9aa2);
 background:var(--card,#15151a);padding:8px 10px;border-radius:8px;border:1px solid var(--border,#2c2c34);
 white-space:pre-wrap;max-height:170px;overflow:auto}
pre{white-space:pre-wrap;font-size:12px;background:var(--card,#15151a);
 padding:10px;border-radius:8px;border:1px solid var(--border,#2c2c34)}
details summary{cursor:pointer;color:var(--muted-foreground,#9a9aa2);font-size:13px;margin:6px 0}
.back{color:var(--accent,#8b8bff);cursor:pointer;font-size:13px}
.wpane{max-height:46vh;overflow-y:auto;border:1px solid var(--border,#333);border-radius:6px;padding:6px 8px;font-size:13px;line-height:1.5}
.wrow{padding:2px 5px;border-radius:4px;cursor:pointer}
.wrow:hover{background:var(--card,#26262c)}
.wrow.now{background:rgba(124,124,255,.20)}
.wrow .t{opacity:.45;font-size:11px;font-family:ui-monospace,monospace}
</style></head><body>
<div id=app>Loading…</div>
<script>
const $=s=>document.querySelector(s);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
// JSON.stringify handles quotes/backslashes; swap its double-quotes for
// entities so it survives sitting inside an onclick="..." attribute.
// No literal backslashes here on purpose: this JS lives in a plain Python
// string, where \\ decodes to \ before the browser ever sees it.
function jsq(s){return esc(JSON.stringify(s)).replace(/"/g,'&quot;')}
const hms=t=>{t|=0;return String(t/3600|0).padStart(2,0)+':'+String(t%3600/60|0).padStart(2,0)+':'+String(t%60).padStart(2,0)};
const STAGES=['download','audio','transcribe','diarize','voiceprint','label','frag','solos','reasr','clean','scan'];
async function j(u,o){return (await fetch(u,o)).json()}
let _jobId=null,_people=[],_lane=null,_timer=null,_voice={},_peopleKey=null,_voiceDone=false,_lanes=[],_meta=null,_hasChat=false,_names={},_nameOv={};
let _watchKey=null,_wt=[],_wc=[],_wCurT=-1,_wCurC=-1,_wFollow=true;
const wm=()=>$('#wmed');
function wcolor(nm){let h=0;for(const c of String(nm))h=(h*31+c.charCodeAt(0))%360;return `hsl(${h} 62% 68%)`}
function wseek(t){const m=wm();if(m){m.currentTime=t;if(m.paused)m.play();}}
function wsync(){
 const m=wm();if(!m)return;const t=m.currentTime;
 $('#wtime').textContent=hms(t);
 const find=(arr,from)=>{let lo=from,hi=arr.length-1;           // walk-forward binary enough for monotonic time
  if(from>=0&&from<arr.length&&arr[from].t<=t&&(from+1>=arr.length||arr[from+1].t>t))return from;
  let a=0,b=arr.length-1,r=-1;while(a<=b){const mid=(a+b)>>1;if(arr[mid].t<=t){r=mid;a=mid+1}else b=mid-1}return r;};
 for(const[pan,key]of[['#wtrans','T'],['#wchat','C']]){
  const arr=key==='T'?_wt:_wc;let idx=find(arr,key==='T'?_wCurT:_wCurC);if(idx<0)continue;
  if(key==='T')_wCurT=idx;else _wCurC=idx;
  const el=$(pan);const rows=el.children;
  if(rows[idx]){
   if(el._cur!=null&&rows[el._cur]&&el._cur!==idx)rows[el._cur].classList.remove('now');
   rows[idx].classList.add('now');el._cur=idx;
   if(_wFollow)el.scrollTop=rows[idx].offsetTop-el.clientHeight/2;}}}
async function renderWatch(J){
 const src=J._video?J._video:(J._audio?J._audio:null);
 if(!src)return;
 const prev=wm()?wm().currentTime:0,wasPlay=wm()?!wm().paused:false;
 $('#watch').innerHTML=`<video id=wmed controls preload=metadata src="/media/${_jobId}/${encodeURIComponent(src)}" style="width:100%;max-height:56vh;background:#000;border-radius:8px"></video>
  <div style="display:flex;gap:10px;align-items:center;font-size:12px;margin:6px 0;color:var(--muted-foreground)">
   <b id=wtime style="font-family:ui-monospace,monospace;color:var(--foreground)">00:00:00</b>
   <select id=wspeed onchange="wm().playbackRate=+this.value" style="font-size:12px">${[0.75,1,1.25,1.5,2].map(r=>`<option${r===1?' selected':''}>${r}×</option>`).join('')}</select>
   <label><input type=checkbox id=wfollow checked onchange="_wFollow=this.checked"> follow</label>
   <span>click any line to jump there</span></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
   <div><h3 style="margin:2px 0 4px;font-size:13px">Transcript</h3><div id=wtrans class=wpane></div></div>
   <div><h3 style="margin:2px 0 4px;font-size:13px">Chat <span id=wchatn style="font-weight:400;opacity:.5;font-size:11px"></span></h3><div id=wchat class=wpane></div></div></div>`;
 const m=wm();m.addEventListener('timeupdate',wsync);m.addEventListener('seeked',wsync);
 m.addEventListener('error',()=>{$('#wtime').textContent='media error — try a different file';});
 for(const pane of['#wtrans','#wchat'])$(pane).addEventListener('wheel',()=>{_wFollow=false;const f=$('#wfollow');if(f)f.checked=false;},{passive:true});
 _wt=(await j(`/api/job/${_jobId}/transcript?limit=99999`)).lines;
 const C=await j(`/api/job/${_jobId}/chat?limit=5000`);
 _wc=C.lines;
 $('#wtrans').innerHTML=_wt.map(l=>`<div class=wrow data-t=${l.t} onclick="wseek(${l.t})"><span class=t>${hms(l.t)}</span> <b style="color:${wcolor(l.label)}">${esc(l.label)}</b> ${esc(l.text)}</div>`).join('')||'<i style=opacity:.5>annotations arrive as the pipeline finishes</i>';
 $('#wchat').innerHTML=_wc.map(l=>`<div class=wrow data-t=${l.t} onclick="wseek(${l.t})"><span class=t>${hms(l.t)}</span> <b style="color:${wcolor(l.user)}">${esc(l.user)}</b> ${esc(l.text)}</div>`).join('')||'<i style=opacity:.5>no chat archive</i>';
 $('#wchatn').textContent=C.total>_wc.length?`(showing ${_wc.length} of ${C.total})`:(C.total?`(${C.total})`:'');
 _wCurT=_wCurC=-1;
 if(prev){m.currentTime=prev;}if(wasPlay)m.play().catch(()=>{});
 wsync();
}
function talkBar(sec,max){const w=Math.max(2,Math.min(100,sec/(max||1)*100));
 return `<span style="display:inline-block;height:5px;width:${w}%;max-width:120px;background:var(--accent,#5b5bd6);border-radius:3px;vertical-align:middle"></span>`}
function personRow(nm,p,maxTalk){
 const v=_voice[nm]||{},solo=_lanes.indexOf(nm);
 const frag=p.n_segments<5;
 const G={'female':['♀','#f778ba','voice-based estimate: female'],
          'male':['♂','#58a6ff','voice-based estimate: male'],
          'ambiguous':['?','#d29922','pitch in the male/female overlap zone']}[v.voice_gender]||['·','#555','pitch not measured'];
 const ns=_names[nm],ov=_nameOv[nm];
 const nmTip=ns?`name suggested from context (${esc(ns.confidence)} confidence):\n${esc((ns.evidence[0]?ns.evidence[0].kind+': '+ns.evidence[0].quote:''))}`:'';
 const nameCell=ov?`<span title="confirmed name for ${esc(nm)}" style="color:#7ee08a;font-weight:700">${esc(ov)}</span> <span style="opacity:.45;font-size:10px">${esc(nm)}</span>`
  :esc(nm)+`${p.joined_late?' <span style=font-size:10px>late</span>':''}${ns?` <span title="${nmTip}" style="color:#7ee08a;font-weight:600;cursor:pointer" onclick="event.stopPropagation();applyName(${jsq(nm)},${jsq(ns.name)})">💡${esc(ns.name)}</span>`:''}`;
 return `<tr${frag?' style="opacity:.55"':''}>
  <td title="${G[2]}${v.f0?' ('+v.f0+' Hz)':''}" style="text-align:center;color:${G[1]};font-size:14px">${G[0]}</td>
  <td class=lane>${nameCell}</td>
  <td style="white-space:nowrap">${hms(p.talk_seconds)} · ${p.n_segments}s ${talkBar(p.talk_seconds,maxTalk)}</td>
  <td style="font-size:12px;color:var(--muted-foreground,#9a9aa2)">${esc(v.desc||(frag?'fragment — too brief to profile':'profile pending…'))}</td>
  <td>${solo>=0?'<span class="st done" title="solo voice clip ready">✓ clip</span>':'<span style="opacity:.35" title="no solo track (lane below threshold)">—</span>'}</td>
  <td><button onclick="openLane(${jsq(nm)})">open</button></td></tr>`}
function renderPeople(){
 const box=$('#peoplebox');if(!box||_lane)return;   // never clobber an open lane
 const majors=_people.filter(([,p])=>p.n_segments>=5);
 const frags=_people.filter(([,p])=>p.n_segments<5);
 const maxT=Math.max(1,...majors.map(([,p])=>p.talk_seconds||0));
 box.innerHTML=`<h2>People <span style="font-weight:400;font-size:12px;color:var(--muted-foreground)">${majors.length} speakers · ${frags.length} fragments</span></h2>
  <div class=pplscroll><table>${majors.map(([nm,p])=>personRow(nm,p,maxT)).join('')}</table></div>
  ${frags.length?`<details style="margin-top:6px"><summary>show ${frags.length} fragment lanes (1-4 utterances: brief/crosstalk voices, no solo tracks)</summary>
   <div class=pplscroll><table>${frags.map(([nm,p])=>personRow(nm,p,0)).join('')}</table></div></details>`:''}
  <div id=laneBox2></div>`;
}

/* ---------------- jobs list ---------------- */
function jobsView(){
 stopTimer();_jobId=null;
 $('#app').innerHTML=`<h1>VODPipe — Twitch voice analysis</h1>
 <div style="display:flex;gap:8px;flex-wrap:wrap">
 <input id=url placeholder="twitch url or local video path" style="flex:1;min-width:260px">
 <input id=seed placeholder="--seed TS=Name (optional)" style="width:200px">
 <button onclick=start()>Start job</button></div>
 <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
 <input id=login placeholder="channel login — queues live + every VOD" style="flex:1;min-width:260px">
 <input id=maxgb type=number value=100 title="max disk GB" style="width:90px">
 <label style="font-size:12px;align-self:center"><input type=checkbox id=qclips> clips</label>
 <button onclick=startQueue()>Queue channel</button></div>
 <div id=queues></div>
 <h2>Jobs</h2><div id=jobs></div>`;
 tick();
}
async function startQueue(){
 const login=$('#login').value.trim();if(!login)return;
 await j('/api/queue',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({login,max_gb:+$('#maxgb').value||100,clips:$('#qclips').checked})});
 $('#login').value='';setTimeout(tick,1500);
}
async function cancelQueue(l){
 if(!confirm('Stop queue for '+l+'? Parts already downloaded stay on disk.'))return;
 await j('/api/queue/'+l+'/cancel',{method:'POST'});setTimeout(tick,800);
}
async function tickQueues(){
 const box=$('#queues');if(!box)return;
 const qs=await j('/api/queues');
 box.innerHTML=qs.length?'<h2>Channel queues</h2>'+qs.map(q=>{
  const c=q.counts||{};const tot=q.items.length;
  const fin=(c.done||0)+(c.cached||0)+(c.offline||0)+(c.failed||0);
  const items=q.items.slice(0,12).map(it=>
   '<span class="st '+it.status+'"><a href="#job/'+it.id+'" style="color:inherit;text-decoration:none">'+it.id+'</a></span>').join('');
  const more=tot>12?'<span style="opacity:.5;font-size:11px"> +'+(tot-12)+' more</span>':'';
  const trimmed=q.dropped_gb?', trimmed '+q.dropped_gb+' GB (disk budget)':'';
  const stop=c.running?'<button onclick="cancelQueue('+JSON.stringify(q.login)+')" style="margin-left:8px;padding:2px 10px;font-size:11px">stop queue</button>':'';
  return '<div class=job><b>'+q.login+'</b> '+fin+'/'+tot+
   ' <span style="opacity:.6;font-size:12px">cap '+q.cap_gb+' GB'+trimmed+'</span>'+stop+
   '<div style="margin-top:4px">'+items+more+'</div></div>'}).join(''):'';
}
async function start(){
 const url=$('#url').value.trim();if(!url)return;
 await j('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({url,seed:$('#seed').value.trim()||undefined})});
 $('#url').value='';$('#seed').value='';setTimeout(tick,1200);
}
async function tick(){
 if(!$('#jobs'))return;
 tickQueues().catch(()=>{});
 let jobs=[];
 try{jobs=await j('/api/jobs');}catch(e){_timer=setTimeout(tick,3000);return;}
 $('#jobs').innerHTML=jobs.map(J=>{
  const st=J.stages||{};let n=0;
  const vals=Object.values(st);
  for(const s of STAGES) if(st[s]&&st[s].status==='done')n++;
  const running=vals.some(v=>v.status==='running');
  const chips=STAGES.map(s=>{const v=st[s];return `<span class="st ${v?v.status:''}">${s}</span>`}).join('');
  const mb=J._size_mb||0;const size=mb>1024?(mb/1024).toFixed(1)+' GB':mb+' MB';
  const retry=!running?`<button onclick="event.preventDefault();rerunJob('${J.id}')" style="float:right;padding:2px 10px;font-size:11px">rerun</button>`:'';
  const M=J._meta||{};
  const title=M.title?esc(String(M.title).slice(0,110)):'<span style="opacity:.55">…</span>';
  const mparts=[M.kind&&M.kind!=='vod'?M.kind.toUpperCase():'',M.streamer?'@'+M.streamer:'',M.game,M.date?String(M.date).slice(0,10):'',M.duration_s?hms(M.duration_s):'',M.view_count?M.view_count+' views':''].filter(Boolean).map(esc).join(' · ');
  return `<div class=job>${retry}<a href="#job/${J.id}" style="color:inherit;text-decoration:none"><b>${title}</b>${M.url?` <a href="${esc(M.url)}" target=_blank style="color:var(--accent,#8b8bff);text-decoration:none">↗</a>`:''}</a>
   <div style="font-size:11px;color:var(--muted-foreground)">${J.id}${mparts?' · '+mparts:''}</div>
   <div style="font-size:11px;color:var(--muted-foreground);margin-top:2px">💾 ${size} · <a style="color:inherit;text-decoration:underline dotted;cursor:pointer" title="${esc(J._dir||'')} — click to open folder" onclick="event.preventDefault();openFolder('${J.id}')">${esc(J._dir||'')}</a></div>
   <div class=bar><i style="width:${n/STAGES.length*100}%"></i></div>${chips}</div>`;
 }).join('')||'<i>none yet</i>';
 _timer=setTimeout(tick,3000);
}
async function rerunJob(id){
 const r=await fetch(`/api/job/${id}/rerun`,{method:'POST'});
 if(r.status===409){alert('Job is already running');return;}
 console.log('rerun',id,await r.json().catch(()=>({})));
 setTimeout(tick,1200);
}
async function openFolder(id){await j(`/api/job/${id}/open`,{method:'POST'});}

/* ---------------- live job view ---------------- */
function stopTimer(){if(_timer){clearTimeout(_timer);_timer=null}}
async function openJob(id){
 stopTimer();_jobId=id;_lane=null;_meta=null;_hasChat=false;_watchKey=null;_wt=[];_wc=[];_peopleKey=null;_voiceDone=false;_voice={};_names={};_nameOv={};
 $('#app').innerHTML=`<span class=back onclick="location.hash='';jobsView()">← jobs</span>
  <h1 id=jtitle>${id}</h1><div id=metabox></div><div id=watch></div><div id=prog></div><div id=logbox></div>
  <div id=peoplebox></div><div id=lanebox></div>
  <div id=searchbox></div><div id=chatbox></div><div id=scanbox></div>`;
 jobTick();
}
async function jobTick(){
 if(_jobId===null||!$('#prog'))return;
 let J=null;
 try{J=await j('/api/job/'+_jobId);}
 catch(e){_timer=setTimeout(jobTick,2500);return;}   // one hiccup must never kill the loop
 _lanes=J.lanes||_lanes;
 let n=0;                       // hoisted: used by the re-arm after the try
 try{
 const st=J.stages||{};
 let act=null;
 for(const s of STAGES){const v=st[s];
  if(v&&v.status==='done')n++;
  else if(!act){const subs=Object.keys(st).filter(k=>k.startsWith(s+':'));
   const rs=subs.find(k=>st[k].status==='running');
   if(rs)act={key:rs,stage:s,sub:parseInt(rs.split(':')[1]),total:subs.length};
   else if(v&&v.status==='running')act={key:s,stage:s,sub:null,total:1};}}
 const running=act?act.stage:null;
 const chips=STAGES.map(s=>{const v=st[s];
  let d=0,t=1;const subs=Object.keys(st).filter(k=>k.startsWith(s+':'));
  if(subs.length){d=subs.filter(k=>st[k].status==='done').length;t=subs.length;}
  return `<span class="st ${v?v.status:''}">${s}${t>1?' '+d+'/'+t:''}</span>`}).join('');
 const a=act&&!J.cancelled?st[act.key]:null;
 const p=(a&&a.prog)||{};
 let frac=0;
 if(a)frac=((act.total>1)?((act.sub+(p.pct!=null?p.pct/100:0))/act.total)
        :(p.pct!=null?p.pct/100:0))/STAGES.length;
 $('#prog').innerHTML=`<div class=bar><i style="width:${Math.min(100,(n/STAGES.length+frac)*100)}%"></i></div>${chips}`;
 if(J.cancelled)$('#prog').innerHTML+=`<div style="font-size:12px;color:#ff7b72;margin-top:4px">cancelled — <button onclick="rerunJob(_jobId)" style="padding:2px 10px;font-size:11px">rerun</button> resumes from cache</div>`;
 else if(Object.values(st).some(v=>v.status==='failed'))
  $('#prog').innerHTML+=`<div style="font-size:12px;color:#ff7b72;margin-top:4px">failed — see log below; fix the cause, then <button onclick="rerunJob(_jobId)" style="padding:2px 10px;font-size:11px">rerun</button> (resumes from cache)</div>`;
 if(!running)$('#prog').innerHTML+=`<div style="font-size:12px;color:var(--muted-foreground);margin-top:4px">
  💾 ${J._size_mb>1024?(J._size_mb/1024).toFixed(1)+' GB':(J._size_mb||0)+' MB'} on disk · <a style="color:var(--accent,#8b8bff);cursor:pointer" title="${esc(J._dir||'')}" onclick="openFolder(_jobId)">${esc(J._dir||'')}</a>
  ${n<STAGES.length?` · <button onclick="rerunJob(_jobId)" style="padding:2px 10px;font-size:11px">rerun / resume</button>`:''}</div>`;
 if(a){
  const parts=[];
  if(p.pct!=null)parts.push(p.pct.toFixed(1)+'%');
  if(p.rate)parts.push(p.rate);
  if(p.eta&&p.eta!=='?')parts.push('ETA '+p.eta);
  if(act.total>1)parts.push('item '+(act.sub+1)+'/'+act.total);
  if(J._partial_mb!=null)parts.push(J._partial_mb+' MB so far');
  if(a.started)parts.push('elapsed '+hms(Date.now()/1000-a.started));
  $('#prog').innerHTML+=`<div style="font-size:12px;color:var(--muted-foreground);margin-top:4px">
   running <b>${esc(act.key)}</b> — ${esc(a.last||'starting…')}
   <button onclick=cancelJob() style="margin-left:8px;padding:2px 10px;font-size:11px">cancel job</button></div>
   ${parts.length?`<div style="font-size:12px;color:var(--accent,#8b8bff)">${parts.join(' · ')}</div>`:''}`;}
 else $('#prog').innerHTML+=`<div style="font-size:12px;color:var(--muted-foreground);margin-top:4px">${J.cancelled?'stopped':(n===STAGES.length?'complete — all results below':'not running — re-run the launcher to resume from cache')}</div>`;

 /* live log tail for whichever stage is current */
 const cur=running||STAGES.find(s=>st[s]&&st[s].status==='failed');
 if(cur){const L=await j(`/api/job/${_jobId}/log?stage=${cur}`);
  $('#logbox').innerHTML=`<details ${running?'open':''}><summary>live log — ${cur}</summary><div id=log>${esc(L.lines.join('\\n'))}</div></details>`;}

 /* artifacts appear as stages land */
 const art=J._artifacts||{};
 if(J._meta&&J._meta!==_meta){_meta=J._meta;
  $('#metabox').innerHTML=`<div style="font-size:13px;color:var(--muted-foreground);margin:-6px 0 10px">
   <b style="color:var(--foreground)">${esc(_meta.title||_jobId)}</b><br>
   ${['streamer','game','date','duration_s','view_count'].filter(k=>_meta[k]).map(k=>
    k==='duration_s'?`length ${hms(_meta[k])}`:
    k==='date'?`${String(_meta.date).slice(0,10)}`:
    k==='view_count'?`${_meta[k]} views`:`${k} ${esc(_meta[k])}`).join(' · ')}
   ${_meta.url?` · <a href="${esc(_meta.url)}" style="color:var(--accent,#8b8bff)" target=_blank>twitch ↗</a>`:''}</div>`;}
 if(art['people.json']){
  const settled=!running;                    // stable after pipeline settles
  const pk=_jobId+':'+(J._frag||0);          // frag stage rewrites people.json
  if(_peopleKey!==pk){
   const P=await j(`/api/job/${_jobId}/people`);
   _people=Object.entries(P).sort((a,b)=>b[1].talk_seconds-a[1].talk_seconds);
   try{_voice=await j(`/api/job/${_jobId}/voices`);}catch(e){_voice={};}
   try{const N=await j(`/api/job/${_jobId}/names`);_names=(N&&N.proposals)||{};_nameOv=(N&&N.overrides)||{};}catch(e){_names={};_nameOv={};}
   _peopleKey=pk; renderPeople();
  } else if(settled&&!_voiceDone){
   try{const v=await j(`/api/job/${_jobId}/voices`);
    if(Object.keys(v).length){_voice=v;_voiceDone=true;renderPeople();if(_lane)openLane(_lane);}}catch(e){}
   try{const N=await j(`/api/job/${_jobId}/names`);
    if(N&&(N.proposals||N.overrides)){_names=N.proposals||{};_nameOv=N.overrides||{};renderPeople();if(_lane)openLane(_lane);}}catch(e){}
  }
 }
 $('#searchbox').innerHTML=$('#searchbox').innerHTML||
  `<h2>Transcript</h2><div style="display:flex;gap:8px;flex-wrap:wrap">
   <input id=q placeholder="keyword…" style="flex:1;min-width:160px" onkeydown="if(event.key==='Enter')doSearch()">
   <select id=laneF><option>ALL</option></select>
   <button onclick=doSearch()>Search</button></div><table id=hits></table>`;
 if(art['labeled.txt']){const sel=$('#laneF');
  if(sel.options.length===1){const P=_people.filter(([,p])=>p.n_segments>=5).map(x=>x[0]);
   sel.innerHTML='<option>ALL</option>'+P.map(x=>`<option>${esc(x)}</option>`).join('');}}
 if(art['chat.json']){_hasChat=true;
  if(!$('#chatbox').innerHTML)$('#chatbox').innerHTML=
   `<h2>Chat replay</h2><div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id=cq placeholder="chat keyword…" style="flex:1;min-width:160px" onkeydown="if(event.key==='Enter')doChat()">
    <button onclick=doChat()>Search chat</button></div><table id=chathits></table>`;}
 /* watch player: render once media + annotations exist; re-render if media or labels change */
 const wk=_jobId+':'+(J._video||J._audio||'none')+':'+(J._frag||0);
 if(art['labeled.txt']&&wk!==_watchKey){_watchKey=wk;await renderWatch(J);}
 if(art['scan_flirting.txt']&&!$('#scanbox').innerHTML){const S=await j(`/api/job/${_jobId}/scan`);
  $('#scanbox').innerHTML=`<h2>Scans</h2><details><summary>register scan (flirting etc.)</summary><pre>${esc(S.text)}</pre></details>`;}
 }catch(e){/* a render error must never stop the poll loop */}
 _timer=setTimeout(jobTick,n===STAGES.length?15000:2500);
}

async function openLane(nm,keep){
 // NEVER fetch here: job state is already cached by jobTick (_lanes) — an
 // await that rejects used to leave _lane set with nothing rendered, and
 // every later click then died silently on the missing box.
 _lane=nm;
 const i=_lanes.indexOf(nm);
 const v=_voice[nm]||{};
 let html=`<span class=back onclick=closeLane()>← all people</span>
  <h2>${esc(nm)} <span style="font-weight:400;font-size:12px;color:var(--muted-foreground)">(${hms(v.talk_seconds||0)} talk · ${v.segs||'?'} segs${v.joined_late?' · joined late':''})</span></h2>`;
 if(v.desc)html+=`<div style="font-size:13px;color:var(--accent,#8b8bff);margin:2px 0 8px">${esc(v.desc)}</div>`;
 if(i>=0)html+=`<audio controls preload=metadata src="/media/${_jobId}/solo_${i}_solo.wav"></audio>
   <div style="font-size:11px;color:var(--muted-foreground)">solo track — only ${esc(nm)}'s segments, stitched; timestamps stay in original VOD time via "clean" tab below</div>`;
 else html+='<i style="opacity:.6">no solo track (lane below solo threshold)</i>';
 html+=`<table id=laneHits></table>`;
 let box=$('#laneBox2');
 if(!box){
  if($('#peoplebox')){$('#peoplebox').insertAdjacentHTML('beforeend','<div id=laneBox2></div>');box=$('#laneBox2');}
  else box=$('#lanebox');
 }
 if(!box){_lane=null;return;}   // page gone (navigated) — don't stay wedged
 box.innerHTML=html;
 try{
  const t=await j(`/api/job/${_jobId}/transcript?lane=${encodeURIComponent(nm)}&limit=500`);
  const tb=$('#laneHits');
  if(tb)tb.innerHTML=t.lines.map(l=>`<tr><td class=t>${hms(l.t)}</td><td>${esc(l.text)}</td></tr>`).join('')||'<tr><td style=opacity:.5>no lines</td></tr>';
 }catch(e){const tb=$('#laneHits');if(tb)tb.innerHTML='<tr><td style="color:#ff7b72">transcript load failed — retry</td></tr>';}
}
function closeLane(){_lane=null;renderPeople();}
async function applyName(lane,name){
 if(!confirm(`Confirm "${name}" as the name for ${lane}?\nStored as a label override (artifacts untouched).`))return;
 try{const r=await j(`/api/job/${_jobId}/confirm_name`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({lane,name})});
  if(r&&r.overrides){_nameOv=r.overrides;renderPeople();if(_lane)openLane(_lane);}}catch(e){alert('confirm failed: '+e)}
}
async function cancelJob(){
 if(!confirm('Cancel this job? Downloaded/transcribed work stays on disk; re-running resumes.'))return;
 const r=await j(`/api/job/${_jobId}/cancel`,{method:'POST'});
 console.log(r);setTimeout(jobTick,800);
}
async function doSearch(){
 const q=$('#q').value.trim(),lane=$('#laneF').value;
 const r=await j(`/api/job/${_jobId}/transcript?q=${encodeURIComponent(q)}&lane=${encodeURIComponent(lane)}&limit=400`);
 $('#hits').innerHTML=r.lines.map(l=>`<tr><td class=t>${hms(l.t)}</td><td class=lane>${esc(l.label)}</td><td>${esc(l.text)}</td>
  ${_hasChat?`<td><a class=t onclick="showChat(${l.t})">💬</a></td>`:''}</tr>`).join('')||'<tr><td>no hits</td></tr>';
}
async function doChat(){
 const q=$('#cq').value.trim();
 const r=await j(`/api/job/${_jobId}/chat?q=${encodeURIComponent(q)}&limit=300`);
 $('#chathits').innerHTML=`<tr><td colspan=3 style=opacity:.6>${r.total} chat messages total, showing ${r.lines.length}</td></tr>`+
  r.lines.map(m=>`<tr><td class=t>${hms(m.t)}</td><td class=lane>${esc(m.user)}</td><td>${esc(m.text)}</td></tr>`).join('')||'<tr><td>no chat hits</td></tr>';
}
async function showChat(t){
 const r=await j(`/api/job/${_jobId}/chat?around=${t}&window=45&limit=60`);
 $('#chatbox').innerHTML=`<h2>Chat around ${hms(t)} <span style="font-weight:400;font-size:12px">±45 s — <a class=t onclick=closeChat()>clear</a></span></h2>
  <table>${r.lines.map(m=>`<tr><td class=t>${hms(m.t)}</td><td class=lane>${esc(m.user)}</td><td>${esc(m.text)}</td></tr>`).join('')||'<tr><td>silent chat then</td></tr>'}</table>`;
 $('#chatbox').scrollIntoView({behavior:'smooth'});
}
function closeChat(){$('#chatbox').innerHTML='';}
function route(){const h=location.hash.slice(1);h.startsWith('job/')?openJob(h.slice(4)):jobsView()}
window.onhashchange=route;
route();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 5001
    global WORKDIR
    if "--workdir" in sys.argv:
        WORKDIR = Path(sys.argv[sys.argv.index("--workdir") + 1])
    WORKDIR.mkdir(parents=True, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
