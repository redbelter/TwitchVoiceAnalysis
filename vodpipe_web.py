#!/usr/bin/env python3
"""
vodpipe_web.py — local dashboard for vodpipe jobs.

  python vodpipe_web.py [--port 5001] [--workdir ~/Downloads/vodpipe]

  http://127.0.0.1:5001          paste url -> job list + live stage progress
  /job/<id>                      people, transcript search, solo players, scans

Localhost-only tool; artifacts on disk are the database (no DB layer).
"""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent
WORKDIR = Path.home() / "Downloads" / "vodpipe"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

_ts_re = re.compile(r"\[(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*(?:(\S+):\s*)?(.*)")


def hms_to_s(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_jobs():
    jobs = []
    for sp in sorted(WORKDIR.glob("*/state.json")):
        try:
            jobs.append(json.loads(sp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return jobs


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
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started_pid": p.pid}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    d = job_dir(jid)
    return json.loads((d / "state.json").read_text(encoding="utf-8"))


@app.get("/api/job/{jid}/log/{stage}")
def api_log(jid: str, stage: str, tail: int = 60):
    d = job_dir(jid)
    p = d / "logs" / f"{stage}.log"
    if not p.exists():
        return {"lines": []}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]
    return {"lines": lines}


@app.get("/api/job/{jid}/people")
def api_people(jid: str):
    d = job_dir(jid)
    p = d / "people.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/job/{jid}/transcript")
def api_transcript(jid: str, q: str = "", lane: str = "", limit: int = 200):
    d = job_dir(jid)
    src = d / "labeled.txt"
    if not src.exists():
        return {"lines": []}
    ql = q.lower()
    out = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        m = _ts_re.match(ln)
        if not m:
            continue
        t = hms_to_s(*m.groups()[:3])
        lab, txt = m[4] or "", m[5]
        if lane and lab != lane:
            continue
        if ql and ql not in txt.lower():
            continue
        out.append({"t": round(t, 1), "label": lab, "text": txt})
        if len(out) >= limit:
            break
    return {"lines": out}


@app.get("/api/job/{jid}/scan")
def api_scan(jid: str):
    d = job_dir(jid)
    p = d / "scan_flirting.txt"
    return {"text": p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""}


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
input,button{background:var(--foreground,#1c1c22);color:var(--foreground,#e8e8ea);
 border:1px solid var(--border,#2c2c34);border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer;background:var(--accent,#5b5bd6);border-color:transparent}
.job{border:1px solid var(--border,#2c2c34);border-radius:8px;padding:10px 12px;margin:8px 0}
.bar{height:6px;border-radius:3px;background:var(--border,#2c2c34);overflow:hidden;margin:6px 0}
.bar i{display:block;height:100%;background:var(--accent,#5b5bd6)}
.st{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;margin:2px 3px 0 0;
 border:1px solid var(--border,#2c2c34)}
.done{color:#7ee08a;border-color:#2f5e39}.failed{color:#ff7b72;border-color:#6e2b2b}
.running{color:#f0c674;border-color:#6e5b2b}
table{border-collapse:collapse;width:100%;font-size:13px}
td{padding:3px 8px;border-bottom:1px solid var(--border,#222);vertical-align:top}
td.t{white-space:nowrap;color:var(--muted-foreground,#9a9aa2);cursor:pointer;text-decoration:underline}
.lane{font-weight:600}
audio{width:100%;margin:4px 0 10px}
pre{white-space:pre-wrap;font-size:12px;background:var(--card,#15151a);
 padding:10px;border-radius:8px;border:1px solid var(--border,#2c2c34)}
.back{color:var(--accent,#8b8bff);cursor:pointer;font-size:13px}
</style></head><body>
<div id=app>Loading…</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const hms=t=>{t|=0;return String(t/3600|0).padStart(2,0)+':'+String(t%3600/60|0).padStart(2,0)+':'+String(t%60).padStart(2,0)};
const STAGES=['download','audio','transcribe','diarize','voiceprint','label','solos','reasr','clean','scan'];
async function j(u,o){return (await fetch(u,o)).json()}
function jobsView(){
 $('#app').innerHTML=`<h1>VODPipe — Twitch voice analysis</h1>
 <div style="display:flex;gap:8px;flex-wrap:wrap">
 <input id=url placeholder="twitch url or local video path" style="flex:1;min-width:260px">
 <input id=seed placeholder="--seed TS=Name (optional)" style="width:200px">
 <button onclick=start()>Start job</button></div>
 <h2>Jobs</h2><div id=jobs></div>`;
 tick();
}
async function start(){
 const url=$('#url').value.trim();if(!url)return;
 await j('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({url,seed:$('#seed').value.trim()||undefined})});
 $('#url').value='';$('#seed').value='';setTimeout(tick,1500);
}
async function tick(){
 if(!$('#jobs'))return;
 const jobs=await j('/api/jobs');
 $('#jobs').innerHTML=jobs.map(J=>{
  const st=J.stages||{};let n=0;
  for(const s of STAGES) if(st[s]&&st[s].status==='done')n++;
  const chips=STAGES.map(s=>{const v=st[s];return `<span class="st ${v?v.status:''}">${s}</span>`}).join('');
  return `<div class=job><b><a href="#job/${J.id}" onclick="openJob('${J.id}');return false">${J.id}</a></b>
   <span style="opacity:.6;font-size:12px"> ${esc((J.url||J.input||'').slice(0,80))}</span>
   <div class=bar><i style="width:${n/STAGES.length*100}%"></i></div>${chips}</div>`;
 }).join('')||'<i>none yet</i>';
 setTimeout(tick,3000);
}
async function openJob(id){
 const J=await j('/api/job/'+id);
 const people=await j(`/api/job/${id}/people`);
 const lanes=J.lanes||[];
 const plist=Object.entries(people).sort((a,b)=>b[1].talk_seconds-a[1].talk_seconds);
 $('#app').innerHTML=`<span class=back onclick="location.hash='';jobsView()">← jobs</span>
  <h1>${id}</h1>
  ${J.stages&&J.stages.label&&J.stages.label.status==='done'?`
  <h2>People</h2><table>${plist.map(([nm,p],i)=>`<tr><td class=lane>${esc(nm)}</td>
   <td>${hms(p.talk_seconds)} talk · ${p.n_segments} segs${p.joined_late?' · joined late':''}</td>
   <td>${nm!=='STREAMER'&&p.n_segments>=10?`<button onclick=showLane('${esc(nm)}')>listen</button>`:''}</td></tr>`).join('')}</table>
  <div id=laneBox></div>
  <h2>Transcript search</h2>
  <div style="display:flex;gap:8px"><input id=q placeholder="keyword…" style="flex:1">
  <input id=laneF placeholder="lane filter e.g. STREAMER" style="width:180px">
  <button onclick=doSearch()>Search</button></div>
  <table id=hits></table>`:'<i>pipeline still running — check chips on jobs page</i>'}`;
 window._jobId=id;
}
async function showLane(nm){
 const J=await j('/api/job/'+window._jobId);
 const i=(J.lanes||[]).findIndex(l=>l===nm);
 let html=`<h2>${esc(nm)} — solo track</h2>`;
 if(i>=0){
  html+=`<audio controls src="/media/${window._jobId}/solo_${i}_solo.wav"></audio>`;
  const t=await j(`/api/job/${window._jobId}/transcript?lane=${encodeURIComponent(nm)}&limit=400`);
  html+=t.lines.map(l=>`<tr><td class=t onclick="seek(${l.t})">${hms(l.t)}</td><td>${esc(l.text)}</td></tr>`).join('');
 } else html+='<i>lane below solo threshold</i>';
 $('#laneBox').innerHTML=html;
}
function seek(t){ /* transcript rows carry original timestamps; deep-link twitch only */
 const h=location.hash.slice(1);const id=h.slice(4);
 if(/^v\d+$/.test(id))window.open('https://www.twitch.tv/videos/'+id.slice(1)+'?t='+Math.floor(t)+'s','_blank');
}
async function doSearch(){
 const q=$('#q').value.trim(),lane=$('#laneF').value.trim();
 const r=await j(`/api/job/${window._jobId}/transcript?q=${encodeURIComponent(q)}&lane=${encodeURIComponent(lane)}&limit=300`);
 $('#hits').innerHTML=r.lines.map(l=>`<tr><td class=t>${hms(l.t)}</td><td class=lane>${esc(l.label)}</td><td>${esc(l.text)}</td></tr>`).join('')||'<tr><td>no hits</td></tr>';
}
(window.onhashchange=()=>{const h=location.hash.slice(1);h.startsWith('job/')?openJob(h.slice(4)):jobsView()})();
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
