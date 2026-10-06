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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent
WORKDIR = Path.home() / "Downloads" / "vodpipe"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

_ts_re = re.compile(r"\[(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*(?:(\S+):\s*)?(.*)")


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
    import vodpipe as vp
    jid = vp.job_id_for(target)
    jdir = WORKDIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "logs").mkdir(exist_ok=True)
    spawn_log = open(jdir / "logs" / "spawn.log", "a", encoding="utf-8")
    # never DEVNULL: tracebacks would vanish and the job would die silently
    p = subprocess.Popen(cmd, stdout=spawn_log, stderr=spawn_log)
    return {"job": jid, "started_pid": p.pid}


@app.post("/api/job/{jid}/cancel")
def api_cancel(jid: str):
    job_dir(jid)
    import vodpipe as vp
    return vp.cancel_job(WORKDIR, jid)


@app.get("/api/job/{jid}")
def api_job(jid: str):
    d = job_dir(jid)
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    st["_artifacts"] = {n: (d / n).exists() for n in
                        ("labeled.txt", "people.json", "scan_flirting.txt")}
    # live download size while the mp4 is still a .part (largest wins —
    # stale parts from earlier runs can coexist)
    parts = list(d.glob("*.part"))
    st["_partial_mb"] = round(max((p.stat().st_size for p in parts), default=0) / 1048576)
    return st


@app.get("/api/job/{jid}/log")
def api_log(jid: str, stage: str, tail: int = 14):
    d = job_dir(jid)
    p = d / "logs" / f"{stage}.log"
    if not p.exists():
        return {"lines": []}
    return {"lines": p.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-tail:]}


@app.get("/api/job/{jid}/people")
def api_people(jid: str):
    p = job_dir(jid) / "people.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@app.get("/api/job/{jid}/transcript")
def api_transcript(jid: str, q: str = "", lane: str = "", limit: int = 400):
    src = job_dir(jid) / "labeled.txt"
    if not src.exists():
        return {"lines": []}
    ql = q.lower()
    out = []
    for ln in src.read_text(encoding="utf-8").splitlines():
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
table{border-collapse:collapse;width:100%;font-size:13px}
td{padding:3px 8px;border-bottom:1px solid var(--border,#222);vertical-align:top}
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
</style></head><body>
<div id=app>Loading…</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const hms=t=>{t|=0;return String(t/3600|0).padStart(2,0)+':'+String(t%3600/60|0).padStart(2,0)+':'+String(t%60).padStart(2,0)};
const STAGES=['download','audio','transcribe','diarize','voiceprint','label','solos','reasr','clean','scan'];
async function j(u,o){return (await fetch(u,o)).json()}
let _jobId=null,_people=[],_lane=null,_timer=null;

/* ---------------- jobs list ---------------- */
function jobsView(){
 stopTimer();_jobId=null;
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
 $('#url').value='';$('#seed').value='';setTimeout(tick,1200);
}
async function tick(){
 if(!$('#jobs'))return;
 const jobs=await j('/api/jobs');
 $('#jobs').innerHTML=jobs.map(J=>{
  const st=J.stages||{};let n=0;
  for(const s of STAGES) if(st[s]&&st[s].status==='done')n++;
  const chips=STAGES.map(s=>{const v=st[s];return `<span class="st ${v?v.status:''}">${s}</span>`}).join('');
  return `<div class=job><a href="#job/${J.id}" style="color:inherit;text-decoration:none"><b>${J.id}</b></a>
   <span style="opacity:.6;font-size:12px"> ${esc((J.url||J.input||'').slice(0,80))}</span>
   <div class=bar><i style="width:${n/STAGES.length*100}%"></i></div>${chips}</div>`;
 }).join('')||'<i>none yet</i>';
 _timer=setTimeout(tick,3000);
}

/* ---------------- live job view ---------------- */
function stopTimer(){if(_timer){clearTimeout(_timer);_timer=null}}
async function openJob(id){
 stopTimer();_jobId=id;_lane=null;
 $('#app').innerHTML=`<span class=back onclick="location.hash='';jobsView()">← jobs</span>
  <h1 id=jtitle>${id}</h1><div id=prog></div><div id=logbox></div>
  <div id=peoplebox></div><div id=lanebox></div>
  <div id=searchbox></div><div id=scanbox></div>`;
 jobTick();
}
async function jobTick(){
 if(_jobId===null||!$('#prog'))return;
 const J=await j('/api/job/'+_jobId);
 const st=J.stages||{};
 let n=0,act=null;
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
 if(J.cancelled)$('#prog').innerHTML+=`<div style="font-size:12px;color:#ff7b72;margin-top:4px">cancelled — re-run the launcher to resume from cache</div>`;
 else if(Object.values(st).some(v=>v.status==='failed'))
  $('#prog').innerHTML+=`<div style="font-size:12px;color:#ff7b72;margin-top:4px">failed — see log below; fix the cause and re-run resumes from cache</div>`;
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
 if(art['people.json']){const P=await j(`/api/job/${_jobId}/people`);
  _people=Object.entries(P).sort((a,b)=>b[1].talk_seconds-a[1].talk_seconds);
  $('#peoplebox').innerHTML=`<h2>People</h2><table>${_people.map(([nm,p],i)=>`<tr>
   <td class=lane>${esc(nm)}</td>
   <td>${hms(p.talk_seconds)} talk · ${p.n_segments} segs${p.joined_late?' · joined late':''}</td>
   <td><button onclick=openLane('${esc(nm)}')>open</button></td></tr>`).join('')}</table><div id=laneBox2></div>`;
  if(_lane)openLane(_lane,true);}
 $('#searchbox').innerHTML=$('#searchbox').innerHTML||
  `<h2>Transcript</h2><div style="display:flex;gap:8px;flex-wrap:wrap">
   <input id=q placeholder="keyword…" style="flex:1;min-width:160px" onkeydown="if(event.key==='Enter')doSearch()">
   <select id=laneF><option>ALL</option></select>
   <button onclick=doSearch()>Search</button></div><table id=hits></table>`;
 if(art['labeled.txt']){const sel=$('#laneF');
  if(sel.options.length===1){const P=_people.length?_people.map(x=>x[0])
   :await j(`/api/job/${_jobId}/people`).then(p=>Object.keys(p));
   sel.innerHTML='<option>ALL</option>'+P.map(x=>`<option>${esc(x)}</option>`).join('');}}
 if(art['scan_flirting.txt']&&!$('#scanbox').innerHTML){const S=await j(`/api/job/${_jobId}/scan`);
  $('#scanbox').innerHTML=`<h2>Scans</h2><details><summary>register scan (flirting etc.)</summary><pre>${esc(S.text)}</pre></details>`;}
 _timer=setTimeout(jobTick,n===STAGES.length?15000:2500);
}

async function openLane(nm,keep){
 _lane=nm;const J=await j('/api/job/'+_jobId);
 const i=(J.lanes||[]).findIndex(l=>l===nm);
 let html=`<h2>${esc(nm)}</h2>`;
 if(i>=0)html+=`<audio controls preload=none src="/media/${_jobId}/solo_${i}_solo.wav"></audio>`;
 else html+='<i style="opacity:.6">no solo track (lane below threshold)</i>';
 html+=`<table id=laneHits></table>`;
 const box=$('#laneBox2')||$('#lanebox');if(!box)return;
 box.innerHTML=html;
 const t=await j(`/api/job/${_jobId}/transcript?lane=${encodeURIComponent(nm)}&limit=500`);
 $('#laneHits').innerHTML=t.lines.map(l=>`<tr><td class=t>${hms(l.t)}</td><td>${esc(l.text)}</td></tr>`).join('');
}
async function cancelJob(){
 if(!confirm('Cancel this job? Downloaded/transcribed work stays on disk; re-running resumes.'))return;
 const r=await j(`/api/job/${_jobId}/cancel`,{method:'POST'});
 console.log(r);setTimeout(jobTick,800);
}
async function doSearch(){
 const q=$('#q').value.trim(),lane=$('#laneF').value;
 const r=await j(`/api/job/${_jobId}/transcript?q=${encodeURIComponent(q)}&lane=${encodeURIComponent(lane)}&limit=400`);
 $('#hits').innerHTML=r.lines.map(l=>`<tr><td class=t>${hms(l.t)}</td><td class=lane>${esc(l.label)}</td><td>${esc(l.text)}</td></tr>`).join('')||'<tr><td>no hits</td></tr>';
}
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
