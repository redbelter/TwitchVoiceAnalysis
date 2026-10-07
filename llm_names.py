"""llm_names — participant-name extraction with a local-network LLM.

Uses an OpenAI-compatible endpoint (Ollama/vLLM proxy on the LAN, default
http://192.168.6.61:11444) with qwen3.8:27b to read the transcript+chat like
a human and list call-participant names with attribution:

  self     speaker announces their own name        -> speaker laneID
  address  someone is called X by name             -> target laneID (who was
           addressed; the model uses dialogue structure, e.g. reply next)
  join     announcement that X arrived/joined      -> target laneID (may be
           null; timing join-event logic handles attribution instead)

No data leaves the LAN. Disabled by env VODPIPE_LLM=0; endpoint override
VODPIPE_LLM_URL / VODPIPE_LLM_MODEL. Side-file only: names_llm.json.
The regex voter (name_suggest) merges these as 'llm-*' evidence.

  python llm_names.py <job_dir> [--max-chunks N]
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("VODPIPE_LLM_URL", "http://192.168.6.61:11444")
MODEL = os.environ.get("VODPIPE_LLM_MODEL", "qwen3.8:27b")
CHUNK_S = 600            # 10-minute chunks
MAX_CHARS = 9000         # per-chunk cap sent to the model
TIMEOUT = 180

PROMPT = """You mine Twitch stream transcripts for NAMES of people in the voice call.
Lines are "[HH:MM:SS] laneID: speech" (call audio via whisper; casing unreliable)
and "[HH:MM:SS] chat:user: message" (Twitch chat).
Return ONLY JSON: {"names":[{"name":"Bob","kind":"self|address|join|gender","said":"male|female (gender kind only, else omit)","ts":"01:02:03","speaker":"laneID|null","target":"laneID|null","quote":"short exact source text"}]}
- kind "self": speaker announces their own name ("I'm X", "my name is X", "this is X"). speaker=that line's laneID, target=null.
- kind "address": text calls/mentions a participant X; speaker=laneID of the line's lane, target=laneID of the participant ADDRESSED (they usually reply right after; from dialogue structure), or null if unclear.
- kind "join": announcement that X is arriving/joining. target=laneID if identifiable, else null.
- kind "gender": any gendered reference to a CALL PARTICIPANT: pronouns ("she said", "what did he mean"), gendered nouns ("girl", "guy", "brother", "sister", "boyfriend", "girlfriend", "lady", "woman", "man", "queen", "king", "sis", "bro"), or gendered nicknames. Set "said" to "male" or "female" and "name" to the participant's name if stated, otherwise the gendered word itself. CRITICAL: pronouns have no name attached — work out WHO is discussed from dialogue flow: the participant just named in the question/answer pair, or the person being talked about in the surrounding lines. If you cannot tell which participant, set both speaker and target to null (skip it). Do NOT guess the speaker is the subject ("I think he is right" spoken by cl0123 about someone else is about SOMEONE ELSE). If the referenced person is clearly not in the call (celebrity, streamer mentioned in chat, game character), skip it.
Use laneIDs EXACTLY as shown. Only CALL PARTICIPANTS (people in the voice call). Exclude game terms, verbs, chat-only usernames, celebrities, random nouns. Whisper text is messy: "ima Simon" counts, "im going" does not.
Example: if the line at 01:14:02 is "cl0106: did Tessa say she wanted in?" then Tessa is female and if Tessa's voice is lane cl0928 (she answered right after, or was named in nearby lines), emit {"name":"Tessa","kind":"gender","said":"female","ts":"01:14:02","speaker":"cl0106","target":"cl0928","quote":"did Tessa say she wanted in"}. If you can name the participant but cannot map them to a lane, keep target null. Emit a gender entry for EVERY gendered reference you can attribute to a participant — these entries are how we learn who is a woman or a man in the call.
If none found return {"names":[]}
"""


def _endpoint_up(timeout=4):
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=timeout):
            return True
    except Exception:
        return False


def _render(jobdir, t0, t1):
    jobdir = Path(jobdir)
    """Interleave labeled.txt + chat.json lines over [t0,t1)."""
    LNR = re.compile(r"^\[(\d+):(\d+):(\d\d(?:\.\d+)?)\]")
    lines = []
    try:
        for ln in (jobdir / "labeled.txt").read_text(encoding="utf-8",
                                                      errors="replace").splitlines():
            m = LNR.match(ln)
            if m:
                t = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
                if t0 <= t < t1:
                    lines.append((t, ln))
    except FileNotFoundError:
        return ""
    try:
        chat = json.loads((jobdir / "chat.json").read_text(encoding="utf-8"))
    except Exception:
        chat = []
    for c in chat:
        t = c.get("t", 0) or 0
        if t0 <= t < t1:
            lines.append((t, f"[{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}] "
                             f"chat:{c.get('user')}: {c.get('text')}"))
    return "\n".join(l for _, l in sorted(lines))[:MAX_CHARS]


def _ts2sec(ts):
    try:
        p = [int(x) for x in str(ts).split(":")]
        while len(p) < 3:
            p.insert(0, 0)
        return p[0] * 3600 + p[1] * 60 + p[2]
    except Exception:
        return None


def _call(content):
    body = {"model": MODEL, "temperature": 0, "stream": False,
            "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": content}]}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
    msg = r["choices"][0]["message"]
    txt = msg.get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run(jobdir, force=False, max_chunks=0):
    if os.environ.get("VODPIPE_LLM") == "0":
        return {"skipped": "disabled by env"}
    jobdir = Path(jobdir)
    out = jobdir / "names_llm.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    labeled = jobdir / "labeled.txt"
    if not labeled.exists():
        return {"skipped": "no labeled.txt"}
    if not _endpoint_up():
        return {"skipped": f"endpoint down: {URL}"}
    # job length from last labeled timestamp
    last = _job_length(labeled)
    votes = []
    t_start = time.time()
    starts = list(range(0, int(last) + CHUNK_S, CHUNK_S))
    if max_chunks:
        starts = starts[:max_chunks]
    for t0 in starts:
        t1 = t0 + CHUNK_S
        chunk = _render(jobdir, t0, t1)
        if len(chunk) < 120:
            continue
        got = None
        for _try in range(2):
            try:
                got = _call(chunk)
                if got is not None:
                    break
            except Exception:
                time.sleep(2)
        for v in (got or {}).get("names", []):
            if not isinstance(v, dict):
                continue
            nm = str(v.get("name") or "").strip()
            kind = str(v.get("kind") or "")
            if not nm or len(nm.split()) > 3 or kind not in ("self", "address", "join", "gender"):
                continue
            votes.append({"name": nm, "kind": kind,
                          "said": str(v.get("said") or "").lower() or None,
                          "ts": _ts2sec(v.get("ts")) or t0,
                          "speaker": v.get("speaker"), "target": v.get("target"),
                          "quote": str(v.get("quote") or "")[:120]})
    res = {"model": MODEL, "chunks_t": CHUNK_S, "secs": round(time.time() - t_start, 1),
           "votes": votes}
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(res, indent=1, ensure_ascii=False))
    return res


GPATH = """You analyze Twitch voice-call transcripts for GENDER evidence about the people IN THE VOICE CALL.
Lines: "[HH:MM:SS] laneID: speech" (call audio via whisper) and "[HH:MM:SS] chat:user: text" (viewers typing — viewers are NOT in the call).
The call participants speak in these voice lanes: {LANES}.
Return ONLY JSON: {"g":[{"said":"male|female","who":"participant name or null","lane":"laneID|null","ts":"HH:MM:SS","quote":"short exact text"}]}
STRICT RULES:
1. QUOTE MUST CONTAIN GENDER EVIDENCE: a pronoun (she/he/her/him/his) or gendered word (girl, guy, bro, sis, lady, woman, man, boyfriend, girlfriend, queen, king, mama, grandma, dude...) referring to a participant. Never guess gender from how a NAME sounds — name-based guessing is forbidden. No gendered word in the quote = no entry.
2. lane = the voice lane of the PERSON REFERRED TO (not the speaker). Work it out from dialogue flow: "did Tessa say she wanted in" asked at 01:14:02, and lane cl0928 answers right after -> that "she" is cl0928. Only lanes from the list above. If unresolvable, lane=null.
3. Chat greetings ("good morning X", "hi X") are NOT gender evidence. Skip chat-only usernames, celebrities, game characters, video-game objects ("he shot her" about an NPC is still a person in the game world — skip ONLY if clearly not a call participant), and people not in the call.
If nothing found return {"g":[]}"""


def _job_length(labeled_path):
    last = 0.0
    LNR = re.compile(r"^\[(\d+):(\d+):(\d\d(?:\.\d+)?)\]")
    for ln in labeled_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LNR.match(ln)
        if m:
            last = max(last, int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]))
    return last


def _call_model(model, system, user, max_tokens=1500, tries=2):
    body = {"model": model, "temperature": 0, "stream": False,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    last = None
    for _ in range(tries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
            msg = r["choices"][0]["message"]
            txt = msg.get("content") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
        except Exception as e:
            last = e
            time.sleep(2)
    if last:
        raise last
    return None


def gender_run(jobdir, force=False, max_chunks=0, model=None):
    """Dedicated gender-evidence pass: the combined name prompt reliably
    finds names but ignores gender instructions, so this gets its own
    focused call per chunk. Writes names_gender.json."""
    if os.environ.get("VODPIPE_LLM") == "0":
        return {"skipped": "disabled by env"}
    jobdir = Path(jobdir)
    out = jobdir / "names_gender.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    labeled = jobdir / "labeled.txt"
    if not labeled.exists():
        return {"skipped": "no labeled.txt"}
    if not _endpoint_up():
        return {"skipped": f"endpoint down: {URL}"}
    # lanes worth resolving: real speakers only, else the list drowns
    lanes = "none"
    try:
        people = json.loads((jobdir / "people.json").read_text(encoding="utf-8"))
        maj = sorted(((k, v.get("talk_seconds", 0)) for k, v in people.items()
                      if v.get("n_segments", 0) >= 5), key=lambda x: -x[1])[:40]
        if maj:
            lanes = ",".join(k for k, _ in maj)
    except Exception:
        pass
    sys_p = GPATH.replace("{LANES}", lanes)
    last = _job_length(labeled)
    entries = []
    t_start = time.time()
    starts = list(range(0, int(last) + CHUNK_S, CHUNK_S))
    if max_chunks:
        starts = starts[:max_chunks]
    for t0 in starts:
        chunk = _render(jobdir, t0, t0 + CHUNK_S)
        if len(chunk) < 120:
            continue
        got = None
        for _try in range(2):
            try:
                got = _call_model(model or MODEL, sys_p, chunk, max_tokens=2000)
                if got is not None:
                    break
            except Exception:
                time.sleep(2)
        for v in (got or {}).get("g", []):
            if not isinstance(v, dict):
                continue
            said = str(v.get("said") or "").lower()
            if said not in ("male", "female"):
                continue
            entries.append({"said": said,
                            "who": str(v.get("who") or "").strip() or None,
                            "lane": v.get("lane"),
                            "ts": _ts2sec(v.get("ts")) or t0,
                            "quote": str(v.get("quote") or "")[:120]})
    res = {"model": model or MODEL, "chunks_t": CHUNK_S,
           "secs": round(time.time() - t_start, 1), "entries": entries}
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    mc = 0
    if "--max-chunks" in sys.argv:
        mc = int(sys.argv[sys.argv.index("--max-chunks") + 1])
    r = run(d, force=True, max_chunks=mc)
    print(json.dumps(r, indent=1, ensure_ascii=False)[:4000])
