"""speaker_sums — a short "what did this person talk about" summary per speaker.

Major speakers (>=5 utterances) get a real LLM summary: their lines are
sampled evenly across the whole timeline (a single 11-hour lane can hold
50k words — we feed a spread, not a prefix), then qwen3.8 returns 1-2
sentences. Fragment lanes get their own utterances quoted instead — cheap
and honest. Side file only (speaker_summaries.json); artifacts untouched.

CLI:  python speaker_sums.py <jobdir> [--force]
"""
import json
import re
import sys
import time
from pathlib import Path

LN_RE = re.compile(r"^\[(\d+):(\d+):(\d\d(?:\.\d+)?)\]\s*([^:]+?)\s*:\s*(.*)$")
MAX_CHARS = 7000          # per-speaker budget sent to the model
MIN_SEGS = 5              # below this it is a fragment lane (no LLM)

PROMPT = """You summarize what ONE person said during a Twitch voice call.
The lines below are ONLY that person's speech (whisper transcript; casing/punctuation unreliable, filler common). {PRON}
Return ONLY JSON: {"sum":"one to two short sentences, third person, about what THIS person talked about: topics, opinions, plans, stories, notable exchanges. Concrete, not vague ('talked about games' is worthless)."}
If the person only exchanged greetings with nothing substantive, summarize that plainly. No advice, no caveats, no quotes longer than a phrase."""


def _lane_lines(jobdir, lane):
    out = []
    for ln in (Path(jobdir) / "labeled.txt").read_text(
            encoding="utf-8", errors="replace").splitlines():
        m = LN_RE.match(ln)
        if m and (m[4] or "").strip() == lane:
            t = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            out.append((t, m[5]))
    return out


def _sample(lines, cap=MAX_CHARS):
    """Even spread across the timeline so late-hour topics are represented."""
    joined = [f"[{int(t)//3600:02d}:{int(t)%3600//60:02d}] {txt}" for t, txt in lines]
    if sum(len(s) for s in joined) <= cap:
        return "\n".join(joined)
    # pick every k-th line, growing k until the result fits the budget
    for k in range(2, 400):
        picked = joined[::k]
        if sum(len(s) for s in picked) <= cap:
            return "\n".join(picked)
    return "\n".join(joined[:cap // 40])


def _pron_hint(lane, gmap):
    g = gmap.get(lane)
    if g == "female":
        return "Evidence says this person is female — use she/her."
    if g == "male":
        return "Evidence says this person is male — use he/him."
    return "Gender unknown — use they/them."


def run(jobdir, force=False):
    jobdir = Path(jobdir)
    out = jobdir / "speaker_summaries.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    import llm_names
    if not (jobdir / "labeled.txt").exists():
        return {"skipped": "no labeled.txt"}
    if not llm_names._endpoint_up():
        return {"skipped": f"endpoint down: {llm_names.URL}"}
    people = json.loads((jobdir / "people.json").read_text(encoding="utf-8"))
    # gender precedence for pronoun hints (same as the UI):
    #   human override > spoken majority from the call > acoustic estimate
    # (voice changers fool pitch+formants; pronouns spoken about a person don't)
    gmap = {}
    try:
        for k, v in json.loads((jobdir / "voice_gender.json").read_text(encoding="utf-8")).items():
            if isinstance(v, dict) and v.get("voice_gender"):
                gmap[k] = v["voice_gender"]
    except Exception:
        pass
    try:
        from collections import Counter
        gp = json.loads((jobdir / "names_gender.json").read_text(encoding="utf-8"))
        votes = {}
        for v in gp.get("entries", []):
            lane, said = v.get("lane"), v.get("said")
            if lane and said in ("male", "female"):
                votes.setdefault(lane, Counter())[said] += 1
        for lane, c in votes.items():
            top = c.most_common(2)
            first, second = top[0][1], (top[1][1] if len(top) > 1 else 0)
            # strict majority of gender statements, minimum 2; a tie stays unknown
            if first >= 2 and first > second:
                gmap[lane] = top[0][0]
    except Exception:
        pass
    try:  # human-confirmed gender wins over everything
        gmap.update({k: v for k, v in json.loads(
            (jobdir / "gender_overrides.json").read_text(encoding="utf-8")).items()
            if v in ("male", "female")})
    except Exception:
        pass
    res, t0 = {}, time.time()
    for lane, p in sorted(people.items(),
                          key=lambda kv: -kv[1].get("talk_seconds", 0)):
        lines = _lane_lines(jobdir, lane)
        if not lines:
            continue
        if p.get("n_segments", len(lines)) < MIN_SEGS:
            txt = " / ".join(t for _, t in lines)
            res[lane] = {"sum": txt[:180] + ("…" if len(txt) > 180 else ""),
                         "kind": "quote"}
            continue
        got = None
        err = None
        for _try in range(2):
            try:
                got = llm_names._call_model(
                    llm_names.MODEL,
                    PROMPT.replace("{PRON}", _pron_hint(lane, gmap)),
                    _sample(lines), max_tokens=160)
                if got:
                    break
            except Exception as e:
                err = str(e)[:80]
                time.sleep(2)
        s = (got or {}).get("sum")
        if s:
            res[lane] = {"sum": str(s).strip(), "kind": "llm",
                         "lines_sampled": len(_sample(lines).splitlines())}
        else:
            res[lane] = {"sum": None, "kind": "error", "note": err}
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(res, indent=1, ensure_ascii=False))
    res["_secs"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    r = run(d, force="--force" in sys.argv)
    for k, v in r.items():
        if isinstance(v, dict) and v.get("sum"):
            print(f"{k:>12} [{v['kind']}] {v['sum'][:110]}")
