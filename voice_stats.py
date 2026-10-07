"""voice_stats — deterministic per-person voice profile for a vodpipe job.

Reads the job's own artifacts (people.json, labeled.txt, solo_N_solo.wav) and
produces voice_stats.json: acoustic stats (pitch via autocorrelation on the
solo track) + speaking-style stats (tempo, fillers, questions, laughter) and
a one-line plain-English description. No models, no network, runs in the app
venv; seconds per lane.

  python voice_stats.py <job_dir> [--force]

Descriptions are heuristics, not facts: pitch is measured, "personality" is
style statistics phrased as speech habits.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FMIN, FMAX = 55.0, 400.0


def _frames(x, size=1024, hop=512):
    n = 1 + max(0, (len(x) - size)) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def f0_from_signal(x, max_frames=2600):
    """(median, iqr, p10) Hz from autocorrelation on float32 mono 16k; None-set if no speech."""
    x = np.concatenate([x, np.zeros(SR)])
    fr = _frames(x)
    if len(fr) == 0:
        return None, None, None
    e = (fr ** 2).mean(axis=1)
    keep = np.argsort(e)[::-1][:max_frames]           # loudest frames = speech
    fr = fr[keep]
    fr = fr - fr.mean(axis=1, keepdims=True)
    w = np.hanning(fr.shape[1])
    fr = fr * w
    spec = np.fft.rfft(fr, n=2048, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = int(SR / FMAX), int(SR / FMIN)
    seg = ac[:, lo:hi]
    best = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), best]
    conf = (seg.max(axis=1) - seg.mean(axis=1)) / (seg.std(axis=1) + 1e-9)
    voiced = (peak > 0.25) & (conf > 3.5)
    if voiced.sum() < 30:
        return None, None, None
    f0 = SR / (best[voiced] + lo)
    return (float(np.median(f0)),
            float(np.percentile(f0, 75) - np.percentile(f0, 25)),
            float(np.percentile(f0, 10)))


def f0_stats(wavpath, max_frames=2600):
    """Median pitch + stability + low-floor from a wav (kept for solo tracks)."""
    x, sr = sf.read(str(wavpath), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:  # resample by naive decimation only for integer ratios
        import soxr  # not guaranteed present; fall back
        raise RuntimeError("solo wavs are 16k by construction")
    return f0_from_signal(x, max_frames)


# --- whole-audio reader (so lanes WITHOUT solo tracks still get pitch) ------
_audio_cache = {}


def lane_pcm(jobdir):
    """Decode job audio.mp3/wav ONCE to int16 mono 16k numpy (shared across lanes)."""
    key = str(jobdir)
    if key in _audio_cache:
        return _audio_cache[key]
    src = None
    for cand in ("audio.mp3", "audio.wav", "audio.m4a"):
        if (jobdir / cand).exists():
            src = jobdir / cand
            break
    if not src:
        return None
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR),
             "-f", "s16le", "-"], capture_output=True)
        if r.returncode != 0 or not r.stdout:
            return None
        x = np.frombuffer(r.stdout, dtype=np.int16)
    except Exception:
        return None
    _audio_cache[key] = x
    return x


def lane_ranges(jobdir):
    """rttm.json [[start,end,'speaker_N'],...] -> {N: [(s,e),...]}"""
    try:
        rttm = json.loads((jobdir / "rttm.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for seg in rttm:
        try:
            s, e, spk = seg[0], seg[1], seg[2]
        except Exception:
            continue
        n = re.search(r"(\d+)$", str(spk))
        if n:
            out.setdefault(int(n.group(1)), []).append((float(s), float(e)))
    return out


def _job_media(jobdir):
    """(pcm int16 mono 16k, transcript segments) — decoded once per job."""
    pcm = lane_pcm(jobdir)
    try:
        segs = json.loads((jobdir / "transcript.json").read_text(encoding="utf-8"))["segments"]
    except Exception:
        segs = None
    return pcm, segs


_name_ranges_cache = {}


def name_time_ranges(jobdir, nm):
    """[(start,end)] for every transcript line spoken by `nm` (per labeled.txt,
    i.e. AFTER fragment consolidation — the final attribution)."""
    key = (str(jobdir), nm)
    if key in _name_ranges_cache:
        return _name_ranges_cache[key]
    pcm, segs = _job_media(jobdir)
    out = []
    if pcm is not None:
        try:
            lines = (jobdir / "labeled.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
        for k, ln in enumerate(lines):
            m = _LN_RE.match(ln.strip())
            if not m or m[4].strip() != nm:
                continue
            # trust labeled.txt's own timestamp; borrow the segment's duration
            # only when the two line up (guards against any index drift)
            s = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            e = s + 3.0
            if segs and k < len(segs):
                d = float(segs[k].get("end", 0)) - float(segs[k].get("start", 0))
                if abs(float(segs[k].get("start", -99)) - s) < 2 and 0.1 <= d <= 30:
                    e = s + d
            out.append((s, e))
    _name_ranges_cache[key] = out
    return out


def f0_for_name(jobdir, nm, max_seconds=900):
    """Pitch for ANY major lane (incl. seeded STREAMER) by cutting its own
    transcript spans out of the job audio — no solo track needed. Other
    speakers bleed in during overlap, but autocorrelation tracks the dominant
    pitch, and gender bands are wide enough that bleed doesn't matter."""
    pcm, _ = _job_media(jobdir)
    spans = name_time_ranges(jobdir, nm)
    if pcm is None or not spans:
        return None, None, None
    chunks, tot = [], 0.0
    for s, e in sorted(spans):
        if tot >= max_seconds:
            break
        a, b = int(s * SR), min(int(e * SR), len(pcm))
        if b - a > SR // 8:                       # >125ms of audio
            chunks.append(pcm[a:b])
            tot += (b - a) / SR
    if not chunks:
        return None, None, None
    x = np.concatenate(chunks).astype("float32") / 32768.0
    return f0_from_signal(x)


def gender_of(med, p10):
    """Voice-only gender ESTIMATE with honesty about the gray zone.
    Population medians overlap ~140-170 Hz; we say so instead of guessing."""
    if med is None:
        return None, None
    if med >= 185:
        return "female", "high"
    if med >= 165 and (p10 is None or p10 >= 125):
        return "female", "likely"
    if med < 120:
        return "male", "high"
    if med < 150:
        return "male", "likely"
    return "ambiguous", None


_LN_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d(?:\.\d+)?)\]\s*\(?\s*([^:()\[\]]+?)\s*\)?:\s*(.*)")
FILL = re.compile(r"\b(?:um|uh|erm)\b", re.I)
LAUGH = re.compile(r"\b(?:ha(?:ha)+|lol+|lmao+)\b", re.I)


def text_stats(labeled_path, lane):
    secs, words, fill, laugh, q, ex, turns, last_t = 0, 0, 0, 0, 0, 0, 0, 0.0
    for ln in labeled_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LN_RE.match(ln.strip())
        if not m or m[4].strip() != lane:
            continue
        t = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        txt = m[5]
        last_t = max(last_t, t)
        turns += 1
        words += len(txt.split())
        fill += len(FILL.findall(txt))
        laugh += len(LAUGH.findall(txt))
        q += txt.rstrip().endswith("?")
        ex += txt.count("!")
    return {"turns": turns, "words": words, "fillers": fill, "laughs": laugh,
            "questions": q, "exclaims": ex, "last_t": last_t}


def pitch_word(med):
    if med is None:
        return "unmeasured pitch"
    return ("very low" if med < 95 else "low" if med < 125 else "medium"
            if med < 175 else "high" if med < 230 else "very high") + " pitch"


def describe_lane(lane, stats, total_words):
    bits = []
    g, conf = stats.get("voice_gender"), stats.get("gender_confidence")
    if g:
        bits.append(g + " voice" + (" (likely)" if conf == "likely" else ""))
    med, iqr = stats.get("f0"), stats.get("f0_iqr")
    if stats.get("f0"):
        stab = "steady" if iqr and iqr < 22 else "expressive" if iqr else ""
        bits.append(f"{pitch_word(stats['f0'])} (~{stats['f0']:.0f} Hz"
                    + (f", {stab}" if stab else "") + ")")
    wpm = stats.get("wpm")
    if wpm:
        bits.append(f"{wpm:.0f} wpm " + ("fast talker" if wpm > 200 else
                    "measured pace" if wpm > 140 else "laid-back pace"))
    if stats.get("fillers_per_100", 0) >= 2.5:
        bits.append("filler-heavy ('um/uh')")
    if stats.get("questions_per_turn", 0) >= 0.25:
        bits.append("asks a lot of questions")
    if stats.get("laughs_per_100", 0) >= 1.5:
        bits.append("laughs a lot")
    if stats.get("avg_words", 0) >= 45:
        bits.append("long storyteller turns")
    elif stats.get("avg_words") and stats["avg_words"] <= 8:
        bits.append("short reactive interjections")
    share = stats.get("word_share", 0)
    if share >= 0.25:
        bits.append(f"dominates the airtime ({share:.0%} of words)")
    elif stats.get("joined_late"):
        bits.append("joined late")
    return "; ".join(bits[:4]).capitalize() + "." if bits else "sparse speech — too few turns to profile."


def run(jobdir, force=False):
    jobdir = Path(jobdir)
    out = jobdir / "voice_stats.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    people = json.loads((jobdir / "people.json").read_text(encoding="utf-8"))
    try:
        lanes = json.loads((jobdir / "state.json").read_text(encoding="utf-8")).get("lanes") or []
    except Exception:
        lanes = []                              # solo_N index == position in this list
    labeled = jobdir / "labeled.txt"
    total_words = sum(p.get("n_segments", 0) for p in people.values()) or 1
    res = {}
    tss = {}
    for nm, p in people.items():
        if p.get("n_segments", 0) < 5:
            continue                        # fragments: nothing to profile
        tss[nm] = text_stats(labeled, nm) if labeled.exists() else {"turns": 0, "words": 0,
                    "fillers": 0, "laughs": 0, "questions": 0, "exclaims": 0, "last_t": 0}
    total_words = sum(t["words"] for t in tss.values()) or 1
    for nm, p in people.items():
        ts = tss.get(nm)
        if ts is None:
            continue
        talk = max(p.get("talk_seconds") or 0, 1e-9)
        st = {"segs": p.get("n_segments"), "words": ts["words"],
              "wpm": ts["words"] / talk * 60 if ts["words"] else None,
              "fillers_per_100": ts["fillers"] / max(ts["words"], 1) * 100,
              "laughs_per_100": ts["laughs"] / max(ts["words"], 1) * 100,
              "questions_per_turn": ts["questions"] / max(ts["turns"], 1),
              "avg_words": ts["words"] / max(ts["turns"], 1),
              "word_share": ts["words"] / total_words,
              "talk_seconds": p.get("talk_seconds"),
              "joined_late": bool(p.get("joined_late"))}
        i = lanes.index(nm) if nm in lanes else -1
        wav = jobdir / f"solo_{i}_solo.wav" if i >= 0 else None
        med = iqr = p10 = None
        if wav and wav.exists():
            try:
                med, iqr, p10 = f0_stats(wav)
            except Exception as e:
                st["f0_err"] = str(e)[:120]
        if med is None:                         # no solo track (or STREAMER):
            try:                                # cut this name's spans from audio
                med, iqr, p10 = f0_for_name(jobdir, nm)
            except Exception:
                pass
        if med is not None:
            st["f0"] = round(med, 1)
            st["f0_iqr"] = round(iqr, 1)
            st["f0_p10"] = round(p10, 1)
            g, conf = gender_of(med, p10)
            st["voice_gender"] = g
            if conf:
                st["gender_confidence"] = conf
        st["desc"] = describe_lane(nm, st, total_words)
        res[nm] = st
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    r = run(d, force="--force" in sys.argv)
    print(json.dumps({k: v["desc"] for k, v in r.items()}, indent=1, ensure_ascii=False))
