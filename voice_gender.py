"""voice_gender — two-factor acoustic gender estimate.

Pitch (f0) alone is unreliable: Discord pitch-shifters, voice training, and
naturally high male voices all land in the "female" pitch band. Vocal-tract
length is the stronger signal — you can shift pitch, you can't shorten your
throat. This module computes:

  f0     median fundamental (autocorrelation, octave-corrected)
  GFD    geometric mean of LPC formants F1..F4 (vocal-tract proxy)
         adult male ~<1100 Hz, adult female ~>1200 Hz on this measure

and combines them. High f0 + male-range GFD = "likely male (pitch-shifted or
naturally high)" — which matches what a listener HEARS, unlike pitch alone.

Writes voice_gender.json; voice_stats.json stays untouched (the dashboard
merges this file over its pitch-only verdict).

  python voice_gender.py <job_dir> [--force]
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000


def _frames(x, size=1024, hop=512):
    n = 1 + max(0, (len(x) - size)) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def f0_octave(x, max_frames=2600):
    """(f0, octave-corrected f0, subharmonic ratio) via autocorrelation."""
    x = np.concatenate([x, np.zeros(SR)])
    fr = _frames(x)
    if len(fr) == 0:
        return None, None, None
    e = (fr ** 2).mean(axis=1)
    keep = np.argsort(e)[::-1][:max_frames]
    fr = fr[keep]
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = fr * np.hanning(fr.shape[1])
    spec = np.fft.rfft(fr, n=2048, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = int(SR / 400.0), int(SR / 55.0)
    seg = ac[:, lo:hi]
    best = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), best]
    conf = (seg.max(axis=1) - seg.mean(axis=1)) / (seg.std(axis=1) + 1e-9)
    voiced = (peak > 0.25) & (conf > 3.5)
    if voiced.sum() < 30:
        return None, None, None
    rows = np.arange(ac.shape[0])[voiced]
    lags = best[voiced]
    f0 = SR / (lags + lo)
    dbl = np.clip(2 * lags + lo, 0, ac.shape[1] - 1)
    ratio = ac[rows, dbl] / (ac[rows, lags + lo] + 1e-9)
    octf = f0.copy()
    octf[ratio >= 0.9] /= 2.0            # double-lag as strong: pitch was halved
    return (float(np.median(f0)), float(np.median(octf)), float(np.median(ratio)))


def _lpc_a(x, order=14):
    n = len(x)
    r = np.correlate(x, x, "full")[n - 1:n + order]
    if r[0] <= 0:
        return None
    r[0] *= 1.000001
    r[0] += 1e-10
    from scipy.linalg import toeplitz
    try:
        return np.linalg.solve(toeplitz(r[:order]), r[1:order + 1])
    except np.linalg.LinAlgError:
        return None


def gfd(x, max_frames=900, nfft=4096):
    """Median geometric-mean formant dispersion using the FIRST FOUR formant
    peaks in ascending frequency (F1..F4) — the literature measure. Taking the
    loudest peaks instead would pick F2..F5 on male voices and fake the scale."""
    x = np.concatenate([x, np.zeros(SR)])
    fr = _frames(x)
    if len(fr) == 0:
        return None
    e = (fr ** 2).mean(axis=1)
    fr = fr[np.argsort(e)[::-1][:max_frames]]
    freqs = np.fft.rfftfreq(nfft, 1.0 / SR)
    band = (freqs >= 250) & (freqs <= 4500)
    fband = freqs[band]
    out = []
    for f in fr:
        f = f - f.mean()
        a = _lpc_a(f * np.hanning(len(f)))
        if a is None:
            continue
        S = 1.0 / np.abs(np.fft.rfft(np.concatenate([[1.0], -a]), nfft)) ** 2
        Sb = 10 * np.log10(S[band] + 1e-12)
        # local maxima that stand >= 3 dB above the preceding valley
        fs, prev_min = [], Sb[0]
        for i in range(1, len(Sb) - 1):
            if Sb[i] > Sb[i - 1] and Sb[i] >= Sb[i + 1] and Sb[i] - prev_min >= 3:
                fs.append(fband[i])
                if len(fs) >= 4:
                    break
            prev_min = min(prev_min, Sb[i])
        if len(fs) < 4:
            continue
        out.append((fs[0] * fs[1] * fs[2] * fs[3]) ** 0.25)
    if len(out) < 40:
        return None
    return float(np.median(out))


_LN_RE = re.compile(r"\[(\d+):(\d+):(\d\d(?:\.\d+)?)\]\s*([^:]+?)\s*:\s*(.*)$")


def lane_audio(jobdir, nm, lanes, max_seconds=900):
    """float32 mono 16k of this lane's speech: solo track if present, else
    cut from audio.mp3 by labeled.txt timestamps."""
    import soundfile as sf
    i = lanes.index(nm) if nm in lanes else -1
    solo = jobdir / f"solo_{i}_solo.wav" if i >= 0 else None
    if solo and solo.exists():
        x, sr = sf.read(str(solo), dtype="float32")
        return (x.mean(axis=1) if x.ndim > 1 else x)
    # fallback: decode job audio once (cached)
    key = "__pcm__" + str(jobdir)
    if key not in lane_audio.__dict__:
        import subprocess
        src = None
        for cand in ("audio.mp3", "audio.wav", "audio.m4a"):
            if (jobdir / cand).exists():
                src = jobdir / cand
                break
        if not src:
            return None
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1",
                            "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
        if r.returncode != 0 or not r.stdout:
            return None
        lane_audio.__dict__[key] = np.frombuffer(r.stdout, dtype=np.int16)
    pcm = lane_audio.__dict__[key]
    spans = []
    try:
        lines = (jobdir / "labeled.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for ln in lines:
        m = _LN_RE.match(ln.strip())
        if m and m[4].strip() == nm:
            s = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            spans.append((s, s + 3.0))       # 3 s window per line; enough for formants
    if not spans:
        return None
    chunks, tot = [], 0.0
    for s, e in sorted(spans):
        if tot >= max_seconds:
            break
        a, b = int(s * SR), min(int(e * SR), len(pcm))
        if b - a > SR // 4:
            chunks.append(pcm[a:b])
            tot += (b - a) / SR
    if not chunks:
        return None
    return np.concatenate(chunks).astype("float32") / 32768.0


def classify(f0, g):
    """Absolute sanity only; job-relative logic in run()."""
    if f0 is None:
        return None, None, "no pitch"
    if g is None:
        if f0 >= 175:
            return "female", "likely", "pitch only — no formant data"
        if f0 < 150:
            return "male", "likely", "pitch only — no formant data"
        return "ambiguous", None, None
    if f0 < 150 and g < 1500:
        return "male", "high", None
    if f0 >= 175 and g >= 1700:
        return "female", "high", None
    return None, None, None


def run(jobdir, force=False):
    jobdir = Path(jobdir)
    out = jobdir / "voice_gender.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    t0 = time.time()
    people = json.loads((jobdir / "people.json").read_text(encoding="utf-8"))
    try:
        lanes = json.loads((jobdir / "state.json").read_text(encoding="utf-8")).get("lanes") or []
    except Exception:
        lanes = []
    raw = {}
    for nm, p in people.items():
        if p.get("n_segments", 0) < 5:
            continue
        x = lane_audio(jobdir, nm, lanes)
        if x is None or len(x) < SR:
            continue
        f0, f0o, sub = f0_octave(x)
        g = gfd(x)
        f0use = f0o if f0o else f0
        raw[nm] = {"f0": f0use and round(f0use, 1),
                   "f0_raw": f0 and round(f0, 1),
                   "gfd": g and round(g),
                   "voice_gender": None, "confidence": None, "note": None}

    # --- job-relative formant reasoning --------------------------------------
    # Absolute GFD thresholds are unreliable on 16 kHz compressed Discord
    # audio (and pitch-shift moves formants too, fooling BOTH measures).
    # Within one job, however, the tract MEASUREMENT spread is comparable,
    # so we rank: a high-pitch voice sitting in the bottom quartile of the
    # job's GFD values gets flagged as pitch/tract disagreement — the same
    # thing a good pair of ears notices. We flag, we don't flip.
    gs = sorted(v["gfd"] for v in raw.values() if v["gfd"])
    q1 = gs[max(0, len(gs) // 4)] if len(gs) >= 8 else None
    for nm, v in raw.items():
        f0, g = v["f0"], v["gfd"]
        gd, conf, note = classify(f0, g)
        if gd is None:                             # no confident absolute call
            if f0 is not None:
                gd = ("female" if f0 >= 175 else
                      "male" if f0 < 150 else "ambiguous")
                conf = ("likely" if gd != "ambiguous" else None)
        if (gd == "female" and g and q1 and g <= q1):
            note = ("high pitch but LOWEST-quartile formants in this job — "
                    "possible pitch shift or voice-changer; trust your ears")
            conf = "likely" if conf == "high" else conf
            v["mismatch"] = True
        v["voice_gender"], v["confidence"], v["note"] = gd, conf, note
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(raw, indent=1, ensure_ascii=False))
    raw["_secs"] = round(time.time() - t0, 1)
    return raw


if __name__ == "__main__":
    d = sys.argv[1]
    r = run(d, force="--force" in sys.argv)
    rows = [(k, v) for k, v in r.items() if isinstance(v, dict)]
    for nm, e in sorted(rows, key=lambda kv: -(kv[1].get("f0") or 0)):
        print(f"{nm:>10} f0 {e['f0'] or 0:>6} GFD {e['gfd'] or 0:>5}  "
              f"{e['voice_gender'] or '?':<10} {e['confidence'] or '':<7} {e['note'] or ''}")
