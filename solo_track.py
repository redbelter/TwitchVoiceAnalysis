#!/usr/bin/env python3
"""
solo_track.py — build a single-speaker "radio edit" track from a voiceprint
lane, then (optionally) re-transcribe it with faster-whisper for a clean
"what did X actually say" transcript that beats the mixed-audio pass.

Usage:
  python solo_track.py AUDIO whisper.json labeled.txt OUT_PREFIX \
      --cluster=cl1805 [--retranscribe]
  python solo_track.py AUDIO whisper.json labeled.txt OUT_PREFIX --cluster=CALL \
      # every labeled lane at once (all voices except STREAMER/(short))

  AUDIO    : the 16 kHz mono mp3/wav the whisper.json was made from
  labeled  : "[HH:MM:SS] clXXXX: text" lines (label_voices.py output)

Produces:
  OUT_prefix_solo.wav        the lane's voice stitched with natural gaps
  OUT_prefix_timeline.json   [solo_t0, solo_t1, orig_t, text-hint]
  OUT_prefix_solo.txt/.srt/.json  (with --retranscribe, via faster-whisper)

Note: words spoken DURING crosstalk are excluded by design — slicing cannot
un-sum overlapping voices. The solo track is "her clean speech", not a stem.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
GAP = 0.45          # silence inserted between speech slices (keeps rhythm)
MAXSLICE = 15.0     # per whisper segment cap, s
TOL = 0.75          # ts match tolerance vs segment start, s


def parse_ts(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def main():
    AUDIO, JSON_IN, LABELED, PREFIX = (Path(sys.argv[1]), Path(sys.argv[2]),
                                       Path(sys.argv[3]), sys.argv[4])
    target = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cluster=")), None)
    CALL_MODE = target == "CALL"        # every labeled lane (streamer excluded)

    segs = json.load(open(JSON_IN, encoding="utf-8"))["segments"]

    if target is None or CALL_MODE:
        want_ts = set()
        for ln in LABELED.read_text(encoding="utf-8").splitlines():
            if "] " not in ln:
                continue
            ts, tail = ln[1:].split("] ", 1)
            lab = tail.split(":", 1)[0].strip()
            if CALL_MODE:
                if lab.startswith("cl"):
                    want_ts.add(parse_ts(ts))
            elif lab == target:
                want_ts.add(parse_ts(ts))
    else:
        # named lanes from people.json (cl-id passthrough still works)
        want_ts = set()
        for ln in LABELED.read_text(encoding="utf-8").splitlines():
            if "] " not in ln:
                continue
            ts, tail = ln[1:].split("] ", 1)
            lab = tail.split(":", 1)[0].strip()
            if lab == target or f"cl{lab}" == target:
                want_ts.add(parse_ts(ts))

    want = [i for i, s in enumerate(segs)
            if any(abs(s["start"] - t) <= TOL for t in want_ts)]
    if not want:
        sys.exit(f"[solo] no segments matched {target} ({len(want_ts)} labels)")

    data, sr = sf.read(str(AUDIO), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = np.interp(np.arange(0, len(data), sr / SR),
                         np.arange(len(data)), data).astype(np.float32)

    out_parts, timeline, total = [], [], 0.0
    for i in sorted(want):
        s = segs[i]
        a = s["start"]
        b = min(s["end"], a + MAXSLICE)
        ch = data[int(a * SR):int(min(b * SR, len(data)))]
        t0 = total
        out_parts.append(ch)
        total += len(ch) / SR
        timeline.append([round(t0, 2), round(total, 2), a, s["text"].strip()])
        out_parts.append(np.zeros(int(GAP * SR), dtype=np.float32))
        total += GAP

    solo = np.concatenate(out_parts)
    out_wav = Path(str(PREFIX) + "_solo.wav")
    sf.write(str(out_wav), solo, SR)
    Path(str(PREFIX) + "_timeline.json").write_text(json.dumps(timeline))
    print(f"[solo] {target or 'dominant'}: {len(want)} slices, {total/60:.1f} min -> {out_wav}")

    if "--retranscribe" in sys.argv:
        cmd = [sys.executable, str(Path(__file__).parent / "twitch_transcribe.py"),
               str(out_wav), "-m", "large-v3", "--lang", "en"]
        r = subprocess.run(cmd)
        print("[solo] retranscribe exit", r.returncode)


if __name__ == "__main__":
    main()
