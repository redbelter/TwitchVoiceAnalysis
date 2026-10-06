#!/usr/bin/env python3
"""
chunk_diar.py — Sortformer diarization in overlapping 90-min chunks (streaming
model stalls past ~6h), stitched to GLOBAL speaker ids by overlap consistency:
lanes co-active inside the 10-min overlap window are the same person.
Writes vod_speakers.rttm.json  ([start, end, label], absolute seconds).
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

AUDIO = Path(sys.argv[1])
TOTAL = float(sys.argv[2])
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "vod_speakers.rttm.json")
WIN = int(__import__("os").environ.get("DIAR_WIN", 5400))
STEP = int(__import__("os").environ.get("DIAR_STEP", 4800))  # overlap = WIN-STEP
TMP = Path(tempfile.mkdtemp(prefix="diarch_"))

from nemo.collections.asr.models import SortformerEncLabelModel

print("[chunk] loading sortformer ...", flush=True)
model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
model = model.to("cuda").eval()
m = model.sortformer_modules
m.chunk_len = 340; m.chunk_right_context = 40; m.fifo_len = 40
m.spkcache_update_period = 300; m.spkcache_len = 188
m._check_streaming_parameters()


def parse(items):
    out = []
    for it in items:
        if isinstance(it, str):
            p = it.split()
            out.append((float(p[0]), float(p[1]), " ".join(p[2:])))
    return out


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


all_regions = []          # [abs_start, abs_end, global_label]
global_speech = []        # (abs_start, abs_end, label) for matching
next_id = 0
t0, start, n = time.time(), 0.0, 0

while start < TOTAL:
    dur = min(WIN, TOTAL - start)
    cpath = TMP / f"c{n:02d}.mp3"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start), "-t", str(dur), "-i", str(AUDIO),
                    "-ar", "16000", "-ac", "1", str(cpath)], check=True)
    with torch.no_grad():
        raw = model.diarize(audio=[str(cpath)], batch_size=1)
    lanes = [(s, e, l) for s, e, l in parse(raw[0] if raw and isinstance(raw[0], list) else raw)
             if e - s > 0.05]

    # only trust the NON-overlap tail for emission; overlap zone used for matching only
    ov_start_rel = max(0.0, start - (start - STEP) if n else 0.0)  # overlap belongs to prev chunk
    by_lane = {}
    for s, e, l in lanes:
        by_lane.setdefault(l, []).append((s, e, l))

    # --- map local lanes -> global via co-activity in the overlap window ---
    mapping = {}
    if n == 0:
        for l in sorted(by_lane):
            mapping[l] = f"speaker_{next_id}"
            next_id += 1
    else:
        rel_ov = (start - (start - STEP)) if start > 0 else 0  # = STEP? recompute:
        # overlap window in absolute time: [prev_start+prev_dur - OV, start+dur_end]
        ov_abs = (start - (WIN - STEP), start)  # 10 min ending at chunk boundary
        # wait: previous chunk covered [start-STEP, start-STEP+WIN]; overlap = [start, start+WIN-STEP]
        ov_abs = (start, start + (WIN - STEP))
        prev = [r for r in global_speech if overlap(r, ov_abs) > 0]
        # per local lane: overlap-time with each global inside ov_abs
        prof = {}
        for local, regs in sorted(by_lane.items()):
            cur_ov = [(s + start, e + start) for s, e, _ in regs]
            labs = {}
            for cs, ce in cur_ov:
                for ps, pe, pl in prev:
                    o = overlap((cs, ce), (ps, pe))
                    if o > 0:
                        labs[pl] = labs.get(pl, 0.0) + o
            prof[local] = labs
        # score each (local, global) pair: co-active time vs the lane's total ov presence
        cands = []  # (score, local, global)
        for local, labs in prof.items():
            tot_ov = sum(max(0.0, min(ce, ov_abs[1]) - max(cs, ov_abs[0]))
                         for cs, ce in [(s + start, e + start) for s, e, _ in by_lane[local]])
            if tot_ov <= 0:
                continue
            for pl, o in labs.items():
                cands.append((o / tot_ov, local, pl))
        cands.sort(reverse=True)  # greedy one-to-one assignment, best first
        used_l, used_g = set(), set()
        for score, local, pl in cands:
            if local in used_l or pl in used_g or score < 0.3:
                continue
            mapping[local] = pl
            used_l.add(local); used_g.add(pl)
        # leftovers: co-active with a mapped lane but unmatched = distinct new voice;
        # co-activity with MAPPED lanes is evidence they are NOT that person.
        for local, regs in sorted(by_lane.items()):
            if local in mapping:
                continue
            tot = sum(e - s for s, e, _ in regs)
            if tot > 30:
                mapping[local] = f"speaker_{next_id}"
                next_id += 1
            else:
                continue

    emitted = 0
    clip_lo = (start + (WIN - STEP)) if n else 0.0  # overlap zone: prev chunk already emitted it
    for s, e, l in lanes:
        if l not in mapping:
            continue
        a, b = max(s + start, clip_lo), e + start
        if b - a <= 0.05:
            continue
        all_regions.append([round(a, 2), round(b, 2), mapping[l]])
        global_speech.append((a, b, mapping[l]))
        emitted += 1

    print(f"[chunk] {start/3600:5.2f}h-{(start+dur)/3600:5.2f}h: {emitted:4d} regions  map={mapping}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    n += 1
    start += STEP

OUT.write_text(json.dumps(all_regions))
import collections
tot = collections.defaultdict(float)
for s, e, l in all_regions:
    tot[l] += e - s
print(f"[chunk] wrote {OUT}: {len(all_regions)} regions ({time.time()-t0:.0f}s)", flush=True)
for l in sorted(tot):
    print(f"[chunk]   {l}: {tot[l]/3600:.2f} h active", flush=True)
