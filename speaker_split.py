#!/usr/bin/env python3
"""
speaker_split.py — diarize VOD audio with NVIDIA Streaming Sortformer (4 spk),
then attach SPEAKER_0X labels to the faster-whisper transcript JSON.

Run with the diar venv python (has NeMo installed):
  python speaker_split.py --audio vod.mp3 --json vod.json --out vod_speakers.txt
"""

import argparse
import json
import re
import time
from pathlib import Path


def load_whisper_json(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["segments"]


def diarize(audio: str):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    print("[diar] loading streaming sortformer 4spk v2.1 ...")
    model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # high-accuracy offline-ish streaming config (from model card "very high latency")
    m = model.sortformer_modules
    m.chunk_len = 340
    m.chunk_right_context = 40
    m.fifo_len = 40
    m.spkcache_update_period = 300
    m.spkcache_len = 188
    m._check_streaming_parameters()

    print(f"[diar] diarizing {audio} ...")
    t0 = time.time()
    # high-level API: returns list (per file) of [(start, end, speaker), ...]
    with torch.no_grad():
        segments = model.diarize(audio=[audio], batch_size=1)
    print(f"[diar] done in {time.time()-t0:.0f}s")
    # NeMo returns RTTM-style strings: "start end speaker_0"
    raw = segments[0] if segments and isinstance(segments[0], list) else segments
    segs = []
    for item in raw:
        if isinstance(item, str):
            parts = item.split()
            segs.append((float(parts[0]), float(parts[1]), " ".join(parts[2:]) or "spk"))
        else:
            segs.append(tuple(item))
    return segs


def speaker_at(spks, a, b):
    """Dominant speaker (max overlap) over [a,b]; None if silent/no activity."""
    best, bestov = None, 0.0
    totals = {}
    for s, e, lab in spks:
        ov = min(e, b) - max(s, a)
        if ov > 0:
            totals[lab] = totals.get(lab, 0.0) + ov
    if totals:
        best = max(totals, key=totals.get)
        bestov = totals[best]
        if bestov < 0.2 * (b - a):
            best = best + "?"  # mostly-unattributable
    return best


def hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--json", required=True, help="faster-whisper transcript json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spk", default=None, help="dump only this speaker, e.g. SPEAKER_01")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="audio file starts this many seconds into the original recording")
    ns = ap.parse_args()

    segs = load_whisper_json(Path(ns.json))
    raw_cache = Path(ns.out).with_suffix(".rttm.json")
    if raw_cache.exists():
        spks = [tuple(s) for s in json.loads(raw_cache.read_text(encoding="utf-8"))]
        print(f"[diar] loaded {len(spks)} cached activity regions")
    else:
        spks = diarize(ns.audio)
        raw_cache.write_text(json.dumps([list(s) for s in spks]), encoding="utf-8")
    if ns.offset:
        spks = [(s + ns.offset, e + ns.offset, l) for s, e, l in spks]
    labels = sorted({s[2] for s in spks})
    print(f"[diar] speakers found: {labels}, {len(spks)} activity regions")

    with open(ns.out, "w", encoding="utf-8") as f:
        for seg in segs:
            lab = speaker_at(spks, seg["start"], seg["end"]) or "???"
            if lab == "???" and ns.spk:
                continue
            if ns.spk and not lab.startswith(ns.spk):
                continue
            f.write(f"[{hms(seg['start'])}] {lab}: {seg['text'].strip()}\n")
    print(f"[diar] wrote {ns.out}")
    # speaker-activity summary
    tot = {}
    for s, e, l in spks:
        tot[l] = tot.get(l, 0) + (e - s)
    for l in labels:
        print(f"[diar]   {l}: {hms(tot.get(l, 0))} active")


if __name__ == "__main__":
    main()
