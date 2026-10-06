#!/usr/bin/env python3
"""
map_clean.py — remap a solo track's whisper output back to ORIGINAL video
timestamps, producing the human-facing clean transcript for one person.

  python map_clean.py <solo_prefix>     # prefix as used in solo_track.py

Reads <prefix>_timeline.json + <prefix>_solo.json, writes <prefix>_clean.txt.
"""
import json
import sys
from pathlib import Path


def ts(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    prefix = Path(sys.argv[1])
    tl = json.load(open(str(prefix) + "_timeline.json"))          # [t0, t1, orig_t, hint]
    solo = json.load(open(str(prefix) + "_solo.json"))["segments"]
    k = 0
    out = []
    for seg in solo:
        while k < len(tl) - 1 and tl[k][1] < seg["start"]:
            k += 1
        out.append((tl[k][2], seg["text"].strip()))
    dst = Path(str(prefix) + "_clean.txt")
    dst.write_text("\n".join(f"[{ts(o)}] {t}" for o, t in out) + "\n", encoding="utf-8")
    print(f"[map_clean] {dst.name}: {len(out)} segments")


if __name__ == "__main__":
    main()
