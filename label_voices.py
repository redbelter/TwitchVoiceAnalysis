#!/usr/bin/env python3
"""
label_voices.py — turn raw voiceprint clusters into named people.

Inputs (from vp_cluster.py + twitch_transcribe.py):
  whisper.json, vp_emb2.npy, vp_idx2.json, vp_lab2.npy
Outputs:
  labeled.txt   "[HH:MM:SS] <label>: text"   (names or clNNNN ids)
  people.json   {label: {name, lane, talk_seconds, first_ts, last_ts, n_segments}}

Labeling logic (session-tuned, keep the invariants):
  1. Auto-streamer: dominant lane by total talk-time, unless --streamer given.
  2. Seeds: --seed TS[=NAME] anchors that segment's cluster as the host
     (default streamer source of truth), or as a NAMED person otherwise.
  3. Joiners: lanes whose first activity is after the median first activity
     are flagged (they joined late — useful for VC-call VODs).
  4. Centroid refinement: fragments/short segments (no embedding or lane<8)
     absorb into nearest stable lane (>=8 segs) at cosine >= 0.45.
     ESTABLISHED lanes are NEVER re-assigned. This invariant matters: a loose
     pass over-relabels 2400+ lines and steals from the streamer lane.

Run with the diar venv python (numpy+scipy only, any python works).
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_ts(s):
    hh, mm, ss = s.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="whisper segments json")
    ap.add_argument("--emb", default="vp_emb2.npy")
    ap.add_argument("--idx", default="vp_idx2.json")
    ap.add_argument("--lab", default="vp_lab2.npy")
    ap.add_argument("--streamer", default=None,
                    help="timestamp (HH:MM:SS) inside a host monologue, e.g. 00:15:00")
    ap.add_argument("--seed", action="append", default=[],
                    metavar="TS[=NAME]", help="anchor cluster at TS as named person")
    ap.add_argument("--out", default="labeled.txt")
    ap.add_argument("--people", default="people.json")
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min-lane", type=int, default=8)
    ns = ap.parse_args()

    segs = json.load(open(ns.json, encoding="utf-8"))["segments"]
    E = np.load(ns.emb)
    IDX = json.load(open(ns.idx))
    lab = np.fromfile(ns.lab, dtype="int32")
    assert len(lab) == len(E) == len(IDX), "emb/idx/lab cache mismatch — rerun vp_cluster"

    lane_of_row = {k: int(lab[k]) for k in range(len(lab))}
    row_of_seg = {i: k for k, i in enumerate(IDX)}

    # lanes by lane-id (not segment order); talk-time from whisper seg durations
    segs_by_lane = collections.defaultdict(list)
    for i, k in row_of_seg.items():
        segs_by_lane[lane_of_row[k]].append(i)
    talk = {L: sum(segs[i]["end"] - segs[i]["start"] for i in v)
            for L, v in segs_by_lane.items()}

    # --- streamer ---
    streamer_lane = None
    if ns.streamer:
        t = parse_ts(ns.streamer)
        j = min(row_of_seg, key=lambda i: abs(segs[i]["start"] - t))
        streamer_lane = lane_of_row[row_of_seg[j]]
    else:
        streamer_lane = max(talk, key=talk.get)

    # --- named seeds ---
    named = {}
    for spec in ns.seed:
        tspec, _, nm = spec.partition("=")
        t = parse_ts(tspec)
        j = min(row_of_seg, key=lambda i: abs(segs[i]["start"] - t))
        L = lane_of_row[row_of_seg[j]]
        named[L] = nm or tspec

    # --- joiners ---
    first = {L: min(segs[i]["start"] for i in v) for L, v in segs_by_lane.items()}
    med = float(np.median(list(first.values())))
    joiners = {L for L, t0 in first.items() if t0 > med + 600}

    # --- display names: dominant=streamer, seeds=names, rest clNNNN by rank ---
    order = sorted(talk, key=lambda L: -talk[L])
    disp, people = {}, {}
    rank = 0
    for L in order:
        if L == streamer_lane:
            d = "STREAMER"
        elif L in named:
            d = named[L]
        else:
            rank += 1
            d = f"cl{L:04d}"
        disp[L] = d
        people[d] = {"lane": L, "name": "streamer host" if d == "STREAMER" else (named.get(L) or ""),
                     "talk_seconds": round(talk[L], 1), "first_ts": hms(first[L]),
                     "last_ts": hms(max(segs[i]["end"] for i in segs_by_lane[L])),
                     "n_segments": len(segs_by_lane[L]),
                     "joined_late": L in joiners and L != streamer_lane}

    centroids = {}
    for L, v in segs_by_lane.items():
        if len(v) >= ns.min_lane:
            c = E[[row_of_seg[i] for i in v]].mean(0)
            centroids[disp[L]] = c / max(np.linalg.norm(c), 1e-9)
    names = list(centroids)
    C = np.stack([centroids[n] for n in names]) if names else np.zeros((0, E.shape[1]))

    # --- write labeled.txt, absorbing fragments into nearest stable lane ---
    out = []
    absorbed = 0
    final_stats = {}          # per FINAL display label -> talk/first/last/count
    for i, s in enumerate(segs):
        txt = s["text"].strip()
        k = row_of_seg.get(i)
        if k is not None:
            L = lane_of_row[k]
            d = disp[L]
            # fragment lanes (< min_lane, unnamed) absorb into nearest centroid
            if (L not in segs_by_lane or len(segs_by_lane[L]) < ns.min_lane) \
                    and L != streamer_lane and L not in named and C.shape[0]:
                sims = C @ E[k]
                b = int(np.argmax(sims))
                if sims[b] >= ns.thr:
                    d = names[b]
                    absorbed += 1
        else:
            # no embedding: neighbor-vote within a 5-segment window (stable lanes only)
            d = "(short)"
            near = [row_of_seg[j] for j in range(max(0, i - 4), min(len(segs), i + 5))
                    if j in row_of_seg and j != i]
            if near and C.shape[0]:
                votes = collections.Counter(disp[lane_of_row[kk]] for kk in near)
                nn, v = votes.most_common(1)[0]
                if v >= 3 and nn in centroids:
                    d = nn
        out.append(f"[{hms(s['start'])}] {d}: {txt}")
        st = final_stats.setdefault(d, {"talk": 0.0, "first": s["start"],
                                        "last": s["end"], "n": 0})
        st["talk"] += s["end"] - s["start"]
        st["first"] = min(st["first"], s["start"])
        st["last"] = max(st["last"], s["end"])
        st["n"] += 1

    Path(ns.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    people = {}
    for d, p in final_stats.items():
        if d == "(short)":
            continue
        lane = None
        for L, nm in disp.items():
            if nm == d:
                lane = L
                break
        people[d] = {"lane": lane,
                     "name": "streamer host" if d == "STREAMER" else (named.get(lane) or ""),
                     "talk_seconds": round(p["talk"], 1), "first_ts": hms(p["first"]),
                     "last_ts": hms(p["last"]), "n_segments": p["n"],
                     "joined_late": bool(lane is not None and lane in joiners
                                         and d != "STREAMER")}
    Path(ns.people).write_text(json.dumps(people, indent=1), encoding="utf-8")
    print(f"[label] {len(people)} people | streamer lane {streamer_lane} "
          f"| fragments absorbed: {absorbed}")
    for d, p in sorted(people.items(), key=lambda x: -x[1]["talk_seconds"])[:8]:
        print(f"[label]   {d}: {hms(p['talk_seconds'])} talk, {p['n_segments']} segs"
              f"{' [joined late]' if p['joined_late'] else ''}")


if __name__ == "__main__":
    main()
