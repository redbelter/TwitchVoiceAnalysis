#!/usr/bin/env python3
"""frag_merge.py — post-process voice fragments into established speakers.

The diarization lanes churn: brief/crosstalk utterances fall into one-shot
"fragment" lanes (an 11 h VOD had 401 single-segment lanes). Those fragments
DO have TitaNet embeddings (vp_emb2.npy) — they were just never folded into a
person. This pass:

  1. Major lanes (>= min_lane segments) -> normalized centroids.
  2. Fragment lanes (size >= protect, i.e. 5-7 segs) need cosine >= hi —
     they may be a real person (e.g. a 6-segment VC joiner); don't steal them
     on a weak match.
  3. Smaller fragments: cosine >= hi merges outright; cosine >= lo merges
     only with a CONVERSATION PRIOR — the candidate speaker talks within
     adj seconds of the fragment (interruptions belong to the active
     conversation). Raw 0.30 similarity alone is NOT accepted: single
     2-second clips routinely score 0.3 against everyone.
  4. Unembedded "(short)" lines: neighborhood vote — among segments within
     ±vote_win seconds whose (final) label is a major, >= vote_min agreeing
     votes assigns the label. Short interjections come from whoever is in
     the conversation.
  5. Rewrites labeled.txt + people.json in place (first run keeps a
     labeled.pre_frag.txt backup), writes fragmerged.json stats.

Never re-labels established lanes — same invariant as label_voices.py.
Deterministic, numpy-only, no GPU.

Run:  python frag_merge.py <jobdir> [--hi 0.40] [--lo 0.30] [--force]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from vodpipe import atomic_write  # same repo; state/artifact files are read mid-write by the dashboard

LINE_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s*([^:]+):\s*(.*)$")


def hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _unit(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def run(jobdir, hi=0.40, lo=0.30, min_lane=8, protect=5, adj=10.0,
        vote_win=12.0, vote_min=2, force=False):
    d = Path(jobdir)
    emb_p, idx_p, lab_p = d / "vp_emb2.npy", d / "vp_idx2.json", d / "vp_lab2.npy"
    tj, labeled, people = d / "transcript.json", d / "labeled.txt", d / "people.json"
    missing = [p.name for p in (tj, labeled) if not p.exists()]
    if missing:
        raise SystemExit(f"frag_merge: missing {' '.join(missing)} — run the pipeline first")
    if not emb_p.exists():  # voiceprint stage never ran — nothing to merge with
        return {"changed": 0, "note": "no vp_emb2.npy — skipped"}

    E = np.load(emb_p, allow_pickle=True).astype(np.float32)
    IDX = json.load(open(idx_p, encoding="utf-8"))
    lab = np.fromfile(lab_p, dtype="int32")
    segs = json.load(open(tj, encoding="utf-8"))["segments"]
    P = json.load(open(people, encoding="utf-8")) if people.exists() else {}

    # lane -> display label (from people.json)
    disp_of_lane = {}
    for nm, p in P.items():
        if p.get("lane") is not None and nm != "(short)":
            disp_of_lane[p["lane"]] = nm

    sizes = collections.Counter(lab.tolist())
    # majors = big raw lanes, PLUS lanes people.json says are real people
    # (>=5 segments) — label_voices absorbed fragments into them, so their
    # RAW row count can be small; protect them from being eaten.
    person_lanes = {p["lane"] for nm, p in P.items()
                    if nm != "(short)" and p.get("lane") is not None
                    and p.get("n_segments", 0) >= 5}
    majors = [L for L, c in sizes.items()
              if c >= min_lane or (L in person_lanes and c >= 2)]
    if not majors:
        raise SystemExit("frag_merge: no major lanes to merge into")
    rows_of_lane = {L: [k for k in range(len(lab)) if lab[k] == L] for L in sizes}
    cents, names = {}, []
    for L in majors:
        nm = disp_of_lane.get(L) or f"cl{L:04d}"
        cents[nm] = _unit(E[rows_of_lane[L]].mean(0))
        names.append(nm)
    name_set = set(names)
    C = np.stack([cents[n] for n in names])

    # conversation prior: when does each major speaker talk?
    talk_times = collections.defaultdict(list)
    for nm in names:
        L = next(L for L in majors if (disp_of_lane.get(L) or f"cl{L:04d}") == nm)
        for k in rows_of_lane[L]:
            i = IDX[k]
            if i is not None and i < len(segs):
                talk_times[nm].append((segs[i]["start"], segs[i]["end"]))
        talk_times[nm].sort()

    import bisect

    def nearby(nm, t0, t1):
        starts = [a for a, _ in talk_times[nm]]
        j = bisect.bisect_left(starts, t0 - adj)
        for a, b in talk_times[nm][j:]:
            if a > t1 + adj:
                break
            if b >= t0 - adj:
                return True
        return False

    # pass A: embedded fragment rows
    new_disp, stats = {}, {"lane_merged": 0, "lane_kept": 0,
                           "seg_merged": 0, "seg_kept": 0}
    for L, size in sorted(sizes.items()):
        if L in majors:
            continue
        rows = rows_of_lane[L]
        big = size >= protect  # could be a real person — be strict
        sims = (C @ _unit(E[rows].mean(0))).astype(float)
        b = int(np.argmax(sims))
        accept = sims[b] >= hi or (not big and sims[b] >= lo
                                   and any(nearby(names[b], segs[IDX[k]]["start"],
                                                  segs[IDX[k]]["end"])
                                           for k in rows if IDX[k] < len(segs)))
        if accept:
            for k in rows:
                i = IDX[k]
                if i < len(segs):
                    new_disp[i] = names[b]
            stats["lane_merged"] += 1
        else:
            stats["lane_kept"] += 1
            for k in rows:
                i = IDX[k]
                if i >= len(segs):
                    continue
                sims2 = (C @ _unit(E[k])).astype(float)
                bb = int(np.argmax(sims2))
                if sims2[bb] >= hi or (sims2[bb] >= lo
                                       and nearby(names[bb], segs[i]["start"], segs[i]["end"])):
                    new_disp[i] = names[bb]
                    stats["seg_merged"] += 1
                else:
                    stats["seg_kept"] += 1

    # effective label per embedded segment (post-merge), for the vote pass
    lines = labeled.read_text(encoding="utf-8").splitlines()
    parsed = []  # match per line, in line order
    for ln in lines:
        parsed.append(LINE_RE.match(ln))

    # pass B: unembedded "(short)" lines via neighborhood vote
    emb_time = {}
    for k in range(len(IDX)):
        i = IDX[k]
        if i < len(segs):
            emb_time[i] = segs[i]
    short_segs = [i for i, m in enumerate(parsed)
                  if i < len(segs) and m and m[4] == "(short)" and i not in emb_time]
    vote_changed = 0
    for i in short_segs:
        t = segs[i]["start"]
        votes = collections.Counter()
        for j in range(max(0, i - 40), min(len(segs), i + 40)):
            if j == i or j not in emb_time:
                continue
            sj = segs[j]
            if abs((sj["start"] + sj["end"]) / 2 - t) > vote_win:
                continue
            lab_j = new_disp.get(j) or (parsed[j][4] if j < len(parsed) and parsed[j] else "")
            if lab_j in name_set:
                votes[lab_j] += 1
        if votes and votes.most_common(1)[0][1] >= vote_min:
            new_disp[i] = votes.most_common(1)[0][0]
            vote_changed += 1

    stats["short_voted"] = vote_changed
    stats["short_left"] = len(short_segs) - vote_changed
    if not new_disp:
        stats.update(changed=0, people_before=len(P), people_after=len(P), hi=hi, lo=lo)
        atomic_write(d / "fragmerged.json", json.dumps(stats, indent=1))
        return {**stats, "note": "nothing accepted"}

    # rewrite labeled.txt + recompute people.json
    if not (d / "labeled.pre_frag.txt").exists():
        atomic_write(d / "labeled.pre_frag.txt", "\n".join(lines) + "\n")
    out, final = [], {}
    changed = 0
    for i, ln in enumerate(lines):
        m = parsed[i]
        if not m:
            out.append(ln)
            continue
        new = new_disp.get(i, m[4])
        if new != m[4]:
            changed += 1
        out.append(f"[{m[1]}:{m[2]}:{m[3]}] {new}: {m[5]}")
        s = final.setdefault(new, {"talk": 0.0, "first": 1e18, "last": 0.0, "n": 0})
        if i < len(segs):
            s["talk"] += segs[i]["end"] - segs[i]["start"]
            s["first"] = min(s["first"], segs[i]["start"])
            s["last"] = max(s["last"], segs[i]["end"])
        s["n"] += 1
    atomic_write(labeled, "\n".join(out) + "\n")

    new_people = {}
    for nm, s in final.items():
        if nm == "(short)":   # unattributed lines stay in labeled.txt but are not a "person"
            continue
        old = P.get(nm, {})
        new_people[nm] = {"lane": old.get("lane"), "name": old.get("name", ""),
                          "talk_seconds": round(s["talk"], 1),
                          "first_ts": hms(s["first"]) if s["first"] < 1e17 else "00:00:00",
                          "last_ts": hms(s["last"]),
                          "n_segments": s["n"],
                          "joined_late": old.get("joined_late", False)}
    atomic_write(people, json.dumps(new_people, indent=1))

    stats.update(changed=changed, people_before=len(P), people_after=len(new_people),
                 hi=hi, lo=lo)
    atomic_write(d / "fragmerged.json", json.dumps(stats, indent=1))
    if changed:
        # downstream artifacts were cut from the OLD labels — drop them so the
        # pipeline rebuilds against the consolidated people (voice descriptions,
        # solo tracks, re-asr, clean maps). Dashboard may hold a read handle
        # mid-stream; failing to delete one file is not fatal.
        for pat in ("voice_stats.json", "solo_*_solo.*", "solo_*_timeline.json",
                    "solo_*_clean.txt", "solo_*_rereasr*", "scan_flirting.txt"):
            for p in d.glob(pat):
                try:
                    p.unlink()
                except OSError:
                    pass
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--hi", type=float, default=0.40, help="merge on acoustics alone")
    ap.add_argument("--lo", type=float, default=0.30, help="merge with conversation prior")
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()
    print("[fragmerge]", json.dumps(run(ns.jobdir, ns.hi, ns.lo, force=ns.force)))
