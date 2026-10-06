#!/usr/bin/env python3
"""
vp_cluster.py v2 — voiceprint identity with VAD masking.
Sortformer's union-of-activity acts as speech mask (its 4 lanes churn, but the
UNION is a decent VAD regardless); whisper segments get embedded on
speech-only audio -> TitaNet embeddings -> average-linkage cosine clustering
-> stable VPxx labels across all 11 h.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as _sf
import torch

AUDIO, JSON_IN = Path(sys.argv[1]), Path(sys.argv[2])
RTTM = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("vod_speakers.rttm.json")
OUT_TXT = Path(sys.argv[4] if len(sys.argv) > 4 else "vod_voices.txt")
THR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
MIN = 1.2          # min clean-speech seconds to embed a segment
CAP = 10.0         # max clean seconds per embedding slice
SR = 16000

print("[vp] loading audio + regions ...", flush=True)
_data, sr = _sf.read(str(AUDIO), dtype="float32")
wav = torch.from_numpy(_data)
if wav.ndim > 1:
    wav = wav.mean(dim=1)
if sr != SR:
    import torchaudio
    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR)[0]

segs = json.load(open(JSON_IN, encoding="utf-8"))["segments"]
regions = json.load(open(RTTM))
# merge union-of-activity intervals (all lanes, sorted)
ivs = sorted([(s, e) for s, e, _ in regions])
merged = []
for s, e in ivs:
    if merged and s <= merged[-1][1] + 0.25:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])


def clean_parts(a, b, max_take=CAP):
    """speech-only sub-intervals of [a,b], greedily up to max_take seconds."""
    out, taken = [], 0.0
    for s, e in merged:
        if e <= a or s >= b or taken >= max_take:
            continue
        cs, ce = max(s, a), min(e, b)
        if ce - cs <= 0:
            continue
        if taken + (ce - cs) > max_take:
            ce = cs + (max_take - taken)
        out.append((cs, ce))
        taken += ce - cs
    return out, taken


work = Path(sys.argv[6] if len(sys.argv) > 6 else ".") / "vp_wavs2"
work.mkdir(exist_ok=True)
print("[vp] slicing + masking ...", flush=True)
usable = []
for i, s in enumerate(segs):
    parts, tot = clean_parts(s["start"], min(s["end"], s["start"] + 20))
    if tot < MIN:
        continue
    chunks = [wav[int(a * SR):int(b * SR)] for a, b in parts if b > a]
    if not chunks:
        continue
    p = work / f"m{i:05d}.wav"
    _sf.write(str(p), torch.cat(chunks).numpy(), SR)
    usable.append((i, p))
print(f"[vp] {len(usable)} embeddable segments", flush=True)

from nemo.collections.asr.models import EncDecSpeakerLabelModel
vp = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to("cuda").eval()

print("[vp] embedding ...", flush=True)
wd = Path(sys.argv[6] if len(sys.argv) > 6 else ".")
EMB_CACHE, IDX_CACHE = wd / "vp_emb2.npy", wd / "vp_idx2.json"
if EMB_CACHE.exists() and IDX_CACHE.exists():
    E = np.load(EMB_CACHE)
    IDX = json.loads(IDX_CACHE.read_text())
    print(f"[vp] loaded cached embeddings {E.shape}", flush=True)
else:
    EMBS, IDX = [], []
    for j, (i, p) in enumerate(usable):
        e = vp.get_embedding(str(p))
        if isinstance(e, (tuple, list)):
            e = e[0]
        if torch.is_tensor(e):
            e = e.detach().float().cpu()
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        EMBS.append(e / max(np.linalg.norm(e), 1e-8))
        IDX.append(i)
        if j and j % 2000 == 0:
            print(f"[vp]  {j}/{len(usable)}", flush=True)
    E = np.stack(EMBS)
    np.save(EMB_CACHE, E)
    IDX_CACHE.write_text(json.dumps(IDX))
print(f"[vp] embedded {E.shape}", flush=True)

print("[vp] clustering ...", flush=True)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
D = np.clip(1.0 - E @ E.T, 0, None)
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
lab = fcluster(Z, t=THR, criterion="distance")
(wd / "vp_lab2.npy").write_bytes(lab.astype("int32").tobytes())
print(f"[vp] {lab.max()} clusters", flush=True)

def hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s2 = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s2:02d}"

lab_of = {i: int(lab[k]) for k, i in enumerate(IDX)}
byc = collections.Counter()
rows = []
for i, s in enumerate(segs):
    if i in lab_of:
        c = f"VP{lab_of[i]:02d}"; byc[c] += 1
    else:
        near = [j for j in range(max(0, i - 4), min(len(segs), i + 5)) if j in lab_of and j != i]
        c = (f"VP{collections.Counter(lab_of[j] for j in near).most_common(1)[0][0]:02d}~"
             if near else "VP??")
    rows.append(f"[{hms(s['start'])}] {c}: {s['text'].strip()}")
OUT_TXT.write_text("\n".join(rows) + "\n", encoding="utf-8")
Path(str(OUT_TXT) + ".map.json").write_text(json.dumps(dict(byc.most_common())))
print(f"[vp] wrote {OUT_TXT}", flush=True)
for c, n in byc.most_common():
    print(f"[vp]   {c}: {n} segs", flush=True)
