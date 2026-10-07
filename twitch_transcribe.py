#!/usr/bin/env python3
"""
twitch_transcribe.py — transcribe a Twitch VOD (or any video/audio) with faster-whisper on GPU.

Pipeline: video -> ffmpeg 16kHz mono mp3 (cached) -> faster-whisper (CTranslate2, CUDA)
Output:   <name>.txt (plain), <name>.srt (subtitles), <name>.json (segments+timing)

Usage:
  python twitch_transcribe.py <video-or-audio>            # large-v3 int8 on CUDA
  python twitch_transcribe.py vod.mp4 -m medium           # smaller/faster model
  python twitch_transcribe.py vod.mp4 --test 5            # smoke test: first 5 min only
  python twitch_transcribe.py vod.mp4 -o C:\\out           # custom output dir
  python twitch_transcribe.py vod.mp4 --lang en --words   # force English + word timestamps (slower)

Notes:
  - First run downloads the model (~1.5 GB for large-v3 int8) to the HF cache.
  - Partial results are flushed to disk every ~5 min of audio, so a crash loses little.
  - vad_filter skips dead air: faster AND fewer hallucinations on long streams.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path

SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3")


def extract_audio(src: Path, audio_path: Path) -> None:
    if audio_path.exists() and audio_path.stat().st_size > 0:
        print(f"[transcribe] reusing cached audio: {audio_path.name}")
        return
    print(f"[transcribe] extracting audio -> {audio_path.name} ...")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-q:a", "4", str(audio_path)]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not audio_path.exists():
        sys.exit("[transcribe] ffmpeg audio extraction failed")


def fmt_hms(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ensure_cuda_dlls():
    """CTranslate2 lazily LoadLibrary's cublas64_12.dll + cudnn64_*.dll, which
    only searches PATH (not os.add_dll_directory). The pip wheels
    nvidia-cublas-cu12 / nvidia-cudnn-cu12 ship them in
    site-packages/nvidia/<pkg>/bin — prepend those to PATH before CT2 loads."""
    import importlib.metadata
    for dist in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            files = importlib.metadata.Distribution.from_name(dist).files
            bins = {str(pathlib.Path(f.locate()).parent) for f in files
                    if f.name and f.name.endswith(".dll")}
            for b in sorted(bins):
                if b not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(b)  # belt-and-braces for ctypes loads
                except OSError:
                    pass
        except importlib.metadata.PackageNotFoundError:
            print(f"[transcribe] warning: {dist} not installed; CUDA may fall back to CPU")


def main():
    ap = argparse.ArgumentParser(description="GPU transcription with faster-whisper.")
    ap.add_argument("input", help="video or audio file (mp4/mp3/wav/...)")
    ap.add_argument("-m", "--model", default="large-v3", choices=SUPPORTED_MODELS)
    ap.add_argument("-o", "--output", default=None, help="output dir (default: beside input)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default="int8", choices=["int8", "float16", "int8_float16"])
    ap.add_argument("--lang", default=None, help="force language, e.g. en (default: auto-detect)")
    ap.add_argument("--words", action="store_true", help="word-level timestamps (slower)")
    ap.add_argument("--stem", default=None, help="output stem override (default: input name)")
    ap.add_argument("--test", type=float, default=None, metavar="MINUTES",
                    help="only transcribe the first N minutes (smoke test)")
    ns = ap.parse_args()

    _ensure_cuda_dlls()

    src = Path(ns.input)
    if not src.exists():
        # tolerate shell-glob leftovers: try wildcard match in parent dir
        matches = list(src.parent.glob(src.name)) if "*" in src.name else []
        if not matches:
            sys.exit(f"Input not found: {src}")
        src = matches[0]

    out_dir = Path(ns.output) if ns.output else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (ns.stem or src.stem)[:80]  # long VOD titles are fine; cap anyway

    audio = out_dir / f"{stem}.mp3"
    extract_audio(src, audio)

    test_limit = ns.test * 60 if ns.test else None

    from faster_whisper import WhisperModel  # deferred: heavy import
    print(f"[transcribe] loading {ns.model} ({ns.device}/{ns.compute}) ...")
    model = WhisperModel(ns.model, device=ns.device, compute_type=ns.compute)

    total = None
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(audio)],
                       capture_output=True, text=True)
    try:
        total = float(r.stdout.strip())
    except ValueError:
        pass
    if test_limit and total:
        total = min(total, test_limit)

    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    json_path = out_dir / f"{stem}.json"

    t0 = time.time()
    segments, info = model.transcribe(
        str(audio),
        language=ns.lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=ns.words,
        without_timestamps=False,
    )
    print(f"[transcribe] language={info.language} (p={info.language_probability:.2f}), "
          f"audio={fmt_hms(total) if total else '?'}")

    seg_list, flush_marks = [], []
    last_flush, last_print = 0.0, 0.0

    def flush():
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"[{fmt_hms(s[0])}] {s[2]}" for s in seg_list) + "\n")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(
                f"{i+1}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n"
                for i, (a, b, t) in enumerate(seg_list)) + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"source": str(src), "model": ns.model, "language": info.language,
                       "segments": [{"start": a, "end": b, "text": t} for a, b, t in seg_list]},
                      f, ensure_ascii=False, indent=1)

    try:
        for seg in segments:
            if test_limit and seg.start > test_limit:
                break
            seg_list.append((seg.start, seg.end, seg.text.strip()))
            now = time.time()
            if now - last_print > 5:
                pct = f" {100*seg.end/total:4.1f}%" if total else ""
                rate = seg.end / max(now - t0, 1)
                eta = fmt_hms((total - seg.end) / rate) if (total and rate > 0) else "?"
                print(f"[transcribe] {pct} | at {fmt_hms(seg.end)} | {rate:.0f}x RT | ETA {eta}")
                last_print = now
            if seg.end - last_flush > 300:  # ~5 min of audio
                flush(); last_flush = seg.end
    except KeyboardInterrupt:
        print("\n[transcribe] interrupted — flushing partial results")
    finally:
        flush()

    words = sum(len(t.split()) for _, _, t in seg_list)
    dt = time.time() - t0
    print(f"[transcribe] DONE: {len(seg_list)} segments, {words} words in {fmt_hms(dt)}")
    print(f"[transcribe] wrote:\n  {txt_path}\n  {srt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
