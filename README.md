# TwitchVoiceAnalysis

Feed it a Twitch VOD / clip / live-channel URL (or a local video file) and it
runs the whole chain **autonomously and resumably**:

```
download → extract audio → transcribe (Whisper large-v3, GPU)
        → diarize (NVIDIA Sortformer VAD) → voiceprint-cluster (TitaNet)
        → label people → per-person "radio edit" solo tracks
        → re-transcribe each solo track → clean per-person transcripts
        → register scans (keyword / flirting / etc.)
```

plus a local web dashboard: paste a URL, watch stages tick, click a person to
**hear their isolated voice** and read their transcript with timestamps that
deep-link back into the VOD.

Everything runs **locally on your GPU** (an 11 h VOD ≈ 90 min end-to-end on an
RTX 4080, ≈2 GB VRAM during whisper). No cloud, no API keys.

## Quick start

```bash
pip install -r requirements.txt          # root env: download + whisper + web
# diar venv (isolated, has NeMo — see requirements-diar.txt for exact commands):
set VODPIPE_DIAR_PY=C:\path\to\venvs\diar\Scripts\python.exe

python vodpipe.py https://www.twitch.tv/videos/1234567890 -f 720
#   -> resumable; re-run the same command anytime to continue
#   -> artifacts land in ~/Downloads/vodpipe/v1234567890/

python vodpipe_web.py --port 5001        # dashboard at http://127.0.0.1:5001
```

Local file instead of a URL? `python vodpipe.py C:\Videos\whatever.mp4` skips
download and runs everything else.

### Taming the cast

Personal streams with Discord calls have more voices than the diarizer's 4
lanes, so **identity comes from voiceprints, not diarization lanes** (lane IDs
recycle; fingerprints don't). Two optional hints go a long way:

```
python vodpipe.py <url> --streamer 00:15:00          # a timestamp inside the host's monologue
python vodpipe.py <url> --seed 02:11:37=alice        # name a voice at the moment they speak
```

Without them the dominant talker is auto-labeled `STREAMER` and everyone else
gets stable `clNNNN` lanes.

## Outputs (per job)

| file | what |
|---|---|
| `transcript.json/.txt/.srt` | full mixed transcript, segment timestamps |
| `labeled.txt` | every line tagged `STREAMER` / person / `clNNNN` |
| `people.json` | per-person talk-time, first/last seen, joined-late flag |
| `solo_N_solo.wav` | one person's voice stitched across the whole VOD ("radio edit") |
| `<lane>_clean.txt` | that person's re-transcribed speech with ORIGINAL video timestamps |
| `scan_flirting.txt` | register-block extractor (generic; regex-editable) |
| `state.json` + `logs/` | stage statuses and full child-process logs |

## Architecture notes (the things that mattered)

- **Two interpreters on purpose.** The root env handles yt-dlp/faster-whisper/
  web; a separate venv holds NeMo (`--diar-python`). Mixing them causes DLL
  and dependency pain on Windows.
- **Voiceprint identity pipeline.** Sortformer supplies *where* speech happens
  (union-of-activity as VAD); TitaNet `nvidia/speakerverification_en_titanet_large`
  embeds each speech-masked whisper segment; average-linkage cosine clustering
  gives lanes stable across 11+ hours. Fragment lanes absorb into stable lanes
  (≥8 segs) at cosine ≥ 0.45 — established lanes are never re-assigned.
- **Solo tracks exclude crosstalk by design.** Slicing cannot un-sum overlapping
  voices; the solo track is "their clean speech", not a music-style stem.
  (A future stage could add speaker *extraction* to recover during-overlap words.)
- **ASCII job ids, `.part`-then-rename writes, atomic caches** — VOD titles are
  full of `⧸`, `!`, and other shell-hostile Unicode; truncation on kill must
  never look like a completed stage.
- **rttm coverage guard:** diarization output is only trusted if it reaches the
  audio end minus 60 s (a streaming-mode truncation bug made this mandatory).
- **Whisper on single-speaker audio beats mixed audio.** The solo re-ASR step
  measurably de-mangles lines (word-error on clean isolated speech is what
  Whisper is trained for) — expect recovered words the mixed pass swallowed.

## Known ceilings

- Whisper large-v3 on clean speech ≈ human-level WER; the remaining error lives
  in crosstalk and sub-1.2 s fragments (never embedded, labeled by majority vote).
- 4-speaker diarizer + voiceprint clustering handles rotating Discord casts;
  pyannote community-1 (needs HF token) or NVIDIA Nemotron-3 (needs an
  unreleased NeMo) would sharpen boundaries further.
- Audio the mic never captured is unrecoverable by anyone.

## Legal / ethics

Download only what your rights allow (your own VODs, per-TOS content). Voice
analysis of identifiable people is sensitive: artifacts stay local, nothing is
uploaded, and `requirements-diar.txt` documents exact model provenance.

## Repo layout

```
vodpipe.py            orchestrator (stage graph, resume, GPU lock, per-stage CLI)
vodpipe_web.py        localhost dashboard (FastAPI, zero build step)
twitch_dl.py          yt-dlp wrapper (VOD/clip/live, trim, cookies)
twitch_transcribe.py  faster-whisper GPU transcription (partial flushes, CUDA DLL fix)
chunk_diar.py         Sortformer, overlapping-window diarization + VAD regions
vp_cluster.py         TitaNet voiceprints -> stable speaker lanes
label_voices.py       streamer/seed anchoring, fragment absorption, people.json
solo_track.py         per-person radio-edit extraction (+ CALL mode)
map_clean.py          solo-track timestamps -> original video timestamps
flirt_scan.py         register-block extractor over labeled transcripts
```
