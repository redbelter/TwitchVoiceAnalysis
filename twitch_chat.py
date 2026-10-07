"""twitch_chat — pull VOD chat replay + video metadata into a vodpipe job.

Twitch has no public VOD-chat API; this walks the same persisted GraphQL
query the VOD player uses (via twitchdl's battle-tested fetchers), paginates
by cursor, dedupes by message id, and writes:

  chat.json   [{t, user, text}, ...] sorted by stream offset
  chat.txt    [HH:MM:SS] user: text          (grep-friendly)
  meta.json   video metadata (title, streamer, date, duration, views, ...)

  python twitch_chat.py <job_dir> [--max-messages N] [--force]

Exit codes: 0 ok | 3 nothing to do (local/clip job, twitchdl missing,
VOD/chat unavailable — the pipeline skips this stage cleanly).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path


def tag(msg):
    print(f"[chat] {msg}", flush=True)


def fetch(video_id, max_messages=120000):
    try:
        from twitchdl import twitch
    except ImportError:
        tag("twitch-dl not installed (pip install twitch-dl) — chat skipped")
        sys.exit(3)
    try:
        video = twitch.get_video(video_id)
    except Exception as e:
        tag(f"video lookup failed: {str(e)[:120]}")
        sys.exit(3)
    if not video:
        tag("no such VOD (expired, or live session without an archive VOD)")
        sys.exit(3)

    comments, cursor, pages = {}, None, 0
    t0 = time.time()
    while pages < 6000:
        try:
            r = twitch.get_comments(video_id, cursor=cursor)
            c = (r or {}).get("comments")
            if not c:
                tag("chat endpoint returned nothing (VOD chat unavailable?)")
                break
        except Exception as e:
            tag(f"page error, stopping with what we have: {str(e)[:100]}")
            break
        edges = c["edges"]
        for e in edges:
            n = e["node"]
            comments[n["id"]] = n
        pages += 1
        if pages % 25 == 0:
            tag(f"{len(comments)} msgs, offset {edges[-1]['node']['contentOffsetSeconds'] if edges else '?'}s, {pages} pages ({time.time()-t0:.0f}s)")
        if not c["pageInfo"]["hasNextPage"] or not edges or len(comments) >= max_messages:
            break
        cursor = edges[-1]["cursor"]

    msgs = sorted(comments.values(), key=lambda n: n["contentOffsetSeconds"])
    tag(f"{len(msgs)} unique messages over {pages} pages in {time.time()-t0:.0f}s")
    return video, msgs


def msg_text(n):
    frags = (n.get("message") or {}).get("fragments") or []
    return " ".join(f.get("text", "") for f in frags if f.get("text")).strip()


def meta_from(jobdir, video):
    jobdir = Path(jobdir)
    # merge on top of whatever meta_job fetched at job start (live titles,
    # clip author, local-file entries) — never blank out known fields
    meta = {}
    prev = jobdir / "meta.json"
    if prev.exists():
        try:
            meta = json.loads(prev.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    info = next(jobdir.glob("*.info.json"), None)     # yt-dlp dump (new downloads)
    if info:
        try:
            d = json.loads(info.read_text(encoding="utf-8"))
            meta.update({k: d.get(k) for k in
                         ("title", "uploader", "uploader_id", "timestamp",
                          "upload_date", "duration", "view_count", "like_count",
                          "categories", "chapters") if d.get(k) is not None})
        except Exception:
            pass
    if video:
        owner = video.get("owner") or video.get("creator") or {}
        game = video.get("game")
        meta.update({"kind": "vod",
                     "title": video.get("title") or meta.get("title"),
                     "note": None,          # live caveat no longer applies — VOD is final
                     "streamer": meta.get("uploader") or owner.get("login") or owner.get("displayName"),
                     "date": video.get("publishedAt") or video.get("recordedAt") or meta.get("upload_date"),
                     "duration_s": int(video.get("lengthSeconds") or meta.get("duration") or 0) or None,
                     "view_count": meta.get("view_count") or int(video.get("viewCount") or 0) or None,
                     "game": game.get("name") if isinstance(game, dict) else game,
                     "url": f"https://www.twitch.tv/videos/{video.get('id')}"})
    return {k: v for k, v in meta.items() if v not in (None, "", [])}


def run(jobdir, max_messages=120000, force=False):
    jobdir = Path(jobdir)
    cj = jobdir / "chat.json"
    if cj.exists() and not force:
        return
    state = json.loads((jobdir / "state.json").read_text(encoding="utf-8"))
    url = state.get("url") or ""
    m = re.search(r"twitch\.tv/videos/(\d+)", url)
    if not m:
        tag("job has no VOD url (local file, clip, or live-only) — skipped")
        sys.exit(3)
    video, msgs = fetch(m.group(1), max_messages)

    slim = [{"t": n["contentOffsetSeconds"],
             "user": (n.get("commenter") or {}).get("displayName")
                     or (n.get("commenter") or {}).get("login") or "?",
             "text": msg_text(n)} for n in msgs]
    from vodpipe import atomic_write
    atomic_write(cj, json.dumps(slim, ensure_ascii=False))
    with open(jobdir / "chat.txt", "w", encoding="utf-8") as f:
        for s in slim:
            f.write(f"[{s['t']//3600:02d}:{s['t']%3600//60:02d}:{s['t']%60:02d}] "
                    f"{s['user']}: {s['text']}\n")
    atomic_write(jobdir / "meta.json",
                 json.dumps(meta_from(jobdir, video), indent=1, ensure_ascii=False))
    tag(f"wrote chat.json ({len(slim)} msgs), chat.txt, meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--max-messages", type=int, default=120000)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()
    run(Path(ns.jobdir), ns.max_messages, ns.force)
