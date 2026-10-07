"""meta_job — fetch video metadata as soon as a job is created.

Titles/views/game make job rows recognizable ("what video is this?") instead
of bare ids. Written to <jobdir>/meta.json so the dashboard list can read it
without any network. Handles every job URL shape:

  twitch.tv/videos/NNN        -> GQL video object (title, streamer, game, ...)
  clips.twitch.tv / clip URL  -> GQL clip object
  twitch.tv/<login> (live)    -> yt-dlp probe, only if actually live
  local file                  -> filename + probed duration (offline)

Every path is best-effort: network down / clip expired -> no meta.json, no
crash. The chat stage still refreshes meta.json at job end (richer, merged).

  python meta_job.py <job_dir> [--force]
"""
import json
import re
import sys
from pathlib import Path


def _t():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _clean(d):
    return {k: v for k, v in d.items() if v not in (None, "", [], 0)}


def _norm_video(v):
    owner = v.get("owner") or v.get("creator") or {}
    game = v.get("game")
    return _clean({
        "kind": "vod", "title": v.get("title"),
        "streamer": owner.get("login") or owner.get("displayName"),
        "date": v.get("publishedAt") or v.get("recordedAt"),
        "duration_s": int(v.get("lengthSeconds") or 0) or None,
        "view_count": int(v.get("viewCount") or 0) or None,
        "game": game.get("name") if isinstance(game, dict) else game,
        "url": f"https://www.twitch.tv/videos/{v.get('id')}" if v.get("id") else None})


def _norm_clip(c, slug):
    game = c.get("game")
    b = c.get("broadcaster") or {}
    return _clean({
        "kind": "clip", "title": c.get("title"),
        "streamer": b.get("login") or b.get("displayName") or c.get("author") or c.get("login"),
        "date": c.get("createdAt"),
        "duration_s": round(float(c.get("durationSeconds") or c.get("duration") or 0)) or None,
        "view_count": int(c.get("viewCount") or 0) or None,
        "game": game.get("name") if isinstance(game, dict) else game,
        "url": c.get("url") or f"https://clips.twitch.tv/{slug}"})


def _live_meta(login):
    """yt-dlp probe of a channel root; returns meta only while live."""
    import json as _j
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "-J", "--skip-download",
                            f"https://www.twitch.tv/{login}"],
                           capture_output=True, text=True, timeout=45,
                           encoding="utf-8", errors="replace")
        d = _j.loads(r.stdout or "{}")
        if d is None:          # yt-dlp prints literal `null` for offline channels
            return None
    except Exception:
        return None
    if not d.get("is_live"):
        return None
    return _clean({"kind": "live", "title": d.get("title") or f"LIVE — {login}",
                   "streamer": login, "game": d.get("category"),
                   "view_count": d.get("concurrent_view") or None,
                   "url": f"https://www.twitch.tv/{login}",
                   "note": "title/views change live; refreshed when recording ends"})


def sync(jobdir, force=False):
    """Fetch + write meta.json for this job. Returns the meta dict ({} = none)."""
    jobdir = Path(jobdir)
    out = jobdir / "meta.json"
    if out.exists() and not force:
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            pass
    sp = jobdir / "state.json"
    if not sp.exists():
        return {}
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    url = st.get("url") or ""
    meta = {}
    try:
        m = re.search(r"twitch\.tv/(?:videos/|p/)(\d{6,})", url)
        c = re.search(r"clips\.twitch\.tv/([\w-]+)|twitch\.tv/\w+/clip/([\w-]+)", url)
        ch = re.search(r"twitch\.tv/([A-Za-z0-9_]+)/?$", url)
        if m:
            from twitchdl import twitch
            v = twitch.get_video(m.group(1))
            meta = _norm_video(v) if v else {}
        elif c:
            from twitchdl import twitch
            slug = c.group(1) or c.group(2)
            cl = twitch.get_clip(slug)
            meta = _norm_clip(cl, slug) if cl else {}
        elif ch and ch.group(1).lower() not in ("videos", "p", "clip", "clips", "directory"):
            meta = _live_meta(ch.group(1).lower()) or {}
        elif not url and st.get("input"):
            import vodpipe
            p = Path(st["input"])
            meta = _clean({"kind": "local", "title": p.stem,
                           "duration_s": round(vodpipe.ffprobe_dur(p) or 0) or None})
    except Exception as e:
        print(f"[meta] fetch failed (non-fatal): {str(e)[:120]}")
        return {}
    if meta:
        meta["fetched"] = _t()
        from vodpipe import atomic_write
        atomic_write(out, json.dumps(meta, indent=1, ensure_ascii=False))
        print(f"[meta] {meta.get('kind')}: {str(meta.get('title'))[:70]}")
    return meta


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(json.dumps(sync(Path(sys.argv[1]), force), indent=1, ensure_ascii=False))
