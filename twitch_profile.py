"""twitch_profile — enumerate everything downloadable for a channel.

Given a login (or twitch.tv/<login> URL), reports:
  * live status right now (is_live)
  * VODs + highlights (twitch.tv/<login>/videos)
  * clips (twitch.tv/<login>/clips)

Used by vodpipe queue mode: live -> L<login> job (records when/while live),
each VOD/clip -> its normal job id, all queued and self-serializing on the
GPU lock. Re-running a queue is safe: existing job dirs are skipped, so
re-queueing a profile acts as "pick up anything new since last time".

  python twitch_profile.py <login-or-url> [--json]
"""
import argparse
import json
import re
import subprocess
import sys

RESERVED = {"videos", "p", "clip", "clips", "directory", "following",
            "subscriptions", "settings", "inventory", "duo", "jobs", "about",
            "front", "products", "tv", "collections", "user", "download",
            "mobile", "wallet", "prime", "turbo", "jobs", "communities"}


def channel_login(s):
    """Return the channel login if s is a bare login or a channel-root URL."""
    s = s.strip()
    m = re.fullmatch(r"(?:https?://)?(?:www\.|m\.)?twitch\.tv/"
                     r"([A-Za-z0-9_]+)/?(?:\?.*)?", s, re.I)
    if m and m.group(1).lower() not in RESERVED:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_]{3,25}", s):   # bare login, e.g. "some_streamer"
        return s
    return None


def _yt(*args):
    return subprocess.run([sys.executable, "-m", "yt_dlp", *args],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def is_live(login):
    r = _yt("-J", "--skip-download", f"https://www.twitch.tv/{login}")
    try:
        return json.loads(r.stdout).get("is_live") is True
    except Exception:
        return False


def _flat(url):
    r = _yt("--flat-playlist", "-J", url)
    try:
        return json.loads(r.stdout).get("entries") or []
    except Exception:
        return []


def _job_id(url):
    """Same derivation as vodpipe.job_id_for (kept ASCII, stable)."""
    import hashlib
    m = re.search(r"twitch\.tv/(?:videos/|p/)?(\d{6,})", url)
    if m:
        return "v" + m.group(1)
    m = re.search(r"clips\.twitch\.tv/([\w-]+)|twitch\.tv/\w+/clip/([\w-]+)", url)
    if m:
        return "c" + (m.group(1) or m.group(2))[:24].replace("-", "_")
    return "x" + hashlib.sha1(url.encode()).hexdigest()[:12]


# ~2.4 Mbps video + audio at 720p ~= 1.2 GB per stream-hour (measured:
# 11h VOD ~= 11.1 GiB). Budget guard uses this estimate.
GB_PER_HOUR = 1.2


def enumerate_profile(login):
    live = is_live(login)
    vods = []
    for e in _flat(f"https://www.twitch.tv/{login}/videos"):
        url = e.get("url") or ""
        vid = str(e.get("id") or "")
        if not vid:
            continue
        if not url:
            url = f"https://www.twitch.tv/videos/{vid.lstrip('0') if vid.isdigit() else vid}"
        kind = "highlight" if vid.startswith("hl") else "vod"
        dur = e.get("duration") or 0
        vods.append({"id": _job_id(url), "kind": kind,
                     "title": e.get("title") or "", "duration": dur,
                     "est_gb": round(dur / 3600 * GB_PER_HOUR, 1), "url": url})
    clips, seen = [], set()
    for e in _flat(f"https://www.twitch.tv/{login}/clips"):
        slug = str(e.get("id") or "")
        url = e.get("url") or f"https://clips.twitch.tv/{slug}"
        if not slug:
            continue
        dur = e.get("duration") or 60
        cid = _job_id(url)
        if cid in seen:          # flat-playlist pagination can repeat slugs
            continue
        seen.add(cid)
        clips.append({"id": cid, "kind": "clip",
                      "title": e.get("title") or "", "duration": dur,
                      "est_gb": round(dur / 3600 * GB_PER_HOUR, 2), "url": url})
    return {"login": login, "live": live,
            "live_url": f"https://www.twitch.tv/{login}", "live_id": "L" + login.lower(),
            "vods": vods, "clips": clips}


def hms(sec):
    if not sec:
        return "?"
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="list everything downloadable for a Twitch channel")
    ap.add_argument("login")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args()
    login = channel_login(ns.login)
    if not login:
        sys.exit(f"not a channel login/url: {ns.login}")
    prof = enumerate_profile(login)
    if ns.json:
        print(json.dumps(prof, ensure_ascii=False))
    else:
        print(f"channel {login}  live={'YES' if prof['live'] else 'no'}")
        for v in prof["vods"]:
            print(f"  {v['id']:>16s}  {v['kind']:9s} {hms(v['duration']):>8s}  {v['title'][:60]}")
        for v in prof["clips"]:
            print(f"  {v['id']:>16s}  {v['kind']:9s} {hms(v['duration']):>8s}  {v['title'][:60]}")
