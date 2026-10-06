#!/usr/bin/env python3
"""
twitch_dl.py — download any Twitch VOD / clip / live stream.

Uses yt-dlp under the hood (handles Twitch's client-id/token extraction,
HLS adaptive streaming, and muxing via ffmpeg).

Usage:
  python twitch_dl.py <twitch-url>                     # best quality -> Downloads
  python twitch_dl.py <url> -o D:\\Videos              # custom output dir
  python twitch_dl.py <url> -f 720                    # cap height at 720p
  python twitch_dl.py <url> --start 00:10:00 --end 00:20:00   # trim a section
  python twitch_dl.py <url> --list                    # list formats, don't download
  python twitch_dl.py <url> --sub                     # live sub only, no chat/emoji extras
  python twitch_dl.py <https://twitch.tv/somechannel> # live channel (grabs stream if up)

Notes:
  - VODs expire (60 days; 7/60 for affiliates) — if the VOD is gone, nothing works.
  - Subscriber-only VODs need --browser chrome (or firefox) to use your login cookie.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

VALID_URL_RE = re.compile(
    r"(twitch\.tv/(videos/\d+|clip/[\w-]+|p/\w+|\w+)(\?.*)?$)", re.I
)

def run_ytdlp(args):
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    return subprocess.run(cmd)

def main():
    ap = argparse.ArgumentParser(description="Download any Twitch VOD/clip/live stream.")
    ap.add_argument("url", help="Twitch URL (video, clip, or channel)")
    ap.add_argument("-o", "--output", default=str(Path.home() / "Downloads"),
                    help="Output directory (default: ~/Downloads)")
    ap.add_argument("-f", "--height", type=int, default=None,
                    help="Max video height, e.g. 1080, 720 (default: best)")
    ap.add_argument("--start", default=None, help="Trim start (HH:MM:SS or seconds)")
    ap.add_argument("--end", default=None, help="Trim end   (HH:MM:SS or seconds)")
    ap.add_argument("--list", action="store_true", help="List formats and exit")
    ap.add_argument("--sub", action="store_true", help="Sub-only: video+audio, skip other assets")
    ap.add_argument("--browser", default=None, choices=["chrome", "firefox", "edge"],
                    help="Use browser cookies (needed for subscriber-only VODs)")
    ns = ap.parse_args()

    url = ns.url if ns.url.startswith("http") else "https://" + ns.url
    if not VALID_URL_RE.search(url):
        sys.exit(f"Not a Twitch URL: {url}")

    os.makedirs(ns.output, exist_ok=True)

    if ns.list:
        sys.exit(run_ytdlp(["-F", url]).returncode)

    sel = f"bv*[height<={ns.height}]+ba/best" if ns.height else "bv+ba/best"
    outtmpl = str(Path(ns.output) / "%(title)s [%(id)s].%(ext)s")

    args = ["-f", sel, "-o", outtmpl, "--merge-output-format", "mp4",
            "--no-playlist", "--embed-metadata", "--progress", "--newline", url]

    if ns.sub:
        i = args.index("-f")
        args[i + 1] = f"{sel}/best"  # plain video+audio only

    if ns.browser:
        args = ["--cookies-from-browser", ns.browser] + args

    # Section trimming (VODs only): yt-dlp syntax is *START-END
    if ns.start or ns.end:
        rng = f"*{ns.start or '00:00:00'}-{ns.end or '99:00:00'}"
        args = [a for a in args if a != url]
        args += ["--download-sections", rng, url]

    print(f"[twitch_dl] {url}\n[twitch_dl] saving to {ns.output}")
    rc = run_ytdlp(args).returncode
    sys.exit(rc)

if __name__ == "__main__":
    main()
