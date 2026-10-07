"""vodpipe_queue — queue a whole Twitch profile (live + VODs + clips).

  python vodpipe_queue.py <login> [--max-gb 100] [--clips] [--no-live]

Enumerates the channel (twitch_profile), trims the list to what fits on disk
(newest first, honoring --max-gb and leaving 20 GB free), then runs vodpipe
jobs sequentially on the shared GPU lock. Progress lands in
<workdir>/queue_<login>.json, which the dashboard renders.

Statuses per item: pending -> running -> done | cached | offline | failed.
Re-running the same queue is cheap: completed job dirs are skipped, and any
new VOD since last run is picked up. Live item: if the channel is offline it
is marked offline and the queue moves on (re-run later to catch the stream).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twitch_profile as tp  # noqa: E402
import vodpipe  # noqa: E402  (pid_alive; main() is __main__-guarded)


def now():
    return time.strftime("%F %T")


def save_q(path, q):
    vodpipe.atomic_write(path, json.dumps(q, indent=1, ensure_ascii=False))


def budget_items(items, workdir, max_gb):
    """Keep newest-first items while they fit: <= max_gb total AND leaves
    20 GB free on the volume. Returns (keep, dropped)."""
    free = shutil.disk_usage(str(workdir)).free / 1e9
    cap = min(max_gb, free - 20)
    out, drop, cum = [], [], 0.0
    for it in items:
        est = it.get("est_gb") or 0
        if it["kind"] == "live" or cum + est <= cap:
            out.append(it)
            cum += est
        else:
            drop.append(it)
    return out, drop, round(cap, 1)


def main():
    ap = argparse.ArgumentParser(prog="vodpipe_queue")
    ap.add_argument("login", help="channel login or twitch.tv/<login> url")
    ap.add_argument("--workdir", default=str(Path.home() / "Downloads" / "vodpipe"))
    ap.add_argument("--max-gb", type=float, default=100.0,
                    help="max disk to use for this queue (default 100)")
    ap.add_argument("--clips", action="store_true", help="also queue clips (small)")
    ap.add_argument("--no-live", action="store_true", help="skip the live-recording item")
    ap.add_argument("--vod-limit", type=int, default=0, help="only N newest VODs")
    ap.add_argument("--height", type=int, default=720)
    ns = ap.parse_args()

    login = tp.channel_login(ns.login)
    if not login:
        sys.exit(f"[queue] not a channel login/url: {ns.login}")
    workdir = Path(ns.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    qpath = workdir / f"queue_{login.lower()}.json"
    print(f"[queue] enumerating {login} ...")
    prof = tp.enumerate_profile(login)

    items = []
    if prof["live"] and not ns.no_live:
        items.append({"id": prof["live_id"], "url": prof["live_url"],
                      "kind": "live", "title": "RECORD LIVE (realtime)", "est_gb": None})
    vods = prof["vods"]
    if ns.vod_limit:
        vods = vods[:ns.vod_limit]
    items += vods
    if ns.clips:
        items += prof["clips"]

    keep, dropped, cap = budget_items(items, workdir, ns.max_gb)
    q = {"login": login, "updated": now(), "cap_gb": cap,
         "dropped_gb": round(sum(d.get("est_gb") or 0 for d in dropped), 1),
         "dropped": [{"id": d["id"], "title": d["title"][:60]} for d in dropped[:20]],
         "items": [{"id": it["id"], "url": it["url"], "kind": it["kind"],
                    "title": (it.get("title") or "")[:70],
                    "est_gb": it.get("est_gb"), "status": "pending"} for it in keep]}
    save_q(qpath, q)
    print(f"[queue] {len(keep)} items ({q['dropped_gb']} GB trimmed by budget), cap {cap} GB")

    for i, it in enumerate(q["items"]):
        jdir = workdir / it["id"]
        st_path = jdir / "state.json"
        if st_path.exists():
            try:
                st = json.loads(st_path.read_text(encoding="utf-8"))
                if all(v.get("status") == "done" for v in st.get("stages", {}).values()):
                    it["status"] = "cached"
                    save_q(qpath, q)
                    continue
                rp = st.get("runner_pid")
                if rp and vodpipe.pid_alive(rp):   # someone else already runs it
                    it["status"] = "running"
                    save_q(qpath, q)
                    continue
            except Exception:
                pass
        it["status"] = "running"
        save_q(qpath, q)
        cmd = [sys.executable, str(HERE / "vodpipe.py"), it["url"],
               "--workdir", str(workdir), "--wait-lock", "-f", str(ns.height)]
        if it["kind"] == "live":
            cmd.append("--live")
        jdir.mkdir(parents=True, exist_ok=True)
        (jdir / "logs").mkdir(exist_ok=True)
        slog = open(jdir / "logs" / "queue-spawn.log", "a", encoding="utf-8")
        slog.write(f"\n===== {now()} queue {login} =====\n")
        slog.flush()
        rc = subprocess.run(cmd, stdout=slog, stderr=slog).returncode
        # read what the job thought of itself
        status = "failed"
        try:
            st = json.loads(st_path.read_text(encoding="utf-8"))
            dl = st.get("stages", {}).get("download", {})
            if dl.get("status") == "waiting":
                status = "offline"
            elif all(v.get("status") == "done" for v in st.get("stages", {}).values()):
                status = "done"
            elif rc == 0:
                status = "partial"
        except FileNotFoundError:
            pass
        it["status"] = status
        q["updated"] = now()
        save_q(qpath, q)
        print(f"[queue] {it['id']} -> {status}")
    q["finished"] = now()
    save_q(qpath, q)
    print(f"[queue] DONE {login}")


if __name__ == "__main__":
    main()
