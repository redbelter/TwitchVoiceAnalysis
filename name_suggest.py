"""name_suggest — infer probable display names for voice lanes from context.

Pure text + timing statistics, no models, no network. Deterministic.

Evidence kinds (each vote carries weight + a quotable timestamp):
  self     "I'm X" / "my name is X" inside a lane's own line      (w=3.0)
  join     "X just joined / is here / hopped in" (chat OR read aloud)
           and exactly one lane STARTS speaking within 15 s       (w=2.0)
  reply    chat message addresses NAME, lane starts speaking
           within -2..+12 s                                       (w=1.0, decays)
  pair     a line addresses NAME and the very next speaking
           turn (different lane) starts within 6 s                (w=0.8)
  channel  the job's Twitch login belongs to the STREAMER lane   (w=2.0)
  acoustic voiceprint centroid matches a strongly-named lane at
           cos>=0.55 (same person, diarization re-split)          (w=2.0)

A (lane, name) pair becomes a PROPOSAL only when its margin over the
runner-up lane is clear; otherwise it's reported as ambiguous. Names are
never written into people.json — this side-writes names_suggested.json and
the dashboard shows proposals with their evidence; the human confirms by
rerunning with --seed, which remains the source of truth.

  python name_suggest.py <job_dir> [--force]
"""
import json
import re
import sys
import time
from pathlib import Path

LN_RE = re.compile(r"\[(\d+):(\d+):(\d\d(?:\.\d+)?)\]\s*([^:]+?)\s*:\s*(.*)$")

# vocative/greeting frames: a NAME is being called.
# Chat is lowercase, whisper is inconsistently cased — match any case and
# judge word-worthiness in _ok_name instead.
ADDR = re.compile(
    r"(?:^|[\s,.!?])(?:hey|hi|yo|hiya|hello|sup|wassup|what's up|whats up|"
    r"good morning|morning|good evening|welcome(?: back)?|gl|gg|ty|thanks|thank you|"
    r"nice|lol|omg|wow|ok(?:ay)?|yeah|yep|no|but|and|so|wait|look|ask|tell|said|"
    r"with|to|from|for|@)\s+([A-Za-z][a-zA-Z]{2,14})\b"
    r"|\b([A-Za-z][a-zA-Z]{2,14})\b[\s,.!?]*(?:is (?:here|back|on)|speaking|talking|"
    r"good morning|are you|you there|where are you)"
    r"|^([A-Za-z][a-zA-Z]{2,14})\b[,\s]+(?:you|your|are|is|where|what|why|how|do|did|can|will|should)\b",
    re.I)
SELF = re.compile(
    r"\b(?:i'?m|i am|my name is)\s+([A-Za-z][a-zA-Z]{2,14})\b"
    r"|\bthis is\s+([A-Z][a-zA-Z]{2,14})\b(?=\s|$|[,.!?])", re.I)

# join announcements: chat or read aloud ("X just joined", "X is here",
# "X hopped in the call") — a voice that FIRST appears right after is X
ANN = re.compile(
    r"\b([A-Za-z][A-Za-z]{2,16})\b[^,.!?]{0,24}?"
    r"\b(?:just\s+)?(?:joined|joins|joining|has joined|is here|came in|got here|"
    r"got on|hopped in|hop in|hops in|jumped in|in the vc|in voice|"
    r"on the call|in the call|on the vc)\b", re.I)

# words that get capitalized but are not names (extend conservatively —
# a false name costs trust; a missed name costs nothing)
STOP = set("""
the a an and but or so yet for nor this that these those here there who what
when where why how all any some none nobody someone something nothing
everything today tomorrow yesterday tonight morning evening night day week
year time life love yeah yep nope okay ok xd twitch stream clip vod vods
discord youtube tiktok instagram twitter snapchat spotify patreon monday
tuesday wednesday thursday friday saturday sunday january february march
april may june july august september october november december christmas
halloween birthday america english fortnite minecraft lmg lol lmao wtf
support mid top jungle bot bots chat chatting mods mod moderator admin
mr mrs ms dr sir maam bro bruh dude guy guys girl girls she him her they
them we you me my mine your yours his its our their us why how
fine back good sorry here tired busy ready sure just still really so not
on off up down happy sad sick confident serious honest curious bored excited
hungry sleepy awake live new old same different right wrong first last next
only even never always maybe probably basically actually literally kinda
sorta gonna wanna hafta tryna totally absolutely definitely obviously
apparently seriously technically going getting saying taking giving liking
sitting standing waiting hoping trying feeling thinking talking asking
telling wondering remembering pretending pretending bout kinda tryin
like pretty about out dead freaking damn crazy man dude girl kid folks
going's can't dont don't cant wont isnt ain't cause cus
""".split())

# small gender hint map for sanity flags only (voice pitch beats it anyway)
NAME_G = {}
for g, names in {
    "female": "abby aleah alina amanda amber amy ana angelica anna ashley "
              "bayan bee bella betty brie casey cass chelsea christina cynthia "
              "dani danielle dee diamond emma erin faith grace hannah harley "
              "heather iris jade jenny jess jessica jo jojo jordan julia kaitlyn "
              "karen kate katie kelly kim kimbra kristen lana lauren lea lexi "
              "lily lindsay lisa liz maddie maren maria mary mckenzie "
              "meg melanie michele molly natasha nicole nina olivia pam rachel "
              "rebecca rita rose samantha sarah shannon shawna sophie stephanie "
              "sue taylor tiffany tina tracey val vanessa veronica vicki yuki "
              "zoe zuri".split(),
    "male": "adam alex andrew anthony aaron ben benjamin brad brandon "
            "brett brian chris christian daniel david dean derek dominic doug "
            "edward eric ethan evan gary george greg henry hugh ian jackson "
            "jacob james jason jay jeff jeremy jesse joe john jonathan jordan "
            "joseph joshua kevin kyle larry lenny leo logan luis marc mark "
            "marty mason matt michael mike nathan nick oliver pat patrick paul "
            "peter ray rick rob roger ronan ross ryan sam scott seth "
            "shane steve steven tony travis tyler vic will zach zachary "
            "arthur bruce clint dale frank gus hank jack jim joe kurt lester "
            "miles pete ralph stan".split(),
}.items():
    for n in names:
        NAME_G[n] = g


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def _ok_name(c):
    c0 = c.strip()
    if not re.fullmatch(r"[A-Za-z][a-zA-Z]{2,14}", c0):
        return False
    n = _norm(c0)
    if n in STOP or len(set(n)) < 2 or len(set(n)) == 1:
        return False                      # "aaa"/"lolol" garbage; real short
                                          # names (Bob, Jim) have 2+ distinct
    # ALL-CAPS shouting ("WHY", "ALICE") counts as non-name unless mixed-case
    if c0.isupper():
        return False
    return True


def _secs(m):
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def load_turns(jobdir):
    """[(t, lane, text)] from labeled.txt."""
    turns = []
    try:
        for ln in (jobdir / "labeled.txt").read_text(encoding="utf-8",
                                                      errors="replace").splitlines():
            m = LN_RE.match(ln.strip())
            if m:
                turns.append((_secs(m), m[4].strip(), m[5]))
    except FileNotFoundError:
        pass
    return turns


def load_chat(jobdir):
    try:
        return json.loads((jobdir / "chat.json").read_text(encoding="utf-8"))
    except Exception:
        return []


def _cand_from(text, pat):
    out = []
    for m in pat.finditer(text or ""):
        for gi, grp in enumerate(m.groups(), start=1):
            if not grp or not _ok_name(grp):
                continue
            # a candidate glued to '-' ("Tic-Tac") is a truncated username —
            # don't guess the first half of someone's handle
            if m.end(gi) < len(text) and text[m.end(gi)] == "-":
                continue
            out.append(grp)
    return out


def run(jobdir, force=False):
    jobdir = Path(jobdir)
    out = jobdir / "names_suggested.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    t0 = time.time()
    turns = load_turns(jobdir)
    chat = load_chat(jobdir)
    people = {}
    try:
        people = json.loads((jobdir / "people.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    voice = {}
    try:
        voice = json.loads((jobdir / "voice_stats.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    # votes: (lane, normname) -> {score, display, evidence:[...]}
    votes = {}

    def vote(lane, name, t, kind, w, quote):
        # display form: keep real casing if the source had it, else Title-case
        disp = name if (name[0].isupper() or name != name.lower()) else name.capitalize()
        k = (lane, _norm(name))
        v = votes.setdefault(k, {"lane": lane, "name": disp, "display": {},
                                 "score": 0.0, "evidence": []})
        v["display"][disp] = v["display"].get(disp, 0) + 1
        v["score"] += w
        if len(v["evidence"]) < 8:
            v["evidence"].append({"kind": kind, "t": round(t, 1),
                                  "quote": (quote or "")[:120]})

    # --- self references (strongest) ----------------------------------------
    for t, lane, text in turns:
        for nm in _cand_from(text, SELF):
            # whisper capitalizes genuine self-introductions ("I'm Alice") but
            # leaves predicates lowercase ("i'm going") — require real casing.
            if nm != nm.lower() and nm[0].isupper() and _norm(nm) not in STOP:
                vote(lane, nm, t, "self", 3.0, text)

    # --- channel login -> STREAMER lane --------------------------------------
    try:
        meta = json.loads((jobdir / "meta.json").read_text(encoding="utf-8"))
        login = (meta.get("streamer") or "").strip()
        if login and "STREAMER" in people and len(login) >= 3:
            disp = login
            # prefer the cased form actually seen in chat messages
            low = _norm(login)
            seen = re.compile(r"\b([A-Za-z][a-zA-Z]{2,14})\b")
            for c in chat[:400]:
                for mm in seen.finditer(c.get("text") or ""):
                    if _norm(mm.group(1)) == low:
                        disp = mm.group(1)
                        break
            vote("STREAMER", disp, 0, "channel", 2.0,
                 f"channel login {login}")
    except Exception:
        pass

    # --- join announcements: "X just joined" -> lane that STARTS speaking ----
    first_t = {}
    for t, lane, _ in turns:
        if lane not in first_t:
            first_t[lane] = t
    events = [(float(c.get("t", 0) or 0), c.get("text") or "", "chat")
              for c in chat]
    events += [(t, text, "voice") for t, _, text in turns]
    for et, text, _src in events:
        for m in ANN.finditer(text):
            nm = m.group(1)
            if not _ok_name(nm):
                continue
            # lanes whose FIRST utterance lands 0..15 s after the announcement
            starters = [ln for ln, ft in first_t.items() if 0 <= ft - et <= 15
                        and ln in people]
            if len(starters) == 1:      # unambiguous newcomer
                vote(starters[0], nm, et, "join", 2.0,
                     f"{_src}: {text}")

    # --- chat-address -> next speaker reply latency ---------------------------
    turn_i = 0
    for c in chat:
        ct = c.get("t", 0) or 0
        text = c.get("text") or ""
        names = _cand_from(text, ADDR)
        if not names:
            continue
        # next turn starting within -2..+12 s of the message
        while turn_i < len(turns) and turns[turn_i][0] < ct - 2:
            turn_i += 1
        if turn_i >= len(turns):
            continue
        t, lane, _ = turns[turn_i]
        if -2 <= t - ct <= 12:
            for nm in names:
                if lane in people:
                    vote(lane, nm, ct, "reply", 1.0,
                         f"chat {c.get('user')}: {text}")

    # --- spoken vocative -> next different lane answers -----------------------
    for i, (t, lane, text) in enumerate(turns[:-1]):
        names = _cand_from(text, ADDR)
        if not names:
            continue
        t2, lane2, _ = turns[i + 1]
        if 0 < t2 - t <= 6 and lane2 != lane and lane2 in people:
            for nm in names:
                if _norm(nm) != _norm(lane):
                    vote(lane2, nm, t, "pair", 0.8, text)

    # --- acoustic propagation: voice-identical lanes inherit a strong name ----
    # frag_merge deliberately protects >=5-seg lanes from merging, so the SAME
    # person re-split by diarization survives as two "people". If one side has
    # a strong name and their voiceprint centroids match at cos>=0.55, the
    # other side is (near-certainly) the same person — propagate it.
    try:
        import numpy as np
        emb_p, idx_p, lab_p = (jobdir / "vp_emb2.npy", jobdir / "vp_idx2.json",
                               jobdir / "vp_lab2.npy")
        if emb_p.exists() and lab_p.exists() and votes:
            E = np.load(emb_p, allow_pickle=True).astype(np.float32)
            raw_lab = np.fromfile(lab_p, dtype="int32")
            lane_of_row = {}
            for pnm, pp in people.items():
                if pnm != "(short)" and pp.get("lane") is not None:
                    lane_of_row[pp["lane"]] = pnm
            cents = {}
            for L in np.unique(raw_lab):
                nm = lane_of_row.get(int(L))
                rows = np.where(raw_lab == L)[0]
                if nm and len(rows):
                    c = E[rows].mean(0)
                    cents[nm] = c / max(np.linalg.norm(c), 1e-9)
            strong = {}
            for (vlane, vnn), v in votes.items():
                # seed only from DIRECT evidence (self/channel/join): weak
                # pair/reply junk must never launder itself via acoustic boost
                vkinds = {ev["kind"] for ev in v["evidence"]}
                if (v["score"] >= 2.0 and vlane in cents
                        and vkinds <= {"self", "channel", "join"} and vkinds):
                    cur = strong.get(vlane)
                    if cur is None or v["score"] > cur["score"]:
                        strong[vlane] = v
            prop = []
            for nm, c in cents.items():
                if nm in strong:
                    continue
                best = None
                for snm, sv in strong.items():
                    s = float(c @ cents[snm])
                    if s >= 0.55 and (best is None or s > best[1]):
                        best = (snm, s)
                if best:
                    prop.append((nm, strong[best[0]], best))
            for nm, sv, best in prop:
                disp = max(sv["display"].items(), key=lambda kv: kv[1])[0]
                vote(nm, disp, 0, "acoustic", 2.0,
                     f"voiceprint cos {best[1]:.2f} vs {best[0]}")
    except Exception:
        pass

    # --- collapse: best name per lane + ambiguity margins --------------------
    by_lane = {}
    name_lanes = {}
    for (lane, nn), v in votes.items():
        name_lanes.setdefault(nn, []).append(v)
    per_lane = {}
    for (lane, nn), v in votes.items():
        per_lane.setdefault(lane, []).append(v)

    for lane, vs in per_lane.items():
        vs.sort(key=lambda v: -v["score"])
        best = vs[0]
        disp = max(best["display"].items(), key=lambda kv: kv[1])[0]
        g_est = (voice.get(lane) or {}).get("voice_gender")
        hint = NAME_G.get(_norm(disp))
        entry = {
            "name": disp,
            "score": round(best["score"], 2),
            "margin": round(best["score"] - (vs[1]["score"] if len(vs) > 1 else 0), 2),
            "evidence": best["evidence"],
            "runners": [{"name": v["name"], "score": round(v["score"], 2)}
                        for v in vs[1:3]],
        }
        # a name some OTHER lane claims stronger = contested — EXCEPT when
        # this lane's evidence is acoustic-only: an acoustic twin legitimately
        # inherits the name the source lane earned outright
        kinds = {ev["kind"] for ev in best["evidence"]}
        rivals = [w for w in name_lanes[_norm(disp)]
                  if w["lane"] != lane and w["score"] > best["score"]]
        if kinds == {"acoustic"}:
            rivals = []
        contested = bool(rivals) or entry["margin"] < 1.0
        conf = ("high" if (not contested and best["score"] >= 2.0)
                else "low" if contested else "medium")
        # derived-only evidence (acoustic twin, lone join event) deserves
        # less trust than a spoken self-intro, however tidy the numbers look
        if conf == "high" and kinds <= {"acoustic"}:
            conf = "medium"
        entry["confidence"] = conf
        if hint and g_est and hint != g_est:
            entry["gender_mismatch"] = f"name reads {hint}, voice reads {g_est}"
        by_lane[lane] = entry

    # keep only proposals worth showing; raw votes stay for the CLI --all
    shown = {lane: e for lane, e in by_lane.items()
             if e["score"] >= 2.0 and e["confidence"] != "low"}
    res = {"generated": round(time.time() - t0, 2),
           "lanes": len(people), "proposals": shown,
           "hidden": {lane: e["name"] for lane, e in by_lane.items()
                      if lane not in shown}}
    from vodpipe import atomic_write
    atomic_write(out, json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    r = run(d, force="--force" in sys.argv)
    for lane, e in sorted(r["proposals"].items(), key=lambda kv: -kv[1]["score"]):
        kinds = ",".join(sorted({ev["kind"] for ev in e["evidence"]}))
        print(f"{lane:>12} -> {e['name']:<14} {e['confidence']:<6} "
              f"score {e['score']} kinds: {kinds}")
