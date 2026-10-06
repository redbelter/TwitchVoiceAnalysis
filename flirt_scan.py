#!/usr/bin/env python3
"""
flirt_scan.py — extract flirting-register conversation blocks from a voice-labeled
transcript (vod_voices_labeled.txt built by vp_cluster.py + labeling step).

Usage:
  python flirt_scan.py vod_voices_labeled.txt vod_flirting.txt
"""
import re
import sys
from pathlib import Path

FLIRT = re.compile(
    r"flirt|crush|baddie|shorty|gyatt|gyat\b|rizz|waifu|wifey|\bbae\b|"
    r"girlfriend|boyfriend|dating|go on a date|\bdate\b|\bdates\b|marry|married|proposal|"
    r"\bkiss\b|kisses|kissing|snog|cuddle|makeout|make out|sliding into|thirsty|talking stage|"
    r"baby girl|babyboy|\bbabe\b|\bbabes\b|sweetheart|sweetie|cutie|shawty|\bbby\b|"
    r"how old are you|\bsingle\b|\btaken\b|relationship|wife material|prom night|"
    r"love struck|cupid|valentine|match made in heaven|pickup line|pick up line|"
    r"she'?s (so |really |kinda |pretty |fine\b)|you'?re (so |really |kinda )?(cute|fine|hot|pretty|sexy)|"
    r"i think (you|she|he)('?s| is) (cute|fine|hot|pretty|nice)|i like you|i want (you|her|him)|"
    r"be my (girlfriend|boyfriend|girl|wife)|my (girlfriend|boyfriend|type)|\bwife\b|\bhusband\b|"
    r"\bthicc\b|thick\b.*(she|her|girl)|fine\b.*(black man|dude|boy)|mine\b.*(baby|shorty)", re.I)

# benign patterns to exclude (streamer catchphrases / game talk / wholesome bits)
BENIGN = re.compile(
    r"beautiful (day|people)|every single day|single day|love you guys|love you[,. ]|"
    r"simpsons|little baby|senior baby|10 year old baby|baby shark|baby yoda|"
    r"i love (valorant|fortnite|today|meeting|having|involving|new people|talking|people|it|my|him)|"
    r"love the way|good morning.*baby girl|baby girl good morning|dating app.*ad|"
    r"(died|kill|dead|shot|hit).*(single|taken)|taken (a|the|his|her|my) (seat|spot|break|down)|"
    r"wife\b.*(mrs|missus|got me|is)\b.*(annoying)", re.I)


def ts(ln):
    m = re.match(r"\[(\d+):(\d+):(\d+)\]", ln)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


def hms(sec):
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "vod_voices_labeled.txt")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "vod_flirting.txt")
    lines = src.read_text(encoding="utf-8").splitlines()

    hits = [i for i, ln in enumerate(lines) if FLIRT.search(ln) and not BENIGN.search(ln)]
    marks = set()
    for i in hits:
        marks.update(range(max(0, i - 2), min(len(lines), i + 3)))

    blocks, cur, lastt = [], [], None
    for j in sorted(marks):
        t = ts(lines[j])
        if t is None:
            continue
        if lastt is not None and t - lastt > 45 and cur:
            blocks.append(cur)
            cur = []
        cur.append((t, lines[j]))
        lastt = t
    if cur:
        blocks.append(cur)

    out_lines = []
    for b in blocks:
        out_lines.append(f"--- {hms(b[0][0])} ---")
        out_lines.extend(l for _, l in b)
        out_lines.append("")
    head = (f"FLIRTING-REGISTER EXTRACT from {src.name}\n"
            f"{len(blocks)} blocks around {len(hits)} trigger lines (catchphrase noise filtered)\n\n")
    out.write_text(head + "\n".join(out_lines), encoding="utf-8")
    print(f"{len(hits)} triggers -> {len(blocks)} blocks -> {out}")


if __name__ == "__main__":
    main()
