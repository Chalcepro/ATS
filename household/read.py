"""Coal reads a text: every word gets its place in the word library, and which words come
together is counted. Reading gives words, not meanings - those still come from class and life.

    py -3.12 -m household.read FILE_OR_FOLDER ...

Any document works: it is turned into text first (to_text.py). If Coal is running (CoalOS
open) the text goes to it there; otherwise the brain file is opened, read into, and saved.
What it has read is listed in the brain (keep["read"]).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

from household.to_text import to_text, tidy

COAL = "http://127.0.0.1:8765"


def read_into(lex, keep: dict, name: str, text: str) -> dict:
    """Coal reads one text into its word library. Returns what it took from it."""
    known = len(lex.ids)
    lines = [l for l in re.split(r"(?<=[.!?])\s+|\n+", text) if l.strip()]
    for line in lines:
        lex.meet(line)
        ws = [w for w in re.findall(r"[a-z']+", line.lower())]
        for a, b in zip(ws, ws[1:]):                      # which words come together
            lex.pairs[f"{a} {b}"] = lex.pairs.get(f"{a} {b}", 0) + 1
    got = {"name": name, "when": time.strftime("%Y-%m-%d %H:%M"), "sentences": len(lines),
           "words": len(text.split()), "new_words": len(lex.ids) - known}
    keep.setdefault("read", []).append(got)
    return got


def give(name: str, text: str) -> dict:
    """To the running Coal if there is one, else straight into the brain file."""
    try:
        req = urllib.request.Request(COAL + "/api/read", json.dumps({"name": name, "text": text}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except OSError:
        from household.lexicon import Lexicon, grow
        from household.model import load, save
        from household.train import SAVE
        brain, rounds = load(SAVE)
        lex = Lexicon.from_dict(brain.keep["lexicon"]) if "lexicon" in brain.keep else grow()
        got = read_into(lex, brain.keep, name, text)
        brain.keep["lexicon"] = dict(lex.__dict__)
        save(brain, SAVE, rounds)
        lex.save(os.path.join(os.path.dirname(SAVE), "lexicon.json"))
        return got


def main(paths: list[str]) -> None:
    files = [os.path.join(d, f) for p in paths for d, _, fs in (os.walk(p) if os.path.isdir(p) else [(os.path.dirname(p), 0, [os.path.basename(p)])]) for f in fs]
    for f in files:
        try:
            t = to_text(f)
        except Exception as e:
            print(f"  skipped {os.path.basename(f)}: {e}")
            continue
        if t and t.strip():
            g = give(os.path.basename(f), tidy(t))
            print(f"  read {g['name']}: {g['words']} words, {g['sentences']} sentences, {g['new_words']} new words")


if __name__ == "__main__":
    main(sys.argv[1:])
