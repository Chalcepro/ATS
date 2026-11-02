"""WordNet (Princeton, wordnetcode.princeton.edu, WordNet 3.1) as class lessons for Coal.

    py -3.12 -m household.dictionary 20        (show the 20 most common words' lessons)

For each word, most common first (WordNet's own usage counts, from tagged real text):
  - what it means           -> its most common sense's definition
  - what kind of word it is -> noun / verb / describing word (adjective) / adverb
  - what kind of thing it is (nouns) -> its wider kinds: "a kind of bear, which is a kind of mammal"
Asked four ways each, like every class lesson; the fourth way is held back for the exam.
The data lives in data/wordnet/dict (gitignored, it is Princeton's); fetch it with
  curl -L -o data/wordnet/wn3.1.dict.tar.gz https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz
"""
from __future__ import annotations

import os
import re
import sys

DICT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wordnet", "dict")
FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}
POS_OF_KEY = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "a"}      # 5 = adjective satellite
KIND = {"n": "a noun: a word for a thing, a person, a place or an idea",
        "v": "a verb: a doing word, an action",
        "a": "an adjective: a describing word, it tells what something is like",
        "r": "an adverb: a describing word for how, when or where something is done"}
_synsets: dict | None = None


def synsets() -> dict:
    """(pos, offset) -> {"words", "gloss", "up"}; up = the wider kinds (hypernyms)."""
    global _synsets
    if _synsets is None:
        _synsets = {}
        for pos, name in FILES.items():
            for line in open(os.path.join(DICT, "data." + name), encoding="utf-8", errors="ignore"):
                if line.startswith(" "):
                    continue                                   # the licence at the top
                head, _, gloss = line.partition(" | ")
                f = head.split()
                n = int(f[3], 16)
                words = [f[4 + 2 * i].replace("_", " ") for i in range(n)]
                k = 4 + 2 * n
                ptrs = [f[k + 1 + 4 * i: k + 5 + 4 * i] for i in range(int(f[k]))]
                up = [(p[2] if p[2] != "s" else "a", p[1]) for p in ptrs if p[0] in ("@", "@i")]
                _synsets[(pos, f[0])] = {"words": words, "gloss": gloss.strip(), "up": up}
    return _synsets


def common(limit: int) -> list[tuple[str, str, str]]:
    """The most used words, as (word, pos, offset of its most used sense)."""
    best: dict[str, tuple[int, str, str]] = {}
    for line in open(os.path.join(DICT, "index.sense"), encoding="utf-8"):
        key, off, _, count = line.split()
        lemma, rest = key.split("%")
        if "_" in lemma or not lemma.isalpha() or len(lemma) < 2:
            continue                                           # single plain words first
        c = int(count)
        if c > best.get(lemma, (-1,))[0]:
            best[lemma] = (c, POS_OF_KEY[rest[0]], off)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    return [(w, pos, off) for w, (c, pos, off) in ranked[:limit]]


def meaning(gloss: str) -> str:
    """The definition without the example sentences ("...; "he did it"")."""
    return gloss.split(";")[0].strip()                        # the first sense only, no examples


def kinds(pos: str, off: str, deep: int = 2) -> list[str]:
    s, out = synsets(), []
    here = s.get((pos, off))
    while here and here["up"] and len(out) < deep:
        here = s.get(here["up"][0])
        if here:
            out.append(here["words"][0])
    return out


def a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def lessons(start: int = 0, count: int = 200) -> list[tuple[str, list[str], str]]:
    out = []
    for w, pos, off in common(start + count)[start:]:
        s = synsets().get((pos, off))
        if not s:
            continue
        out.append(("dictionary", [f"What does {w} mean?", f"What is the meaning of {w}?", f"Define {w}.",
                                   f"Tell me what the word {w} means."], meaning(s["gloss"])))
        out.append(("dictionary", [f"What kind of word is {w}?", f"Is {w} a noun, a verb or a describing word?",
                                   f"What type of word is {w}?", f"Which kind of word is {w}: naming, doing or describing?"], KIND[pos]))
        up = kinds(pos, off)
        if pos == "n" and up:
            chain = f"a kind of {up[0]}" + (f", which is a kind of {up[1]}" if len(up) > 1 else "")
            out.append(("dictionary", [f"What kind of thing is {a(w)}?", f"What group does {a(w)} belong to?",
                                       f"{a(w).capitalize()} is a type of what?", f"Tell me what family of things {a(w)} is in."], chain))
    return out


if __name__ == "__main__":
    for subject, asks, answer in lessons(0, int(sys.argv[1]) if len(sys.argv) > 1 else 20):
        print(f"{asks[0]:<45} -> {answer}")
