"""Coal's word layer: every word gets its own small vector, and sections read what kind of word it is.

    py -3.12 -m household.wordlayer test        (train a fresh layer on a copy, test it on held-back words)

A word's vector = its own row (grows as new words arrive) + the pieces of its spelling (so a word never
seen still lands somewhere sensible: "-ly", "-tion", "un-").
Sections are small heads, one per kind of grouping, each tag a yes/no, so one word can sit in several
("plan" is a noun and a verb):
  kind   - noun / verb / adjective / adverb          (WordNet)
  group  - animal, food, artifact, feeling, ...      (WordNet's 45 groups)
  topic  - physics, chemistry, music, law, ...       (WordNet's subject domains; grows as new ones turn up)
  number - a number word or digits
New sections and new tags are added on the fly (grow()); what was learned stays where it was.
Words are also linked: a word's vector learns to sit near the words of its own definition.
"""
from __future__ import annotations

import os
import random
import re
import sys
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from household.dictionary import DICT, FILES, synsets

DIM = 48
BUCKETS = 20000                                     # spelling pieces share these rows
NUMBERS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
           "billion first second third half dozen").split()
# WordNet's lexicographer files, in their fixed order (lex_filenum 00-44)
LEXNAMES = ("adj.all adj.pert adv.all noun.Tops noun.act noun.animal noun.artifact noun.attribute noun.body noun.cognition "
            "noun.communication noun.event noun.feeling noun.food noun.group noun.location noun.motive noun.object noun.person "
            "noun.phenomenon noun.plant noun.possession noun.process noun.quantity noun.relation noun.shape noun.state "
            "noun.substance noun.time verb.body verb.change verb.cognition verb.communication verb.competition verb.consumption "
            "verb.contact verb.creation verb.emotion verb.motion verb.perception verb.possession verb.social verb.stative "
            "verb.weather adj.ppl").split()
KINDS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def pieces(word: str) -> list[int]:
    w = f"<{word}>"
    grams = [w[i:i + n] for n in (3, 4) for i in range(len(w) - n + 1)]
    return [zlib.crc32(g.encode()) % BUCKETS for g in grams]


class WordLayer(nn.Module):
    def __init__(self, words: list[str] | None = None, sections: dict[str, list[str]] | None = None):
        super().__init__()
        self.words = list(words or [])
        self.index = {w: i for i, w in enumerate(self.words)}
        self.rows = nn.Parameter(torch.randn(max(1, len(self.words)), DIM) * 0.1)
        self.spell = nn.Parameter(torch.randn(BUCKETS, DIM) * 0.1)
        self.ctx = nn.Parameter(torch.randn(max(1, len(self.words)), DIM) * 0.1)   # a word as it appears in definitions
        self.sections = {k: list(v) for k, v in (sections or {}).items()}
        self.heads = nn.ModuleDict({k: nn.Linear(DIM, len(v)) for k, v in self.sections.items()})

    # ---- growing ------------------------------------------------------------------------
    def add(self, word: str) -> int:
        if word not in self.index:
            self.index[word] = len(self.words)
            self.words.append(word)
            with torch.no_grad():
                grow = lambda p: nn.Parameter(torch.cat([p.data, torch.randn(1, DIM) * 0.1]))
                if len(self.words) > self.rows.shape[0]:
                    self.rows, self.ctx = grow(self.rows), grow(self.ctx)
        return self.index[word]

    def grow(self, section: str, tag: str) -> int:
        """A new section, or a new tag in one: a new head row, the old rows untouched."""
        tags = self.sections.setdefault(section, [])
        if tag not in tags:
            tags.append(tag)
            old = self.heads[section] if section in self.heads else None
            new = nn.Linear(DIM, len(tags))
            if old is not None:
                with torch.no_grad():
                    new.weight[:-1], new.bias[:-1] = old.weight, old.bias
            self.heads[section] = new
        return tags.index(tag)

    # ---- reading ------------------------------------------------------------------------
    def vec(self, words: list[str]) -> torch.Tensor:
        own = torch.stack([self.rows[self.index[w]] if w in self.index else torch.zeros(DIM) for w in words])
        spell = torch.stack([self.spell[pieces(w)].mean(0) for w in words])
        return own + spell

    @torch.no_grad()
    def tags(self, word: str, sure: float = 0.5) -> dict[str, list[tuple[str, float]]]:
        v = self.vec([word])
        out = {}
        for k, head in self.heads.items():
            p = torch.sigmoid(head(v))[0]
            out[k] = sorted([(self.sections[k][i], float(p[i])) for i in range(len(p)) if p[i] >= sure], key=lambda t: -t[1])
        return out

    @torch.no_grad()
    def near(self, word: str, n: int = 8) -> list[str]:
        v = Fn.normalize(self.vec([word]), dim=1)
        allv = Fn.normalize(self.vec(self.words), dim=1)
        best = (allv @ v.T)[:, 0].topk(n + 1).indices
        return [self.words[i] for i in best if self.words[i] != word][:n]

    # ---- keeping ------------------------------------------------------------------------
    def to_keep(self) -> dict:
        return {"words": self.words, "sections": self.sections, "state": {k: v.detach().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_keep(cls, keep: dict) -> "WordLayer":
        wl = cls(keep["words"], keep["sections"])
        wl.load_state_dict(keep["state"])
        return wl


# ---- what WordNet says about each word ------------------------------------------------------
def facts(limit: int | None = None) -> dict[str, dict]:
    """word -> {"kind": {...}, "group": {...}, "topic": {...}, "number": {...}, "gloss": [definition words]}"""
    s, out = synsets(), {}
    lex = {}
    for pos, name in FILES.items():                 # lex_filenum and topic per synset
        for line in open(os.path.join(DICT, "data." + name), encoding="utf-8", errors="ignore"):
            if line.startswith(" "):
                continue
            f = line.split()
            topics = [(f[i + 2], f[i + 1]) for i in range(len(f) - 2) if f[i] == ";c"]
            lex[(pos, f[0])] = (int(f[1]), topics, f[2])
    for (pos, off), syn in s.items():
        filenum, topics, ss = lex[(pos, off)]
        for w in syn["words"]:
            w = w.lower()
            if not re.fullmatch(r"[a-z]+", w):
                continue
            d = out.setdefault(w, {"kind": set(), "group": set(), "topic": set(), "number": set(), "gloss": set()})
            d["kind"].add(KINDS[ss])
            d["group"].add(LEXNAMES[filenum].split(".")[1])
            for tpos, toff in topics:
                t = s.get(("a" if tpos == "s" else tpos, toff))
                if t:
                    d["topic"].add(t["words"][0].lower())
            if len(d["gloss"]) < 30:
                d["gloss"].update(x for x in re.findall(r"[a-z]+", syn["gloss"].split(";")[0].lower()) if len(x) > 2)
    for w in NUMBERS:
        out.setdefault(w, {"kind": set(), "group": set(), "topic": set(), "number": set(), "gloss": set()})["number"].add("number")
    for n in range(0, 101):
        out[str(n)] = {"kind": set(), "group": set(), "topic": set(), "number": {"number"}, "gloss": set()}
    if limit:
        out = dict(list(out.items())[:limit])
    return out


def build(fs: dict[str, dict]) -> WordLayer:
    wl = WordLayer()
    common = {}
    for d in fs.values():
        for t in d["topic"]:
            common[t] = common.get(t, 0) + 1
    for w, d in fs.items():
        wl.add(w)
        for sec in ("kind", "group", "number"):
            for t in d[sec]:
                wl.grow(sec, t)
        for t in d["topic"]:
            if common[t] >= 15:                     # a subject with enough words to learn from
                wl.grow("topic", t)
    return wl


def train(wl: WordLayer, fs: dict[str, dict], words: list[str], steps: int = 4000, batch: int = 256, log=print) -> None:
    opt = torch.optim.Adam(wl.parameters(), lr=3e-3)
    for step in range(steps):
        b = random.sample(words, min(batch, len(words)))
        v = wl.vec(b)
        loss = 0
        for sec, head in wl.heads.items():
            y = torch.zeros(len(b), len(wl.sections[sec]))
            have = torch.zeros(len(b))
            for i, w in enumerate(b):
                for t in fs[w][sec]:
                    if t in wl.sections[sec]:
                        y[i, wl.sections[sec].index(t)] = 1
                have[i] = 1 if sec != "topic" or fs[w]["topic"] else 0.2   # most words have no subject: weigh that lightly
            loss = loss + (Fn.binary_cross_entropy_with_logits(head(v), y, reduction="none").mean(1) * have).mean()
        # links: a word sits near the words of its definition, away from random words
        pairs = [(i, wl.index[g]) for i, w in enumerate(b) for g in random.sample(sorted(fs[w]["gloss"]), min(3, len(fs[w]["gloss"]))) if g in wl.index]
        if pairs:
            a = v[[p[0] for p in pairs]]
            pos = wl.ctx[[p[1] for p in pairs]]
            neg = wl.ctx[torch.randint(len(wl.words), (len(pairs), 5))]
            loss = loss + 0.5 * (-Fn.logsigmoid((a * pos).sum(1)).mean() - Fn.logsigmoid(-(neg @ a.unsqueeze(2)).squeeze(2)).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            log(f"  step {step}: loss {float(loss):.3f}")


@torch.no_grad()
def score(wl: WordLayer, fs: dict[str, dict], words: list[str]) -> dict[str, float]:
    """Per section: of the words that have a tag there, how often the surest tag is one of theirs."""
    out = {}
    for sec, head in wl.heads.items():
        have = [w for w in words if fs[w][sec]]
        if not have:
            continue
        p = head(wl.vec(have)).argmax(1)
        out[sec] = sum(wl.sections[sec][int(p[i])] in fs[w][sec] for i, w in enumerate(have)) / len(have)
    return out


def test() -> None:
    random.seed(0)
    torch.manual_seed(0)
    fs = facts()
    words = sorted(fs)
    random.shuffle(words)
    held = words[:len(words) // 10]                 # never trained, not even their own row: spelling only
    seen = words[len(words) // 10:]
    wl = build({w: fs[w] for w in seen})
    print(f"{len(fs)} words, held back {len(held)}; sections: " + ", ".join(f"{k} {len(v)}" for k, v in wl.sections.items()))
    train(wl, fs, seen)
    for name, ws in [("trained words", seen[:3000]), ("never-seen words", held)]:
        print(name + ": " + ", ".join(f"{k} {v:.0%}" for k, v in score(wl, fs, ws).items()))
    base = {sec: max((sum(t in fs[w][sec] for w in held if fs[w][sec]) / max(1, sum(1 for w in held if fs[w][sec])))
                     for t in wl.sections[sec]) for sec in wl.sections}
    print("guessing the most common tag: " + ", ".join(f"{k} {v:.0%}" for k, v in base.items()))
    for w in ["guitar", "plan", "power", "quickly", "happy", "seven", "dog", "electron", "acid", "violin", "run"] + held[:6]:
        t = wl.tags(w)
        print(f"  {w:<12} " + "  ".join(f"{k}: " + ",".join(x for x, _ in v[:3]) for k, v in t.items() if v) + f"   near: {', '.join(wl.near(w, 5))}")
    torch.save(wl.to_keep(), os.path.join(os.path.dirname(DICT), "..", "..", "checkpoints", "wordlayer_test.pt"))


if __name__ == "__main__":
    test() if sys.argv[1:2] == ["test"] else print(__doc__)
