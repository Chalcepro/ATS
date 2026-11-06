"""Layer 1 of understanding (Chalce, 11-06): what each word IS in this sentence - noun, verb, adjective, ...

    py -3.12 -m household.grammar export     (once: turn NLTK's tagged Brown corpus into data/grammar/, no NLTK after)
    py -3.12 -m household.grammar test       (teach a tagger on a copy and examine it; TRAINS - needs the charger)

Not "name" or "place" or "thing": those were slots for one kind of sentence. Every English sentence is built from
the same word types, so the reader learns those, and layer 2 works out who / does what / to what / where / when
from them ("on the table" = a preposition + a noun = where).

What the tagger reads for each word:
  - Coal's word layer vector (it already holds WordNet: which words CAN be nouns, verbs, ...; spelling pieces
    like "-ly", "-ing" for words it never met);
  - how often this word was each type in a million words of real English (Brown), if it was seen there;
and the words around it, both ways (BiGRU), because "run" is a verb in "i run" and a noun in "a long run".
The exam holds back whole sentences, and scores words the tagger never met separately.
"""
from __future__ import annotations

import json
import os
import random

import torch
import torch.nn as nn

from household.understand import words

TAGS = ["noun", "verb", "adjective", "adverb", "pronoun", "determiner", "preposition", "number", "conjunction",
        "particle", "other", "interjection"]
UPOS = {"NOUN": "noun", "PROPN": "noun", "VERB": "verb", "AUX": "verb", "ADJ": "adjective", "ADV": "adverb",
        "PRON": "pronoun", "DET": "determiner", "ADP": "preposition", "NUM": "number", "CCONJ": "conjunction",
        "SCONJ": "conjunction", "PART": "particle", "INTJ": "interjection", "SYM": "other", "X": "other"}
UNIVERSAL = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb", "PRON": "pronoun", "DET": "determiner",
             "ADP": "preposition", "NUM": "number", "CONJ": "conjunction", "PRT": "particle", "X": "other"}
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "grammar")
SENTS = os.path.join(HERE, "brown.txt")                      # one sentence a line: word/tag word/tag ...
UD = os.path.join(os.path.dirname(HERE), "ud")               # Universal Dependencies English-EWT (web, email, forum talk)
UD_SENTS = os.path.join(HERE, "ud_{}.txt")                   # word/tag/head/relation ... (head -1 = the root)
LEXICON = os.path.join(HERE, "lexicon.json")                 # word -> [count per tag]
HELD_EVERY = 10                                              # every 10th sentence is exam only
PRIOR = len(TAGS) + 1                                        # tag shares + "never seen in Brown"


# ---- export once from NLTK ----------------------------------------------------------------------
def export():
    from nltk.corpus import brown
    os.makedirs(HERE, exist_ok=True)
    counts, n = {}, 0
    with open(SENTS, "w", encoding="utf-8") as f:
        for i, sent in enumerate(brown.tagged_sents(tagset="universal")):
            toks = []
            for w, t in sent:
                if t == ".":
                    continue
                tag = TAGS.index(UNIVERSAL.get(t, "other"))
                for piece in words(w):                     # Coal's own way of cutting words ("didn't" -> did, 't)
                    toks.append(f"{piece}/{tag}")
                    if i % HELD_EVERY:                      # the lexicon only counts taught sentences
                        counts.setdefault(piece, [0] * len(TAGS))[tag] += 1
            if toks:
                f.write(" ".join(toks) + "\n")
                n += 1
    json.dump(counts, open(LEXICON.replace("lexicon", "lexicon_brown"), "w", encoding="utf-8"), separators=(",", ":"))
    print(f"{n} sentences -> {SENTS}, {len(counts)} words (Brown)")
    export_ud()


def _ud_file(name):
    """One UD .conllu file -> sentences of (piece, tag, head piece, relation); punctuation dropped."""
    out, rows = [], []
    for line in open(os.path.join(UD, f"en_ewt-ud-{name}.conllu"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if rows:
                out.append(rows)
            rows = []
        elif not line.startswith("#"):
            f = line.split("\t")
            if f[0].isdigit():
                rows.append(f)
    sents = []
    for rows in out:
        first, toks = {}, []                                   # token id -> index of its first piece
        for f in rows:
            if f[3] == "PUNCT":
                continue
            for k, piece in enumerate(words(f[1])):
                if not k:
                    first[f[0]] = len(toks)
                toks.append([piece, TAGS.index(UPOS.get(f[3], "other")), f[6], f[7].split(":")[0] if not k else "piece"])
        for t in toks:
            t[2] = first.get(t[2], -1) if t[3] != "piece" else -2  # -1 root (or punctuation head), -2 = part of a word
        if toks:
            sents.append(toks)
    return sents


def export_ud():
    counts = {}
    for name in ("train", "dev", "test"):
        sents = _ud_file(name)
        with open(UD_SENTS.format(name), "w", encoding="utf-8") as f:
            for toks in sents:
                f.write(" ".join(f"{w}/{t}/{h}/{r}" for w, t, h, r in toks) + "\n")
                if name == "train":
                    for w, t, _, _ in toks:
                        counts.setdefault(w, [0] * len(TAGS))[t] += 1
        print(f"UD {name}: {len(sents)} sentences")
    brown = json.load(open(LEXICON.replace("lexicon", "lexicon_brown"), encoding="utf-8")) if os.path.exists(
        LEXICON.replace("lexicon", "lexicon_brown")) else {}
    for w, c in brown.items():                                   # Brown only for words UD never had (its own
        if w not in counts:                                      # habits differ: "my" is a determiner there)
            counts[w] = c + [0] * (len(TAGS) - len(c))
    json.dump(counts, open(LEXICON, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"{len(counts)} words -> {LEXICON}")


def load_ud(name):
    out = []
    for line in open(UD_SENTS.format(name), encoding="utf-8"):
        q = [p.split("/") for p in line.split()]
        out.append(([a[0] for a in q], [int(a[1]) for a in q], [int(a[2]) for a in q], [a[3] for a in q]))
    return out


def load():
    sents = []
    for line in open(SENTS, encoding="utf-8"):
        pairs = [p.rsplit("/", 1) for p in line.split()]
        sents.append(([w for w, _ in pairs], [int(t) for _, t in pairs]))
    return [s for i, s in enumerate(sents) if i % HELD_EVERY], [s for i, s in enumerate(sents) if not i % HELD_EVERY]


# ---- the tagger ---------------------------------------------------------------------------------
class Tagger(nn.Module):
    def __init__(self, wl, lexicon: dict, hidden: int = 64):
        super().__init__()
        self.wl, self.lexicon = wl, lexicon
        dim = wl.rows.shape[1]
        self.gru = nn.GRU(dim + PRIOR, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, len(TAGS))

    def prior(self, w: str) -> torch.Tensor:
        c = self.lexicon.get(w)
        if not c:
            return torch.tensor([0.0] * len(TAGS) + [1.0])
        t = torch.tensor(c, dtype=torch.float)
        return torch.cat([t / t.sum(), torch.zeros(1)])

    def forward(self, sents: list[list[str]]):
        n, dim = max(len(s) for s in sents), self.wl.rows.shape[1]
        x = torch.zeros(len(sents), n, dim + PRIOR)
        for i, s in enumerate(sents):
            with torch.no_grad():
                x[i, :len(s), :dim] = self.wl.vec(s)
            x[i, :len(s), dim:] = torch.stack([self.prior(w) for w in s])
        return self.out(self.gru(x)[0])

    @torch.no_grad()
    def tag(self, text: str) -> list[tuple[str, str]]:
        s = words(text)
        return list(zip(s, [TAGS[i] for i in self(s and [s])[0].argmax(-1).tolist()])) if s else []


def _batch(sents):
    m = max(len(w) for w, _ in sents)
    y = torch.full((len(sents), m), -100)
    for i, (_, t) in enumerate(sents):
        y[i, :len(t)] = torch.tensor(t)
    return [w for w, _ in sents], y


def teach(tagger, taught, steps=3000, log=print, seed=0, drop=0.1):
    """drop: a share of words is taught as if never seen in Brown, so unseen words are practised too."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for n, p in tagger.named_parameters() if not n.startswith("wl.")], 2e-3)
    lex = tagger.lexicon
    for step in range(steps):
        sents, y = _batch(rng.sample(taught, 32))
        hide = {w for s in sents for w in s if rng.random() < drop}
        tagger.lexicon = {k: v for k, v in lex.items() if k not in hide} if hide else lex
        loss = nn.functional.cross_entropy(tagger(sents).flatten(0, 1), y.flatten())
        tagger.lexicon = lex
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 499:
            log(f"  step {step + 1}: loss {loss.item():.3f}")


@torch.no_grad()
def exam(tagger, held, n=2000) -> dict:
    right = total = new_right = new_total = 0
    for i in range(0, min(n, len(held)), 64):
        sents, y = _batch(held[i:i + 64])
        p = tagger(sents).argmax(-1)
        for j, s in enumerate(sents):
            for k, w in enumerate(s):
                ok = int(p[j, k] == y[j, k])
                right, total = right + ok, total + 1
                if w not in tagger.lexicon:
                    new_right, new_total = new_right + ok, new_total + 1
    return {"words right": right / total, "never-met words right": new_right / max(new_total, 1),
            "never-met words": new_total}


def test():
    from household.model import load as load_brain
    from household.train import SAVE
    from household.wordlayer import WordLayer
    brain, _ = load_brain(SAVE)                                  # read only: nothing is saved into Coal
    wl = WordLayer.from_keep(brain.keep["mind"]["wordlayer"])
    for p in wl.parameters():
        p.requires_grad_(False)
    taught = [(w, t) for w, t, _, _ in load_ud("train") + load_ud("dev")]   # chat-like web English, UD habits
    held = [(w, t) for w, t, _, _ in load_ud("test")]
    tagger = Tagger(wl, json.load(open(LEXICON, encoding="utf-8")))
    print(f"teaching the tagger on {len(taught)} sentences (copy; Coal's brain is not touched)")
    teach(tagger, taught, steps=int(os.environ.get("STEPS", "3000")))
    print("  exam (sentences it never saw): " + " | ".join(
        f"{k} {v:.0%}" if isinstance(v, float) else f"{k} {v}" for k, v in exam(tagger, held).items()))
    from household.talk_lines import MINE
    for line in ["I have a blue ball", "my phone is on the table", "where is my phone", "STOOOOP", "yes",
                 "btw can you fix this file", "we ran to the shop because it rained", "i went for a long run",
                 *[v[0] for v in MINE.values()]]:
        print("  " + "  ".join(f"{w}/{t}" for w, t in tagger.tag(line)))
    torch.save({k: v for k, v in tagger.state_dict().items() if not k.startswith("wl.")},
               os.path.join(os.path.dirname(SAVE), "grammar_test.pt"))


if __name__ == "__main__":
    import sys
    {"export": export, "export_ud": export_ud, "test": test}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
