"""Layer 2 of understanding (Chalce, 11-06): how the words of a sentence hang together.

    py -3.12 -m household.relations test     (teach on a copy and examine; TRAINS - needs the charger)

For every word the reader says three things, all learned from 14,000 marked sentences of web English (UD EWT):
  its type      noun, verb, adjective, ...                        (layer 1, learned again here, jointly)
  its head      the word it belongs to ("blue" -> "ball", "ball" -> "have", "on" -> "table", "table" -> "is")
  its link      how it belongs: who does it (nsubj), to what (obj), whose (poss), what it is like (amod),
                a place / time / way hung on with a preposition (obl + its "case" word), ...
From that tree the frame of a sentence is read off: who / does / what / like / whose / and the preposition
phrases ("on the table", "at night", "because it rained"). Which of those is a place or a time is not a rule:
the word layer already knows WordNet's groups (location, time, ...), and the notebook learns it from use.
"""
from __future__ import annotations

import json
import os
import random

import torch
import torch.nn as nn

from household.grammar import TAGS, LEXICON, PRIOR, load_ud
from household.understand import words

RELS = ["root", "nsubj", "obj", "iobj", "obl", "nmod", "poss", "amod", "advmod", "det", "case", "aux", "cop", "mark",
        "cc", "conj", "compound", "xcomp", "ccomp", "advcl", "acl", "nummod", "discourse", "vocative", "expl",
        "flat", "appos", "parataxis", "csubj", "fixed", "dep", "piece", "other"]


class Parser(nn.Module):
    def __init__(self, wl, lexicon: dict, hidden: int = 128):
        super().__init__()
        self.wl, self.lexicon = wl, lexicon
        dim = wl.rows.shape[1]
        self.gru = nn.GRU(dim + PRIOR, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        h2 = 2 * hidden
        self.tag = nn.Linear(h2, len(TAGS))
        self.root = nn.Parameter(torch.zeros(h2))                     # the "head" of the main word
        self.dep, self.head = nn.Linear(h2, 128), nn.Linear(h2, 128)  # word i looks for its head j
        self.rel = nn.Sequential(nn.Linear(2 * h2, 128), nn.ReLU(), nn.Linear(128, len(RELS)))

    def prior(self, w):
        c = self.lexicon.get(w)
        if not c:
            return torch.tensor([0.0] * len(TAGS) + [1.0])
        t = torch.tensor(c, dtype=torch.float)
        return torch.cat([t / t.sum(), torch.zeros(1)])

    def forward(self, sents, gold_heads=None):
        B, n, dim = len(sents), max(len(s) for s in sents), self.wl.rows.shape[1]
        x = torch.zeros(B, n, dim + PRIOR)
        mask = torch.zeros(B, n, dtype=torch.bool)
        for i, s in enumerate(sents):
            with torch.no_grad():
                x[i, :len(s), :dim] = self.wl.vec(s)
            x[i, :len(s), dim:] = torch.stack([self.prior(w) for w in s])
            mask[i, :len(s)] = True
        h, _ = self.gru(x)
        cand = torch.cat([self.root.expand(B, 1, -1), h], 1)          # position 0 = root, word j = j + 1
        arc = torch.einsum("bid,bjd->bij", self.dep(h), self.head(cand))
        arc = arc.masked_fill(~torch.cat([torch.ones(B, 1, dtype=torch.bool), mask], 1)[:, None, :], -1e9)
        heads = gold_heads if gold_heads is not None else arc.argmax(-1)  # rel reads the head it is given / picks
        hh = cand[torch.arange(B)[:, None], heads.clamp(min=0)]
        return self.tag(h), arc, self.rel(torch.cat([h, hh], -1))

    @torch.no_grad()
    def parse_many(self, texts: list[str]) -> list[list[dict]]:
        """Many sentences in one go (much faster than one by one)."""
        ss = [words(t) for t in texts]
        keep = [i for i, s in enumerate(ss) if s]
        out = [[] for _ in texts]
        if not keep:
            return out
        t, a, r = self([ss[i] for i in keep])
        heads, tags, rels = a.argmax(-1).tolist(), t.argmax(-1).tolist(), r.argmax(-1).tolist()
        for b, i in enumerate(keep):
            out[i] = [{"i": j, "word": w, "type": TAGS[tags[b][j]], "head": heads[b][j] - 1, "link": RELS[rels[b][j]]}
                      for j, w in enumerate(ss[i])]
        return out

    @torch.no_grad()
    def parse(self, text: str) -> list[dict]:
        s = words(text)
        if not s:
            return []
        t, a, r = self([s])
        heads = a[0].argmax(-1).tolist()
        return [{"i": i, "word": w, "type": TAGS[t[0, i].argmax()], "head": heads[i] - 1, "link": RELS[r[0, i].argmax()]}
                for i, w in enumerate(s)]


def frame(parsed: list[dict]) -> dict:
    """Read the sentence's frame off the tree: who / does / what / like / whose / with-what (prepositions)."""
    if not parsed:
        return {}
    kids = lambda j, *links: [p for p in parsed if p["head"] == j and (not links or p["link"] in links)]
    roots = [p for p in parsed if p["head"] == -1] or [parsed[0]]
    size = lambda r: sum(1 for p in parsed if p["head"] == r["i"])
    root = max(roots, key=size)                                        # "yo coal | my charger is ...": the real one
    def phrase(p):                                                     # a word with its describers, in order
        near = [q for q in parsed if q["head"] == p["i"] and q["link"] in ("amod", "compound", "nummod", "det", "poss")]
        return " ".join(q["word"] for q in sorted(near + [p], key=lambda q: q["i"]))
    def subtree(j):
        out, todo, seen = [], [j], set()
        while todo:
            k = todo.pop()
            if k in seen:                                           # a guessed tree can loop: never go round twice
                continue
            seen.add(k)
            out.append(parsed[k])
            todo += [q["i"] for q in parsed if q["head"] == k]
        return sorted(out, key=lambda q: q["i"])
    f = {"does": root["word"]}
    for k, links in (("who", ("nsubj", "csubj", "expl")), ("what", ("obj", "xcomp", "ccomp")), ("to", ("iobj",))):
        got = kids(root["i"], *links)
        if got:
            f[k] = [phrase(p) for p in got]
    if root["type"] in ("noun", "adjective", "pronoun") and kids(root["i"], "cop"):
        case = kids(root["i"], "case")
        f["is"] = " ".join([case[0]["word"]] if case else []) + (" " if case else "") + phrase(root)                                         # "my phone is red": the "does" is "is"
        f["does"] = kids(root["i"], "cop")[0]["word"]
    extras = []
    for p in parsed:
        if p["link"] in ("obl", "nmod", "advcl", "acl") and p["head"] != -1:
            case = kids(p["i"], "case", "mark")
            if p["link"] in ("advcl", "acl"):                               # a whole clause: "because it rained"
                it = " ".join(q["word"] for q in subtree(p["i"]) if not case or q is not case[0])
            else:
                it = phrase(p)
            extras.append({"by": case[0]["word"] if case else "", "it": it, "on": parsed[p["head"]]["word"]})
        if p["link"] == "advmod" and p["head"] == root["i"]:
            extras.append({"by": "", "it": p["word"], "on": root["word"]})
    if extras:
        f["with"] = extras
    q = [p["word"] for p in parsed if p["word"] in ("what", "where", "when", "who", "why", "how", "which", "whose")]
    if q:
        f["asks"] = q[0]
    return f


def _batch(rows):
    m = max(len(r[0]) for r in rows)
    T, H, R = (torch.full((len(rows), m), -100) for _ in range(3))
    for i, (w, t, h, r) in enumerate(rows):
        T[i, :len(t)] = torch.tensor(t)
        H[i, :len(h)] = torch.tensor([x + 1 if x >= 0 else (0 if x == -1 else -100) for x in h])
        R[i, :len(r)] = torch.tensor([RELS.index(x) if x in RELS else RELS.index("other") for x in r])
    return [r[0] for r in rows], T, H, R


def teach(parser, rows, steps=4000, log=print, seed=0, drop=0.1):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for n, p in parser.named_parameters() if not n.startswith("wl.")], 2e-3)
    ce = nn.functional.cross_entropy
    lex = parser.lexicon
    parser.train()
    for step in range(steps):
        sents, T, H, R = _batch(rng.sample(rows, 32))
        hide = {w for s in sents for w in s if rng.random() < drop}   # practise words it never met
        parser.lexicon = {k: v for k, v in lex.items() if k not in hide} if hide else lex
        t, a, r = parser(sents, gold_heads=H.clamp(min=0))
        parser.lexicon = lex
        loss = ce(t.flatten(0, 1), T.flatten()) + ce(a.flatten(0, 1), H.flatten()) + ce(r.flatten(0, 1), R.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 499:
            log(f"  step {step + 1}: loss {loss.item():.3f}")
    parser.eval()


@torch.no_grad()
def exam(parser, rows) -> dict:
    tt = hh = ll = n = 0
    for i in range(0, len(rows), 64):
        sents, T, H, R = _batch(rows[i:i + 64])
        t, a, r = parser(sents)
        m = H != -100
        hd = a.argmax(-1) == H
        tt += ((t.argmax(-1) == T) & m).sum().item()
        hh += (hd & m).sum().item()
        ll += (hd & (r.argmax(-1) == R) & m).sum().item()
        n += m.sum().item()
    return {"type right": tt / n, "head right": hh / n, "head + link right": ll / n}


TRY = ["I have a blue ball", "my phone is on the table", "where is my phone", "dad has a black car",
       "we ran to the shop because it rained", "i went for a long run yesterday", "can you fix this file for me",
       "my laptop is kinda slow today", "Kemi put her umbrella under the chair at night",
       "who has the remote", "btw can you edit this", "i tried something yesterday can you remember"]


def load_test():
    """The parser saved by test(), on Coal's word layer (read only)."""
    from household.model import load as load_brain
    from household.train import SAVE
    from household.wordlayer import WordLayer
    brain, _ = load_brain(SAVE)
    wl = WordLayer.from_keep(brain.keep["mind"]["wordlayer"])
    sd = torch.load(os.path.join(os.path.dirname(SAVE), os.environ.get("OUT", "relations_test.pt")))
    parser = Parser(wl, json.load(open(LEXICON, encoding="utf-8")), hidden=sd["tag.weight"].shape[1] // 2)
    parser.load_state_dict(sd, strict=False)
    return parser.eval()


def show(parser, lines=TRY):
    for line in lines:
        p = parser.parse(line)
        print(f"  {line}  ->  {frame(p)}")


def test():
    from household.model import load as load_brain
    from household.train import SAVE
    from household.wordlayer import WordLayer
    brain, _ = load_brain(SAVE)                                    # read only: nothing is saved into Coal
    wl = WordLayer.from_keep(brain.keep["mind"]["wordlayer"])
    for p in wl.parameters():
        p.requires_grad_(False)
    parser = Parser(wl, json.load(open(LEXICON, encoding="utf-8")), hidden=int(os.environ.get("HIDDEN", "128")))
    rows = load_ud("train") + load_ud("dev")
    print(f"teaching layers 1+2 on {len(rows)} sentences (copy; Coal's brain is not touched)")
    teach(parser, rows, steps=int(os.environ.get("STEPS", "4000")))
    print("  exam (sentences it never saw): " + " | ".join(f"{k} {v:.0%}" for k, v in exam(parser, load_ud("test")).items()))
    for line in TRY:
        p = parser.parse(line)
        print(f"\n  {line}\n    " + "  ".join(f"{x['word']}/{x['type'][:4]}>{p[x['head']]['word'] if x['head'] >= 0 else 'ROOT'}:{x['link']}" for x in p))
        print(f"    frame: {frame(p)}")
    torch.save({k: v for k, v in parser.state_dict().items() if not k.startswith("wl.")},
               os.path.join(os.path.dirname(SAVE), os.environ.get("OUT", "relations_test.pt")))


if __name__ == "__main__":
    import sys
    {"test": test, "exam": lambda: (lambda p: (print(exam(p, load_ud("test"))), show(p)))(load_test())}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
