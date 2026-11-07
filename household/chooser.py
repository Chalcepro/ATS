"""Coal's chooser (Chalce, 11-06): he picks for himself which of the things he knows answers a question.

    py -3.12 -m household.chooser test       (teach on a copy and examine; TRAINS - run it in its own window)

The loop, all learned:
  1. look   - the things he knows that share words with the question (a wide net, ~24; cheap)
  2. read   - the question and each candidate, word by word, both ways (his word layer under it)
  3. pick   - score every candidate against the question; the best one is read
  4. again? - the picked fact is added to what he is thinking about and he looks once more (a second or
              third step: keys -> drawer -> kitchen), or he picks "enough"
  5. answer - in the last picked fact, he points at the words that answer (start and end)
No rule says what a definition looks like, which source to trust or how places chain: lessons teach it.

Lessons (all made with random words in the roles, so he learns how, not which words):
  - stories told in many wordings: who put what where, when, why; things inside other things (2-3 steps);
    the answer hidden among facts that only mention the same words;
  - "what is X": a sentence that says what X is, among sentences that only mention X;
  - which subject: the same word told in two places, the earlier talk says which one is meant.
The exam uses stories and words it never practised.
"""
from __future__ import annotations

import os
import random

import torch
import torch.nn as nn

from household.understand import words

STEPS_MAX = 3
NET = 24


class Chooser(nn.Module):
    def __init__(self, wl, hidden: int = 96):
        super().__init__()
        self.wl = wl
        dim = wl.rows.shape[1]
        self.enc = nn.GRU(dim + 1, hidden, batch_first=True, bidirectional=True)   # +1: "this word is in the question"
        h2 = 2 * hidden
        self.think = nn.GRUCell(h2, h2)                       # what he is thinking about, step by step
        self.score = nn.Sequential(nn.Linear(4 * h2 + 1, h2), nn.ReLU(), nn.Linear(h2, 1))
        self.enough = nn.Parameter(torch.zeros(h2))           # the "i have enough" candidate
        self.start, self.end = nn.Linear(2 * h2, 1), nn.Linear(2 * h2, 1)

    def _enc(self, sents, qwords):
        n, dim = max(len(s) for s in sents), self.wl.rows.shape[1]
        x = torch.zeros(len(sents), n, dim + 1)
        mask = torch.zeros(len(sents), n, dtype=torch.bool)
        for i, s in enumerate(sents):
            with torch.no_grad():
                x[i, :len(s), :dim] = self.wl.vec(s)
            x[i, :len(s), dim] = torch.tensor([float(w in qwords) for w in s])
            mask[i, :len(s)] = True
        h, _ = self.enc(x)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return h, pooled, mask

    def forward(self, question, facts, picks=None, steps=STEPS_MAX):
        """question: words; facts: list of word lists. picks (teaching): the right fact per step (-1 = enough).
        Returns per-step scores (facts + enough), the picked facts, and start/end scores over the last pick."""
        q = set(question)
        hq, pq, _ = self._enc([question], set())
        state, out, chosen = pq[0], [], []
        mind = set(q)                                         # the words he is thinking about: the question + what he picked
        for t in range(steps):
            hf, pf, mf = self._enc(facts, mind)               # each look lights up what is on his mind now
            overlap = torch.tensor([[len(mind & set(f)) / max(1, len(mind))] for f in facts])
            s = self.score(torch.cat([state.expand(len(facts), -1), pf, state * pf, (state - pf).abs(), overlap], 1))[:, 0]
            s_enough = (state * self.enough).sum()[None] if t else torch.tensor([-1e4])   # never "enough" before a pick
            scores = torch.cat([s, s_enough])
            out.append(scores)
            k = picks[t] if picks is not None else int(scores.argmax()) if scores.argmax() < len(facts) else -1
            if k == -1:
                break
            chosen.append(k)
            mind |= set(facts[k])
            state = self.think(pf[k][None], state[None])[0]
            last_h, last_m = hf, mf
        last = chosen[-1] if chosen else 0
        if not chosen:
            last_h, last_m = hf, mf
        tok = torch.cat([last_h[last], state.expand(last_h.shape[1], -1)], 1)
        st = self.start(tok)[:, 0].masked_fill(~last_m[last], -1e4)
        en = self.end(tok)[:, 0].masked_fill(~last_m[last], -1e4)
        return out, chosen, st, en

    @torch.no_grad()
    def answer(self, question: str, facts: list[str]):
        q = words(question)
        fs = [words(f) or ["-"] for f in facts]
        if not q or not fs:
            return None, []
        _, chosen, st, en = self(q, fs)
        if not chosen:
            return None, []
        s = int(st.argmax())
        e = s + int(en[s:].argmax())
        return " ".join(fs[chosen[-1]][s:e + 1]), [facts[c] for c in chosen]


# ---- lessons ------------------------------------------------------------------------------------
PLACES = ["in the {p}", "on the {p}", "under the {p}", "inside the {p}", "behind the {p}", "next to the {p}"]
TIMES = ["yesterday", "last night", "this morning", "on {d}", "at noon", "today", "after dinner"]
DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
PUT = ["{w} put the {t} {pl}", "{w} left the {t} {pl}", "the {t} is {pl}", "{w} kept the {t} {pl}",
       "{w} dropped the {t} {pl}", "{w} hid the {t} {pl}", "{w} placed the {t} {pl}"]
WHY = ["because it was {a}", "because {w2} asked", "since it was {a}", "because the {t2} was {a}"]
ASK = {"where": ["where is the {t}", "where did {w} put the {t}", "where 's the {t}", "where can i find the {t}",
                 "do you know where the {t} is"],
       "when": ["when did {w} put the {t} there", "when was the {t} left there", "what time did {w} do that"],
       "who": ["who put the {t} there", "who left the {t}", "who had the {t}"],
       "why": ["why did {w} put the {t} there", "why was the {t} left there"]}


def _story(rng, vocab):
    """A small told story, its question, the facts that answer it (in order), and the answer words."""
    pick = lambda n: rng.sample(vocab, n)
    t, p1, p2, p3, w, w2, t2, a = pick(8)
    hops = rng.choice([1, 1, 2, 2, 3])
    chain = [t, p1, p2, p3][:hops + 1]
    when = rng.choice(TIMES).format(d=rng.choice(DAYS))
    why = rng.choice(WHY).format(a=a, w2=w2, t2=t2)
    facts, gold = [], []
    for i in range(hops):                                     # t in p1, p1 in p2, p2 in p3
        pl = rng.choice(PLACES).format(p=chain[i + 1])
        tell = rng.choice(PUT).format(w=w, t=chain[i], pl=pl)
        if i == 0:
            tell += " " + when + " " + why if rng.random() < .7 else ""
        facts.append(tell)
        gold.append(len(facts) - 1)
    kind = rng.choice(list(ASK)) if facts[0].count(" ") > 6 else "where"
    if kind == "where" and hops > 1 and rng.random() < .5:  # "is the keys in the kitchen" -> keys -> drawer -> kitchen
        q = rng.choice(["is the {t} in the {p}", "is the {t} somewhere in the {p}", "did {w} leave the {t} in the {p}"]
                       ).format(t=t, w=w, p=chain[-1])
    else:
        q = rng.choice(ASK[kind]).format(t=t, w=w)
        if kind == "where":                                   # plain where: the nearest place, as a person says it
            chain, gold = chain[:2], gold[:1]
    if kind == "where":
        ans = rng.choice(PLACES).format(p=chain[-1]).split()   # the last place, as told
        last = facts[gold[-1]].split()
        idx = last.index(chain[-1])
        ans = last[idx - 2:idx + 1] if last[idx - 1] == "the" else last[idx - 1:idx + 1]
    else:
        gold = gold[:1]
        ans = {"when": when, "who": w, "why": why}[kind].split()
    for _ in range(rng.randint(3, 10)):                       # facts that only share words with the question
        d1, d2 = pick(2)
        facts.append(rng.choice(PUT).format(w=rng.choice([w, d1]), t=rng.choice([t, d1]) if rng.random() < .5 else d2,
                                            pl=rng.choice(PLACES).format(p=rng.choice([p1, d2]))))
    return q, facts, gold, ans


def _define(rng, vocab):
    x, a, b, c, d = rng.sample(vocab, 5)
    good = rng.choice([f"a {x} is a {a} that {b} the {c}", f"the {x} is the {a} of a {b}", f"{x} is a way of {a} {b}",
                       f"an {x} is a kind of {a} used to {b} {c}", f"a {x} means a {a} with a {b}"])
    noise = [f"{x} can be {a}", f"the {x} should be {b}", f"use {x} with {c}", f"{x} is faster than {d}",
             f"many {x} are {a}", f"this {x} is {b} in the {c}", f"you can {a} the {x}", f"{x} and {d} are {c}"]
    facts = [good] + rng.sample(noise, rng.randint(3, 7))
    q = rng.choice([f"what is a {x}", f"what is {x}", f"what does {x} mean", f"tell me what a {x} is", f"define {x}"])
    ans = good.split()[good.split().index("is" if " is " in good else "means") + 1:]
    return q, facts, [0], ans


COLOURS = "red blue green black white old new big small broken".split()


def _has(rng, vocab):
    """who has / what opens: the one doing it is the answer, among others doing something like it."""
    w, t, k, d1, d2, d3 = rng.sample(vocab, 6)
    if rng.random() < .5:
        a, b = rng.sample(COLOURS, 2)
        good = rng.choice([f"my {w} has a {a} {t}", f"{w} has a {a} {t}", f"{w} owns a {a} {t}", f"my {w} bought a {a} {t}"])
        noise = [f"my {d1} has a {b} {t}", f"{d2} has a {a} {d3}", f"the {a} {t} is nice", f"my {d1} wants a {t}"]
        q = rng.choice([f"who has a {a} {t}", f"who owns the {a} {t}", f"who bought a {a} {t}"])
    else:
        good = rng.choice([f"{w} opens {k} files", f"{w} {t} opens {k} files", f"{w} can open {k} files"])
        noise = [f"{d1} opens {d2} files", f"{w} is an app on this device", f"the {k} folder has {rng.randint(2, 90)} {k} files",
                 f"{d3} is an app on this device", f"{k} files are big"]
        q = rng.choice([f"what opens {k} files", f"what can open {k} files", f"which app opens {k} files"])
    ans = good.split()[good.split().index(w) - (1 if good.startswith("my ") else 0):good.split().index(w) + 1]
    if "opens" in good and f"{w} {t}" in good:
        ans = [w, t]
    return q, [good] + rng.sample(noise, rng.randint(2, len(noise))), [0], ans


REAL = []                                                     # sentences he really read / found (wrong options)


def _real(rng, swap):
    """A real read sentence, sometimes with one of the lesson's words put in, so it 'mentions' it."""
    x = rng.choice(REAL).split()[:30]
    if swap and rng.random() < .6:
        i = rng.randrange(len(x))
        x[i] = rng.choice(swap)
    return " ".join(x)


DEFS = []                                                     # (word, its real definition) from WordNet


def _define_real(rng, vocab):
    """A real definition among sentences that only say something ABOUT the thing (is not.., is very.., uses of it)."""
    x, g = rng.choice(DEFS)
    other = [d for _, d in rng.sample(DEFS, 3)]
    art = ("an " if x[0] in "aeiou" else "a ")
    good = rng.choice([f"{art}{x} is {g}", f"{x} is {g}", f"the {x} is {g}", f"{x} refers to {g}",
                       f"{x} , or {rng.choice(vocab)} , is {g}", f"in science {x} is {g}"])
    ans = g.split()
    noise = [f"{x} is not {rng.choice(other)}", f"{x} is not an accident", f"{x} is very {rng.choice(vocab)}",
             f"{x} is important", f"when {x} is {rng.choice(vocab)} it can {rng.choice(vocab)}",
             f"some people think {x} is caused by {rng.choice(vocab)}", f"{art}{rng.choice(vocab)} is {other[0]}",
             f"{x} happens because of {other[1]}", f"there are many kinds of {x}", f"{x} was named by {rng.choice(vocab)}"]
    if REAL:
        noise += [_real(rng, [x]) for _ in range(3)]
    q = rng.choice([f"what is {art}{x}", f"what is {x}", f"what does {x} mean", f"define {x}", f"tell me what {x} is",
                    f"what is meant by {x}"])
    return q, [good] + rng.sample(noise, rng.randint(3, 8)), [0], ans


def lesson(rng, vocab):
    r = rng.random()
    q, facts, gold, ans = (_story if r < .5 else _define if r < .58 else _has if r < .72 else   # v5 mix (v4 lost stories)
                           _define_real if DEFS else _define)(rng, vocab)
    if REAL:
        qw = [w for w in words(q) if len(w) > 3]
        facts = facts + [_real(rng, qw) for _ in range(rng.randint(2, 12))]
    order = list(range(len(facts)))
    rng.shuffle(order)
    facts = [facts[i] for i in order]
    gold = [order.index(g) for g in gold]
    last = words(facts[gold[-1]])
    aw = [x for w in ans for x in words(w)]
    s = next((i for i in range(len(last)) if last[i:i + len(aw)] == aw), None)
    if s is None:
        return None
    return words(q), [words(f) for f in facts], gold, (s, s + len(aw) - 1)


def teach(ch, vocab, steps=6000, log=print, seed=0):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for n, p in ch.named_parameters() if not n.startswith("wl.")], 1e-3)
    ce = nn.functional.cross_entropy
    total = 0.0
    for step in range(steps):
        loss = 0
        for _ in range(8):
            L = None
            while L is None:
                L = lesson(rng, vocab)
            q, facts, gold, (s, e) = L
            picks = gold + [-1]
            out, _, st, en = ch(q, facts, picks=picks, steps=len(picks))
            for t, sc in enumerate(out):
                loss = loss + ce(sc[None], torch.tensor([picks[t] if picks[t] >= 0 else len(facts)]))
            loss = loss + ce(st[None], torch.tensor([s])) + ce(en[None], torch.tensor([e]))
        opt.zero_grad(); (loss / 8).backward(); opt.step()
        total += loss.item() / 8
        if step % 250 == 249:
            log(f"  step {step + 1}: loss {total / 250:.3f}", flush=True)
            total = 0.0


@torch.no_grad()
def exam(ch, vocab, n=400, seed=99):
    rng = random.Random(seed)
    right_pick = right_ans = 0
    by = {}
    for _ in range(n):
        L = None
        while L is None:
            L = lesson(rng, vocab)
        q, facts, gold, (s, e) = L
        _, chosen, st, en = ch(q, facts)
        ok_pick = chosen[:len(gold)] == gold
        ps = int(st.argmax()); pe = ps + int(en[ps:].argmax())
        ok_ans = ok_pick and (ps, pe) == (s, e)
        kind = f"{len(gold)} step" if len(gold) > 1 else "1 step"
        a, b = by.get(kind, (0, 0))
        by[kind] = (a + ok_ans, b + 1)
        right_pick += ok_pick
        right_ans += ok_ans
    return {"picked the right facts": right_pick / n, "answered exactly": right_ans / n,
            **{f"  {k}": a / b for k, (a, b) in sorted(by.items())}}


def test():
    from household.model import load
    from household.train import SAVE
    from household.wordlayer import WordLayer
    brain, _ = load(SAVE)                                     # read only
    wl = WordLayer.from_keep(brain.keep["mind"]["wordlayer"])
    for p in wl.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    every = [w for w in wl.words if w.isalpha() and 3 <= len(w) <= 9]
    rng = random.Random(1)
    rng.shuffle(every)
    practise, held = every[:15000], every[15000:20000]       # the exam uses words it never practised with
    import json
    know = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reading", "knowledge.jsonl")
    if os.path.exists(know):
        REAL.extend(t for t in (json.loads(x)["text"] for x in open(know, encoding="utf-8")) if 4 <= len(t.split()) <= 30)
    try:
        from household import device
        REAL.extend(device.meanings())
    except Exception:
        pass
    print(f"{len(REAL)} real sentences as wrong options", flush=True)
    try:
        from household.dictionary import meaning, synsets
        for (pos, _), v in synsets().items():
            w = v["words"][0].lower()
            g = meaning(v["gloss"]).lower().replace("(", "").replace(")", "")
            if pos == "n" and w.isalpha() and 3 <= len(g.split()) <= 25:
                DEFS.append((w, g))
    except Exception:
        pass
    print(f"{len(DEFS)} real definitions", flush=True)
    ch = Chooser(wl)
    print(f"teaching the chooser (copy; Coal's brain is not touched)", flush=True)
    teach(ch, practise, steps=int(os.environ.get("STEPS", "6000")))
    for name, v in (("practised words", practise), ("words it never saw", held)):
        print(f"  exam, {name}: " + " | ".join(f"{k} {x:.0%}" for k, x in exam(ch, v).items()), flush=True)
    for q, facts in [("where are my keys", ["i put my keys in the drawer", "the drawer is in the kitchen", "my keys are old",
                                            "i like the kitchen"]),
                     ("what is a closure", ["closures over modules can be more interesting",
                                            "a closure is the combination of a function and the lexical environment",
                                            "closures are useful"]),
                     ("why did dad leave the bag", ["dad left the bag in the car because it was heavy", "dad has a bag",
                                                     "the car is red"])]:
        a, used = ch.answer(q, facts)
        print(f"  {q} -> {a}   (thought: {' -> '.join(used)})", flush=True)
    torch.save({k: v for k, v in ch.state_dict().items() if not k.startswith("wl.")},
               os.path.join(os.path.dirname(SAVE), os.environ.get("OUT", "chooser_test.pt")))


if __name__ == "__main__":
    import sys
    {"test": test}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
