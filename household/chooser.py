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
        hf, pf, mf = self._enc(facts, q)
        state, out, chosen = pq[0], [], []
        for t in range(steps):
            overlap = torch.tensor([[len(q & set(f)) / max(1, len(q))] for f in facts])
            s = self.score(torch.cat([state.expand(len(facts), -1), pf, state * pf, (state - pf).abs(), overlap], 1))[:, 0]
            s_enough = (state * self.enough).sum()[None] if t else torch.tensor([-1e4])   # never "enough" before a pick
            scores = torch.cat([s, s_enough])
            out.append(scores)
            k = picks[t] if picks is not None else int(scores.argmax()) if scores.argmax() < len(facts) else -1
            if k == -1:
                break
            chosen.append(k)
            state = self.think(pf[k][None], state[None])[0]
        last = chosen[-1] if chosen else 0
        tok = torch.cat([hf[last], state.expand(hf.shape[1], -1)], 1)
        st = self.start(tok)[:, 0].masked_fill(~mf[last], -1e4)
        en = self.end(tok)[:, 0].masked_fill(~mf[last], -1e4)
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
    q = rng.choice(ASK[kind]).format(t=t, w=w)
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


def lesson(rng, vocab):
    q, facts, gold, ans = (_story if rng.random() < .65 else _define)(rng, vocab)
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
               os.path.join(os.path.dirname(SAVE), "chooser_test.pt"))


if __name__ == "__main__":
    import sys
    {"test": test}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
