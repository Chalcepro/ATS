"""Coal understands what he is told (Chalce, 11-06): not the words he matches, but who has what and what it is like.

    py -3.12 -m household.understand test       (teach a reader on a copy and examine it on wordings and words it never saw)

The reader is learned. It reads a sentence through Coal's word layer, word by word in both directions, and says
  - for every word, what it is in this sentence: the speaker ("i", "my"), Coal ("you"), someone named, the thing,
    what the thing is like, or nothing;
  - for every "what it is like" word, which kind: a colour, a size, or how it is (new, broken, heavy...);
  - for the whole sentence, what it does: tells something, asks what colour / what size / what it is like,
    asks yes or no, asks what someone has, asks who has it, greets, or something else.
Lessons are made from many wordings; the exam holds back whole wordings and whole sets of words (names, things,
colours), so passing it means the reader generalised, not that it remembered sentences.

What is understood goes into a notebook - facts, not text: {owner: {thing: {kind: word}}} - and answers come
from the notebook, said from Coal's side ("my ball" becomes "your ball"). Anything not understood goes to the
writer as before. When asked something it was never told, Coal says so and asks.
"""
from __future__ import annotations

import random
import re

import torch
import torch.nn as nn

ROLES = ["-", "me", "you", "name", "thing", "quality"]
QUALS = ["colour", "size", "state"]
KINDS = ["tell", "ask_colour", "ask_size", "ask_state", "ask_yesno", "ask_things", "ask_owner", "greet", "other"]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|'[a-z]+", text.lower().replace("’", "'"))


# ---- lessons: many wordings; the exam keeps some wordings and some words back ----------------------------
TRAIN_W = {
    "name": "dan tolu ada sam mum dad grace john femi mary".split(),
    "thing": "ball car phone cup shirt bag bike book chair hat dog cat box pen laptop shoe kite lamp table door".split(),
    "colour": "blue red green yellow black white brown pink".split(),
    "size": "big small tiny huge".split(),
    "state": "new old broken heavy clean dirty soft hard".split(),
}
HELD_W = {
    "name": "kemi ryan lucy".split(),
    "thing": "guitar bottle jacket watch rabbit umbrella".split(),
    "colour": "purple orange grey".split(),
    "size": "large little".split(),
    "state": "wet shiny".split(),
}
# {S} subject (i / you / dan), {P} possessive (my / your / dan 's), {M} mine/yours/dan 's, {H} have/has, {D} do/does
TELL = ["{S} {H} a {Q} {T}", "{S} got a {Q} {T}", "{S} 've got a {Q} {T}", "{P} {T} is {Q}", "{S} bought a {Q} {T}",
        "there is a {Q} {T} and it is {M}", "{S} own a {Q} {T}", "look at {P} {Q} {T}", "{P} {T} is really {Q}",
        "{S} just got a {T} and it is {Q}", "so {S} {H} a {Q} {T}", "the {Q} {T} is {M}", "{P} {T} is very {Q}",
        "{S} {H} got a {Q} {T}", "{P} {T} looks {Q}"]
TELL_HELD = ["{P} {T} happens to be {Q}", "{S} recently picked up a {Q} {T}", "guess what {S} {H} this {Q} {T}"]
ASK = {"ask_colour": ["what colour is {P} {T}", "what color is {P} {T}", "what colour 's {P} {T}", "which colour is {P} {T}",
                      "do you know the colour of {P} {T}", "what is the colour of {P} {T}"],
       "ask_size": ["how big is {P} {T}", "what size is {P} {T}", "is {P} {T} big or small"],
       "ask_state": ["what is {P} {T} like", "how is {P} {T}", "what 's {P} {T} like"],
       "ask_yesno": ["is {P} {T} {Q}", "{P} {T} is {Q} right", "is the {T} {Q}", "{D} {S} have a {Q} {T}"],
       "ask_things": ["what {D} {S} have", "what {H} {S} got", "what things {D} {S} have", "what is {M}"],
       "ask_owner": ["who has a {Q} {T}", "who has the {T}", "who owns a {Q} {T}", "whose {T} is {Q}"]}
ASK_HELD = {"ask_colour": ["tell me the colour of {P} {T}"], "ask_size": ["how large is {P} {T}"],
            "ask_state": ["describe {P} {T}"], "ask_yesno": ["is it true that {P} {T} is {Q}"],
            "ask_things": ["what {D} {S} own"], "ask_owner": ["whose is the {Q} {T}"]}
GREET = ["hi", "hello", "hey", "good morning", "what is up", "what 's up", "how are you", "good evening", "hi there"]
GREET_HELD = ["yo", "howdy", "morning"]
OTHER = ["what is a guitar", "why is the sky blue", "two plus two", "tell me a story", "what does have mean",
         "i am tired", "thank you", "can you help me", "what time is it", "where do fish live", "i like music",
         "open the file", "what is your name", "play a song", "the weather is nice today", "what is a noun"]
OTHER_HELD = ["how do birds fly", "close the window", "who made you"]
FUNCTION = set("i you my your mine yours me he she it is are was the a an and or of to in on at what who whose which how why do does did have has got had this that there here so just really very look own bought colour color size big small coal hey btw okay".split())
LEAD = ["", "", "", "coal ", "so ", "hey coal ", "btw ", "okay so "]


def _owner(rng, w, pool=None):
    k = rng.choice(["me", "me", "you", "name"])
    if k == "me":
        return k, {"S": [("i", "me")], "P": [("my", "me")], "M": [("mine", "me")], "H": "have", "D": "do"}
    if k == "you":
        return k, {"S": [("you", "you")], "P": [("your", "you")], "M": [("yours", "you")], "H": "have", "D": "do"}
    n = rng.choice(pool) if pool and rng.random() < .5 else rng.choice(w["name"])
    return n, {"S": [(n, "name")], "P": [(n, "name"), ("'s", "-")], "M": [(n, "name"), ("'s", "-")], "H": "has", "D": "does"}


def lesson(rng, held: bool, pool=None):
    """One sentence with its truth: (words, roles, qual kinds, sentence kind)."""
    w = HELD_W if held else TRAIN_W
    if held:                                                           # the exam mixes held words into known wordings too
        w = {k: (HELD_W[k] if rng.random() < .7 else TRAIN_W[k]) for k in TRAIN_W}
    r = rng.random()
    if r < .4:
        kind, pat = "tell", rng.choice(TELL_HELD if held and rng.random() < .6 else TELL)
    elif r < .85:
        kind = rng.choice(list(ASK))
        pat = rng.choice(ASK_HELD[kind] if held and rng.random() < .6 else ASK[kind])
    elif r < .93:
        kind, pat = "greet", rng.choice(GREET_HELD if held else GREET)
    else:
        kind, pat = "other", rng.choice(OTHER_HELD if held else OTHER)
    qk = rng.choice(QUALS) if kind not in ("ask_colour", "ask_size") else kind[4:]
    if kind == "ask_colour":
        qk = "colour"
    q = rng.choice(w[qk])
    t = rng.choice(pool) if pool and rng.random() < .5 else rng.choice(w["thing"])   # any word can be the thing:
    _, own = _owner(rng, w, pool)                                                  # it is known by its place
    toks = []
    for piece in (rng.choice(LEAD) + pat).split():
        m = re.fullmatch(r"\{(\w)\}", piece)
        if not m:
            toks += [(x, "-", None) for x in words(piece)]
        elif m[1] == "Q":
            toks.append((q, "quality", qk))
        elif m[1] == "T":
            toks.append((t, "thing", None))
        elif m[1] in "SPM":
            toks += [(a, b, None) for a, b in own[m[1]]]
        else:
            toks.append((own[m[1]], "-", None))
    return [a for a, _, _ in toks], [ROLES.index(b) for _, b, _ in toks], \
        [QUALS.index(c) if c else -100 for _, _, c in toks], KINDS.index(kind)


# ---- the reader ---------------------------------------------------------------------------------
class Reader(nn.Module):
    def __init__(self, wl, hidden: int = 96):
        super().__init__()
        self.wl = wl
        dim = wl.rows.shape[1]
        self.gru = nn.GRU(dim + 1, hidden, batch_first=True, bidirectional=True)
        self.role = nn.Linear(2 * hidden, len(ROLES))
        self.qual = nn.Linear(2 * hidden + dim, len(QUALS))       # the sentence around it + the word itself
        self.kind = nn.Linear(4 * hidden, len(KINDS))

    def forward(self, sents: list[list[str]]):
        n = max(len(s) for s in sents)
        dim = self.wl.rows.shape[1]
        x = torch.zeros(len(sents), n, dim + 1)
        for i, s in enumerate(sents):
            with torch.no_grad():
                x[i, :len(s), :dim] = self.wl.vec(s)
            x[i, :len(s), dim] = 1                                      # a word is here (not padding)
        h, _ = self.gru(x)
        lens = torch.tensor([len(s) for s in sents])
        last = h[torch.arange(len(sents)), lens - 1]
        pooled = torch.cat([last, h.max(1).values], 1)
        return self.role(h), self.qual(torch.cat([h, x[..., :dim]], 2)), self.kind(pooled)

    @torch.no_grad()
    def read(self, text: str) -> dict:
        s = words(text)
        if not s:
            return {"kind": "other", "sure": 1.0, "words": []}
        r, q, k = self(s and [s])
        kp = k[0].softmax(-1)
        return {"words": s, "roles": [ROLES[i] for i in r[0].argmax(-1).tolist()],
                "quals": [QUALS[i] for i in q[0].argmax(-1).tolist()],
                "kind": KINDS[int(kp.argmax())], "sure": float(kp.max())}


def batch(rng, n, held, pool=None):
    L = [lesson(rng, held, pool) for _ in range(n)]
    m = max(len(a[0]) for a in L)
    roles = torch.full((n, m), -100)
    quals = torch.full((n, m), -100)
    for i, (s, r, q, _) in enumerate(L):
        roles[i, :len(s)] = torch.tensor(r)
        quals[i, :len(s)] = torch.tensor(q)
    return [a[0] for a in L], roles, quals, torch.tensor([a[3] for a in L])


def teach(reader, steps=4000, log=print, seed=0):
    rng = random.Random(seed)
    held = {w for ws in HELD_W.values() for w in ws}
    pool = [w for w in reader.wl.words if w.isalpha() and 3 <= len(w) <= 9 and w not in held and w not in FUNCTION][:20000]
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for n, p in reader.named_parameters() if not n.startswith("wl.")], 2e-3)
    ce = nn.functional.cross_entropy
    for step in range(steps):
        s, r, q, k = batch(rng, 64, False, pool)
        pr, pq, pk = reader(s)
        loss = ce(pr.flatten(0, 1), r.flatten()) + ce(pq.flatten(0, 1), q.flatten()) + ce(pk, k)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 999:
            log(f"  step {step + 1}: loss {loss.item():.3f}")


@torch.no_grad()
def exam(reader, held=True, n=1000, seed=99) -> dict:
    """Sentence right = every word's part and the sentence's kind right."""
    rng = random.Random(seed)
    s, r, q, k = batch(rng, n, held)
    pr, pq, pk = reader(s)
    ok_k = pk.argmax(-1) == k
    m = r != -100
    ok_r = ((pr.argmax(-1) == r) | ~m).all(1)
    mq = q != -100
    ok_q = ((pq.argmax(-1) == q) | ~mq).all(1)
    return {"kind": ok_k.float().mean().item(), "parts": ok_r.float().mean().item(),
            "quality kind": ok_q.float().mean().item(), "whole sentence": (ok_k & ok_r & ok_q).float().mean().item()}


# ---- the notebook: what Coal understood, and answers from it -------------------------------------
SIDE = {"me": "your", "you": "my"}                                    # said back from Coal's side


class Notebook:
    def __init__(self, facts: dict | None = None):
        self.facts = facts or {}                                      # owner -> thing -> kind -> word
        self.rng = random.Random()

    @staticmethod
    def _parts(u):
        get = lambda role: [w for w, r in zip(u["words"], u["roles"]) if r == role]
        owner = next((r if r in ("me", "you") else w for w, r in zip(u["words"], u["roles"]) if r in ("me", "you", "name")), None)
        quals = [(w, q) for w, r, q in zip(u["words"], u["roles"], u["quals"]) if r == "quality"]
        things = get("thing")
        return owner, (things[0] if things else None), quals

    def _whose(self, owner):
        return SIDE.get(owner, f"{owner}'s")

    def answer(self, u: dict) -> str | None:
        kind = u["kind"]
        owner, thing, quals = self._parts(u)
        if kind == "greet":
            return self.rng.choice(["hi chalce", "hello", "hey, i'm here", "hi, what's up?"])
        if kind == "tell" and thing and quals:
            box = self.facts.setdefault(owner or "me", {}).setdefault(thing, {})
            for w, q in quals:
                box[q] = w
            desc = " ".join(w for w, _ in quals)
            whose = self._whose(owner or "me")
            return self.rng.choice([f"okay, {whose} {thing} is {desc}.", f"got it: a {desc} {thing}.",
                                    f"noted, {whose} {thing} is {desc}.", f"a {desc} {thing}, okay."])
        if kind.startswith("ask_") and kind[4:] in QUALS and thing:
            q = kind[4:]
            have = self.facts.get(owner or "me", {}).get(thing, {})
            whose = self._whose(owner or "me")
            if q in have:
                return self.rng.choice([f"{whose} {thing} is {have[q]}.", f"it's {have[q]}.", f"{have[q]}, you told me."])
            return f"i don't know what {q} {whose} {thing} is. what {q} is it?"
        if kind == "ask_yesno" and thing:
            have = self.facts.get(owner or "me", {}).get(thing)
            whose = self._whose(owner or "me")
            if have is None:
                return f"i don't know about {whose} {thing}. tell me?"
            for w, q in quals:
                if q in have:
                    return f"yes, {whose} {thing} is {w}." if have[q] == w else f"no, {whose} {thing} is {have[q]}."
            return f"i don't know that yet. i know it's {' and '.join(have.values())}."
        if kind == "ask_things":
            got = self.facts.get(owner or "me", {})
            whose = self._whose(owner or "me")
            if not got:
                return "you haven't told me about anything yet." if (owner or "me") == "me" else "i don't know yet."
            items = [f"a {' '.join(v.values())} {t}".replace("  ", " ") for t, v in got.items()]
            who = {"me": "you have", "you": "i have"}.get(owner or "me", f"{owner} has")
            return f"{who} " + ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] + "."
        if kind == "ask_owner" and thing:
            who = [o for o, ts in self.facts.items() if thing in ts and all(ts[thing].get(q) == w for w, q in quals)]
            if not who:
                return f"i don't know who has the {thing}."
            name = {"me": "you", "you": "i"}.get(who[0], who[0])
            return f"{name} {'do' if name in ('you', 'i') else 'does'}."
        return None


def test():
    import os
    from household.model import load
    from household.train import SAVE
    from household.wordlayer import WordLayer
    brain, _ = load(SAVE)                                              # read only: nothing is saved
    wl = WordLayer.from_keep(brain.keep["mind"]["wordlayer"])
    for p in wl.parameters():
        p.requires_grad_(False)
    reader = Reader(wl)
    print("teaching the reader (copy; Coal's brain is not touched)")
    teach(reader)
    for held, name in ((False, "wordings and words it practised"), (True, "wordings AND words it never saw")):
        r = exam(reader, held)
        print(f"  {name:34s} " + " | ".join(f"{k} {v:.0%}" for k, v in r.items()))
    nb = Notebook()
    talk = ["hi coal", "what is up coal", "I have a blue ball", "what colour is my ball?", "is my ball red?",
            "is my ball blue?", "guess what, I have this purple guitar", "what color's my guitar", "how big is my guitar",
            "my guitar is huge", "how big is my guitar", "Kemi's umbrella is shiny", "who has the umbrella?",
            "what does Kemi have", "what do I have?", "you have a red hat", "what colour is your hat", "why is the sky blue"]
    print("\na chat (the reader + the notebook; '(writer)' = not understood, goes to the writer as before):")
    for t in talk:
        u = reader.read(t)
        a = nb.answer(u) if u["kind"] != "other" else None
        print(f"  you > {t}\n  coal> {a or '(writer)'}      [{u['kind']} {u['sure']:.0%}]")
    torch.save({"reader": {k: v for k, v in reader.state_dict().items() if not k.startswith("wl.")}},
               os.path.join(os.path.dirname(SAVE), "understand_test.pt"))


if __name__ == "__main__":
    import sys
    {"test": test}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
