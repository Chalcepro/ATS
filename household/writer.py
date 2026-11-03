"""Coal writes its answer word by word, reading the question through its word layer (Chalce, 11-02).

    py -3.12 -m household.writer test        (train on a copy, A/B against the picker; nothing is saved into the brain)

The picker (live.reply) can only hand back one stored answer. The writer reads each word of the question
as its word-layer vector, with:
  - whether the word is in Coal's lessons at all (its own memory, not a rule), and
  - the word's section tags as extra things to read ("guitar" brings "noun", "artifact", "music"),
then writes one word at a time. Each word is either chosen from the words it has ever written, or copied
straight from what it is reading (a question word or a tag), so it can say "a guitar is a kind of
instrument" without that sentence ever being taught.

"I don't know" is a lesson like any other: questions about words that are in none of its lessons, and
nonsense words, are answered "i don't know." in class. When to say it is learned, not ruled.
"""
from __future__ import annotations

import os
import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from household.typist import tokens
from household.wordlayer import DIM, WordLayer

H = 128
LONGEST = 24
DONT_KNOW = "i don't know."
STOP = set("a an the is are was what which who whom how why when where do does did you your me my i it of to in on for "
           "and or be can tell about mean means meaning word kind type thing things group belong define someone say says "
           "back that this with know".split())


def content(q: str) -> list[str]:
    return [w for w in tokens(q) if re.fullmatch(r"[a-z]+", w) and w not in STOP]


class Writer(nn.Module):
    def __init__(self, wl: WordLayer, out_words: list[str] | None = None):
        super().__init__()
        self.wl = wl
        self.out = list(out_words or ["<end>"])
        self.out_at = {w: i for i, w in enumerate(self.out)}
        self.out_fix = nn.Parameter(torch.zeros(len(self.out), DIM))
        self.enc_in = nn.Linear(DIM + 5, H)
        self.enc = nn.GRU(H, H, batch_first=True, bidirectional=True)
        self.dec = nn.GRUCell(DIM + 2 * H, 2 * H)
        self.att = nn.Linear(2 * H, 2 * H, bias=False)
        self.proj = nn.Linear(4 * H, DIM)
        self.gate = nn.Linear(4 * H + DIM, 1)
        self._vec: dict[str, torch.Tensor] = {}
        self._tags: dict[str, list[str]] = {}

    def add_out(self, w: str) -> int:
        if w not in self.out_at:
            self.out_at[w] = len(self.out)
            self.out.append(w)
            self.out_fix = nn.Parameter(torch.cat([self.out_fix.data, torch.zeros(1, DIM)]))
        return self.out_at[w]

    # the word layer is read, not trained here: its sections keep their meaning
    def vec(self, w: str) -> torch.Tensor:
        if w not in self._vec:
            with torch.no_grad():
                self._vec[w] = self.wl.vec([w])[0]
        return self._vec[w]

    def tags(self, w: str) -> list[str]:
        if w not in self._tags:
            t = self.wl.tags(w, sure=0.6)
            self._tags[w] = [x for sec in ("kind", "group", "topic", "number") for x, _ in t.get(sec, [])[:2]]
        return self._tags[w]

    def reading(self, q: str, known: set[str], recent: list[str] = (), recalled: list[str] = ()) -> tuple[list[str], torch.Tensor]:
        """What it reads: the question's words, each content word's tags, what was just said,
        and what it recalled from its record. Flags: question, known, tag, from memory, just now."""
        rows, names = [], []

        def put(w: str, flags: list[float]) -> None:
            names.append(w)
            rows.append(torch.cat([self.vec(w), torch.tensor(flags)]))
        for w in tokens(q)[:30]:
            put(w, [1, float(w in known), 0, 0, 0])
        for w in dict.fromkeys(content(q)):
            for t in self.tags(w):
                put(t, [0, 1, 1, 0, 0])
        for line in recent:
            for w in tokens(line)[:30]:
                put(w, [0, 1, 0, 1, 1])
        for line in recalled:
            for w in tokens(line)[:40]:
                put(w, [0, 1, 0, 1, 0])
        return names, torch.stack(rows)

    def forward(self, batch: list[tuple[list[str], torch.Tensor]], targets: list[list[str]] | None = None, greedy: int = 0):
        """Teacher-forced loss when targets are given; else greedy writing for `greedy` steps."""
        B = len(batch)
        L = max(len(n) for n, _ in batch)
        x = torch.zeros(B, L, DIM + 5)
        mask = torch.zeros(B, L, dtype=torch.bool)
        ext = list(self.out)
        ext_at = dict(self.out_at)
        src = torch.zeros(B, L, dtype=torch.long)
        for b, (names, rows) in enumerate(batch):
            x[b, :len(rows)] = rows
            mask[b, :len(rows)] = True
            for i, n in enumerate(names):
                if n not in ext_at:
                    ext_at[n] = len(ext)
                    ext.append(n)
                src[b, i] = ext_at[n]
        mem, _ = self.enc(torch.relu(self.enc_in(x)))
        mem = mem.masked_fill(~mask.unsqueeze(2), 0)
        h = mem.sum(1) / mask.sum(1, keepdim=True)
        V = torch.stack([self.vec(w) for w in self.out]) + self.out_fix
        prev = torch.zeros(B, DIM)
        steps = max(len(t) for t in targets) + 1 if targets else greedy
        loss, written = 0.0, [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool)
        for s in range(steps):
            score = (mem @ self.att(h).unsqueeze(2)).squeeze(2).masked_fill(~mask, -1e9)
            a = score.softmax(1)
            ctx = (a.unsqueeze(2) * mem).sum(1)
            h = self.dec(torch.cat([prev, ctx], 1), h)
            hc = torch.cat([h, ctx], 1)
            p_vocab = (self.proj(hc) @ V.T).softmax(1)
            g = torch.sigmoid(self.gate(torch.cat([hc, prev], 1)))
            p = torch.zeros(B, len(ext))
            p[:, :len(self.out)] = g * p_vocab
            p = p.scatter_add(1, src, (1 - g) * a)
            if targets:
                want = torch.tensor([ext_at.get((t + ["<end>"])[s], 0) if s <= len(t) else 0 for t in targets])
                live = torch.tensor([s <= len(t) for t in targets], dtype=torch.float32)
                loss = loss + (-(p.gather(1, want.unsqueeze(1)).squeeze(1) + 1e-9).log() * live).sum()
                nxt = [(t + ["<end>"])[s] if s <= len(t) else "<end>" for t in targets]
            else:
                for b in range(B):                       # no stutter: not the same word twice running
                    for w in written[b][-1:]:
                        p[b, ext_at[w]] = 0
                ids = p.argmax(1)
                nxt = [ext[int(i)] for i in ids]
                for b in range(B):
                    if not done[b]:
                        if nxt[b] == "<end>":
                            done[b] = True
                        else:
                            written[b].append(nxt[b])
                if done.all():
                    break
            prev = torch.stack([self.vec(w) for w in nxt])
        if targets:
            return loss / sum(len(t) + 1 for t in targets)
        return [re.sub(r" ([?!.,])", r"\1", " ".join(w)) for w in written]

    @torch.no_grad()
    def write(self, questions: list[str], known: set[str], record=None, recent: list[list[str]] | None = None) -> list[str]:
        self.eval()
        reads = [self.reading(q, known, (recent or [[]] * len(questions))[i], record.recall(q) if record else ())
                 for i, q in enumerate(questions)]
        return self(reads, greedy=LONGEST)


# ---- lessons ---------------------------------------------------------------------------------
def known_words(pairs: list[tuple[str, str]]) -> set[str]:
    return {w for q, _ in pairs for w in content(q)}


DONT_KNOW_ASKS = ["What is a {w}?", "What does {w} mean?", "Do you know what {w} is?", "Tell me about {w}.",
                  "What is the meaning of {w}?", "Define {w}.", "What kind of thing is a {w}?"]


def nonsense(r: random.Random) -> str:
    c, v = "bcdfghjklmnprstvwz", "aeiou"
    return "".join(r.choice(c) + r.choice(v) for _ in range(r.randint(2, 3))) + r.choice(["", "p", "n", "t", "x"])


def dont_know_lessons(known: set[str], pool: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    r = random.Random(seed)
    words = [w for w in pool if w not in known]
    out = []
    for i in range(n):
        w = r.choice(words) if i % 3 else nonsense(r)
        out.append((r.choice(DONT_KNOW_ASKS).format(w=w), DONT_KNOW))
    return out


def train(wr: Writer, pairs: list[tuple[str, str]], known: set[str], epochs: int = 12, log=print, record=None,
          extra: list[tuple[str, str, list[str]]] = ()) -> None:
    """pairs: (question, answer); with a record, each lesson reads what it recalls (never its own wording).
    extra: (question, answer, what was just said) - lessons that need the last few lines."""
    for _, a in list(pairs) + [(q, a) for q, a, _ in extra]:
        for w in tokens(a):
            wr.add_out(w)
    r = random.Random(9)
    talk = [q for q, _ in pairs]
    noise = lambda: [f"{r.choice(['you said', 'i said'])} : {r.choice(talk)}" for _ in range(r.randint(0, 3))] if record else []
    data = [(wr.reading(q, known, noise(), record.recall(q, skip=q) if record else ()), tokens(a)[:LONGEST - 1]) for q, a in pairs]
    data += [(wr.reading(q, known, rec, record.recall(q) if record else ()), tokens(a)[:LONGEST - 1]) for q, a, rec in extra]   # as in a real chat: recall is on
    params = [p for n, p in wr.named_parameters() if not n.startswith("wl.")]
    opt = torch.optim.Adam(params, lr=2e-3)
    for e in range(epochs):
        wr.train()
        random.shuffle(data)
        total = 0.0
        for i in range(0, len(data), 64):
            b = data[i:i + 64]
            loss = wr([r for r, _ in b], [t for _, t in b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss) * len(b)
        log(f"  epoch {e + 1}/{epochs}: loss {total / len(data):.3f}")


def overlap(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return 2 * len(x & y) / max(1, len(x) + len(y))


def test() -> None:
    from household import live
    from household.auto_class import ANSWERS
    from household.class_bank import BANK
    from household.dictionary import lessons
    from household.model import load
    from household.train import SAVE
    from household.wordlayer import facts
    random.seed(0)
    torch.manual_seed(0)
    brain, _ = load(SAVE)                                  # read only: the brain file is never written here
    A = live.Answers(os.devnull, brain.keep.get("answers"))
    live.rarity(A)
    pairs = [(q, A.texts[i]) for q, i in A.pairs]
    bank = BANK + lessons(0, 500)
    held = [(asks[-1], ans) for _, asks, ans in bank]
    held_q = {q for q, _ in held}
    pairs = [(q, a) for q, a in pairs if q not in held_q]  # the held-back wordings are never taught
    known = known_words(pairs)
    wl_path = os.path.join(os.path.dirname(SAVE), "wordlayer_test.pt")
    wl = WordLayer.from_keep(torch.load(wl_path))
    pool = [w for w in facts() if w.isalpha() and len(w) > 3]
    random.shuffle(pool)
    idk_train = dont_know_lessons(known, pool[: len(pool) // 2], len(pairs) // 6, seed=1)
    idk_test = dont_know_lessons(known, pool[len(pool) // 2:], 200, seed=2)     # other untaught words, other nonsense
    print(f"{len(pairs)} taught pairs, {len(idk_train)} 'i don't know' lessons, {len(held)} held-back wordings, {len(known)} known words")
    wr = Writer(wl)
    train(wr, pairs + idk_train, known)
    sample = random.Random(3).sample(held, min(400, len(held)))
    wrote = []
    for i in range(0, len(sample), 64):
        wrote += wr.write([q for q, _ in sample[i:i + 64]], known)
    picked = [live.reply(brain, A, q)[0] for q, _ in sample]
    exact_w = sum(w == re.sub(r" ([?!.,])", r"\1", " ".join(tokens(a))) for w, (_, a) in zip(wrote, sample)) / len(sample)
    exact_p = sum(p == a for p, (_, a) in zip(picked, sample)) / len(sample)
    ov_w = sum(overlap(w, a) for w, (_, a) in zip(wrote, sample)) / len(sample)
    ov_p = sum(overlap(p or "", a) for p, (_, a) in zip(picked, sample)) / len(sample)
    idk_on_taught = sum(w == DONT_KNOW for w in wrote) / len(sample)
    idk_w = wr.write([q for q, _ in idk_test], known)
    idk_ok = sum(w == DONT_KNOW for w in idk_w) / len(idk_w)
    print(f"held-back wordings ({len(sample)}):  writer exact {exact_w:.0%}, words in common {ov_w:.0%}   |   picker exact {exact_p:.0%}, words in common {ov_p:.0%}")
    print(f"'i don't know' on untaught/nonsense: {idk_ok:.0%};  wrongly on things it was taught: {idk_on_taught:.0%}")
    for (q, a), w, p in list(zip(sample, wrote, picked))[:12]:
        print(f"  {q}\n     writer: {w}\n     picker: {p}\n     teacher: {a}")
    chat = ["what is my name?", "what is a guitar?", "is a dog an animal?", "what is 3 times 4?", "what kind of word is happy?",
            "what kind of word is quickly?", "what is a violin?", "what is a florp?", "what does electron mean?", "i am sad today",
            "what should i do if i am hungry?", "the stove is hot, should i touch it?", "what is a plan?", "what group does a guitar belong to?"]
    for q, w in zip(chat, wr.write(chat, known)):
        print(f"  CHAT {q} -> {w}")
    torch.save({"out": wr.out, "state": {k: v for k, v in wr.state_dict().items() if not k.startswith("wl.")}},
               os.path.join(os.path.dirname(SAVE), "writer_test.pt"))


# ---- remembering what was said: class lessons with made-up names and facts, never the same ones twice ----
NAMES = ("amara tomas lina kofi yuki mateo nadia oskar priya leon sana ivan zara diego mila felix noor hugo"
         " aiko bruno chen dalia emil fatima goran hana ines jonas kemi luca maya nils olu paulo rosa sven tara").split()
NAMES_NEW = "bea chalce dario elif gael ilse jun kaito lara moses".split()
COLOURS = "red blue green yellow purple orange black white pink brown grey gold".split()
PLACES = "a small town|the city|a village|the coast|the mountains|a farm|an island|the desert".split("|")
FACTS = [("my name is {x}", "what is my name?", "your name is {x}.", "name"),
         ("call me {x}", "do you remember my name?", "your name is {x}.", "name"),
         ("my favourite colour is {c}", "what is my favourite colour?", "your favourite colour is {c}.", "colour"),
         ("i live in {p}", "where do i live?", "you live in {p}.", "place")]


def chat_lessons(n: int, names: list[str], pool: list[str], seed: int) -> list[tuple[str, str, list[str]]]:
    """(question, answer, the last few lines) - the fact is somewhere in the last lines, among others."""
    r, out = random.Random(seed), []
    for _ in range(n):
        noise = [f"{r.choice(['you said', 'i said'])} : {r.choice(pool)}" for _ in range(r.randint(0, 2))]
        kind = r.random()
        if kind < 0.15:
            x = r.choice(names)
            # not "i am {x}": "i am sad" is a feeling, not a name
            out.append((r.choice(["my name is {x}", "call me {x}", "you can call me {x}"]).format(x=x), f"nice to meet you, {x}.", noise))
        elif kind < 0.6:
            said, ask, ans, _ = r.choice(FACTS)
            fill = {"x": r.choice(names), "c": r.choice(COLOURS), "p": r.choice(PLACES)}
            lines = noise + [f"you said : {said.format(**fill)}"]
            r.shuffle(lines)
            out.append((ask, ans.format(**fill), lines))
        elif kind < 0.8:
            prev = r.choice(pool)
            reply = [f"i said : {r.choice(pool)}"] if r.random() < 0.7 else []      # in a real chat its own answer comes after the question
            out.append((r.choice(["what did i just ask you?", "what was my last question?"]), f"you asked {prev}", noise[:1] + [f"you said : {prev}"] + reply))
        else:
            prev = r.choice(pool)
            asked = [f"you said : {r.choice(pool)}"] if r.random() < 0.7 else []
            out.append((r.choice(["what did you just say?", "what was your last answer?"]), f"i said {prev}", noise[:1] + asked + [f"i said : {prev}"]))
    return out


def test_recall() -> None:
    from household import live
    from household.class_bank import BANK
    from household.dictionary import lessons
    from household.memory import Record
    from household.model import load
    from household.train import SAVE
    from household.wordlayer import facts
    random.seed(0)
    torch.manual_seed(0)
    brain, _ = load(SAVE)                                  # read only
    A = live.Answers(os.devnull, brain.keep.get("answers"))
    live.rarity(A)
    held = [(asks[-1], ans) for _, asks, ans in BANK + lessons(0, 500)]
    held_q = {q for q, _ in held}
    pairs = [(q, A.texts[i]) for q, i in A.pairs if q not in held_q]
    have = {q for q, _ in pairs}
    pairs += [(q, ans) for _, asks, ans in BANK for q in asks[:-1] if q not in have]
    known = known_words(pairs)
    wl = WordLayer.from_keep(torch.load(os.path.join(os.path.dirname(SAVE), "wordlayer_test.pt")))
    wr = Writer(wl)
    record = Record(wr, [{"when": "", "kind": "lesson", "text": q, "answer": a} for q, a in pairs])
    pool = [w for w in facts() if w.isalpha() and len(w) > 3]
    random.shuffle(pool)
    # 1 in 12: at 1 in 6 it said "i don't know" to a third of what it was taught
    idk_train = dont_know_lessons(known, pool[: len(pool) // 2], len(pairs) // 12, seed=1)
    idk_test = dont_know_lessons(known, pool[len(pool) // 2:], 200, seed=2)
    lines = [q for q, _ in pairs]
    chat_train = chat_lessons(1500, NAMES, lines, seed=4)
    chat_test = chat_lessons(200, NAMES_NEW, lines, seed=5)            # names it never practised with
    print(f"{len(pairs)} taught pairs in the record, {len(idk_train)} 'i don't know', {len(chat_train)} remembering lessons")
    train(wr, pairs + idk_train, known, record=record, extra=chat_train)
    sample = random.Random(3).sample(held, 400)
    wrote = []
    for i in range(0, 400, 64):
        wrote += wr.write([q for q, _ in sample[i:i + 64]], known, record)
    picked = [live.reply(brain, A, q)[0] for q, _ in sample]
    norm = lambda a: re.sub(r" ([?!.,])", r"\1", " ".join(tokens(a)))
    ex_w = sum(w == norm(a) for w, (_, a) in zip(wrote, sample)) / 400
    ex_p = sum(p == a for p, (_, a) in zip(picked, sample)) / 400
    ov_w = sum(overlap(w, a) for w, (_, a) in zip(wrote, sample)) / 400
    ov_p = sum(overlap(p or "", a) for p, (_, a) in zip(picked, sample)) / 400
    idk_w = wr.write([q for q, _ in idk_test], known, record)
    reads = [wr.reading(q, known, rec, record.recall(q)) for q, _, rec in chat_test]
    wr.eval()
    with torch.no_grad():
        chat_w = wr(reads, greedy=LONGEST)
    print(f"held-back wordings: writer+recall exact {ex_w:.0%}, words in common {ov_w:.0%}   |   picker exact {ex_p:.0%}, words in common {ov_p:.0%}")
    print(f"'i don't know' on untaught/nonsense {sum(w == DONT_KNOW for w in idk_w) / len(idk_w):.0%}; wrongly on taught {sum(w == DONT_KNOW for w in wrote) / 400:.0%}")
    print(f"remembering what was said (names it never practised with): {sum(w == norm(a) for w, (_, a, _) in zip(chat_w, chat_test)) / len(chat_test):.0%}")
    for (q, a), w, p in list(zip(sample, wrote, picked))[:10]:
        print(f"  {q}\n     writer+recall: {w}\n     picker: {p}\n     teacher: {a}")
    for (q, a, rec), w in list(zip(chat_test, chat_w))[:6]:
        print(f"  [{' / '.join(rec)}] {q} -> {w}   (want: {a})")
    # the chat from earlier, with a live record: what it heard goes in as it goes
    for line in ["my name is chalce", "what is my name?", "what is a guitar?", "what did i just ask you?", "what is a florp?", "i'm feeling stressed today",
                 "what kind of word is quickly?", "is a dog an animal?", "what is a violin?", "what does buy mean?", "i am sad today"]:
        rec = record.recent(3)
        out = wr.write([line], known, record, [rec])[0]
        record.add("heard", line)
        record.add("said", out)
        print(f"  CHAT {line} -> {out}")


if __name__ == "__main__":
    {"test": test, "recall": test_recall}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
