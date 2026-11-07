"""Coal's mind (11-03): the word layer, the writer, the record and the maths section, put into the real Coal.

    py -3.12 -m household.mind build      (backs the brain up, then puts the tested parts into it)
    py -3.12 -m household.mind chat       (a quick chat with the result, in the terminal)

Everything lives in the brain file (brain.keep["mind"]), so the one file is still all of Coal:
  wordlayer - words as vectors, in sections (wordlayer.py)
  writer    - writes answers word by word, reading the question, the last lines and what it recalls (writer.py)
  record    - every lesson, everything heard and said, found again by meaning (memory.py)
  maths     - the column cell, which works sums out column by column (maths.py)
  understand (11-06, optional) - the reader (what Chalce means), the parser (word types + how words hang
              together), and what he was told or read, kept as meanings; he reasons over them (comprehend.py)
A lesson taught in class goes into the record at once, so Coal can use it before any retraining.
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import torch

from household.maths import ColumnCell, OPS, read_sum, work
from household.memory import Record
from household.wordlayer import WordLayer
from household.writer import Writer

SAYS = {"add": "plus", "subtract": "minus", "times": "times"}
GREET = ["hi chalce", "hey, i'm here", "yo", "hi, what's up?"]
SITE_NAMES = {"chat": "what you told me", "device": "this device", "mdn-js": "javascript", "react": "react", "simple-wikipedia": "plain english",
              "en-wikipedia": "general knowledge", "w3schools-js": "javascript", "w3schools-html": "html"}


QUESTIONS = ("what", "where", "when", "who", "why", "which", "how", "is", "are", "did", "does", "do", "can")


class Mind:
    def __init__(self, keep: dict):
        self._wl_keep = keep["wordlayer"]                  # it does not change in chat: saved as it came
        self.wl = WordLayer.from_keep(keep["wordlayer"])
        self.writer = Writer(self.wl, keep["writer"]["out"])
        self.writer.load_state_dict(keep["writer"]["state"], strict=False)
        self.writer.eval()
        self.known = set(keep["writer"]["known"])
        self.record = Record(self.writer, keep["record"])
        self.cell = ColumnCell()
        self.cell.load_state_dict(keep["maths"])
        self.cell.eval()
        self.memory = self.reader = None
        self.pending = None                                # a question waiting on "which one did you mean?"
        u = keep.get("understand")
        if u:
            self._understand(u)

    def _understand(self, u: dict) -> None:
        import random
        from household.comprehend import Memory
        from household.relations import Parser
        from household.understand import Reader
        self.reader = Reader(self.wl)
        self.reader.load_state_dict(u["reader"], strict=False)
        self.reader.eval()
        parser = Parser(self.wl, u["lexicon"], hidden=u["parser"]["tag.weight"].shape[1] // 2)
        parser.load_state_dict(u["parser"], strict=False)
        parser.eval()
        self.memory = Memory(parser)
        for k in u.get("facts", []) + u.get("read", []):
            self.memory.kept.append(dict(k, words=set(k["words"]), nouns=set(k.get("nouns", []))))
        self._u = u
        self.rng = random.Random()
        self.chooser = None                                  # he picks answers himself (chooser.py), if taught
        from household.train import SAVE
        cp = os.path.join(os.path.dirname(os.environ.get("COAL_BRAIN") or SAVE), os.environ.get("COAL_CHOOSER", "chooser_test.pt"))
        if os.environ.get("COAL_PICK", "chooser") == "chooser" and os.path.exists(cp):   # COAL_PICK=rules: the old way
            import torch
            from household.chooser import Chooser
            st = torch.load(cp, map_location="cpu")
            self.chooser = Chooser(self.wl, hidden=st["enc.weight_hh_l0"].shape[1])
            self.chooser.load_state_dict(st, strict=False)
            self.chooser.eval()
        self.look_around()

    def look_around(self) -> int:
        """What device is he on? He looks (device.py) and keeps it as meanings, like anything he reads.
        Done fresh each start: the device may be a different one, or have new apps."""
        from household import device
        from household.relations import frame
        m = self.memory
        m.kept = [k for k in m.kept if k.get("site") != "device"]
        try:
            lines = device.meanings()
        except Exception:                                    # a device that shows nothing: he goes on without it
            return 0
        return self._take(lines, "device")

    def _take(self, lines, site, src=None) -> int:
        """Read lines and keep their meanings, marked with where they came from."""
        from household.relations import frame
        m, n = self.memory, 0
        for i in range(0, len(lines), 64):
            for parsed in m.parser.parse_many(lines[i:i + 64]):
                if len(parsed) >= 3:
                    m.keep(frame(parsed), parsed)
                    m.kept[-1]["site"] = site
                    if src:
                        m.kept[-1]["from"], m.kept[-1]["looked"] = src, True   # looked up in a chat: kept in the brain
                    n += 1
        return n

    FAILED = ("you didn't tell me", "i'm not sure", "no, not that i know")
    MINE = {"i", "my", "me", "mine", "we", "our", "us"}

    def after_fail(self, text, parsed, f, asks):
        """He didn't know. Something not about us -> he goes and looks it up, reads it, keeps it, tries again."""
        from household import browse, device
        thing = " ".join(p["word"] for p in parsed if p["type"] in ("noun", "adjective") and p["word"] not in ("thing", "kind"))
        if not thing or self.MINE & {p["word"] for p in parsed} or not device.online():
            return None
        got = browse.look(thing) or (browse.look(thing.split()[-1]) if " " in thing else None)
        if not got:
            return None
        site, url, sents = got
        sents = [re.sub(r"\[\d+\]", "", x) for x in sents if len(x.split()) <= 40][:150]
        self._take(sents, site, url)
        if self.chooser is not None:                             # he reads the top of the page and picks himself
            said, used = self.chooser.answer(text, sents[:12])
            out = f"{said}. (read in {SITE_NAMES.get(site, site)}: {used[-1]})" if said else None
        else:
            out = self.pick(text, parsed, f, asks, site=site)
        if out and not out.startswith(self.FAILED):
            return f"i didn't know, so i looked it up. {out}"
        return None

    @staticmethod
    def _plain(k):
        return {a: (sorted(b) if isinstance(b, set) else b) for a, b in k.items()}

    def to_keep(self) -> dict:
        return {"wordlayer": self._wl_keep,
                "writer": {"out": self.writer.out, "known": sorted(self.known),
                           "state": {k: v for k, v in self.writer.state_dict().items() if not k.startswith("wl.")}},
                "record": self.record.events, "maths": self.cell.state_dict(),
                **({"understand": dict(self._u, facts=[self._plain(k) for k in self.memory.kept if k.get("site") == "chat"],
                                       read=self._u.get("read", []) + [self._plain(k) for k in self.memory.kept
                                                                       if k.get("looked")])}
                   if self.memory else {})}

    def understood(self, text: str) -> str | None:
        """What Chalce means, said back from what Coal knows. None = not understood (the writer tries)."""
        from household.comprehend import ASK_VERBS
        from household.relations import frame
        from household.understand import MEANS, SAY
        m = self.memory
        if self.pending:                                     # "closure in javascript or in english?" -> "javascript"
            f, parsed, asks, sites, asked = self.pending
            pick = next((s for s in sites if SITE_NAMES.get(s, s) in text.lower() or s in text.lower()), None)
            self.pending = None
            if pick:
                return self.pick(asked, parsed, f, asks, site=pick)
        u = self.reader.read(text)
        self._last_read = (text, u["kind"]) if len(u["words"]) <= 6 else None   # a "no" next can fix this reading
        parsed = m.parser.parse(text)
        f = frame(parsed) if parsed else {}
        lead = next((p for p in parsed if p["type"] != "interjection" and p["word"] not in ("btw", "so", "coal", "okay")),
                    parsed[0]) if parsed else None
        asks = f.get("asks") or (lead["word"] if lead and lead["type"] == "verb" and lead["word"] in ASK_VERBS else None)
        if asks and m._find(parsed):                         # a question about something he knows: answer that first
            pass
        elif u["kind"] == "greet" and u["sure"] > .5:
            return self.rng.choice(GREET)
        elif u["kind"] in MEANS and asks and any(p["type"] == "noun" for p in parsed):
            self.reader_learns(text, "other", why=f"grammar sees a question about a thing, reader said {u['kind']}")
        elif u["kind"] in MEANS and u["sure"] > .4 and len(u["words"]) <= 6:
            return self.rng.choice(SAY[u["kind"]])
        if len(parsed) < 2:
            return None
        if not asks:
            if u["kind"] == "other" or "does" not in f:
                return None
            return m.keep(f, parsed)
        sites, found = m.senses(parsed)
        if len(sites) > 1 and asks in ("what", "which", "how") and not self.MINE & {p["word"] for p in parsed}:         # the word means different things in different places
            thing = next((p["word"] for p in parsed if p["type"] == "noun"), "that")
            self.pending = (f, parsed, asks, sites, text)
            names = [SITE_NAMES.get(s, s) for s in sites]
            return f"{thing} in {', in '.join(names[:-1])} or in {names[-1]}?"
        asks_what_it_is = asks in ("what", "who") and f.get("does") in ("is", "are") and not sites
        out = None if asks_what_it_is else self.pick(text, parsed, f, asks)   # nothing he holds says what it is
        if not out or out.startswith(self.FAILED):              # what he does after failing is the point
            out = self.after_fail(text, parsed, f, asks) or out
        return out

    def pick(self, text, parsed, f, asks, site=None):
        """What he was told: he picks the answer himself (the chooser). Reading/device: the rules, until he beats them."""
        m = self.memory
        if self.chooser is None:
            return m.answer(f, parsed, asks, site=site)
        cands = [] if site else m.net(parsed, "chat")            # what he was told first, like a person
        if not cands and not site and self.MINE & {p["word"] for p in parsed}:   # about us: only the chat can know
            thing = next((p["word"] for p in reversed(parsed) if p["type"] == "noun"), "that")
            return f"you haven't told me about your {thing} yet."
        if not cands:                                            # reading: the rules still beat him there (11-07 test)
            return m.answer(f, parsed, asks, site=site)
        said, used = self.chooser.answer(text, [k["text"] for k in cands])
        if not said:
            return None
        src = next((k.get("site", "chat") for k in cands if k["text"] == used[-1]), "chat")
        from household.comprehend import flip
        return flip(said) if src == "chat" else f"{said}. (read in {SITE_NAMES.get(src, src)}: {used[-1]})"

    def reply(self, text: str) -> str:
        """Hear a line, answer it, and keep both in the record."""
        recent = self.record.recent(3)
        s = read_sum(text)
        told = self._corrected(text)                         # "no, it's in the car": the fix is kept like any telling
        if s:
            a, op, b = s
            out = f"{a} {SAYS[op]} {b} is {work(self.cell, a, b, op)}"
        else:
            out = (self.understood(told or text) if self.memory else None)
            if told and self.chooser is not None and getattr(self, "_fixing", None):
                self.chooser_learns(*self._fixing)              # the answer was wrong: he practises the right one
            out = ("okay, i've got it now." if told and out else out) or                 self.writer.write([text], self.known, self.record, [recent])[0] or "i don't know."
        self.record.add("heard", text)
        self.record.add("said", out)
        first = text.lower().split()[:1]
        self._last_q = (text, out) if first and first[0] in QUESTIONS else None
        return out

    def reader_learns(self, text, kind=None, why="", steps=40, wrong=None):
        """The reader learns from this one line, now (a few small steps), while re-practising lessons he
        already knows so the new one doesn't wipe the old. Logged; the reader is saved with the brain."""
        import json
        import random
        import torch.nn.functional as F
        from household.understand import KINDS, batch, words
        rd = self.reader
        if not hasattr(self, "_rd_opt"):
            self._rd_opt = torch.optim.Adam([p for n, p in rd.named_parameters() if not n.startswith("wl.")], 5e-4)
            self._rd_rng = random.Random()
            self._rd_pool = [w for w in self.wl.words if w.isalpha() and 3 <= len(w) <= 9][:20000]
        line = words(text)
        want = KINDS.index(kind) if kind else None
        bad = KINDS.index(wrong) if wrong else None
        rd.train()
        for step in range(steps):                                # until he reads it right (at most `steps`)
            if step % 5 == 0 and step:
                rd.eval()
                now = rd.read(text)["kind"]
                if (kind and now == kind) or (not kind and now != wrong):
                    break
                rd.train()
            s, r, q, k = batch(self._rd_rng, 31, False, self._rd_pool)
            pr, pq, pk = rd(s + [line])
            loss = (F.cross_entropy(pr[:-1].flatten(0, 1), r.flatten(), ignore_index=-100)
                    + F.cross_entropy(pq[:-1].flatten(0, 1), q.flatten(), ignore_index=-100)
                    + F.cross_entropy(pk[:-1], k))
            if want is not None:
                loss = loss + 3 * F.cross_entropy(pk[-1:], torch.tensor([want]))
            else:                                                # not that: less sure of the wrong one, nothing forced
                loss = loss - 3 * torch.log1p(-pk[-1].softmax(-1)[bad].clamp(max=.999))
            self._rd_opt.zero_grad(); loss.backward(); self._rd_opt.step()
        rd.eval()
        self._u["reader"] = {k: v for k, v in rd.state_dict().items() if not k.startswith("wl.")}
        os.makedirs(os.path.dirname(self.LESSONS), exist_ok=True)
        with open(os.path.join(os.path.dirname(self.LESSONS), "reader_live.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"line": text, "kind": kind, "now": rd.read(text)["kind"], "why": why}) + "\n")

    LESSONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lessons", "corrections.jsonl")

    def _corrected(self, text):
        """Told he was wrong right after answering -> the question, his answer and the fix are saved as a lesson
        for his next training (that is how failing teaches him). Returns the fix without its "no"."""
        last = getattr(self, "_last_q", None)
        w = text.lower().replace(",", " ").split()
        lr, self._last_read = getattr(self, "_last_read", None), None
        if lr and not last and w and w[0] in ("no", "nope", "wrong", "nah"):   # "no" after he took a line one way: misread
            fix = self.reader.read(" ".join(w[1:]))              # "no, i said stop": the fix says what it was
            from household.understand import MEANS
            kind = fix["kind"] if fix["kind"] in MEANS + ["greet"] and fix["kind"] != lr[1] and fix["sure"] > .6 else None
            self.reader_learns(lr[0], kind, wrong=lr[1], why=f"told no after he read it as {lr[1]}")
        if not last or not w or w[0] not in ("no", "nope", "wrong", "nah", "actually") or len(w) < 3:
            return None
        import json
        os.makedirs(os.path.dirname(self.LESSONS), exist_ok=True)
        with open(self.LESSONS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"asked": last[0], "said": last[1], "told": text}, ensure_ascii=False) + "\n")
        fix = " ".join(text.replace(",", " ").split()[1:])
        self._fixing = (last[0], fix)
        return fix

    def chooser_learns(self, question, fix, steps=60):
        """Told the right answer: he finds where it is (a thing he was told that holds the fix, or the fix itself)
        and practises picking it and pointing at it, with old lessons replayed, until he answers it right."""
        import json
        import random
        import torch.nn.functional as F
        from household import chooser as C
        from household.understand import words
        self._fixing = None
        m, ch = self.memory, self.chooser
        parsed = m.parser.parse(question)
        cands = m.net(parsed, "chat")
        if not cands:
            return
        texts = [k["text"] for k in cands]
        fw = [x for x in words(fix) if x not in ("so", "it", "is", "now")]
        q = words(question)
        def find(fs):                                            # the newest told thing holding the fix words in a row
            for i, f in enumerate(fs):
                for n in range(len(fw), 1, -1):
                    for a in range(len(fw) - n + 1):
                        run = fw[a:a + n]
                        for j in range(len(f) - n + 1):
                            if f[j:j + n] == run and not set(run) <= set(q):
                                return i, j, j + n - 1
            return None
        fs = [words(t) or ["-"] for t in texts]
        hit = find(fs)
        if not hit:
            return
        gold, a, b = hit
        if not hasattr(self, "_ch_opt"):
            self._ch_opt = torch.optim.Adam([p for n, p in ch.named_parameters() if not n.startswith("wl.")], 3e-4)
            self._ch_rng = random.Random()
            self._ch_pool = [w for w in self.wl.words if w.isalpha() and 3 <= len(w) <= 9][:15000]
        ce = F.cross_entropy
        for step in range(steps):
            if step % 5 == 0:
                ch.eval()
                said, _ = ch.answer(question, texts)
                if said == " ".join(fs[gold][a:b + 1]):
                    break
            ch.train()
            out, _, st, en = ch(q, fs, picks=[gold, -1], steps=2)
            loss = 3 * (ce(out[0][None], torch.tensor([gold])) + ce(out[1][None], torch.tensor([len(fs)]))
                        + ce(st[None], torch.tensor([a])) + ce(en[None], torch.tensor([b])))
            for _ in range(3):                                   # old lessons, so the new one doesn't wipe them
                L = None
                while L is None:
                    L = C.lesson(self._ch_rng, self._ch_pool)
                lq, lf, lg, (ls, le) = L
                picks = lg + [-1]
                o, _, s2, e2 = ch(lq, lf, picks=picks, steps=len(picks))
                loss = loss + sum(ce(sc[None], torch.tensor([picks[t] if picks[t] >= 0 else len(lf)])) for t, sc in enumerate(o))
                loss = loss + ce(s2[None], torch.tensor([ls])) + ce(e2[None], torch.tensor([le]))
            self._ch_opt.zero_grad(); loss.backward(); self._ch_opt.step()
        ch.eval()
        self._ch_dirty = True
        with open(os.path.join(os.path.dirname(self.LESSONS), "chooser_live.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"asked": question, "fix": fix, "picked": texts[gold],
                                "now": ch.answer(question, texts)[0], "steps": step}) + "\n")

    def taught(self, question: str, answer: str) -> None:
        """A class lesson: into the record now, usable straight away by recall."""
        self.record.add("lesson", question, answer)
        self.known.update(w for w in question.lower().split() if w.isalpha())


def build() -> None:
    from household import live
    from household.class_bank import BANK
    from household.model import load, save
    from household.train import SAVE
    here = os.path.dirname(SAVE)
    need = {p: os.path.join(here, p) for p in ("wordlayer_test.pt", "writer_recall_test.pt", "maths_cell_test.pt")}
    missing = [p for p, f in need.items() if not os.path.exists(f)]
    if missing:
        sys.exit(f"missing {missing}: run the tests first (wordlayer test, writer recall, maths trial)")
    shutil.copy(SAVE, SAVE.replace(".pt", "_before_mind.pt"))
    brain, rounds = load(SAVE)
    A = live.Answers(os.devnull, brain.keep.get("answers"))
    pairs = [(q, A.texts[i]) for q, i in A.pairs]
    have = {q for q, _ in pairs}
    pairs += [(q, ans) for _, asks, ans in BANK for q in asks if q not in have]      # the bank, every wording
    w = torch.load(need["writer_recall_test.pt"])
    brain.keep["mind"] = {"wordlayer": torch.load(need["wordlayer_test.pt"]),
                          "writer": {"out": w["out"], "known": w["known"], "state": w["state"]},
                          "record": [{"when": "", "kind": "lesson", "text": q, "answer": a} for q, a in pairs],
                          "maths": torch.load(need["maths_cell_test.pt"])}
    Mind(brain.keep["mind"])                                                           # it must load before it is saved
    save(brain, SAVE, rounds)
    print(f"Coal has a mind: {len(pairs)} lessons in its record, {len(w['out'])} words it writes with. "
          f"Brain file {os.path.getsize(SAVE) / 1e6:.0f} MB (backup: {os.path.basename(SAVE.replace('.pt', '_before_mind.pt'))})")


def build_understanding(out_name: str = "household_brain_trend_understand_test.pt") -> str:
    """A COPY of Coal's brain with the understanding put into its mind (Coal's own file is only read)."""
    import json
    from household.grammar import LEXICON
    from household.model import load, save
    from household.train import SAVE
    here = os.path.dirname(SAVE)
    brain, rounds = load(SAVE)
    reader = torch.load(os.path.join(here, "understand_test.pt"))["reader"]
    best = os.path.join(here, "relations_test2.pt")
    parser = torch.load(best if os.path.exists(best) else os.path.join(here, "relations_test.pt"))
    read = []
    know = os.path.join(os.path.dirname(here), "data", "reading", "knowledge.jsonl")
    if os.path.exists(know):
        read = [json.loads(x) for x in open(know, encoding="utf-8")]
    brain.keep["mind"]["understand"] = {"reader": reader, "parser": parser, "facts": [], "read": read,
                                        "lexicon": json.load(open(LEXICON, encoding="utf-8"))}
    Mind(brain.keep["mind"])                                                  # it must load before it is saved
    path = os.path.join(here, out_name)
    save(brain, path, rounds)
    print(f"test copy with understanding: {path} ({os.path.getsize(path) / 1e6:.0f} MB, {len(read)} meanings read)")
    return path


def chat() -> None:
    from household.model import load
    from household.train import SAVE
    brain, _ = load(os.environ.get("COAL_BRAIN") or SAVE)
    m = Mind(brain.keep["mind"])
    print("Talking to Coal's mind (nothing is saved). 'quit' to stop.")
    while (line := input("you > ").strip()).lower() not in ("quit", "exit"):
        print("coal >", m.reply(line))


if __name__ == "__main__":
    {"build": build, "chat": chat, "understand_test": build_understanding}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
