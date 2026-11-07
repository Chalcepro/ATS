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
import shutil
import sys

import torch

from household.maths import ColumnCell, OPS, read_sum, work
from household.memory import Record
from household.wordlayer import WordLayer
from household.writer import Writer

SAYS = {"add": "plus", "subtract": "minus", "times": "times"}
GREET = ["hi chalce", "hey, i'm here", "yo", "hi, what's up?"]
SITE_NAMES = {"chat": "what you told me", "mdn-js": "javascript", "react": "react", "simple-wikipedia": "plain english",
              "en-wikipedia": "general knowledge", "w3schools-js": "javascript", "w3schools-html": "html"}


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

    @staticmethod
    def _plain(k):
        return {a: (sorted(b) if isinstance(b, set) else b) for a, b in k.items()}

    def to_keep(self) -> dict:
        return {"wordlayer": self._wl_keep,
                "writer": {"out": self.writer.out, "known": sorted(self.known),
                           "state": {k: v for k, v in self.writer.state_dict().items() if not k.startswith("wl.")}},
                "record": self.record.events, "maths": self.cell.state_dict(),
                **({"understand": dict(self._u, facts=[self._plain(k) for k in self.memory.kept if k.get("site") == "chat"])}
                   if self.memory else {})}

    def understood(self, text: str) -> str | None:
        """What Chalce means, said back from what Coal knows. None = not understood (the writer tries)."""
        from household.comprehend import ASK_VERBS
        from household.relations import frame
        from household.understand import MEANS, SAY
        m = self.memory
        if self.pending:                                     # "closure in javascript or in english?" -> "javascript"
            f, parsed, asks, sites = self.pending
            pick = next((s for s in sites if SITE_NAMES.get(s, s) in text.lower() or s in text.lower()), None)
            self.pending = None
            if pick:
                return m.answer(f, parsed, asks, site=pick)
        u = self.reader.read(text)
        parsed = m.parser.parse(text)
        f = frame(parsed) if parsed else {}
        lead = next((p for p in parsed if p["type"] != "interjection" and p["word"] not in ("btw", "so", "coal", "okay")),
                    parsed[0]) if parsed else None
        asks = f.get("asks") or (lead["word"] if lead and lead["type"] == "verb" and lead["word"] in ASK_VERBS else None)
        if asks and m._find(parsed):                         # a question about something he knows: answer that first
            pass
        elif u["kind"] == "greet" and u["sure"] > .5:
            return self.rng.choice(GREET)
        elif u["kind"] in MEANS and u["sure"] > .4 and len(u["words"]) <= 6:
            return self.rng.choice(SAY[u["kind"]])
        if len(parsed) < 2:
            return None
        if not asks:
            if u["kind"] == "other" or "does" not in f:
                return None
            return m.keep(f, parsed)
        sites, found = m.senses(parsed)
        if len(sites) > 1 and asks in ("what", "which", "how"):         # the word means different things in different places
            thing = next((p["word"] for p in parsed if p["type"] == "noun"), "that")
            self.pending = (f, parsed, asks, sites)
            names = [SITE_NAMES.get(s, s) for s in sites]
            return f"{thing} in {', in '.join(names[:-1])} or in {names[-1]}?"
        return m.answer(f, parsed, asks)

    def reply(self, text: str) -> str:
        """Hear a line, answer it, and keep both in the record."""
        recent = self.record.recent(3)
        s = read_sum(text)
        if s:
            a, op, b = s
            out = f"{a} {SAYS[op]} {b} is {work(self.cell, a, b, op)}"
        else:
            out = self.understood(text) if self.memory else None
            out = out or self.writer.write([text], self.known, self.record, [recent])[0] or "i don't know."
        self.record.add("heard", text)
        self.record.add("said", out)
        return out

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
