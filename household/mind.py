"""Coal's mind (11-03): the word layer, the writer, the record and the maths section, put into the real Coal.

    py -3.12 -m household.mind build      (backs the brain up, then puts the tested parts into it)
    py -3.12 -m household.mind chat       (a quick chat with the result, in the terminal)

Everything lives in the brain file (brain.keep["mind"]), so the one file is still all of Coal:
  wordlayer - words as vectors, in sections (wordlayer.py)
  writer    - writes answers word by word, reading the question, the last lines and what it recalls (writer.py)
  record    - every lesson, everything heard and said, found again by meaning (memory.py)
  maths     - the column cell, which works sums out column by column (maths.py)
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

    def to_keep(self) -> dict:
        return {"wordlayer": self._wl_keep,
                "writer": {"out": self.writer.out, "known": sorted(self.known),
                           "state": {k: v for k, v in self.writer.state_dict().items() if not k.startswith("wl.")}},
                "record": self.record.events, "maths": self.cell.state_dict()}

    def reply(self, text: str) -> str:
        """Hear a line, answer it, and keep both in the record."""
        recent = self.record.recent(3)
        s = read_sum(text)
        if s:
            a, op, b = s
            out = f"{a} {SAYS[op]} {b} is {work(self.cell, a, b, op)}"
        else:
            out = self.writer.write([text], self.known, self.record, [recent])[0] or "i don't know."
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


def chat() -> None:
    from household.model import load
    from household.train import SAVE
    brain, _ = load(SAVE)
    m = Mind(brain.keep["mind"])
    print("Talking to Coal's mind (nothing is saved). 'quit' to stop.")
    while (line := input("you > ").strip()).lower() not in ("quit", "exit"):
        print("coal >", m.reply(line))


if __name__ == "__main__":
    {"build": build, "chat": chat}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
