"""The virtual class (Chalce, 10-30): Coal keeps learning while nobody is in the classroom.

The teacher (class_bank.py) asks each question; Coal answers first and is marked, then
learns the teacher's answer, the same way as in the live class (live.learn: only the
class inputs and the reply head move, so the household is untouched). Every row goes into
coal_class.jsonl marked "virtual", and new words go into the library.

One way of asking each question is held back and never taught. Before and after, Coal is
asked those: if it gets them, it learned the idea, not the sentence. The answers Chalce
taught in the live class are checked too, so the virtual class can not crowd them out.

Run:  python -m household.auto_class
"""
from __future__ import annotations

import os
import shutil

import torch

from household import classroom, live, typist
from household.class_bank import BANK
from household.lexicon import Lexicon, grow
from household.model import load, save
from household.train import SAVE

CLASS = os.path.join(os.path.dirname(SAVE), "coal_class.jsonl")
ANSWERS = os.path.join(os.path.dirname(SAVE), "coal_answers.json")


def score(brain, answers, pairs) -> float:
    return sum(live.reply(brain, answers, q)[0] == a for q, a in pairs) / max(1, len(pairs))


def main() -> None:
    shutil.copy(SAVE, SAVE.replace(".pt", "_before_virtual_class.pt"))
    brain, rounds = load(SAVE)
    keep = brain.keep
    answers = live.Answers(ANSWERS, keep.get("answers"))
    lex = Lexicon.from_dict(keep["lexicon"]) if "lexicon" in keep else grow()
    his = [(q, answers.texts[i]) for q, i in answers.pairs]             # taught live, by Chalce's class
    asked = {q for q, _ in his}
    bank = [l for l in BANK if not asked & set(l[1])]                   # a question he answered stays his answer
    held = [(asks[-1], ans) for _, asks, ans in bank]
    probe = torch.rand(8, 202)                                          # household senses: must not change
    with torch.no_grad():
        house = brain(probe)["act"]
    print(f"before: held-back questions {score(brain, answers, held):.0%}, live-class answers {score(brain, answers, his):.0%}")

    words = 0
    for subject, asks, answer in bank:
        for q in asks[:-1]:
            coal = live.reply(brain, answers, q)[0] or "I don't know yet."
            classroom.keep(CLASS, -1, "coal", coal, {"question": q, "virtual": True,
                                                     "verdict": "right" if coal == answer else "wrong"})
            classroom.keep(CLASS, -1, "teacher", answer, {"question": q, "virtual": True, "subject": subject})
            answers.teach(q, answer)
            words += len([w for t in (q, answer) for w in lex.meet(t)])
    live.learn(brain, answers, steps=400)
    typist.teach(brain, [(q, answers.texts[i]) for q, i in answers.pairs])     # and to type them itself

    with torch.no_grad():
        moved = float((brain(probe)["act"] - house).abs().max())
    print(f"after:  held-back questions {score(brain, answers, held):.0%}, live-class answers {score(brain, answers, his):.0%}")
    print(f"taught {len(bank)} lessons, {len(answers.texts)} answers known, {words} new words, household moved {moved:.1e}")
    for q, a in held[:: max(1, len(held) // 8)]:
        print(f"  {q!r} -> {live.reply(brain, answers, q)[0]!r}  (teacher: {a!r})")
    brain.keep.update(answers=answers.to_dict(), lexicon=dict(lex.__dict__))
    save(brain, SAVE, rounds)
    lex.save(os.path.join(os.path.dirname(SAVE), "lexicon.json"))
    print("saved", SAVE)


if __name__ == "__main__":
    main()
