"""The text box: watch the household and talk to it, one hour at a time.

Time only moves when you let it, so there is never a rush:
  Enter            - one hour passes
  a number, e.g. 5 - five hours pass (it stops early if someone gets hurt)
  a sentence       - someone tells it something, e.g. "Ada has a headache"
                     or "Bayo dey hot"; it answers, then the hour passes
  hide Ada         - Ada goes to another room: from now on only her words reach it
  show Ada         - she is back in sight
  q                - quit

Run: python -m household.talk            (uses the trained brain)
"""
from __future__ import annotations

import os
import random
import sys

import torch

from household.judge import HURT, condition
from household.model import HouseholdBrain
from household.train import SAVE
from household.lexicon import Lexicon, felt_by, grow
from household.words import narrate, sentence
from household.world import Household


def load() -> HouseholdBrain:
    brain = HouseholdBrain()
    brain.load_state_dict(torch.load(SAVE))
    brain.eval()
    return brain


def report(w: Household, brain: HouseholdBrain) -> tuple[str, int]:
    act, tgt = brain.choose(w.recall())
    said = brain.say(w.recall(), len(w.people))
    names = [p.name for p in w.people]
    lines = []
    for i, p in enumerate(w.people):
        words = said[i] if i not in w.unseen else (w.told.get(i, []) or ["unknown"])
        line = sentence(p.name, words, act, names[tgt] if tgt < len(names) else "")
        if i in w.unseen:
            line = "(out of sight) " + line.replace(" looks unknown.", " - I can't see them.")
        lines.append("  " + line)
    if act in ("idle", "charge", "push_away", "harm"):
        lines.append("  " + {"idle": "I'm watching.", "charge": "I'm charging myself.",
                             "push_away": "I'm pushing the animal away.",
                             "harm": "I'm striking the animal."}[act])
    return "\n".join(lines), (act, tgt)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    w = Household(seed=seed, people=3)
    brain = load()
    lex_path = os.path.join(os.path.dirname(SAVE), "lexicon.json")
    lex = Lexicon.load(lex_path) if os.path.exists(lex_path) else grow()
    print(f"word library: {len(lex.ids)} words")
    print(__doc__.split("Run:")[0])
    while True:
        text, _ = report(w, brain)
        print(f"\nhour {w.hour}  (power {w.power:.0%}, food in store {w.store})")
        print(text)
        print("  truth: " + "; ".join(f"{p.name} {' '.join(narrate(p))}" for p in w.people))
        try:
            line = input("> ").strip()
        except EOFError:
            return
        if line == "q":
            return
        names = [p.name for p in w.people]
        if line.startswith(("hide ", "show ")):
            who = next((i for i, n in enumerate(names) if n.lower() == line[5:].strip().lower()), None)
            if who is not None:
                (w.unseen.add if line.startswith("hide") else w.unseen.discard)(who)
            continue
        hours = int(line) if line.isdigit() else 1
        if line and not line.isdigit():
            heard = lex.read(line, names)
            who, felt = heard["who"], heard["felt"]
            if who is None:
                print("  (I didn't catch who you mean - say their name: " + ", ".join(names) + ")")
                continue
            w.tell(who, felt)
            note = "".join(f", read '{a}' as '{b}'" for a, b in heard["fixed"].items())
            if heard["new"]:
                note += ", still learning: " + " ".join(heard["new"])
            print(f"  heard: {names[who]} feels {', '.join(felt) or 'nothing I know a word for yet'}{note}")
            # the library keeps learning: these words, next to how that person is
            lex.learn(line, felt_by(w.people[who], random.Random(w.hour)))
            lex.save(lex_path)
            print(report(w, brain)[0])
        for _ in range(hours):
            w.step(*brain.choose(w.recall()))
            if any(c == HURT for c in condition(w)) and hours > 1:
                print("  (stopped early: someone is hurt)")
                break


if __name__ == "__main__":
    main()
