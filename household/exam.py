"""Exams. Some hazards are held back: the model never sees them while it
learns, and they are only ever used here. Passing on a held-back hazard means
it read the body; failing means it memorised the causes it was shown.

Run: python -m household.exam            (an untrained model, as a baseline)
"""
from __future__ import annotations

import torch

from household.judge import best_action, condition, foresight
from household.world import CARE, Household

HELD_BACK = ["bad water"]                  # never in training data: familiar signs, new cause


def lives(policy, hazards, seeds=range(100, 140), people=1, days=3, partners=None) -> float:
    """Fraction of people alive after `days` with this carer. With partners,
    each person also gets one of those at the same time, so no single repeated
    action can carry them through."""
    alive = total = 0
    for s in seeds:
        w = Household(seed=s, people=people, hazards=[])
        for i in range(people):
            w.strike(hazards[(s + i) % len(hazards)], i)
            if partners:
                w.strike(partners[(s * 3 + i) % len(partners)], i)
        w.allowed, w.rate = list(hazards), 0.012
        for _ in range(24 * days):
            w.step(*policy(w))
        alive += sum(p.alive for p in w.people)
        total += people
    return alive / total


def readings(brain, hazards, seeds=range(200, 220), hours=48) -> dict[str, float]:
    """How well the model's condition and foresight match the judge's, as
    balanced accuracy (each answer counts equally, so always saying "fine"
    scores no better than chance). The first hours go untended so there is
    trouble to read."""
    hits: dict[tuple, list[int]] = {}
    for s in seeds:
        w = Household(seed=s, people=1, hazards=[])
        w.strike(hazards[s % len(hazards)])
        for t in range(hours):
            with torch.no_grad():
                out = brain(torch.tensor([w.recall()], dtype=torch.float32))
            c, f = condition(w)[0], foresight(w)[0]
            hits.setdefault(("condition", c), []).append(int(out["condition"][0, 0].argmax()) == c)
            hits.setdefault(("foresight", f), []).append(int(out["foresight"][0, 0] > 0) == f)
            w.step(*(best_action(w) if t >= hours // 3 else ("idle", 0)))
    score = {}
    for head in ("condition", "foresight"):
        recalls = [sum(v) / len(v) for (h, _), v in hits.items() if h == head]
        score[head] = sum(recalls) / len(recalls)
    return score


def report(brain, trained_on) -> None:
    judge = lambda w: best_action(w)
    idle = lambda w: ("charge" if w.power < 0.2 else "idle", 0)
    model = lambda w: brain.choose(w.recall())
    dangers = [h for h in trained_on if h != "exertion"]
    for label, hz, partners in (("trained hazards", trained_on, None), ("HELD BACK + one", HELD_BACK, dangers)):
        # the bar to beat: the best carer that only ever does one thing (and charges)
        trick, trick_v = max(((a, lives(lambda w, a=a: ("charge" if w.power < 0.2 else a, 0), hz, partners=partners))
                              for a in CARE), key=lambda x: x[1])
        print(f"{label:16} alive after 3 days:  judge {lives(judge, hz, partners=partners):.0%}"
              f"   model {lives(model, hz, partners=partners):.0%}"
              f"   one trick ({trick}) {trick_v:.0%}   nobody {lives(idle, hz, partners=partners):.0%}")
        r = readings(brain, hz)
        print(f"{'':16} model reads condition {r['condition']:.0%}, foresight {r['foresight']:.0%}")
    from household.memory_test import score as memory_score
    print(f"who is getting worse (needs memory): model {memory_score(model):.0%}   guessing 50%")
    from household.teachings import score
    print("teachings, exam seeds only:")
    for name, v in score(model, lambda w: brain.say(w.recall(), len(w.people))).items():
        print(f"  {name:18} model {v:.0%}")


if __name__ == "__main__":
    import os
    import sys
    from household.hazards import HAZARDS
    from household.model import HouseholdBrain
    from household.train import SAVE
    torch.manual_seed(0)
    brain = HouseholdBrain()
    if "--blank" not in sys.argv and os.path.exists(SAVE):
        brain.load_state_dict(torch.load(SAVE))
        from household.world import MEMORY
        print("trained brain:", SAVE, "  memory:", MEMORY)
    else:
        print("blank brain")
    report(brain, [h for h in HAZARDS if h not in HELD_BACK])
