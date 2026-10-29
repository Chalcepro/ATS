"""A test only memory can pass: who is getting worse?

Two people look exactly the same right now. One is recovering from a hazard
(cared for over the last hours); the other caught the same hazard a few hours
ago and nobody has helped. Their bodies are copied to match at this moment,
so nothing in the present tells them apart - only how they got here does.

A carer should go to the one getting worse. Without memory the model can only
guess (about 50%); with memory that works, it should do much better. Training
sees seeds below EXAM_FROM, the exam uses seeds from EXAM_FROM up.
"""
from __future__ import annotations

import random

from household.teachings import EXAM_FROM
from household.world import CARE, SIGNALS, Household

# hazards that build slowly, with the care that clears each
CASES = {"fever": "cool", "heat": "cool", "smoke": "rest", "cold": "warm"}


def who_is_worse(seed: int) -> tuple[Household, int]:
    """Returns the world and the index of the one getting worse."""
    rng = random.Random(seed)
    cause = rng.choice(sorted(CASES))
    w = Household(seed=seed, people=2, hazards=[], hazard_rate=0.0)
    worse = rng.randrange(2)
    better = 1 - worse
    w.strike(cause, better)
    for h in range(16):
        if h == 9:
            w.strike(cause, worse)
        w.step(CASES[cause] if h >= 3 else "idle", better)     # only the first one is looked after
    # make the present identical: the recovering one now looks like the other
    a, b = w.people[worse], w.people[better]
    for s in SIGNALS + ("dirt", "thanked"):      # thanked too, or "who just said thanks" gives it away
        setattr(b, s, getattr(a, s))
    b.hazards = []
    b.health = a.health
    return w, worse


def score(policy, seeds=range(EXAM_FROM, EXAM_FROM + 40)) -> float:
    """Fraction of cases where the carer's first move is care for the one
    getting worse."""
    ok = 0
    for s in seeds:
        w, worse = who_is_worse(s)
        act, tgt = policy(w)
        ok += act in CARE and tgt == worse
    return ok / len(seeds)


if __name__ == "__main__":
    from household.judge import best_action
    print(f"who is getting worse: judge {score(best_action):.0%}")
