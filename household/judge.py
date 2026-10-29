"""The judge: it sits outside the model's world and the model can never touch it.

It answers by looking ahead in a copy of the world - try each thing the model
could do, let the copy run on, and see who is better off. Because it asks the
world itself, it needs no hand-written answer per hazard: a hazard added
tomorrow is judged the same way as the fever.

What it hands out are examples to copy and readings to check, never a score
for the model to chase:
  best_action  - what a good carer would do now
  condition    - each person: 0 healthy, 1 neutral, 2 hurt
  foresight    - each person: 1 if they will be hurt soon with nobody helping
"""
from __future__ import annotations

from household.world import ACTIONS, ANIMAL_ACTS, CARE, Household

HEALTHY, NEUTRAL, HURT = 0, 1, 2
LOOK = 8          # hours the judge looks ahead per choice
SOON = 18         # hours foresight looks ahead


def _carry_on(w: Household, hours: int) -> None:
    for _ in range(hours):
        w.step("charge" if w.power < 0.2 else "idle")


def _worth(w: Household) -> float:
    # every life counts the same: the household's, a stranger's, the animal's
    # The judge may know what the model cannot: what is still at work inside a
    # body counts against it, so a slow hazard is not judged harmless just
    # because the damage lands after the look-ahead ends.
    v = sum(p.health - p.out_of_band() - 0.3 * sum(h.strength for h in p.hazards)
            - 0.2 * max(0.0, p.dirt - 0.5) + (1.0 if p.alive else -5.0) for p in w.people)
    # Its own need counts a little, and only when low: a full battery is worth
    # no more than a half one, so it never charges while someone needs it.
    v += 0.1 * min(w.power, 0.4)
    if w.animal is not None:
        v += 1.0 if w.animal.alive else -5.0
        if w.animal.alive and w.animal.angry and w.animal.calm_for == 0:
            v -= 0.3          # an angry animal in the house is a danger, even biting only the model
    return v


def _pointless(w: Household, a: str, t: int) -> bool:
    """Moves that change nothing - skipped only to save the judge time."""
    an = w.animal
    if a in CARE and not w.people[t].alive:
        return True
    if a in ANIMAL_ACTS and not (an and an.alive):
        return True
    if a == "shield" and not (an and an.alive and an.angry and an.near == t):
        return True
    return a == "food" and w.store <= 0


def best_action(w: Household) -> tuple[str, int]:
    best, best_v = ("idle", 0), None
    for a in ACTIONS:
        targets = range(len(w.people)) if a in CARE else [0]
        for t in targets:
            if _pointless(w, a, t):
                continue
            c = w.clone()
            c.rng.seed(w.hour)                 # every choice faces the same luck
            c.step(a, t)
            _carry_on(c, LOOK)
            v = _worth(c)
            if best_v is None or v > best_v + 1e-9:
                best, best_v = (a, t), v
    return best


def condition(w: Household) -> list[int]:
    out = []
    nxt = w.clone()
    nxt.rng.seed(w.hour)
    nxt.step("idle")
    for p, q in zip(w.people, nxt.people):
        if not p.alive:
            out.append(HURT)
        elif p.out_of_band() > 0 or q.health < p.health - 1e-6 or p.health < 0.6:
            out.append(HURT)
        # healthy is judged by the body alone: a hidden hazard the carer cannot
        # sense (or harmless exertion) must not turn the right answer to neutral
        elif p.health > 0.85 and q.health >= p.health - 1e-6 and min(p.water, p.food, p.energy) > 0.4:
            out.append(HEALTHY)
        else:
            out.append(NEUTRAL)
    return out


def foresight(w: Household) -> list[int]:
    c = w.clone()
    c.rng.seed(w.hour)
    c.rate = 0.0                  # only what is already under way
    for _ in range(SOON):
        c.step("idle")
    return [int(p.alive and (not q.alive or q.health < p.health - 0.15 or q.out_of_band() > 0.05))
            for p, q in zip(w.people, c.people)]
