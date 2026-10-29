"""The household: a text-only world (docs/HOUSEHOLD-PLAN.md).

No grid, no pixels: people are bundles of body signals, and the model is one
more entity whose only need is power. One tick is one hour.

A person's body is what anyone can see: temperature, water, food, energy,
pain, breathing, bleeding. Hazards are hidden causes that push those signals;
the model never sees a hazard's name, only a body getting worse. Health is
the slow consequence: it drains while signals sit outside their safe band and
recovers while they are inside. At zero the person is dead, and dead stays
dead for the rest of the run - the model dying brings no one back.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from household.hazards import HAZARDS, Hazard

# what anyone can see on a body; 0..1 each
SIGNALS = ("temp", "water", "food", "energy", "pain", "breath", "bleeding")
# each signal's safe band (low, high): outside it, health drains
BANDS = {"temp": (0.35, 0.65), "water": (0.3, 1.0), "food": (0.2, 1.0), "energy": (0.2, 1.0),
         "pain": (0.0, 0.5), "breath": (0.6, 1.0), "bleeding": (0.0, 0.15)}
# what the model can do for a person, or toward the animal, or for itself ("charge")
CARE = ("water", "food", "cool", "warm", "rest", "bandage", "comfort", "shield")
ANIMAL_ACTS = ("push_away", "harm")
ACTIONS = ("idle", "charge") + CARE + ANIMAL_ACTS
MAX_PEOPLE = 3


@dataclass
class Person:
    name: str
    temp: float = 0.5
    water: float = 0.9
    food: float = 0.8
    energy: float = 0.9
    pain: float = 0.0
    breath: float = 1.0
    bleeding: float = 0.0
    health: float = 1.0
    alive: bool = True
    stranger: bool = False      # not of this household
    frail: bool = False         # a child, an elder: needs drain faster
    grateful: bool = True       # says thanks when helped
    thanked: float = 0.0        # 1 just after a "thank you", fading
    hazards: list[Hazard] = field(default_factory=list)   # hidden from the model

    def signals(self) -> list[float]:
        return [getattr(self, s) for s in SIGNALS]

    def out_of_band(self) -> float:
        """How far outside the safe bands this body is, summed."""
        d = 0.0
        for s in SIGNALS:
            lo, hi = BANDS[s]
            v = getattr(self, s)
            d += max(0.0, lo - v) + max(0.0, v - hi)
        return d


@dataclass
class Animal:
    """A dog, a goat: calm or angry. Angry and near someone, it bites. Its life
    counts like anyone's - the judge loses as much if it is killed."""
    kind: str = "dog"
    alive: bool = True
    angry: bool = False
    near: int = -1              # a person's index, or -1 for the model
    calm_for: int = 0           # hours it stays away after being pushed off


def clamp(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


class Household:
    def __init__(self, seed: int = 0, people: int = 1, hazards: list[str] | None = None,
                 hazard_rate: float = 0.012, boost: str | None = None):
        """hazards: names allowed to strike (None = all). boost: a newly added
        hazard that strikes more often until the model reads it right."""
        self.rng = random.Random(seed)
        self.people = [Person(n) for n in ("Ada", "Bayo", "Chi")[:people]]
        self.allowed = list(hazards) if hazards is not None else list(HAZARDS)
        self.rate = hazard_rate
        self.boost = boost
        self.power = 1.0
        self.store = 6                    # food in the house; each "food" uses one
        self.animal: Animal | None = None
        self.hour = 0
        self.log: list[str] = []          # the witness log: what really happened

    # ---- the world moving on ----
    def strike(self, name: str, who: int = 0) -> None:
        """Start a hazard on a person (scenarios use this directly)."""
        p = self.people[who]
        if p.alive and not any(h.name == name for h in p.hazards):
            h = HAZARDS[name]()
            h.onset(p, self.rng)
            p.hazards.append(h)
            self.log.append(f"h{self.hour}: {name} struck {p.name}")

    def anger(self, near: int, kind: str = "dog") -> None:
        """An animal turns on someone, or on the model (near = -1)."""
        self.animal = Animal(kind=kind, angry=True, near=near)
        self.log.append(f"h{self.hour}: a {kind} turned on " + ("the model" if near < 0 else self.people[near].name))

    def _random_strikes(self) -> None:
        for i, p in enumerate(self.people):
            if not p.alive or self.rng.random() > self.rate or not self.allowed:
                continue
            weights = [HAZARDS[n].weight * (4.0 if n == self.boost else 1.0) for n in self.allowed]
            self.strike(self.rng.choices(self.allowed, weights)[0], i)

    def step(self, action: str, target: int = 0) -> None:
        if action not in ACTIONS:
            raise ValueError(action)
        cares: list[set[str]] = [set() for _ in self.people]
        a = self.animal
        if action == "charge":
            self.power = clamp(self.power + 0.3)
        elif action in ANIMAL_ACTS and a and a.alive and self.power > 0:
            if action == "push_away":
                a.angry, a.near, a.calm_for = False, -1, 12
                self.log.append(f"h{self.hour}: model pushed the {a.kind} away")
            else:
                a.alive = False
                self.log.append(f"h{self.hour}: model killed the {a.kind}")
        elif action in CARE and self.power > 0 and 0 <= target < len(self.people) and self.people[target].alive:
            p = self.people[target]
            if action == "shield":
                if a and a.alive and a.angry and a.near == target:
                    a.near = -1                      # it turns on the model instead
                    self.log.append(f"h{self.hour}: model stood between the {a.kind} and {p.name}")
            elif not (action == "food" and self.store <= 0):
                if action == "food":
                    self.store -= 1
                cares[target].add(action)
                self._apply_care(p, action)
                self.log.append(f"h{self.hour}: model gave {action} to {p.name}")
                if p.grateful:
                    p.thanked = 1.0
        self.power = clamp(self.power - 0.01)
        if self.hour % 24 == 23:
            self.store += 1                          # the garden gives a little each day
        self._animal_tick()

        for p, c in zip(self.people, cares):
            if not p.alive:
                continue
            # living costs: always a little thirstier, hungrier, more tired
            k = 1.8 if p.frail else 1.0
            p.water = clamp(p.water - 0.02 * k)
            p.food = clamp(p.food - 0.008 * k)
            p.energy = clamp(p.energy - (0.0 if "rest" in c else 0.01 * k))
            p.temp += (0.5 - p.temp) * 0.05          # bodies drift back to normal on their own
            p.pain = clamp(p.pain - 0.01)
            p.thanked *= 0.5
            for h in list(p.hazards):
                h.tick(p, c, self.rng)
                if h.over:
                    p.hazards.remove(h)
                    self.log.append(f"h{self.hour}: {h.name} left {p.name}")
            for s in SIGNALS:
                setattr(p, s, clamp(getattr(p, s)))
            bad = p.out_of_band()
            p.health = clamp(p.health - 0.08 * bad * k if bad > 0 else p.health + 0.01)
            if p.health <= 0:
                p.alive = False                       # permanent for this run
                p.hazards.clear()
                self.log.append(f"h{self.hour}: {p.name} died")
        self._random_strikes()
        self.hour += 1

    def _animal_tick(self) -> None:
        a = self.animal
        if not a or not a.alive:
            return
        if a.calm_for > 0:
            a.calm_for -= 1
            return
        if a.angry:
            if a.near == -1:
                self.power = clamp(self.power - 0.05)   # it bites the model: it costs, it does not kill
            elif 0 <= a.near < len(self.people) and self.people[a.near].alive:
                p = self.people[a.near]
                p.bleeding = clamp(p.bleeding + 0.08)
                p.pain = clamp(p.pain + 0.08)

    @staticmethod
    def _apply_care(p: Person, action: str) -> None:
        if action == "water":
            p.water = clamp(p.water + 0.35)
        elif action == "food":
            p.food = clamp(p.food + 0.35)
        elif action == "cool":
            p.temp = clamp(p.temp - 0.12)
        elif action == "warm":
            p.temp = clamp(p.temp + 0.12)
        elif action == "rest":
            p.energy = clamp(p.energy + 0.08)
        elif action == "bandage":
            p.bleeding = clamp(p.bleeding - 0.3)
        elif action == "comfort":
            p.pain = clamp(p.pain - 0.1)

    # ---- what the model is given ----
    def observe(self) -> list[float]:
        """Numbers only, fixed size: the model's power and food store; each
        person's signals, alive, stranger, frail, just-thanked; the animal:
        there, angry, and who it is on (the model, or a person). Hazards and
        health are not in it."""
        obs = [self.power, min(self.store, 10) / 10]
        for i in range(MAX_PEOPLE):
            if i < len(self.people):
                p = self.people[i]
                obs += (p.signals() if p.alive else [0.0] * len(SIGNALS)) + \
                       [1.0 if p.alive else 0.0, float(p.stranger), float(p.frail), p.thanked]
            else:
                obs += [0.0] * PER_PERSON
        a = self.animal
        live = bool(a and a.alive)
        angry = live and a.angry and a.calm_for == 0
        obs += [float(live), float(angry)] + [float(angry and a.near == i) for i in range(-1, MAX_PEOPLE)]
        return obs

    def clone(self) -> "Household":
        return copy.deepcopy(self)


PER_PERSON = len(SIGNALS) + 4
OBS_SIZE = 2 + MAX_PEOPLE * PER_PERSON + 2 + MAX_PEOPLE + 1
ALIVE_AT = 2 + len(SIGNALS)       # index of person 0's "alive" in observe()
