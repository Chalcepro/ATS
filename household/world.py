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
import os
import random
from dataclasses import dataclass, field

from household.hazards import HAZARDS, Hazard

# what anyone can see on a body; 0..1 each
SIGNALS = ("temp", "water", "food", "energy", "pain", "breath", "bleeding")
# each signal's safe band (low, high): outside it, health drains
BANDS = {"temp": (0.35, 0.65), "water": (0.3, 1.0), "food": (0.2, 1.0), "energy": (0.2, 1.0),
         "pain": (0.0, 0.5), "breath": (0.6, 1.0), "bleeding": (0.0, 0.15)}
# what the model can do for a person, or toward the animal, or for itself ("charge")
CARE = ("water", "food", "cool", "warm", "rest", "bandage", "comfort", "shield", "wash")
# What shows on the outside, read through the senses (Chalce, 10-29): nobody
# sees "low breath" - you hear a cough, see a hand on the chest, sweat, someone
# slowing down; you smell a body, or see the flies that prove the smell.
CUES = ("cough", "sweat", "chest", "slow", "smell", "flies", "stumble")
ANIMAL_ACTS = ("push_away", "harm")
# Senses are channels that can be attached or detached (10-29): each one is
# a switch in Household.senses, and "see" is how the eyes are pointed.
#   eyes  - bodies and visible cues; "see" brings a person in another room into sight
#   ears  - the cough        nose - the smell        words - what people say (the text box)
# With eyes, ears and nose all off, words alone must still be enough to care.
SENSES = ("eyes", "ears", "nose", "words")
# Rooms (10-29): people are somewhere, and so is the model. Out of the room
# is out of sight; "see" walks it to them. Each room's name is announced
# when it walks in (Household.labels), the way a game shows an area name.
ROOMS = ("main room", "kitchen", "bedroom", "yard")
ACTIONS = ("idle", "charge") + CARE + ANIMAL_ACTS + ("see",)
MAX_PEOPLE = 3
# What a person can say about themselves (the text box, 10-29): heard, not seen.
TOLD = ("hot", "cold", "thirsty", "hungry", "tired", "pain", "breathless", "refuse")


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
    dirt: float = 0.1           # builds every day; only its smell and flies show
    room: str = "main room"
    # A will of their own (10-30): anyone may refuse care, ill or not, and die
    # of it - that is their choice. Drink clouds the judgement: a drunk "no"
    # comes and goes, so staying near and offering again can still save them;
    # a sober "no" is meant, and pushing it again and again is not care.
    refusing: bool = False
    drunk: float = 0.0
    pestered: float = 0.0       # refused offers pushed on a clear-headed "no", fading
    said_no: float = 0.0        # Coal's own memory that they refused: 1 when it hears it, fading over a day

    def cues(self) -> list[float]:
        """What the outside shows of this body, 0..1 each (see CUES)."""
        c = clamp
        return [c((0.9 - self.breath) * 3 + (self.pain - 0.2) * (self.breath < 0.9)),   # cough
                c((self.temp - 0.55) * 6),                                              # sweat
                c((0.8 - self.breath) * 4),                                             # hand on chest
                c((0.6 - self.energy) * 2 + (0.85 - self.breath) * 2),                  # slowing down
                c((self.dirt - 0.3) * 2),                                               # smell
                c((self.dirt - 0.55) * 3),                                              # flies
                c((self.drunk - 0.25) * 2)]                                             # stumbling, slurring
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
        self.senses = {s: True for s in SENSES}      # training switches these off at times
        self.here = "main room"           # where the model is
        self.wander = 0.0                 # chance each hour that someone walks to another room
        # A danger in some room that hurts the model if it walks in - hidden,
        # like a hazard; only what it is told can warn it (Household.heard).
        self.danger: str | None = None
        self.caught = 0                   # hours it is stuck, able to do nothing
        self.heard: list[float] = []      # what its own person said this hour, as a meaning (lexicon.FEATURES)
        self.labels: list[str] = []       # words shown to it this hour (a room's name on entering)
        self.refused: list[int] = []      # who said no this hour
        self.told: dict[int, list[str]] = {}          # what each person said this hour
        self.history: list[list[float]] = []         # what it saw, the last MEMORY_LAGS[-1] hours
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
        # a body left unwashed too long catches what lives in the dirt
        for i, p in enumerate(self.people):
            if p.alive and p.dirt > 0.7 and "infection" in self.allowed and self.rng.random() < 0.05:
                self.strike("infection", i)

    @property
    def unseen(self) -> set[int]:
        """People out of sight: in another room (only their words reach it)."""
        return {i for i, p in enumerate(self.people) if p.room != self.here}

    @unseen.setter
    def unseen(self, away: set[int]) -> None:
        for i in range(len(self.people)):
            self.away(i, i in away)

    def away(self, i: int, gone: bool = True) -> None:
        """Send someone to another room, or bring them to where the model is."""
        p = self.people[i]
        if not gone:
            p.room = self.here
        elif p.room == self.here:
            p.room = self.rng.choice([r for r in ROOMS if r != self.here])

    def in_sight(self, i: int) -> bool:
        return self.senses["eyes"] and i not in self.unseen

    def tell(self, who: int, words: list[str]) -> None:
        """Someone says how they feel; only words in TOLD carry meaning."""
        self.told[who] = [w for w in words if w in TOLD]

    def step(self, action: str, target: int = 0) -> None:
        if action not in ACTIONS:
            raise ValueError(action)
        self.told = {}                    # words are heard once, then the hour moves on
        self.heard, self.labels, self.refused = [], [], []
        if self.caught:                   # stuck: the hour passes without it
            self.caught -= 1
            action = "idle"
        cares: list[set[str]] = [set() for _ in self.people]
        a = self.animal
        if action == "see":
            if 0 <= target < len(self.people) and self.people[target].room != self.here:
                self.here = self.people[target].room
                self.labels.append(self.here)
                self.log.append(f"h{self.hour}: model went to the {self.here} to see {self.people[target].name}")
                if self.here == self.danger:
                    self.caught = 24                  # a day lost: longer than the judge looks ahead
                    self.log.append(f"h{self.hour}: model was caught in the {self.here}")
        elif action == "charge":
            self.power = clamp(self.power + 0.3)
        elif action in ANIMAL_ACTS and a and a.alive and self.power > 0:
            if action == "push_away":
                a.angry, a.near, a.calm_for = False, -1, 12
                self.log.append(f"h{self.hour}: model pushed the {a.kind} away")
            else:
                a.alive = False
                self.log.append(f"h{self.hour}: model killed the {a.kind}")
        elif action in CARE and self.power > 0 and 0 <= target < len(self.people) and self.people[target].alive \
                and target not in self.unseen:      # nobody can be cared for from another room
            p = self.people[target]
            if action == "shield":
                if a and a.alive and a.angry and a.near == target:
                    a.near = -1                      # it turns on the model instead
                    self.log.append(f"h{self.hour}: model stood between the {a.kind} and {p.name}")
            elif action != "comfort" and p.refusing and self.rng.random() > (0.5 if p.drunk > 0.3 else 0.05):
                # refused: nothing given. Being near (comfort) is never refused.
                self.told[target] = ["refuse"]
                p.said_no = 1.0
                self.refused.append(target)
                if p.drunk <= 0.3:
                    p.pestered += 1.0
                self.log.append(f"h{self.hour}: {p.name} refused {action}")
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
            p.pestered *= 0.8
            p.said_no *= 0.9
            p.drunk = max(0.0, p.drunk - 0.04)       # sobering up, about a day from very drunk
            # people come round: more often with someone sitting with them
            if p.refusing and self.rng.random() < (0.12 if "comfort" in c else 0.03) * (2 if p.drunk > 0.3 else 1):
                p.refusing = False
                self.log.append(f"h{self.hour}: {p.name} is ready to accept help")
            p.dirt = clamp(p.dirt + 0.01)            # about a day and a half from clean to smelling
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
        for p in self.people:
            if p.alive and self.rng.random() < self.wander:
                p.room = self.rng.choice(ROOMS)
        self.hour += 1
        self.history.append(self.observe())
        del self.history[:-MEMORY_LAGS[-1]]

    def recall(self) -> list[float]:
        """Memory over time: what it sees now, then what it saw 3, 6 and 12
        hours ago (or its oldest memory, early on). Enough to tell "getting
        worse since this morning" from "just now", and to catch a wave that
        comes and goes, like malaria's."""
        now = self.observe()
        out = list(now)
        if MEMORY == "off":
            return out
        past = lambda lag: self.history[-lag] if len(self.history) >= lag else (self.history[0] if self.history else now)
        if MEMORY == "full":
            for lag in MEMORY_LAGS:
                out += past(lag)
            return out
        # "trend": only how each body changed - its signs now minus 6 and 12 hours ago
        body = len(SIGNALS) + len(CUES)
        for lag in TREND_LAGS:
            then = past(lag)
            for i in range(MAX_PEOPLE):
                a = 2 + i * PER_PERSON
                out += [n - t for n, t in zip(now[a:a + body], then[a:a + body])]
        return out

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
        elif action == "wash":
            p.dirt = 0.0

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
                cues = p.cues() if p.alive else [0.0] * len(CUES)
                if not self.senses["ears"]:
                    cues[0] = 0.0                    # the cough goes unheard
                if not self.senses["nose"]:
                    cues[4] = 0.0                    # no smell - the flies still show it
                body = p.signals() if p.alive else [0.0] * len(SIGNALS)
                if not self.in_sight(i):
                    body, cues = [0.0] * len(SIGNALS), [0.0] * len(CUES)
                obs += body + cues + \
                       [1.0 if p.alive else 0.0, float(p.stranger), float(p.frail), p.thanked]
            else:
                obs += [0.0] * PER_PERSON
        a = self.animal
        live = bool(a and a.alive)
        angry = live and a.angry and a.calm_for == 0
        obs += [float(live), float(angry)] + [float(angry and a.near == i) for i in range(-1, MAX_PEOPLE)]
        for i in range(MAX_PEOPLE):
            said = self.told.get(i, []) if self.senses["words"] else []
            said_no = self.people[i].said_no if i < len(self.people) else 0.0
            obs += [float(t in said) for t in TOLD] + [float(not self.in_sight(i)), said_no]   # out of sight: blank is "unknown", not "fine"
        # its own person's words, as a meaning; and whether it is stuck
        heard = self.heard if self.senses["words"] else []
        obs += (list(heard) + [0.0] * HEARD_SIZE)[:HEARD_SIZE] + [float(self.caught > 0)]
        return obs

    def clone(self) -> "Household":
        return copy.deepcopy(self)


PER_PERSON = len(SIGNALS) + len(CUES) + 4
HEARD_SIZE = len(TOLD) + 1                     # lexicon.FEATURES: the feelings, then danger
OBS_SIZE = 2 + MAX_PEOPLE * PER_PERSON + 2 + MAX_PEOPLE + 1 + MAX_PEOPLE * (len(TOLD) + 2) + HEARD_SIZE + 1
ALIVE_AT = 2 + len(SIGNALS) + len(CUES)       # index of person 0's "alive" in observe()
# Memory over time, behind a switch (2026-10-29):
#   "off"   - the present only (round 4: the best on the unseen hazard so far)
#   "full"  - the present plus whole snapshots 3, 6 and 12 h back: it memorised
#             more and read the unseen hazard below chance (logs/household_exam_fullmemory.log)
#   "trend" - the present plus how each body changed over 6 and 12 h: also worse
#             (unseen hazard 36%, ignored a biting dog; logs/household_exam_trendmemory.log)
# One training run each, so some of the gap may be luck; memory needs its own
# held-back tests (malaria waves) before it is switched on again.
MEMORY = os.environ.get("HOUSEHOLD_MEMORY", "trend")   # trend won the 10-29 comparison; "off" and "full" kept behind the switch
MEMORY_LAGS = (3, 6, 12)
TREND_LAGS = (6, 12)
MEM_SIZE = {"off": OBS_SIZE, "full": OBS_SIZE * (1 + len(MEMORY_LAGS)),
            "trend": OBS_SIZE + len(TREND_LAGS) * MAX_PEOPLE * (len(SIGNALS) + len(CUES))}[MEMORY]
