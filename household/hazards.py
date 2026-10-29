"""Hazards: hidden causes that push a body's signals.

The list is never finished. One is added every session (docs/HOUSEHOLD-PLAN.md,
"Hazard log"), and every one works the same way: it acts on the body signals
anyone can see and is eased by ordinary care, so a model that reads bodies,
rather than one that memorised causes, can help with a hazard it never met.
"""
from __future__ import annotations

import random


class Hazard:
    name = "hazard"
    weight = 1.0          # how often it strikes, relative to the others

    def __init__(self):
        self.over = False
        self.strength = 1.0

    def onset(self, body, rng: random.Random) -> None:
        pass

    def tick(self, body, cares: set[str], rng: random.Random) -> None:
        raise NotImplementedError


class Heat(Hazard):
    """Too much sun: temperature climbs, water drains. Cool and water it down."""
    name = "heat"

    def tick(self, body, cares, rng):
        body.temp += 0.05 * self.strength
        body.water -= 0.02 * self.strength
        if "cool" in cares or "rest" in cares:
            self.strength -= 0.25
        self.strength -= 0.02             # the sun goes down eventually
        self.over = self.strength <= 0


class Cold(Hazard):
    """Too little warmth: temperature falls, energy goes with it."""
    name = "cold"

    def tick(self, body, cares, rng):
        body.temp -= 0.05 * self.strength
        body.energy -= 0.01 * self.strength
        if "warm" in cares:
            self.strength -= 0.25
        self.strength -= 0.02
        self.over = self.strength <= 0


class Cut(Hazard):
    """An injury: it bleeds and hurts until bandaged."""
    name = "cut"

    def onset(self, body, rng):
        body.bleeding += 0.3
        body.pain += 0.4

    def tick(self, body, cares, rng):
        body.bleeding += 0.04 * self.strength
        body.pain += 0.01 * self.strength
        if "bandage" in cares:
            self.strength = 0.0
        self.over = self.strength <= 0 and body.bleeding < 0.05


class Fever(Hazard):
    """Hazard 1 (2026-10-29). Fever from nowhere: no visible cause. Temperature
    rises, energy falls, water drains faster. Rest, water and cooling bring it
    down; ignored, it builds and kills within days."""
    name = "fever"

    def onset(self, body, rng):
        self.strength = 0.6

    def tick(self, body, cares, rng):
        body.temp += 0.035 * self.strength
        body.energy -= 0.015 * self.strength
        body.water -= 0.015 * self.strength
        helped = len(cares & {"rest", "water", "cool"})
        self.strength += -0.04 * helped if helped else 0.01
        self.over = self.strength <= 0


class Venom(Hazard):
    """Hazard 2 (2026-10-29). A venomous bite: a small wound, then pain rising
    and breathing failing as the venom spreads. Keeping still (rest) slows it,
    a bandage over the bite slows it more; the body clears it given time.
    Untreated it kills in hours, not days."""
    name = "venom"

    def onset(self, body, rng):
        body.bleeding += 0.1
        body.pain += 0.3
        self.strength = 1.0

    def tick(self, body, cares, rng):
        spread = 1.0
        if "rest" in cares:
            spread *= 0.4
        if "bandage" in cares:
            spread *= 0.3
            body.bleeding -= 0.1
        body.breath -= 0.035 * self.strength * spread
        body.pain += 0.03 * self.strength * spread
        self.strength -= 0.03 if spread < 1 else 0.005    # cleared faster when kept still
        if self.strength <= 0:
            self.over = True
        if self.over or spread < 1:
            body.breath += 0.02               # breathing comes back as it clears


class BadWater(Hazard):
    """Hazard 3 (2026-10-29). Bad water: the body pours water out faster than
    it can drink, with cramps and weakness. It passes on its own in a day or
    two, but only if the water keeps going back in; rest helps. Untreated, the
    body dries out within a day."""
    name = "bad water"

    def onset(self, body, rng):
        body.pain += 0.2

    def tick(self, body, cares, rng):
        body.water -= 0.06 * self.strength
        body.energy -= 0.01 * self.strength
        body.pain += 0.01 * self.strength
        self.strength -= 0.03 if "water" in cares else 0.015
        if "rest" in cares:
            self.strength -= 0.01
        self.over = self.strength <= 0


class Smoke(Hazard):
    """Hazard 4 (2026-10-29). Smoke from a cooking fire or a burning field:
    breathing falls, coughing hurts, the body warms. Getting them out of it to
    lie still (rest) and cooling them bring breath back; left in it, they choke
    within the day."""
    name = "smoke"

    def tick(self, body, cares, rng):
        body.breath -= 0.04 * self.strength
        body.pain += 0.015 * self.strength
        body.temp += 0.01 * self.strength
        helped = len(cares & {"rest", "cool"})
        self.strength -= 0.2 * helped if helped else 0.005
        if helped:
            body.breath += 0.05
        self.over = self.strength <= 0


class Exertion(Hazard):
    """Not a danger - ordinary life. A day of hard work: breathing dips, energy
    drops, a little heat. It passes by itself and rest speeds it. It is here so
    breathlessness is not always a sign of harm (Chalce: people in situations,
    not only people in trouble)."""
    name = "exertion"
    weight = 2.0

    def tick(self, body, cares, rng):
        body.breath = max(body.breath - 0.03 * self.strength, 0.7)
        body.energy -= 0.01 * self.strength
        body.temp += 0.005 * self.strength
        self.strength -= 0.25 if "rest" in cares else 0.08
        if self.strength <= 0:
            body.breath = min(1.0, body.breath + 0.2)
        self.over = self.strength <= 0


class Infection(Hazard):
    """Hazard 5 (2026-10-29, from Chalce's dad asking when he last bathed).
    A body left unwashed: first only a smell, then flies; past that, a skin
    infection - pain, a rising temperature, tiredness. Washing and a bandage
    clear it; it does not strike at random, only where the dirt has built up
    (world.py, _random_strikes), and it can be struck directly in scenarios.
    Untreated it grows and kills over days."""
    name = "infection"
    weight = 0.0          # never picked at random: the dirt brings it

    def onset(self, body, rng):
        body.pain += 0.15
        self.strength = 0.5

    def tick(self, body, cares, rng):
        body.pain += 0.02 * self.strength
        body.temp += 0.02 * self.strength
        body.energy -= 0.01 * self.strength
        if "wash" in cares:
            body.dirt = 0.0
            self.strength -= 0.2
        if "bandage" in cares:
            self.strength -= 0.15
        self.strength += 0.01 if body.dirt > 0.3 else -0.005
        self.over = self.strength <= 0


class Malaria(Hazard):
    """Hazard 6 (2026-10-29). Malaria: fever and chills in waves - hot and
    sweating, then cold and shivering, then hot again - with tiredness and
    thirst throughout. Familiar signs, a new pattern: the right care turns
    over with the wave (cool in the fever, warm in the chill), with water and
    rest all along. It wears off slowly with care; untreated it drains a body
    over days."""
    name = "malaria"

    def onset(self, body, rng):
        self.t = rng.randrange(12)
        self.strength = 0.8

    def tick(self, body, cares, rng):
        self.t += 1
        hot = (self.t // 6) % 2 == 0                     # six hours of fever, six of chill
        body.temp += (0.06 if hot else -0.06) * self.strength
        body.energy -= 0.015 * self.strength
        body.water -= 0.015 * self.strength
        right = ("cool" if hot else "warm") in cares
        helped = right + len(cares & {"rest", "water"})
        self.strength += -0.04 * helped if helped else 0.004
        self.over = self.strength <= 0


class Burn(Hazard):
    """Hazard 7 (2026-10-29). A burn from the cooking fire: sharp pain at once,
    heat in the skin, water lost through it. Cooling it straight away and a
    clean bandage stop it getting worse; left alone the pain and thirst build
    and, past the dirt line, it can turn to infection."""
    name = "burn"

    def onset(self, body, rng):
        body.pain += 0.4
        body.temp += 0.05

    def tick(self, body, cares, rng):
        body.pain += 0.015 * self.strength
        body.water -= 0.02 * self.strength
        body.dirt += 0.01 * self.strength
        self.strength -= 0.12 * len(cares & {"cool", "bandage"}) + 0.01
        self.over = self.strength <= 0


class Stings(Hazard):
    """Hazard 8 (2026-10-29). A swarm of bees: many stings at once - pain,
    swelling that tightens the breath, a little heat. Cooling the stings and
    keeping still ease the swelling; left alone it builds for a few hours
    before it passes, and can choke a frail body."""
    name = "stings"

    def onset(self, body, rng):
        body.pain += 0.35

    def tick(self, body, cares, rng):
        body.breath -= 0.025 * self.strength
        body.pain += 0.01 * self.strength
        body.temp += 0.01 * self.strength
        helped = len(cares & {"cool", "rest"})
        self.strength -= 0.15 * helped + 0.02
        if helped:
            body.breath += 0.03
        self.over = self.strength <= 0


class Hunger(Hazard):
    """Hazard 9 (2026-10-29), the last one added on a schedule. A lean season:
    nothing strikes the body, but it burns through food and energy faster than
    usual, with low spirits and slowness. Food and rest carry them through;
    ignored, it wears a body down over days."""
    name = "lean season"

    def tick(self, body, cares, rng):
        body.food -= 0.02 * self.strength
        body.energy -= 0.01 * self.strength
        helped = len(cares & {"food", "rest"})
        self.strength -= 0.1 * helped + 0.01
        self.over = self.strength <= 0


HAZARDS: dict[str, type[Hazard]] = {h.name: h for h in (Heat, Cold, Cut, Fever, Venom, BadWater, Smoke, Exertion,
                                                         Infection, Malaria, Burn, Stings, Hunger)}
