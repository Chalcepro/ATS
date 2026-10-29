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


HAZARDS: dict[str, type[Hazard]] = {h.name: h for h in (Heat, Cold, Cut, Fever, Venom)}
