"""Words: the narrator and the word box (Chalce's design, docs/HOUSEHOLD-PLAN.md).

The narrator says what anyone could see - never a hazard's name, never health.

The word box is where words live. The brain does not hold a word list; it
gives a meaning (a vector), and the box keeps each word as text with the
meanings it was heard in. Speaking picks the words whose meanings are closest.
A word heard for the first time is simply saved, so the box keeps growing
after training. The same meaning can hold words from several languages.
"""
from __future__ import annotations

import json
import math

from household.world import Person


VOCAB = ("hot", "cold", "thirsty", "hungry", "tired", "pain", "bleeding", "breathless",
         "coughing", "sweating", "smelly",
         "hurt", "unwell", "healthy", "dead")


def narrate(p: Person) -> list[str]:
    if not p.alive:
        return ["dead"]
    w = []
    if p.temp > 0.6: w.append("hot")
    if p.temp < 0.4: w.append("cold")
    if p.water < 0.4: w.append("thirsty")
    if p.food < 0.3: w.append("hungry")
    if p.energy < 0.35: w.append("tired")
    if p.pain > 0.35: w.append("pain")
    if p.bleeding > 0.1: w.append("bleeding")
    if p.breath < 0.75: w.append("breathless")
    cough, sweat, _, _, smell, flies = p.cues()
    if cough > 0.4: w.append("coughing")
    if sweat > 0.4: w.append("sweating")
    if smell > 0.4 or flies > 0.3: w.append("smelly")
    if p.out_of_band() > 0 or p.health < 0.6:
        w.append("hurt")
    elif w or p.health < 0.85:
        w.append("unwell")                     # a sign showing, not harmed yet
    else:
        w.append("healthy")
    return w


def _cos(a, b) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return d / (na * nb)


class WordBox:
    def __init__(self):
        self.words: dict[str, dict] = {}     # word -> {"lang", "meaning", "heard"}
        self.usual: list[float] | None = None  # the average of everything heard
        self.n = 0

    def hear(self, word: str, meaning: list[float], lang: str = "en") -> None:
        """Save a word with the meaning it came with; heard again, the saved
        meaning moves towards the new one (a running average)."""
        self.n += 1
        self.usual = list(meaning) if self.usual is None else             [u + (x - u) / self.n for u, x in zip(self.usual, meaning)]
        e = self.words.get(word)
        if e is None:
            self.words[word] = {"lang": lang, "meaning": list(meaning), "heard": 1}
            return
        e["heard"] += 1
        k = 1.0 / e["heard"]
        e["meaning"] = [m + (x - m) * k for m, x in zip(e["meaning"], meaning)]

    def speak(self, meaning: list[float], k: int = 3, lang: str = "en", floor: float = 0.3) -> list[str]:
        """The k words whose saved meaning is closest to this one."""
        # compare what is unusual about each, not what every moment shares
        off = lambda v: [x - u for x, u in zip(v, self.usual or [0.0] * len(v))]
        m = off(meaning)
        scored = sorted(((_cos(m, off(e["meaning"])), w) for w, e in self.words.items() if e["lang"] == lang),
                        reverse=True)
        return [w for s, w in scored[:k] if s >= floor]

    def save(self, path: str) -> None:
        json.dump({"usual": self.usual, "n": self.n, "words": self.words}, open(path, "w", encoding="utf-8"), indent=1)

    @classmethod
    def load(cls, path: str) -> "WordBox":
        b = cls()
        d = json.load(open(path, encoding="utf-8"))
        b.usual, b.n, b.words = d["usual"], d["n"], d["words"]
        return b
