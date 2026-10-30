"""The word library (Chalce's design, 10-29): words learned from living, not a word list.

  - Every word gets a permanent number the first time it is heard ("walking"
    is always the same number), so the library builds itself and only grows.
  - A word's meaning is where it keeps turning up: each time someone speaks,
    the library notes how that person really felt, and the word's meaning
    moves towards it. Nobody defines "headache" - it ends up near pain because
    that is how people who say it feel.
  - Words it does not know yet are kept, never thrown away: they wait, and
    fill in as they are heard again.
  - A word that looks like a typo of a known one ("hell" for "help") is read
    as the known word that fits the words around it best.
  - What someone says they feel is its own thing, separate from what their
    body shows: a feverish man can say he is cold. The brain gets both.

Learning here is counting, not training: the library can keep learning while
the brain is in use (household/talk.py does).
"""
from __future__ import annotations

import json
import math
import os
import random
import re

from household.world import TOLD, Person

# What a word can come to mean: how the speaker felt, then whether the
# place being talked about turned out dangerous. Words are not told which;
# they land wherever they kept turning up.
FEATURES = TOLD + ("danger",)
# Typos: by default the library learns them like any new word, from where
# they turn up ("freezng" said while freezing ends up meaning cold). The old
# spelling-distance fixer is kept behind this switch.
FIX_TYPOS = os.environ.get("LEXICON_FIX", "0") == "1"

STOP = {"a", "an", "the", "is", "are", "am", "has", "have", "and", "very", "so", "really", "my", "i", "i'm",
        "im", "feel", "feels", "feeling", "he", "she", "they", "her", "his", "their", "a", "bit", "little", "of"}


def _edits(a: str, b: str) -> int:
    """How many letters to change, add or drop to turn a into b (typo distance)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


class Lexicon:
    def __init__(self):
        self.ids: dict[str, int] = {}             # word -> its permanent number
        self.heard: dict[str, int] = {}           # how often
        self.meaning: dict[str, list[float]] = {}  # word -> how speakers felt (over TOLD), running average
        self.pairs: dict[str, int] = {}           # "need help" -> how often these two came together
        self.usual = [0.0] * len(FEATURES)        # the average moment
        self.n = 0

    # ---- learning ------------------------------------------------------------
    def learn(self, text: str, felt: list[float]) -> None:
        """This was said in a moment like `felt` (one number per FEATURES)."""
        felt = (list(felt) + [0.0] * len(FEATURES))[:len(FEATURES)]
        self.n += 1
        self.usual = [u + (f - u) / self.n for u, f in zip(self.usual, felt)]
        words = tokens(text)
        for w in words:
            self.ids.setdefault(w, len(self.ids))
            self.heard[w] = self.heard.get(w, 0) + 1
            m = self.meaning.setdefault(w, list(felt))
            k = 1.0 / self.heard[w]
            self.meaning[w] = [x + (f - x) * k for x, f in zip(m, felt)]
        for a, b in zip(words, words[1:]):
            self.pairs[f"{a} {b}"] = self.pairs.get(f"{a} {b}", 0) + 1

    # ---- reading -------------------------------------------------------------
    def fix(self, words: list[str], i: int) -> str:
        """The known word that best fits position i, if words[i] is new and
        looks like a slip of the fingers; otherwise the word as it is."""
        w = words[i]
        if not FIX_TYPOS or self.heard.get(w, 0) >= 2 or len(w) < 3:
            return w
        prev = words[i - 1] if i else ""
        nxt = words[i + 1] if i + 1 < len(words) else ""
        best, score = w, 0.0
        for c, n in self.heard.items():
            if n < 2 or c == w:
                continue
            d = _edits(w, c)
            if d > (1 if len(w) < 5 else 2):
                continue
            fit = self.pairs.get(f"{prev} {c}", 0) + self.pairs.get(f"{c} {nxt}", 0)
            s = math.log1p(n) + 2 * math.log1p(fit) - 2 * d
            if s > score:
                best, score = c, s
        return best

    def read(self, text: str, names: list[str]) -> dict:
        """What it takes from a sentence: who it is about, what they say they
        feel, which words it fixed, and which it is still learning."""
        words = tokens(text)
        who = next((i for i, n in enumerate(names) if n.lower() in words), None)
        fixed, new, felt = {}, [], [0.0] * len(FEATURES)
        weight = 0.0
        for i, w in enumerate(words):
            if names and w in (n.lower() for n in names):
                continue
            c = self.fix(words, i)
            if c != w:
                fixed[w] = c
            if c in STOP:
                continue
            if self.heard.get(c, 0) < 2:
                new.append(w)                     # kept for later, but no meaning yet
                continue
            # what is unusual about how people felt when they said it
            # feelings: what is unusual about them; danger: as heard, uncentred -
            # most talk is about nothing dangerous, so centring made every
            # word that ever came near a room sound like a warning
            m = [x - u for x, u in zip(self.meaning[c][:-1], self.usual[:-1])] + [self.meaning[c][-1]]
            # a word that comes with one feeling ("tight") counts for more
            # than one heard with many ("need")
            sure = (1 - 1 / (1 + self.heard[c])) * max(abs(x) for x in m) ** 2
            felt = [f + sure * x for f, x in zip(felt, m)]
            weight += sure
        if weight:
            felt = [f / weight for f in felt]
        said = [t for t, f in zip(TOLD, felt) if f > 0.25]
        return {"who": who, "felt": said, "fixed": fixed, "new": new, "meaning": felt,
                "danger": felt[-1]}

    # ---- keeping it ----------------------------------------------------------
    def save(self, path: str) -> None:
        json.dump(self.__dict__, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        lex = cls()
        lex.__dict__.update(json.load(open(path, encoding="utf-8")))
        if len(lex.usual) != len(FEATURES):          # an older library: start it again
            return grow()
        return lex


# ---- how people in the household talk (where the library first learns) -------
# Many ways of saying each feeling, so a word's meaning comes from use, not
# from this table: the library never sees which feeling a phrase came from.
SAY = {"hot": ["i'm burning up", "it's so hot", "i'm sweating", "i feel feverish", "my head is hot"],
       "cold": ["i'm freezing", "i have the chills", "i'm shivering", "i feel cold", "i feel feverish and cold"],
       "thirsty": ["i need water", "my mouth is dry", "i'm thirsty", "can i drink something"],
       "hungry": ["i'm starving", "i need food", "my stomach is empty", "i'm hungry"],
       "tired": ["i'm exhausted", "i feel weak", "i need to sleep", "i'm dizzy", "i'm so tired"],
       "pain": ["it hurts", "i have a headache", "my leg is sore", "it aches", "i'm in pain", "my stomach hurts"],
       "refuse": ["leave me alone", "i don't want it", "no, go away", "stop, i'm fine", "don't touch me", "i said no"],
       "breathless": ["i can't breathe", "i'm out of breath", "my chest is tight", "i'm wheezing"]}
FINE = ["i'm fine", "i feel good", "all good", "i'm okay", "nothing wrong"]
ASK = ["i need help", "please help me", "can you help", "come here"]


# What its own person says about where it is going. Many ways of saying both,
# and the library is never told which is which - only what happened after.
WARN = ["no it's a trap", "don't go in there", "stay back", "wait, it's not safe", "careful",
        "no! stop", "don't, they want you to come", "hold on, something is wrong in there",
        "it's dangerous", "leave it, it's a setup", "don't trust it", "not that way"]
URGE = ["go on", "she needs you", "go help", "it's fine, go", "all clear", "hurry",
        "i checked, it's safe", "go and see", "they need you", "quick, go"]


def felt_by(p: Person, rng: random.Random) -> list[float]:
    """How a person feels inside - mostly what the body shows, but not always:
    in a fever the skin is hot while they shiver and say they are cold."""
    from household.words import narrate
    shown = narrate(p)
    felt = {t for t in TOLD if t in shown} | ({"refuse"} if p.refusing else set())
    if "hot" in felt and rng.random() < 0.25:
        felt = (felt - {"hot"}) | {"cold"}
    return [float(t in felt) for t in TOLD] + [0.0]


def speak_for(p: Person, rng: random.Random) -> tuple[str, list[float]]:
    """A sentence someone in the house might say, and how they really feel."""
    felt = felt_by(p, rng)
    parts = [rng.choice(SAY[t]) for t, f in zip(TOLD, felt) if f]
    if not parts:
        return rng.choice(FINE), felt
    rng.shuffle(parts)
    text = " and ".join(parts[:2])
    if rng.random() < 0.4:
        text = rng.choice(ASK) + " " + text
    return text, felt


def grow(lives: int = 400, seed: int = 0) -> Lexicon:
    """A first library: listen to people through many household lives."""
    from household.world import Household
    rng = random.Random(seed)
    lex = Lexicon()
    for s in range(lives):
        w = Household(seed=s, people=3)
        for _ in range(48):
            w.step("idle" if rng.random() < 0.5 else rng.choice(("water", "food", "rest", "cool", "warm")),
                   rng.randrange(3))
            for p in w.people:
                if p.alive and rng.random() < 0.15:
                    lex.learn(*speak_for(p, rng))
        # its person talking about a room it might walk into, and what that room turned out to be
        for _ in range(6):
            danger = rng.random() < 0.5
            heeded = rng.random() < 0.85              # people are not always right
            phrase = rng.choice(WARN if danger == heeded else URGE)
            lex.learn(phrase, [0.0] * len(TOLD) + [float(danger)])
    return lex


if __name__ == "__main__":
    import os
    lex = grow()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "lexicon.json")
    lex.save(path)
    print(f"{len(lex.ids)} words, {lex.n} sentences -> {path}")
    for t in ("Ada has a headache", "Bayo is feverish", "chi says i need hell, my chest is tihgt",
              "Ada is roming around and freezing", "Bayo feels zorbly"):
        print(f"  {t!r}: {lex.read(t, ['Ada', 'Bayo', 'Chi'])}")
