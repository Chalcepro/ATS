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


# ---- the text box: sentences out, sentences in ------------------------------
DOING = {"water": "giving {n} water", "food": "feeding {n}", "cool": "cooling {n} down",
         "warm": "warming {n}", "rest": "getting {n} to rest", "bandage": "bandaging {n}",
         "comfort": "sitting with {n}", "shield": "standing between {n} and the animal",
         "wash": "washing {n}", "charge": "charging myself", "idle": "watching",
         "push_away": "pushing the animal away", "harm": "striking the animal",
         "see": "going to see {n}"}

# words people use for how they feel -> the meaning the brain knows (world.TOLD)
HEARD = {"hot": "hot", "fever": "hot", "feverish": "hot", "burning": "hot", "sweating": "hot",
         "cold": "cold", "chills": "cold", "shivering": "cold", "freezing": "cold",
         "thirsty": "thirsty", "dry": "thirsty", "water": "thirsty",
         "hungry": "hungry", "starving": "hungry", "food": "hungry",
         "tired": "tired", "weak": "tired", "exhausted": "tired", "sleepy": "tired", "dizzy": "tired",
         "pain": "pain", "hurts": "pain", "hurt": "pain", "ache": "pain", "headache": "pain",
         "stomachache": "pain", "sore": "pain", "cut": "pain", "burn": "pain", "sting": "pain",
         "breathless": "breathless", "breathe": "breathless", "breathing": "breathless",
         "choking": "breathless", "wheezing": "breathless", "cough": "breathless", "coughing": "breathless",
         # a few in Nigerian Pidgin and Yoruba, so the box is not English-only
         "belle": "hungry", "ebi": "hungry", "ongbe": "thirsty", "gbona": "hot", "otutu": "cold",
         "dey pain": "pain", "dey hot": "hot", "dey cold": "cold", "don tire": "tired"}


def sentence(name: str, words: list[str], act: str = "", target_name: str = "") -> str:
    """What it says about one person, and what it is doing, in plain English."""
    say = {"pain": "in pain", "smelly": "unwashed"}
    w = [say.get(x, x) for x in words if x not in ("hurt", "unwell", "healthy", "dead")]
    state = [x for x in words if x in ("hurt", "unwell", "healthy", "dead")]
    if "dead" in state:
        out = f"{name} has died."
    elif w:
        signs = w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1]
        out = f"{name} is {signs}" + (" - badly hurt." if "hurt" in state else ".")
    else:
        out = f"{name} looks {state[0] if state else 'all right'}."
    if act and target_name == name and act in DOING:
        out += " I'm " + DOING[act].format(n=name) + "."
    return out


def understand(text: str, names: list[str]) -> tuple[int, list[str]] | None:
    """'Ada has a headache' -> (0, ["pain"]). Who is named, and what they feel.
    Unknown words are simply passed over; None if nobody is named."""
    t = " " + text.lower().replace(",", " ").replace(".", " ").replace("'s", " ") + " "
    who = next((i for i, n in enumerate(names) if f" {n.lower()} " in t), None)
    if who is None:
        return None
    felt = [m for k, m in HEARD.items() if f" {k} " in t or (" " not in k and f" {k}" in t and len(k) > 4)]
    return who, sorted(set(felt))
