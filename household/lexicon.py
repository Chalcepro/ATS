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

from household.world import DOINGS, ROOMS, TOLD, Person

# What a word can come to mean: how the speaker felt, then whether the
# place being talked about turned out dangerous. Words are not told which;
# they land wherever they kept turning up.
# ...and, last, what kind of thing was said to it (the communication class, 10-30):
# a greeting, a question, thanks. These come after everything the brain hears
# (world.HEARD_NAMES), so the brain's inputs are unchanged; they are for answering.
TALK = ("greet", "bye", "thank", "ask_how", "ask_where", "ask_doing", "ask_is", "ask_name", "yes", "no")
FEATURES = TOLD + ("danger",) + ROOMS + DOINGS + TALK
DANGER = len(TOLD)
FEEL = slice(0, len(TOLD))
ROOM = slice(DANGER + 1, DANGER + 1 + len(ROOMS))
DOING = slice(ROOM.stop, ROOM.stop + len(DOINGS))
TALKING = slice(DOING.stop, DOING.stop + len(TALK))


def moment(feel=(), danger: float = 0.0, room: str | None = None, doing: str | None = None,
           talk: str | None = None) -> list[float]:
    """A moment as the library stores it."""
    return ([float(t in feel) for t in TOLD] + [danger] + [float(r == room) for r in ROOMS]
            + [float(d == doing) for d in DOINGS] + [float(t == talk) for t in TALK])
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
    return re.findall(r"[a-z']+|\?", text.lower())     # a question mark is heard too


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
        # pairs of words are learned as units too, so "not cold" can come to
        # mean something other than "cold" - from use, not from a grammar rule
        for a, b in zip(words, words[1:]):
            pair = f"{a} {b}"
            self.pairs[pair] = self.pairs.get(pair, 0) + 1
            m = self.meaning.setdefault(pair, list(felt))
            k = 1.0 / self.pairs[pair]
            self.meaning[pair] = [x + (f - x) * k for x, f in zip(m, felt)]

    def meet(self, text: str) -> list[str]:
        """Words met without a moment it can read (class, 10-30): each gets its
        permanent id and counts as heard, so it is known and typo-fixing works on
        it, but its meaning is left alone - that only comes from moments.
        Returns the words that were new."""
        new = []
        for w in tokens(text):
            if w not in self.ids:
                self.ids[w] = len(self.ids)
                new.append(w)
            self.heard[w] = self.heard.get(w, 0) + 1
        return new

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
        peak = [0.0] * len(TOLD)                  # the strongest any one word points at each feeling
        for w in words:                           # each word on its own, even inside a pair ("hungry ?")
            if w not in STOP and self.heard.get(w, 0) >= 2:
                peak = [max(a, x - u) for a, x, u in zip(peak, self.meaning[w][FEEL], self.usual[FEEL])]
        skip = False
        for i, w in enumerate(words):
            if skip:
                skip = False
                continue
            if names and w in (n.lower() for n in names):
                continue
            c = self.fix(words, i)
            if c in STOP:                         # "is", "the": never the start of a unit, so "is not hot" reads as "not hot"
                if c != w:
                    fixed[w] = c
                continue
            pair = f"{c} {words[i + 1]}" if i + 1 < len(words) else ""
            if self.pairs.get(pair, 0) >= 3 and pair in self.meaning:
                c, skip = pair, True               # a known pair reads as one unit
            if c != w:
                fixed[w] = c
            if c in STOP:
                continue
            if self.heard.get(c, 0) < 2 and self.pairs.get(c, 0) < 3:
                new.append(w)                     # kept for later, but no meaning yet
                continue
            # what is unusual about how people felt when they said it
            # feelings: what is unusual about them; danger: as heard, uncentred -
            # most talk is about nothing dangerous, so centring made every
            # word that ever came near a room sound like a warning
            mc = self.meaning[c]
            m = [x - u for x, u in zip(mc[FEEL], self.usual[FEEL])] + mc[DANGER:]
            # a word that comes with one feeling ("tight") counts for more
            # than one heard with many ("need")
            n = self.heard.get(c) or self.pairs.get(c, 0)
            sure = (1 - 1 / (1 + n)) * max(abs(x) for x in m) ** 2
            felt = [f + sure * x for f, x in zip(felt, m)]
            weight += sure
        if weight:
            felt = [f / weight for f in felt]
        said = [t for t, f in zip(TOLD, felt) if f > 0.25]
        rooms, doings, talks = felt[ROOM], felt[DOING], felt[TALKING]
        where = ROOMS[rooms.index(max(rooms))] if max(rooms) > 0.35 else None
        doing = DOINGS[doings.index(max(doings))] if max(doings) > 0.35 else None
        talk = TALK[talks.index(max(talks))] if talks and max(talks) > 0.3 else None
        return {"who": who, "felt": said, "fixed": fixed, "new": new, "meaning": felt,
                "danger": felt[DANGER], "where": where, "doing": doing, "talk": talk,
                "about": [TOLD[peak.index(max(peak))]] if max(peak) > 0.15 else []}

    # ---- keeping it ----------------------------------------------------------
    def save(self, path: str) -> None:
        json.dump(self.__dict__, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        return cls.from_dict(json.load(open(path, encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict) -> "Lexicon":
        lex = cls()
        lex.__dict__.update(d)
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
    return moment(felt)


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
            lex.learn(phrase, moment(danger=float(danger)))
        for text, m in english_class(w, rng):
            lex.learn(text, m)
        for text, m in talk_class(w, rng):
            lex.learn(text, m)
    return lex


# ---- the English class (Chalce, 10-30): its teacher narrates what is really happening ----
# Simple sentences, each said at the moment it is true, so every word lands
# on the room, the doing or the feeling it goes with. The teacher varies its
# words; the library is never told which word means what.
MOVE = ["walks to", "goes to", "heads to", "runs to", "is going to", "went into"]
DO = {"cool": ["cooling", "cooling down", "fanning"], "warm": ["warming", "covering", "wrapping up"],
      "water": ["giving water to", "bringing a drink to"], "food": ["feeding", "bringing food to"],
      "rest": ["letting", "settling"], "bandage": ["bandaging", "dressing the wound of"],
      "comfort": ["sitting with", "staying with", "keeping company with"], "see": ["going to see", "checking on"],
      "idle": ["waiting", "watching", "staying put"]}
ASKS = {"cool": ["cool {n} down", "{n} needs cooling"], "warm": ["warm {n} up", "cover {n}"],
        "water": ["give {n} water", "{n} needs a drink"], "food": ["feed {n}", "get {n} something to eat"],
        "bandage": ["bandage {n}", "dress the wound"], "comfort": ["sit with {n}", "stay with {n}"],
        "see": ["go and see {n}", "check on {n}", "go to {n}"], "idle": ["stop", "wait", "stay here", "hold on"]}
NEGATE = ["{n} is not {f}", "{n} isn't {f}", "{n} is not {f} any more"]


def english_class(w, rng: random.Random, hours: int = 24) -> list[tuple[str, list[float]]]:
    """Lessons from one household life: who goes where, what is being done,
    what someone is not feeling, and the room names shown on entering."""
    from household.words import narrate
    out = []
    w.wander = 0.1
    for _ in range(hours):
        before = [p.room for p in w.people]
        act = rng.choice(("idle", "see") + tuple(ASKS))
        who = rng.randrange(len(w.people))
        n = w.people[who].name
        if rng.random() < 0.5:                    # its person asks for it, then it happens
            out.append((rng.choice(ASKS[act]).format(n=n), moment(doing=act)))
        w.step(act if act in ("idle", "see") or w.people[who].alive else "idle", who)
        out.append((f"coal is {rng.choice(DO[act])} {n}" if act != "idle" else f"coal is {rng.choice(DO[act])}",
                    moment(doing=act)))
        for label in w.labels:                    # the room's name, shown on entering
            out.append((label, moment(room=label)))
        for p, was in zip(w.people, before):
            if p.room != was:
                out.append((f"{p.name} {rng.choice(MOVE)} the {p.room}", moment(room=p.room)))
        p = rng.choice(w.people)
        shown = narrate(p)
        absent = [f for f in ("hot", "cold", "thirsty", "hungry", "tired") if f not in shown]
        if absent and rng.random() < 0.5:
            out.append((rng.choice(NEGATE).format(n=p.name, f=rng.choice(absent)), moment()))
    return out



# ---- the communication class (Chalce, 10-30): being spoken to, not just about feelings ----
# Its person greets it, asks it things, thanks it, answers it. Each line is
# heard in a moment marked with what kind of line it was; nothing says which
# word does it - "where" comes to mean asking-where because that is when it
# turns up. Many ways of saying each, with names, rooms and feelings mixed in so
# those words do not get mistaken for the question itself.
TALKS = {
    "greet": ["hello", "hi", "hey", "hi coal", "hello coal", "good morning", "good evening", "hey there",
              "morning coal", "hello there", "good afternoon", "hey coal, i'm back"],
    "bye": ["bye", "goodbye", "see you later", "good night", "bye coal", "i'm going out", "see you soon",
            "i'm off", "night coal", "talk later"],
    "thank": ["thank you", "thanks", "thanks coal", "thank you so much", "cheers", "well done", "good job",
              "nice work", "thanks for helping {n}"],
    "ask_how": ["how is everyone?", "how are you?", "how is {n}?", "how is {n} doing?", "is everyone okay?",
                "how are things?", "how's {n}?", "how are they?", "any news?", "how is the house?"],
    "ask_where": ["where is {n}?", "where's {n}?", "where did {n} go?", "where are you?", "which room is {n} in?",
                  "where are you going?", "where is everyone?", "where did you go?"],
    "ask_doing": ["what are you doing?", "what are you up to?", "what's happening?", "what are you doing now?",
                  "what is coal doing?", "what's going on?", "what did you do?", "why are you doing that?"],
    "ask_is": ["is {n} {f}?", "is {n} {f} now?", "does {n} look {f}?", "is {n} still {f}?", "does {n} seem {f}?",
               "do you think {n} is {f}?", "{n} {f}?"],
    "ask_name": ["what's your name?", "who are you?", "what are you called?", "what is your name?",
                 "what should i call you?"],
    "yes": ["yes", "yeah", "yep", "sure", "okay", "ok", "that's right", "correct", "go ahead", "do it"],
    "no": ["no", "nope", "not now", "don't", "no thanks", "never mind", "leave it", "nah"],
}
TELL = ["{n} is {f}", "{n} looks {f}", "{n} seems {f}", "{n} feels {f}", "{n} is really {f}", "{n} got {f}"]
FEELINGS = ("hot", "cold", "thirsty", "hungry", "tired", "okay", "hurt", "bleeding", "awake", "fine")


def talk_class(w, rng: random.Random, lines: int = 12) -> list[tuple[str, list[float]]]:
    """Lessons from one household life: lines said to Coal, each in the moment
    of what it is - a greeting, a question of some kind, thanks, yes or no."""
    out = []
    for _ in range(lines):
        act = rng.choice(TALK)
        p = rng.choice(w.people)
        text = rng.choice(TALKS[act]).format(n=p.name, f=rng.choice(FEELINGS))
        if rng.random() < 0.15:                    # people drop the question mark
            text = text.replace("?", "")
        out.append((text, moment(room=p.room if act == "ask_where" else None, talk=act)))
        if rng.random() < 0.5:                     # and plain telling, so a statement is not taken for a question
            f = rng.choice(TOLD[:7])
            out.append((rng.choice(TELL).format(n=p.name, f=f), moment(feel=[f])))
    return out


if __name__ == "__main__":
    import os
    lex = grow()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "lexicon.json")
    lex.save(path)
    print(f"{len(lex.ids)} words, {lex.n} sentences -> {path}")
    for t in ("Ada has a headache", "Bayo is feverish", "chi says i need hell, my chest is tihgt",
              "Ada is roming around and freezing", "Bayo feels zorbly"):
        print(f"  {t!r}: {lex.read(t, ['Ada', 'Bayo', 'Chi'])}")
