"""Page lessons (Chalce, 10-30): building pieces of a web page from plain asks.

Coal does not write the code. It picks, for each part of the piece, one of a few named
choices - what it is, where it goes, how wide, how tall, how it looks - and build()
turns those choices into HTML and CSS, the way words.py turns the words head into
sentences. The teacher knows what was asked and teaches by example, like the desk
teacher. Changes ("make it taller") start from the piece already there, so the ask
and what is on the page both reach the brain.
"""
from __future__ import annotations

import random
import re
import zlib

PARTS = {
    "element": ("bar", "button", "sidebar", "box", "list"),
    "place": ("top", "bottom", "left", "right", "middle"),
    "width": ("full", "half", "narrow"),
    "height": ("thin", "normal", "tall"),
    "color": ("blue", "dark", "light", "red", "green"),
    "corners": ("square", "round"),
}
PAGE_WORDS = 64
DEFAULT = {"element": "box", "place": "middle", "width": "half", "height": "normal", "color": "light", "corners": "square"}

# what each word in an ask sets; the teacher reads asks with this, the brain never sees it
SAYS = {
    "element": {"bar": ["bar", "header", "navbar", "toolbar", "strip", "banner"], "button": ["button"],
                "sidebar": ["sidebar", "side panel", "panel down the"], "box": ["box", "card", "panel"],
                "list": ["list", "menu items"]},
    "place": {"top": ["top", "above", "header"], "bottom": ["bottom", "footer", "below"], "left": ["left"],
              "right": ["right"], "middle": ["middle", "centre", "center"]},
    "width": {"full": ["full width", "across", "whole width", "wide"], "half": ["half"], "narrow": ["narrow", "slim", "small"]},
    "height": {"thin": ["thin", "short", "shorter", "flat"], "tall": ["tall", "taller", "big", "bigger", "high"],
               "normal": ["normal height"]},
    "color": {c: [c] for c in ("blue", "dark", "light", "red", "green")} | {"dark": ["dark", "black"], "light": ["light", "white"]},
    "corners": {"round": ["round", "rounded", "soft corners"], "square": ["square", "sharp"]},
}


def page_names() -> list[str]:
    now = [f"page.now.{p}.{c}" for p, cs in PARTS.items() for c in cs]
    return [f"page.word{i}" for i in range(PAGE_WORDS)] + now + ["page.asked"]


PAGE_SIZE = len(page_names())


def page_obs(ask: str, now: dict | None) -> list[float]:
    bag = [0.0] * PAGE_WORDS
    for w in re.findall(r"[a-z]+", ask.lower()):
        bag[zlib.crc32(w.encode()) % PAGE_WORDS] = 1.0
    have = [float(bool(now) and now.get(p) == c) for p, cs in PARTS.items() for c in cs]
    return bag + have + [1.0]


def teacher(ask: str, now: dict | None) -> dict:
    """What was wanted: everything the ask names, the rest kept from what is there
    (or the usual for a new piece). A bar at the top is full width unless said."""
    a = " " + ask.lower() + " "
    out = dict(now or DEFAULT)
    for part, choices in SAYS.items():
        for choice, words in choices.items():
            if any(re.search(r"\b" + re.escape(w) + r"\b", a) for w in words):
                out[part] = choice
    if not now and out["element"] == "bar" and not any(w in a for w in ("half", "narrow", "slim", "small")):
        out["width"] = "full"
    if not now and out["element"] == "sidebar":
        out["width"], out["height"] = "narrow", "tall"
        if out["place"] not in ("left", "right"):
            out["place"] = "left"
    return out


def build(c: dict) -> str:
    """The chosen parts as HTML and CSS."""
    tag = {"bar": "nav", "button": "button", "sidebar": "aside", "box": "div", "list": "ul"}[c["element"]]
    w = {"full": "100%", "half": "50%", "narrow": "200px"}[c["width"]]
    h = {"thin": "32px", "normal": "56px", "tall": "100%" if c["element"] == "sidebar" else "160px"}[c["height"]]
    bg = {"blue": "#1f57c9", "dark": "#222", "light": "#f2f2f2", "red": "#d33", "green": "#2a8a3e"}[c["color"]]
    fg = "#222" if c["color"] == "light" else "#fff"
    place = {"top": "top: 0; left: 0;", "bottom": "bottom: 0; left: 0;", "left": "top: 0; left: 0;",
             "right": "top: 0; right: 0;", "middle": "top: 50%; left: 50%; transform: translate(-50%, -50%);"}[c["place"]]
    r = "8px" if c["corners"] == "round" else "0"
    return (f'<{tag} style="position: fixed; {place} width: {w}; height: {h}; '
            f'background: {bg}; color: {fg}; border-radius: {r};"></{tag}>')


NEW = ["make a bar at the top", "put a blue bar across the top", "add a header", "a dark navbar at the top",
       "make a thin red strip at the bottom", "add a footer bar", "put a sidebar on the left", "a dark side panel on the right",
       "make a round green button", "add a button in the middle", "put a box in the middle", "a light card with round corners",
       "make a list on the left", "add a small blue button at the bottom", "a tall box on the right", "make a narrow dark sidebar"]
CHANGE = ["make it taller", "make it thinner", "move it to the bottom", "move it to the top", "make it blue",
          "make it dark", "round the corners", "make the corners square", "make it full width", "make it narrow",
          "move it to the left", "make it red", "make it shorter", "move it to the right"]
EXAM_NEW = ["can you add a green banner along the top", "i want a slim sidebar on the right", "pop a red button in the centre",
            "give me a rounded white card in the middle", "add a black toolbar at the bottom"]
EXAM_CHANGE = ["can you make it a bit bigger", "now move it down to the bottom", "change it to green", "give it soft corners",
               "make it the whole width"]


FRAMES = ["make a {d} {e} {p}", "add a {d} {e} {p}", "put a {d} {e} {p}", "i need a {d} {e} {p}", "a {d} {e} {p}"]
WHERE = {"top": "at the top", "bottom": "at the bottom", "left": "on the left", "right": "on the right", "middle": "in the middle"}


def mixed(rng: random.Random) -> str:
    """A new piece said with any of the words the teacher knows, so every word is met."""
    pick = lambda part: rng.choice(rng.choice(list(SAYS[part].values())))
    looks = [pick(q) for q in ("color", "width", "height", "corners") if rng.random() < 0.4]
    e = pick("element")
    place = rng.choice(list(SAYS["place"].values()))
    p = rng.choice([WHERE.get(k, "") for k in WHERE] + [f"at the {w}" for w in place])
    return " ".join(rng.choice(FRAMES).format(d=" ".join(looks), e=e, p=p).split())


def lessons(n: int, seed: int, new=NEW, change=CHANGE) -> list[tuple[list[float], dict]]:
    """n moments: (what the brain sees, the teacher's choice for every part)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if new is NEW and rng.random() < 0.35:
            ask, now = mixed(rng), None
        elif rng.random() < 0.5:
            ask, now = rng.choice(new), None
        else:
            ask, now = rng.choice(change), teacher(rng.choice(new), None)
            if new is NEW and rng.random() < 0.4:      # any known word as a change
                q = rng.choice(("color", "width", "height", "corners", "place"))
                w = rng.choice(rng.choice(list(SAYS[q].values())))
                ask = rng.choice(["make it {w}", "now {w}", "can you make it {w}", "change it to {w}", "move it to the {w}"]).format(w=w)
        out.append((page_obs(ask, now), teacher(ask, now)))
    return out


def page_score(choose) -> dict[str, float]:
    """On asks never taught: every part right, and parts right."""
    rows = lessons(2000, seed=20_000, new=EXAM_NEW, change=EXAM_CHANGE)
    whole = parts = 0
    for obs, want in rows:
        got = choose(obs)
        ok = sum(got[p] == want[p] for p in PARTS)
        parts += ok
        whole += ok == len(PARTS)
    return {"whole piece right": whole / len(rows), "parts right": parts / (len(rows) * len(PARTS))}


if __name__ == "__main__":
    for ask, now in [("make a bar at the top", None), ("make it taller", teacher("make a bar at the top", None)),
                     ("put a sidebar on the left", None), ("make a round green button", None)]:
        c = teacher(ask, now)
        print(ask, "->", c, "\n   ", build(c))
