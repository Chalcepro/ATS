"""The classroom (Chalce, 10-30): small lessons, English / computer / a little biology, in turn.

A question is put; Coal answers and Chalce answers; the teacher marks each one:
  right - it says what was wanted
  close - a good way to put it, but not quite (some of what was wanted is there)
  wrong - and the teacher gives the answer
Nothing is cleaned before it is kept: "nah", "NAAAAH", "NaH" are stored exactly as
typed or said, beside the plain form the teacher read them as, so Coal meets how
people really write. The lessons are the teacher's, written here; adding one is
adding a line.
"""
from __future__ import annotations

import json
import re
import time

# (subject, question, accepted answers, words a close answer has, what the teacher adds)
LESSONS = [
    ("English", "Someone walks in and says 'hiya'. What are they doing?", ["greeting you", "saying hello", "greeting"],
     ["hello", "greet", "hi"], "'Hiya' is a casual hello."),
    ("computer", "What is a folder for?", ["holding files", "keeping files together", "storing files"],
     ["files", "keep", "hold", "store", "organise"], "A folder keeps files together so they are easy to find."),
    ("English", "What does 'NAAAAH' mean?", ["no", "nah"], ["no", "not"],
     "Stretched letters and capitals are the same word said strongly: 'nah', 'NAH', 'NaH' all mean no."),
    ("biology", "Why do people drink water?", ["the body needs water", "to stay alive", "so they don't dehydrate"],
     ["body", "needs", "thirst", "alive", "dehydrat"], "The body loses water all day; without it a person weakens in days."),
    ("computer", "What happens to a file you delete?", ["it goes to the recycle bin", "it goes to the bin"],
     ["bin", "recycle", "trash", "gone"], "It goes to the Recycle Bin first; it is only gone when the bin is emptied."),
    ("English", "What do you answer when someone says 'thank you'?", ["you're welcome", "no problem", "anytime"],
     ["welcome", "problem", "pleasure"], "'You're welcome' is the usual; 'no problem' is casual."),
    ("computer", "What is the difference between Save and Save As?", ["save as makes a new copy with a new name",
     "save as asks for a new name"], ["new", "name", "copy", "same"], "Save writes over the same file; Save As asks for a new name or place."),
    ("English", "Someone types 'brb'. What do they mean?", ["be right back"], ["back", "right"],
     "'brb' is 'be right back': they are leaving for a moment."),
    ("biology", "What does a fever tell you?", ["the body is fighting something", "they are ill"],
     ["fight", "ill", "sick", "infection", "body"], "A fever is the body heating itself to fight an illness."),
    ("computer", "What is the mouse pointer?", ["it shows where the person is pointing", "the arrow the person moves"],
     ["person", "move", "point", "arrow", "mouse"], "The pointer is the person's hand on the screen: when it moves, they moved it."),
    ("English", "Is 'what's up' a question you must answer with facts?", ["no it is a greeting", "no"],
     ["greet", "hello", "no"], "'What's up' is mostly a greeting; 'not much' is a fine answer."),
    ("computer", "Where should a photo usually go?", ["photos", "the photos folder", "pictures"],
     ["photo", "picture", "image"], "Photos go in the Photos folder - unless the person keeps them elsewhere on purpose."),
    ("English", "What is the difference between 'ok' and 'OK!!!'?", ["the same word said more strongly", "more excited"],
     ["same", "strong", "excit", "loud"], "Same word; the capitals and marks show feeling, not a new meaning."),
    ("biology", "Why do people sleep?", ["the body and brain rest and repair", "to rest"],
     ["rest", "repair", "tired", "brain", "body"], "Sleep lets the body and brain rest and repair; without it people grow slow and ill."),
    ("computer", "What does Ctrl+Z do?", ["undo", "undoes the last thing"], ["undo", "back"], "Ctrl+Z undoes the last change."),
]


def plain(text: str) -> str:
    """How the teacher reads an answer: lower case, letters run together squeezed
    ('NAAAAH' -> 'nah'), punctuation gone. The raw text is still kept."""
    t = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)            # three or more of a letter -> one
    words = t.split()
    if len(words) > 1 and all(len(w) == 1 for w in words):   # spelled out: "n a h", "N_A_H"
        words = ["".join(words)]
    return " ".join(words)


def mark(n: int, text: str) -> dict:
    subject, q, answers, keys, note = LESSONS[n % len(LESSONS)]
    p = plain(text)
    if not p:
        verdict = "wrong"
    elif any(f" {plain(a)} " in f" {p} " for a in answers):        # whole words, so "no" is not in "know"
        verdict = "right"
    else:
        hits = sum(w.startswith(k) for k in keys for w in p.split())
        verdict = "close" if hits else "wrong"
    say = {"right": "Right. " + note,
           "close": f"That's a good way to put it, but it's more: {answers[0]}. " + note,
           "wrong": f"Not quite - the answer is: {answers[0]}. " + note}[verdict]
    return {"verdict": verdict, "plain": p, "teacher": say}


def question(n: int) -> dict:
    subject, q, *_ = LESSONS[n % len(LESSONS)]
    return {"n": n % len(LESSONS), "subject": subject, "question": q, "of": len(LESSONS)}


def keep(path: str, n: int, who: str, raw: str, marked: dict) -> None:
    """Every answer, exactly as given, with the teacher's reading and mark."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "lesson": n, "question": LESSONS[n % len(LESSONS)][1],
                            "who": who, "raw": raw, **marked}) + "\n")
