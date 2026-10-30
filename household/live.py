"""Live learning in class (Chalce, 10-30): Coal learns the moment an answer is handed in.

Every answer the teacher gives gets a permanent number in a growing list (like a word
in the library). The brain has a reply head that picks one of them for the question in
front of it, which reaches the brain as its words (a hashed bag, like the desk asks), so
a question worded differently but sharing its words can land on the same answer.

When an answer is handed in, Coal trains for a moment on every class question it has
had so far. Only the first layer's class inputs and the reply head move: the household,
desk and page inputs are zero in a class question, so their weights get no change, and
every later layer is left alone - class can not un-teach the household.
"""
from __future__ import annotations

import json
import os
import re
import zlib

import torch
import torch.nn.functional as F

CLASS_WORDS = 64
MAX_ANSWERS = 256          # slots for answers; each one taught takes the next


def class_names() -> list[str]:
    return [f"class.word{i}" for i in range(CLASS_WORDS)] + ["class.asked"]


CLASS_SIZE = len(class_names())


def class_obs(question: str) -> list[float]:
    bag = [0.0] * CLASS_WORDS
    for w in re.findall(r"[a-z]+", question.lower()):
        w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w   # "folders" is "folder"
        bag[zlib.crc32(w.encode()) % CLASS_WORDS] = 1.0
    return bag + [1.0]


class Answers:
    """The answers Coal has been taught, and every (question, answer) it met in class."""

    def __init__(self, path: str, d: dict | None = None):
        self.path = path                                 # a readable copy; the brain file holds the real one
        if d is None:
            d = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        self.texts: list[str] = d.get("texts", [])
        self.pairs: list[list] = d.get("pairs", [])     # [question, answer id]

    def teach(self, question: str, answer: str) -> int:
        if answer not in self.texts:
            if len(self.texts) >= MAX_ANSWERS:
                raise RuntimeError("Coal's answer slots are full; the reply head needs widening")
            self.texts.append(answer)
        i = self.texts.index(answer)
        self.pairs.append([question, i])
        json.dump(self.to_dict(), open(self.path, "w", encoding="utf-8"), indent=1)
        return i

    def to_dict(self) -> dict:
        return {"texts": self.texts, "pairs": self.pairs}


@torch.no_grad()
def reply(brain, answers: Answers, question: str) -> tuple[str | None, float]:
    """Coal's pick among the answers it knows, and how sure it is."""
    if not answers.texts:
        return None, 0.0
    logits = brain(torch.tensor([class_obs(question)], dtype=torch.float32))["reply"][0, :len(answers.texts)]
    p = logits.softmax(0)
    i = int(p.argmax())
    return answers.texts[i], float(p[i])


def learn(brain, answers: Answers, steps: int = 60) -> float:
    """A few quick steps on everything met in class so far; returns the last loss."""
    obs = torch.tensor([class_obs(q) for q, _ in answers.pairs], dtype=torch.float32)
    want = torch.tensor([i for _, i in answers.pairs])
    first = brain.body[0]
    moving = [first.weight, *brain.reply.parameters()]
    was = [p.requires_grad for p in brain.parameters()]
    for p in brain.parameters():
        p.requires_grad_(False)
    for p in moving:
        p.requires_grad_(True)
    opt = torch.optim.Adam(moving, lr=1e-2)
    brain.train()
    for _ in range(steps):
        logits = brain(obs)["reply"][:, :len(answers.texts)]
        loss = F.cross_entropy(logits, want)
        opt.zero_grad()
        loss.backward()
        opt.step()
    brain.eval()
    for p, r in zip(brain.parameters(), was):
        p.requires_grad_(r)
    return float(loss.detach())
