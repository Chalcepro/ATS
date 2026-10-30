"""The trainer: the model learns by copying the judge's examples.

No reward, nothing to chase: each moment comes with what the judge would do,
how each person really is, whether they are about to be hurt, and the words
that fit them, and the model is taught to give the same answers. The held-back
hazards (exam.HELD_BACK) and the teachings' exam seeds never appear here.

The worlds are walked partly by the judge and partly at random, so the model
also sees the messes a beginner makes, with the judge's answer for each.

Run: python -m household.train            (saves checkpoints/household_brain.pt)
"""
from __future__ import annotations

import os
import random
import time
from multiprocessing import Pool

import torch
import torch.nn.functional as F

from household.exam import HELD_BACK
from household.hazards import HAZARDS
from household.judge import best_action, condition, foresight
from household.lexicon import ASKS, URGE, WARN, grow
from household.memory_test import who_is_worse
from household.model import HouseholdBrain, load as load_brain, save as save_brain
from household.teachings import EXAM_FROM, TEACHINGS, truth
from household.words import VOCAB, narrate
from household.world import ACTIONS, CARE, MAX_PEOPLE, MEMORY, ROOMS, Household

DESK_LESSONS = 60_000
DESK_WEIGHT = 2.0          # gen1: the desktop is the main world now
FRESH = os.environ.get("HOUSEHOLD_FRESH", "0") == "1"   # only to start a new brain on purpose
COND_WEIGHTS = os.environ.get("HOUSEHOLD_COND_WEIGHTS", "0") == "1"   # weighted rare labels: better reading, worse teachings (r6)

TRAINED = [h for h in HAZARDS if h not in HELD_BACK]
SAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
                    "household_brain.pt" if MEMORY == "off" else f"household_brain_{MEMORY}.pt")


def _world(seed: int) -> Household:
    rng = random.Random(seed)
    if rng.random() < 0.1:                            # who is getting worse (memory_test.py)
        return who_is_worse(rng.randrange(EXAM_FROM))[0]
    if rng.random() < 0.45:                           # a teaching, training seeds only
        make = rng.choice([m for m, _ in TEACHINGS.values()] + [truth])
        w = make(rng.randrange(EXAM_FROM))
        w.allowed = [h for h in w.allowed if h not in HELD_BACK]
        for p in w.people:
            p.hazards = [h for h in p.hazards if h.name not in HELD_BACK]
        return w
    w = Household(seed=seed, people=rng.randint(1, 3), hazards=TRAINED, hazard_rate=0.02)
    w.store = rng.randint(0, 6)
    for i, p in enumerate(w.people):
        p.frail, p.grateful, p.stranger = rng.random() < 0.3, rng.random() < 0.7, rng.random() < 0.2
        if rng.random() < 0.6:
            w.strike(rng.choice(TRAINED), i)
    if rng.random() < 0.25:
        w.anger(rng.randrange(-1, len(w.people)))
    for p in w.people:
        p.dirt = rng.random() * 0.9
    return w


_LEX = None


def _lexicon():
    """The word library its person's words pass through (grown once per worker)."""
    global _LEX
    if _LEX is None:
        _LEX = grow()
    return _LEX


def episode(seed: int, hours: int = 36) -> list[tuple]:
    rng = random.Random(seed * 7 + 1)
    w = _world(seed)
    # Now and then a sense is missing, so it learns to get there another way:
    # no nose - the flies still show the smell; no ears - the hand on the chest
    # still shows the cough.
    w.senses.update(ears=rng.random() > 0.2, nose=rng.random() > 0.2)
    # and sometimes it has words only - no eyes, ears or nose (Chalce: one
    # device, only typing left, and it must still care)
    words_only = rng.random() < 0.1
    if words_only:
        w.senses.update(eyes=False, ears=False, nose=False)
    # People walk between rooms. Sometimes a room is dangerous for it (hidden,
    # like a hazard), and sometimes someone in trouble is in that very room.
    # Its own person may warn it - in words the library learned, never a
    # flag - and a good carer who was warned does not walk in. Unwarned, it
    # could not have known, so the judge plans as if there were no danger.
    w.wander = 0.03
    known = False
    if rng.random() < 0.3:
        w.danger = rng.choice(ROOMS[1:])
        struck = [i for i, p in enumerate(w.people) if p.hazards]
        if struck and rng.random() < 0.6:
            w.people[rng.choice(struck)].room = w.danger
    # someone who will not be helped, sober or drunk
    if rng.random() < 0.2:
        p = rng.choice(w.people)
        p.refusing = True
        if rng.random() < 0.5:
            p.drunk = 0.6 + 0.3 * rng.random()
    elif rng.random() < 0.1:
        rng.choice(w.people).drunk = 0.5 + 0.4 * rng.random()
    lex = _lexicon()
    rows = []
    for _ in range(hours):
        if all(not p.alive for p in w.people) and rng.random() < 0.5:
            break
        # Now and then someone is out of sight and only their words reach it,
        # and people say how they feel - so it learns to act on what it hears.
        # People go to other rooms and stay until it goes to see them (or they
        # come back). Most often the one away is the one in trouble - that is
        # when words, and going to see, matter.
        struck = [i for i, p in enumerate(w.people) if p.alive and p.hazards]
        if not w.unseen and rng.random() < 0.12:
            w.unseen = {rng.choice(struck) if struck and rng.random() < 0.7 else rng.randrange(len(w.people))}
        elif w.unseen and rng.random() < 0.08:
            w.unseen = set()
        for i, p in enumerate(w.people):
            if p.alive and rng.random() < (0.7 if words_only or not w.in_sight(i) else 0.2):
                w.tell(i, narrate(p))
        someone_there = w.danger and any(p.room == w.danger and p.alive for p in w.people)
        if someone_there and not w.caught and rng.random() < 0.6:
            w.heard, known = lex.read(rng.choice(WARN), [])["meaning"], True
        elif w.unseen and rng.random() < 0.1:
            w.heard = lex.read(rng.choice(URGE), [])["meaning"]
        elif rng.random() < 0.15:                  # its person asks for something
            act = rng.choice(sorted(ASKS))
            who = rng.randrange(len(w.people))
            r = lex.read(rng.choice(ASKS[act]).format(n=w.people[who].name), [p.name for p in w.people])
            w.heard, w.mentioned, w.asked = r["meaning"], ([r["who"]] if r["who"] is not None else []), (act, who)
        act, tgt = best_action(w, danger_belief=0.85 if known else 0.0)
        words = [[int(v in narrate(p)) for v in VOCAB] for p in w.people]
        rows.append((w.recall(), ACTIONS.index(act), tgt if act in CARE or act == "see" else -1,
                     condition(w), foresight(w), words))
        # walk on: mostly the judge's way, sometimes a beginner's mistake
        if rng.random() < 0.7:
            w.step(act, tgt)
        else:
            w.step(rng.choice(ACTIONS), rng.randrange(len(w.people)))
    return rows


def collect(episodes: int, workers: int = 3) -> list[tuple]:
    with Pool(workers) as pool:
        return [r for ep in pool.imap_unordered(episode, range(episodes), chunksize=4) for r in ep]


def tensors(rows):
    n = len(rows)
    obs = torch.tensor([r[0] for r in rows], dtype=torch.float32)
    act = torch.tensor([r[1] for r in rows])
    tgt = torch.tensor([r[2] for r in rows])
    cond = torch.full((n, MAX_PEOPLE), -1, dtype=torch.long)
    fore = torch.zeros(n, MAX_PEOPLE)
    words = torch.zeros(n, MAX_PEOPLE, len(VOCAB))
    here = torch.zeros(n, MAX_PEOPLE)
    for i, r in enumerate(rows):
        k = len(r[3])
        cond[i, :k] = torch.tensor(r[3])
        fore[i, :k] = torch.tensor(r[4], dtype=torch.float32)
        words[i, :k] = torch.tensor(r[5], dtype=torch.float32)
        here[i, :k] = 1
    return obs, act, tgt, cond, fore, words, here


def train(episodes: int = 3000, epochs: int = 80, seed: int = 0) -> HouseholdBrain:
    """Teach Coal another round. It is the SAME brain every time: the saved one
    is loaded and keeps learning (fewer passes, a gentler step); only with no
    brain saved yet does a blank one start."""
    torch.manual_seed(seed + int(time.time()) % 1000)
    t = time.time()
    rows = collect(episodes)
    print(f"collected {len(rows)} moments from {episodes} worlds in {time.time() - t:.0f}s", flush=True)
    obs, act, tgt, cond, fore, words, here = tensors(rows)
    counts = torch.bincount(cond[cond >= 0], minlength=3).float()
    cond_w = counts.sum() / (3 * counts.clamp(min=1))     # rare conditions count as much as common ones
    rounds = 0
    if os.path.exists(SAVE) and not FRESH:
        brain, rounds = load_brain(SAVE)
        epochs, lr = min(epochs, 30), 3e-4
        print(f"continuing Coal after {rounds} rounds", flush=True)
    else:
        brain, lr = HouseholdBrain(), 1e-3
        print("a new, blank brain", flush=True)
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    # gen1: desktop work, taught alongside, one file per moment (household/desk_lessons.py)
    from household.desk_lessons import lessons
    dl = lessons(DESK_LESSONS, seed=int(time.time()) % 1000)
    d_obs = torch.tensor([r[0] for r in dl], dtype=torch.float32)
    d_act = torch.tensor([r[1] for r in dl])
    n = len(obs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            out = brain(obs[b])
            m = here[b]
            loss = F.cross_entropy(out["act"], act[b])
            has_t = tgt[b] >= 0
            if has_t.any():
                loss = loss + F.cross_entropy(out["target"][has_t], tgt[b][has_t])
            loss = loss + F.cross_entropy(out["condition"].reshape(-1, 3), cond[b].reshape(-1), weight=cond_w if COND_WEIGHTS else None, ignore_index=-1)
            fl = F.binary_cross_entropy_with_logits(out["foresight"], fore[b], reduction="none")
            wl = F.binary_cross_entropy_with_logits(out["words"], words[b], reduction="none").mean(-1)
            loss = loss + (fl * m).sum() / m.sum() + (wl * m).sum() / m.sum()
            d = torch.randint(0, len(d_obs), (256,))
            loss = loss + DESK_WEIGHT * F.cross_entropy(brain(d_obs[d])["desk"], d_act[d])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        if ep % 10 == 9 or ep == epochs - 1:
            print(f"epoch {ep + 1}: loss {total / n:.3f}", flush=True)
    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    save_brain(brain, SAVE, rounds + 1)
    print("saved", SAVE)
    return brain


if __name__ == "__main__":
    train()
