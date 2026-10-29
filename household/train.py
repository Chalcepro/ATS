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
from household.model import HouseholdBrain
from household.teachings import EXAM_FROM, TEACHINGS, truth
from household.words import VOCAB, narrate
from household.world import ACTIONS, CARE, MAX_PEOPLE, Household

TRAINED = [h for h in HAZARDS if h not in HELD_BACK]
SAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "household_brain.pt")


def _world(seed: int) -> Household:
    rng = random.Random(seed)
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


def episode(seed: int, hours: int = 36) -> list[tuple]:
    rng = random.Random(seed * 7 + 1)
    w = _world(seed)
    # Now and then a sense is missing, so it learns to get there another way:
    # no nose - the flies still show the smell; no ears - the hand on the chest
    # still shows the cough.
    w.senses = {"ears": rng.random() > 0.2, "nose": rng.random() > 0.2}
    rows = []
    for _ in range(hours):
        if all(not p.alive for p in w.people) and rng.random() < 0.5:
            break
        act, tgt = best_action(w)
        words = [[int(v in narrate(p)) for v in VOCAB] for p in w.people]
        rows.append((w.recall(), ACTIONS.index(act), tgt if act in CARE else -1,
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


def train(episodes: int = 2000, epochs: int = 80, seed: int = 0) -> HouseholdBrain:
    torch.manual_seed(seed)
    t = time.time()
    rows = collect(episodes)
    print(f"collected {len(rows)} moments from {episodes} worlds in {time.time() - t:.0f}s", flush=True)
    obs, act, tgt, cond, fore, words, here = tensors(rows)
    brain = HouseholdBrain()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)
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
            loss = loss + F.cross_entropy(out["condition"].reshape(-1, 3), cond[b].reshape(-1), ignore_index=-1)
            fl = F.binary_cross_entropy_with_logits(out["foresight"], fore[b], reduction="none")
            wl = F.binary_cross_entropy_with_logits(out["words"], words[b], reduction="none").mean(-1)
            loss = loss + (fl * m).sum() / m.sum() + (wl * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        if ep % 10 == 9 or ep == epochs - 1:
            print(f"epoch {ep + 1}: loss {total / n:.3f}", flush=True)
    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    torch.save(brain.state_dict(), SAVE)
    print("saved", SAVE)
    return brain


if __name__ == "__main__":
    train()
