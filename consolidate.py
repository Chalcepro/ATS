"""Hold every rung at once, instead of one at a time.

The problem this exists for
---------------------------
`train_curriculum.py` trains one rung, promotes, and moves on. Rehearsal
mixes a fraction of earlier rungs into that training, and it helps, but the
thing being *scored* is always a single rung - so the optimiser is always
being asked to get better at one room, and the others are only defended.

The result is a ladder that ratchets: each pass gains the rung it is on and
gives back a little of the rungs below, and after enough passes the `passed`
list is a history rather than a description. Measured on a brain claiming
nineteen rungs:

    senior  19x19  42% against a 50% bar
    satchel   5x5  72% against an 85% bar
    armed   19x19  45% against a 50% bar
    warden  15x15  40% against a 45% bar
    warden  17x17  38% against a 45% bar
    warden  19x19  42% against a 45% bar

What this does instead
----------------------
Interleaved training. Every episode picks a rung, weighted toward whichever
ones are currently furthest below their bar, and they are all scored. No rung
is "the one being trained", so there is no rung the optimiser is free to
trade away.

This is the standard answer to catastrophic forgetting and it is not free:
it will not push any single rung as high as training that rung alone would.
It is for after the ladder is climbed, to make the claim true, not for
climbing it.

    python consolidate.py                    # hold everything it claims
    python consolidate.py --also warden      # ...and anything matching this
    python consolidate.py --episodes 4000
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import deque

import torch

import brain
import curriculum as C
from mind.continual_learner import ContinualLearner
from model_rl import RLPolicy
from train_curriculum import greedy_pass_rate, run_episode, rung_key


def all_rungs():
    out = []
    for st in C.default_ladder():
        r = st
        while r is not None:
            out.append(r)
            r = r.grown()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000,
                    help="total interleaved episodes before giving up")
    ap.add_argument("--also", default=None,
                    help="also hold rungs whose name contains this")
    ap.add_argument("--check-every", type=int, default=400)
    ap.add_argument("--check-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    policy = RLPolicy()
    learner = ContinualLearner(policy)
    progress = brain.load(policy, learner) or {}
    passed = list(progress.get("passed") or [])
    episodes_total = int(progress.get("episodes") or 0)

    keep = []
    for r in all_rungs():
        name = rung_key(r)
        if name in passed or (a.also and a.also in name):
            keep.append(r)
    if not keep:
        print("nothing claimed as passed - climb the ladder first")
        return 2

    print("\n  holding %d rung(s) at once\n" % len(keep))
    envs = {rung_key(r): C.CurriculumEnv(r, seed=a.seed + 4000 + i)
            for i, r in enumerate(keep)}
    wins = {rung_key(r): deque(maxlen=40) for r in keep}
    rng = random.Random(a.seed + 17)
    started = time.time()

    def weights():
        """Favour the rungs currently furthest below their bar.

        A rung comfortably above its bar still gets picked sometimes - that
        is what stops it drifting - but the episodes go where the deficit is.
        """
        w = []
        for r in keep:
            k = rung_key(r)
            seen = wins[k]
            rate = (sum(seen) / len(seen)) if seen else 0.0
            w.append(1.0 + 8.0 * max(0.0, r.pass_rate - rate))
        return w

    def report():
        print("  %-20s %7s %6s" % ("rung", "window", "bar"))
        for r in keep:
            k = rung_key(r)
            seen = wins[k]
            rate = (sum(seen) / len(seen)) if seen else 0.0
            print("    %-18s %6.0f%% %5.0f%%  %s"
                  % (k, 100 * rate, 100 * r.pass_rate,
                     "" if rate >= r.pass_rate else "<"))

    for ep in range(1, a.episodes + 1):
        r = rng.choices(keep, weights=weights())[0]
        k = rung_key(r)
        _total, info = run_episode(envs[k], policy, learner, None, ep)
        wins[k].append(1 if info.get("success") else 0)
        episodes_total += 1

        if ep % a.check_every == 0:
            print("\n  --- %d episodes, %.0fs ---" % (ep, time.time() - started))
            report()
            # Greedy is what promotion and diag_ladder both use; the rolling
            # window above is the sampling policy and reads high.
            short = []
            for rr in keep:
                g = greedy_pass_rate(C.CurriculumEnv(rr, seed=a.seed), policy,
                                     rr, episodes=a.check_episodes)
                if g < rr.pass_rate:
                    short.append((rung_key(rr), g, rr.pass_rate))
            brain.save(policy, learner,
                       {"passed": passed, "episodes": episodes_total})
            if not short:
                print("\n  every rung clears its bar under greedy play.")
                print("  brain saved. %d episodes." % episodes_total)
                return 0
            print("  still short under greedy: %s"
                  % ", ".join("%s %.0f%%/%.0f%%" % (n, 100 * g, 100 * b)
                              for n, g, b in short))

    brain.save(policy, learner, {"passed": passed, "episodes": episodes_total})
    print("\n  gave up after %d episodes; brain saved." % a.episodes)
    report()
    return 1


if __name__ == "__main__":
    sys.exit(main())
