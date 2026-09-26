"""What does the brain still play, rung by rung?

`passed` is a historical claim. It records what was true when a rung was
cleared, and every rung trained afterwards moves the same weights - so the
list can say thirteen while the thing it describes plays four of them.

This asks the only question that means anything: greedy pass rate, now, on
every rung of the ladder including the grown ones. Greedy, not sampled,
because sampling is exploration and a near-uniform policy stumbles onto the
goal in a small room often enough to look like competence.

    python diag_ladder.py
    python diag_ladder.py --episodes 40 --only senior
"""
from __future__ import annotations

import argparse
import sys

import brain
import curriculum as C
from model_rl import RLPolicy
from train_curriculum import greedy_pass_rate, rung_key


def main(argv=None):
    ap = argparse.ArgumentParser()
    # 40, matching greedy_pass_rate's own default, because that is the number
    # the trainer promotes and skips on - a reading taken at a different
    # sample size is not comparable with the decision it is meant to inform.
    #
    # Measured the hard way: a 20-episode pass called `avoid 7x7` lost at 65%
    # against a 75% bar, and the trainer immediately re-measured it at 80% and
    # skipped it. One rung of six was sampling noise. The rungs that were
    # genuinely gone - senior 15x15 at 35%, satchel7 at 5% - sat far enough
    # below their bars that no sample size would have rescued them, which is
    # the difference to look for.
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--only", default=None, help="substring of a rung name")
    a = ap.parse_args(argv)

    policy = RLPolicy()
    progress = brain.load(policy) or {}
    policy.eval()
    passed = set(progress.get("passed") or [])

    print("\n  brain says %d rungs passed\n" % len(passed))
    print("  %-20s %7s %6s  %-9s %s"
          % ("rung", "greedy", "bar", "claimed", ""))
    print("  " + "-" * 60)

    lost = []
    for st in C.default_ladder():
        r = st
        while r is not None:
            name = rung_key(r)
            if a.only and a.only not in name:
                r = r.grown()
                continue
            g = greedy_pass_rate(C.CurriculumEnv(r, seed=0), policy, r,
                                 episodes=a.episodes)
            claim = "passed" if name in passed else "-"
            if g >= r.pass_rate:
                verdict = "ok"
            elif name in passed:
                verdict = "LOST"
                lost.append((name, g, r.pass_rate))
            else:
                verdict = "below"
            print("  %-20s %6.0f%% %5.0f%%  %-9s %s"
                  % (name, 100 * g, 100 * r.pass_rate, claim, verdict))
            r = r.grown()

    print("")
    if lost:
        print("  %d rung(s) are marked passed and are not being played:" % len(lost))
        for name, g, bar in lost:
            print("     %-20s %.0f%% against a %.0f%% bar" % (name, 100 * g, 100 * bar))
        print("")
        print("  That is catastrophic forgetting, not a hard rung. Rehearsal")
        print("  (train_stage's `rehearse`) is what is supposed to stop it.")
        return 1
    print("  every rung it claims, it still plays.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
