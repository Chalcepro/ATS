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
import math
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
    ap.add_argument("--repair", action="store_true",
                    help="write the measured truth back into the brain's "
                         "`passed` list, adding rungs it clears and removing "
                         "ones it does not")
    a = ap.parse_args(argv)

    policy = RLPolicy()
    progress = brain.load(policy) or {}
    policy.eval()
    passed = set(progress.get("passed") or [])

    print("\n  brain says %d rungs passed\n" % len(passed))
    print("  %-20s %7s %6s  %-9s %s"
          % ("rung", "greedy", "bar", "claimed", ""))
    print("  " + "-" * 60)

    lost, near, ok_now = [], [], set()
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
            # How many standard errors below the bar, so a reading is not
            # called a loss when it is a coin-flip.
            #
            # A greedy rate is a binomial proportion: at 40 episodes the
            # standard error near 45% is 7.9 points, so a rung reading 40%
            # against a 45% bar is indistinguishable from one sitting exactly
            # on it. The first version of this file called every such rung
            # LOST and printed "that is catastrophic forgetting" underneath,
            # which is a confident claim built on a coin-flip - and it sent
            # me chasing drift that was mostly sampling.
            se = math.sqrt(max(g * (1.0 - g), 1e-9) / max(1, a.episodes))
            short = (r.pass_rate - g) / se if se > 0 else 0.0
            if g >= r.pass_rate:
                verdict = "ok"
                ok_now.add(name)
            elif short < 1.5:
                verdict = "close (%.1f SE)" % short
                near.append((name, g, r.pass_rate))
            elif name in passed:
                verdict = "LOST"
                lost.append((name, g, r.pass_rate))
            else:
                verdict = "below"
            print("  %-20s %6.0f%% %5.0f%%  %-9s %s"
                  % (name, 100 * g, 100 * r.pass_rate, claim, verdict))
            r = r.grown()

    if a.repair:
        # `passed` is what the GUI uses to choose the next rung to train, so
        # a rung missing from it is one the GUI will grind on however well
        # the brain plays it - 600 episodes went into `avoid 7x7` at 90%
        # for exactly that reason. This makes the list say what was measured.
        # Edited in place, NOT re-saved from the policy.
        #
        # brain.save(policy, learner=None) writes the weights and drops the
        # optimizer state, the adapted entropy coefficient and the tick
        # count, because those live on the learner. Calling it from a
        # read-only diagnostic therefore threw away Adam's moments and two
        # million ticks of history to change one list. Loading the blob and
        # putting back exactly one key cannot do that.
        import torch as _torch

        import brain as _brain
        now = sorted(ok_now)
        added = [k for k in now if k not in passed]
        dropped = [k for k in passed if k not in ok_now]
        blob = _torch.load(_brain.DEFAULT_PATH, map_location="cpu",
                           weights_only=False)
        prog = dict(blob.get("progress") or {})
        prog["passed"] = now
        blob["progress"] = prog
        tmp = _brain.DEFAULT_PATH.with_suffix(".pt.tmp")
        _torch.save(blob, tmp)
        tmp.replace(_brain.DEFAULT_PATH)
        print("")
        print("  repaired: %d rungs recorded" % len(now))
        for k in added:
            print("     + %s (clears its bar, was not listed)" % k)
        for k in dropped:
            print("     - %s (listed, does not clear it)" % k)
        if not added and not dropped:
            print("     nothing to change - the list already matched")

    print("")
    if near:
        print("  %d rung(s) sit within sampling noise of their bar:" % len(near))
        for name, g, bar in near:
            print("     %-20s %.0f%% against a %.0f%%" % (name, 100 * g, 100 * bar))
        print("  At %d episodes the error on a rate near 45%% is %.0f points, so"
              % (a.episodes, 100 * math.sqrt(0.45 * 0.55 / max(1, a.episodes))))
        print("  these are not distinguishable from passing. Re-run one with")
        print("  --episodes 150 --only <name> before treating it as a loss.")
        print("")
    if lost:
        print("  %d rung(s) are marked passed and are clearly not being played:"
              % len(lost))
        for name, g, bar in lost:
            print("     %-20s %.0f%% against a %.0f%% bar" % (name, 100 * g, 100 * bar))
        print("")
        print("  Beyond 1.5 standard errors, so this is drift rather than a")
        print("  coin-flip. Rehearsal (train_stage's `rehearse`) is what is")
        print("  supposed to stop it; consolidate.py is what repairs it.")
        return 1
    if near:
        return 0
    print("  every rung it claims, it still plays.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
