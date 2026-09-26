"""Is the combat tier a fight, and is it a winnable one?

Four numbers decide whether a combat rung is worth training, and every one
of them has a failure mode that looks fine from the outside:

  encounter   how often a hostile comes within sight. The whole ask was
              "higher chance of encountering an enemy". A rung can carry
              four hostiles and still be a rung about walking.
  oracle      a competent player's success rate. Below ~90% the rung is not
              hard, it is broken, and no amount of training fixes it.
  random      a random walker's success rate. This is the floor the pass
              bar has to clear to mean anything.
  died        how often the oracle dies. A competent player dying is the
              clock, the bite or the pursuit being wrong, not difficulty.

It also reports what the oracle DID - armed itself, how many it killed, how
many blows it took - because "oracle clears 95%" is equally true of a rung
where it fought its way across and one where it strolled past furniture,
and those are not the same rung.

    py -3 diag_combat.py
    py -3 diag_combat.py --episodes 200
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

import config_rl
import curriculum as C


def play(stage, episodes, seed0, chooser):
    got = dict(win=0, dead=0, met=0, armed=0, kills=0, hits=0, steps=0,
               ret=0.0)
    for k in range(episodes):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        done, info, total = False, {}, 0.0
        while not done:
            _, r, done, info = env.step(chooser(env, stage))
            total += r
        got["win"] += bool(info.get("success"))
        got["dead"] += bool(info.get("dead"))
        got["met"] += bool(info.get("met"))
        got["armed"] += bool(info.get("armed"))
        got["kills"] += info.get("kills", 0)
        got["hits"] += info.get("hits_taken", 0)
        got["steps"] += info.get("steps", 0)
        got["ret"] += total
    n = float(episodes)
    return {k: v / n for k, v in got.items()}


def _random_chooser(rng):
    def pick(env, stage):
        allowed = [i for i, ok in enumerate(env.action_mask()) if ok]
        return rng.choice(allowed)
    return pick


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--grow-to", type=int, default=19)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    rand = _random_chooser(rng)

    rungs = []
    for st in C.combat_tier(grid=a.grid, grow_to=a.grow_to):
        r = st
        while r is not None:
            rungs.append(r)
            r = r.grown()

    print("\n  %d episodes a rung\n" % a.episodes)
    print("  %-14s %8s %7s %7s %7s %7s %6s %6s %7s"
          % ("rung", "encounter", "oracle", "random", "died", "armed",
             "kills", "hits", "return"))
    print("  " + "-" * 82)

    bad = []
    for st in rungs:
        o = play(st, a.episodes, a.seed, C.oracle_step)
        r = play(st, a.episodes, a.seed, rand)
        name = "%s %dx%d" % (st.name, st.grid, st.grid)
        print("  %-14s %7.0f%% %6.0f%% %6.0f%% %6.0f%% %6.0f%% %6.2f %6.2f %+7.2f"
              % (name, 100 * o["met"], 100 * o["win"], 100 * r["win"],
                 100 * o["dead"], 100 * o["armed"], o["kills"], o["hits"],
                 o["ret"]))
        if o["ret"] <= 0:
            bad.append((name, "a competent agent scores negative"))
        if o["win"] < 0.85:
            bad.append((name, "oracle clears only %.0f%%" % (100 * o["win"])))
        if o["met"] < 0.60:
            bad.append((name, "hostiles met in only %.0f%% of episodes"
                        % (100 * o["met"])))
        if r["win"] > st.pass_rate * 0.6:
            bad.append((name, "random walker gets %.0f%% against a %.0f%% bar"
                        % (100 * r["win"], 100 * st.pass_rate)))

    print("")
    if bad:
        for name, why in bad:
            print("  PROBLEM  %-14s %s" % (name, why))
        return 1
    print("  every rung is winnable by a competent agent, not by luck,")
    print("  and the hostiles are actually met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
