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


def _policy_chooser():
    """The trained brain, playing greedily.

    Kept in the same table as the oracle and the random walker on purpose.
    "The rung trains to 8%" says nothing about WHY; the same row next to a
    competent player's says whether it is failing to fight, failing to arm
    itself, or failing to cross the room at all.
    """
    import torch

    import brain
    from model_rl import RLPolicy

    policy = RLPolicy()
    brain.load(policy)
    policy.eval()
    state = {"hx": None, "env": None}

    def pick(env, stage):
        if state["env"] is not env:
            state["env"] = env
            state["hx"] = policy.initial_hidden(1)
        s = torch.tensor(env._state(), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(env.action_mask(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _v, state["hx"] = policy(s, action_mask=m, hx=state["hx"])
        return int(torch.argmax(logits))
    return pick


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--grow-to", type=int, default=19)
    ap.add_argument("--policy", action="store_true",
                    help="also play the trained brain, greedily")
    ap.add_argument("--only", default=None,
                    help="one rung name: armed | hunted | warden")
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    rand = _random_chooser(rng)

    rungs = []
    for st in C.combat_tier(grid=a.grid, grow_to=a.grow_to):
        if a.only and st.name != a.only:
            continue
        r = st
        while r is not None:
            rungs.append(r)
            r = r.grown()

    print("\n  %d episodes a rung\n" % a.episodes)
    pol = _policy_chooser() if a.policy else None
    head = ["rung", "encounter", "oracle", "random", "died", "armed",
            "kills", "hits", "return"]
    print("  %-14s %8s %7s %7s %7s %7s %6s %6s %7s" % tuple(head))
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
        if pol is not None:
            b = play(st, a.episodes, a.seed, pol)
            print("  %-14s %7.0f%% %6.0f%% %6s %6.0f%% %6.0f%% %6.2f %6.2f %+7.2f"
                  % ("  brain", 100 * b["met"], 100 * b["win"], "-",
                     100 * b["dead"], 100 * b["armed"], b["kills"],
                     b["hits"], b["ret"]))
            print("  %-14s steps %.0f of %d" % ("", b["steps"], st.max_steps))

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
