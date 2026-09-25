"""Did training the senior tier un-teach the bearing?

The recorded harm, from trail_rung():

    nursery only       detour 1.98  sensitivity 0.6311  argmax 4.0/4
    trail only         detour 8.12  sensitivity 0.0210  argmax 1.0/4
    nursery -> trail   detour 3.96  sensitivity 0.1297  argmax 1.5/4

    (an untrained network probes at ~0.02-0.05, chance argmax is 1.0/4)

The cause was not an ambiguous bearing but that the episode no longer ends at
the reward, so the critic learns "how many legs remain" rather than "how far
to this goal".

Why this compares before and after, not rung against rung
---------------------------------------------------------
The first version of this file probed each senior rung separately and printed
three rows. They were identical to four decimal places, because **the state
vector does not encode the rung**: at the same spot with the same goal, the
observation for the five-leg trail, the two-leg trail and the single goal is
byte-identical - 0 differing slots out of the whole vector. It was measuring
one thing three times and labelling it three rungs, and it could never have
detected the thing it existed to detect.

(It also set `env.remaining`, which does not exist on CurriculumEnv. That
silently created an unused attribute and changed nothing.)

The bearing lives in the weights, not in the rung, so the only comparison
that means anything is the same probe before and after training. This stores
its reading and diffs against the last one.

    python diag_senior.py            measure, and compare with last time
    python diag_senior.py --save     measure and make this the baseline
"""
from __future__ import annotations

import json
import os
import sys

import torch

import brain
import config_rl
import curriculum as C
from model_rl import RLPolicy

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
WANT = (config_rl.ACT_MOVE_FORWARD, config_rl.ACT_MOVE_BACKWARD,
        config_rl.ACT_MOVE_LEFT, config_rl.ACT_MOVE_RIGHT)

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "logs", "bearing.json")


def _open_spot(env, grid):
    """A tile whose four neighbours are all walkable, or None.

    The action mask blocks moves into walls, so probing a tile against a wall
    measures the mask rather than the bearing. Insisting on the exact centre
    found five usable rooms in two hundred, which is not a sample.
    """
    for y in range(1, grid - 1):
        for x in range(1, grid - 1):
            try:
                if env._tile(x, y) != C.T_FLOOR:
                    continue
                if all(env._tile(x + dx, y + dy) == C.T_FLOOR
                       for dx, dy in DIRS):
                    return x, y
            except Exception:
                continue
    return None


def probe(policy, stage, trials=300, seed0=770000):
    """Put a goal one tile away in each direction; does the policy follow it?

    `argmax` is how many of the four directions it steps toward, out of four.
    `sensitivity` is how far the four action distributions sit apart when only
    the bearing changed. They can disagree - a brain can point the right way
    while barely changing its distribution - and argmax is the one to trust
    for "does it use the bearing at all".
    """
    moved, correct, seen = [], 0, 0
    for k in range(trials):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        spot = _open_spot(env, stage.grid)
        if spot is None:
            continue
        env.ax, env.ay = spot
        seen += 1

        dists = []
        for (dx, dy), want in zip(DIRS, WANT):
            env.goals = [(spot[0] + dx, spot[1] + dy)]
            s = torch.tensor(env._state(), dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(env.action_mask(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _v, _h = policy(s, m, hx=policy.initial_hidden(1))
            p = torch.softmax(logits, -1).squeeze(0)
            dists.append(p)
            if int(torch.argmax(p)) == want:
                correct += 1
        moved.append(float(torch.stack(dists).std(dim=0).mean()))

    if not seen:
        return 0.0, 0.0, 0
    return sum(moved) / len(moved), correct / seen, seen


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    policy = RLPolicy()
    try:
        progress = brain.load(policy) or {}
    except Exception as exc:
        print("no trained brain to probe (%s)" % exc)
        return 1
    policy.eval()

    # One room, because one brain. The rung it came from is not visible in the
    # observation, so probing several would repeat the same number.
    rungs = [s for s in C.default_ladder() if s.name.startswith("senior")]
    stage = rungs[0] if rungs else C.default_ladder()[-1]

    sens, arg, seen = probe(policy, stage)
    passed = list(progress.get("passed") or [])
    now = {"sensitivity": round(sens, 4), "argmax": round(arg, 3),
           "rooms": seen, "rungs_passed": len(passed),
           "grid": stage.grid}

    print("")
    print("  probed on a %dx%d room, %d usable rooms" % (stage.grid, stage.grid,
                                                         seen))
    print("  argmax       %.2f / 4      (chance is 1.00)" % arg)
    print("  sensitivity  %.4f        (untrained is about 0.02-0.05)" % sens)
    print("  rungs passed %d" % len(passed))

    before = None
    if os.path.isfile(STORE):
        try:
            before = json.load(open(STORE))
        except Exception:
            before = None

    if before:
        print("")
        print("  last reading: argmax %.2f/4, sensitivity %.4f, %d rungs passed"
              % (before.get("argmax", 0) * 1, before.get("sensitivity", 0),
                 before.get("rungs_passed", 0)))
        d_arg = arg - before.get("argmax", 0)
        print("  change:       argmax %+0.2f, sensitivity %+0.4f"
              % (d_arg, sens - before.get("sensitivity", 0)))
        if d_arg < -0.5:
            print("")
            print("  THE BEARING FELL. That is the trail doing here what it did")
            print("  next to the nursery. Drop the trail rungs rather than")
            print("  trying to tune around them.")
        elif arg < 1.5:
            print("  argmax is at chance - the bearing is not being used.")
        else:
            print("  the bearing held.")
    else:
        print("\n  no previous reading stored")

    if "--save" in argv or before is None:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        json.dump(now, open(STORE, "w"), indent=2)
        print("  stored as the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
