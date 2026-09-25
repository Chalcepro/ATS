"""Does the senior trail un-teach the bearing, the way the old one did?

This is the open question in the tier, and it is the one that decides whether
the first rung should exist at all.

The recorded harm, from trail_rung():

    nursery only       detour 1.98  sensitivity 0.6311  argmax 4.0/4
    trail only         detour 8.12  sensitivity 0.0210  argmax 1.0/4
    nursery -> trail   detour 3.96  sensitivity 0.1297  argmax 1.5/4

    (an untrained network probes at ~0.02-0.05)

The cause was not an ambiguous bearing but that the episode no longer ends at
the reward, so the critic learns "how many legs remain" rather than "how far
to this goal". The senior tier's middle rung cuts the trail from five legs to
two to walk that back before the compass rung asks for the bearing alone.

That is reasoning, not evidence. Run this after training the tier:

    python diag_senior.py

If sensitivity falls from senior-trail to senior-short to senior, the trail is
doing here what it did next to the nursery, and the first rung should go.
"""
from __future__ import annotations

import sys

import torch

import brain
import config_rl
import curriculum as C
from model_rl import RLPolicy

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
# The action each bearing should produce, in the same order as DIRS.
WANT = (config_rl.ACT_MOVE_FORWARD, config_rl.ACT_MOVE_BACKWARD,
        config_rl.ACT_MOVE_LEFT, config_rl.ACT_MOVE_RIGHT)


def probe(policy, stage, trials=200, seed0=770000):
    """Put a goal one tile away in each direction; does the policy follow it?

    `sensitivity` is how much the action distribution moves when only the
    bearing changes - an untrained network sits at 0.02-0.05 because its
    output barely depends on the goal at all. `argmax` is how many of the four
    directions it actually steps toward.
    """
    moved, correct, seen = [], 0, 0
    for k in range(trials):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        # Any tile with all four neighbours walkable, not just the centre.
        # Insisting on the middle found five usable rooms out of two hundred,
        # which is not a sample.
        spot = _open_spot(env, stage.grid)
        if spot is None:
            continue
        env.ax, env.ay = spot
        mid_x, mid_y = spot
        seen += 1

        dists = []
        for (dx, dy), want in zip(DIRS, WANT):
            env.goals = [(mid_x + dx, mid_y + dy)]
            if stage.sequence:
                env.remaining = stage.sequence
            s = torch.tensor(env._state(), dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(env.action_mask(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _v, _h = policy(s, m, hx=policy.initial_hidden(1))
            p = torch.softmax(logits, -1).squeeze(0)
            dists.append(p)
            if int(torch.argmax(p)) == want:
                correct += 1
        stacked = torch.stack(dists)
        # How far apart the four distributions are: the spread across the
        # bearings, averaged over actions.
        moved.append(float(stacked.std(dim=0).mean()))

    if not seen:
        return 0.0, 0.0, 0
    # `correct` counts up to four per room, so the per-room score is already
    # out of four. Dividing by rooms and then multiplying by four again
    # reported 12.8/4, which is not a number that can exist.
    return sum(moved) / len(moved), correct / seen, seen


def _open_spot(env, grid):
    """A tile whose four neighbours are all walkable, or None.

    The action mask blocks moves into walls, so probing a tile against a wall
    measures the mask rather than the bearing.
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


def main():
    policy = RLPolicy()
    try:
        brain.load(policy)
    except Exception as exc:
        print("no trained brain to probe (%s)" % exc)
        return 1
    policy.eval()

    rungs = [s for s in C.default_ladder() if s.name.startswith("senior")]
    if not rungs:
        print("no senior rungs in the ladder")
        return 1

    print("\n%-16s %5s %6s %13s %10s %7s"
          % ("rung", "grid", "legs", "sensitivity", "argmax", "rooms"))
    print("-" * 62)
    scores = []
    for st in rungs:
        sens, arg, seen = probe(policy, st)
        scores.append((st.name, sens))
        print("%-16s %5d %6d %13.4f %8.2f/4 %7d"
              % (st.name, st.grid, st.sequence, sens, arg, seen))

    print("")
    print("  untrained reference is about 0.02-0.05; chance argmax is 1.00/4")
    print("  The two can disagree, and did: the brain trained to junior probes")
    print("  at 0.0510 on a 15x15 room it has never seen, which reads as")
    print("  untrained, while its argmax is 3.23/4, which plainly is not.")
    print("  Sensitivity asks how much the whole distribution moves; argmax")
    print("  only asks whether the best action points the right way. For")
    print("  'does it follow the bearing at all', trust argmax.")
    if len(scores) >= 2:
        first, last = scores[0][1], scores[-1][1]
        if last < first * 0.5:
            print("  sensitivity FELL %.4f -> %.4f across the tier."
                  % (first, last))
            print("  That is the trail un-teaching the bearing again. Drop the")
            print("  senior-trail rung rather than trying to tune around it.")
        else:
            print("  sensitivity held (%.4f -> %.4f)." % (first, last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
