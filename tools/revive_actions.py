"""Find actions the trained brain can never pick, and give them a fair chance.

An action the mask hides gets no gradient while hidden, so its logit stays
put while the ones it does sample drift. Far enough below and its probability
rounds to nothing: never sampled, so never learned, so unmasking it changes
nothing. Measured 2026-10-17 on the brain after a 7-hour run: interact sat 83
below the moves (p ~ 1e-38) although E had been always available all run, and
attack 23 below (p ~ 1e-10), so it never pressed E and never won a fight room.

This plays the brain in the world for --ticks, reports each action's median
gap below the median allowed action, and with --apply gives every action
whose gap is over --gap the average of the live actions' rows
(RLPolicy.revive_actions). It writes brain.pt, model_rl.pt and model_best.pt
and keeps a copy of each as <name>.pre_revive.pt first.

    python tools/revive_actions.py               # report only
    python tools/revive_actions.py --apply
"""

from __future__ import annotations

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import argparse
import shutil
import statistics

import torch

import brain
import config_rl
from agent import ACTION_NAMES
from ats_env import ATSEnvironment
from model_rl import RLPolicy


def measure(policy, ticks):
    """Per action: median gap below the median allowed logit, and how often allowed."""
    env = ATSEnvironment()
    state, hx = env.reset(), policy.initial_hidden(1)
    n = policy.action_size
    gaps, allowed = [[] for _ in range(n)], [0] * n
    for _ in range(ticks):
        mask = env.agent.get_action_mask(env.world, env.day_night)
        with torch.no_grad():
            raw, _, hx = policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0), hx=hx)
        raw = raw[0]
        live = sorted(raw[i].item() for i in range(n) if mask[i])
        median = live[len(live) // 2]
        for i in range(n):
            gaps[i].append(median - raw[i].item())
            allowed[i] += mask[i]
        masked = raw + (1 - torch.tensor(mask, dtype=torch.float32)) * -1e8
        action = int(torch.distributions.Categorical(logits=masked).sample())
        state, _, done, _ = env.step(action)
        if done:
            state, hx = env.reset(), policy.initial_hidden(1)
    return [statistics.median(g) for g in gaps], [a / ticks for a in allowed]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--gap", type=float, default=15.0,
                    help="revive actions more than this far below the median allowed logit")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    shape = brain.saved_shape()
    if not shape:
        sys.exit("no brain at %s" % brain.DEFAULT_PATH)
    policy = RLPolicy(hidden_size=shape["hidden_size"])
    if not brain.load(policy):
        sys.exit("brain did not load")
    policy.eval()

    gaps, allowed = measure(policy, args.ticks)
    dead = [i for i, g in enumerate(gaps) if g > args.gap]
    print("\naction          gap below median   allowed")
    for i, (g, a) in enumerate(zip(gaps, allowed)):
        print("%2d %-14s %8.1f %s   %5.1f%%"
              % (i, ACTION_NAMES.get(i, str(i)), g, " DEAD" if i in dead else "     ", 100 * a))
    if not dead:
        print("\nnothing to revive")
        return
    if not args.apply:
        print("\n%d dead; --apply to revive them" % len(dead))
        return

    live = [i for i in range(policy.action_size) if i not in dead]
    # Each file keeps its own weights (model_best is an older snapshot than
    # brain.pt); only the dead rows are rewritten, from that file's live rows.
    for path in (brain.DEFAULT_PATH, config_rl.MODEL_PATH, config_rl.BEST_MODEL_PATH):
        if not path.exists():
            continue
        shutil.copy2(path, path.with_name(path.stem + ".pre_revive.pt"))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("policy", "model_state_dict", "state_dict"):
            if key in blob:
                one = RLPolicy(hidden_size=shape["hidden_size"])
                one.load_state_dict(blob[key])
                one.revive_actions(dead, live)
                blob[key] = one.state_dict()
        torch.save(blob, path)
        print("revived %d action(s) in %s (old copy kept as %s)"
              % (len(dead), path.name, path.stem + ".pre_revive.pt"))

    policy.revive_actions(dead, live)
    after, _ = measure(policy, min(args.ticks, 1000))
    print("after: " + ", ".join("%s %.1f" % (ACTION_NAMES.get(i, i), after[i]) for i in dead))


if __name__ == "__main__":
    main()
