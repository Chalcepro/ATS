"""Measure *how* the agent moves, not just what it scores.

The reward curve can look fine while the policy is doing something
degenerate — the long-standing symptom here was "the agent walks in a
straight line". Reward alone never caught that, because with a flat
per-new-tile exploration reward a straight line is close to optimal.

Two action-semantics-agnostic metrics, both read off positions only:

  straightness  net displacement / path length, per episode.
                1.0 = perfectly straight line, ~0 = wandering in place.

  approach      of the steps where a nearest object exists and the agent
                moved, the fraction that *reduced* manhattan distance to
                it. This is the goal-seeking test: before the directional
                perception work the agent could not see which way anything
                was, so it had no way to beat chance here.

Compare a checkpoint against `--random` (the chance baseline) — an agent
that navigates should beat it on `approach` while scoring *lower* on
`straightness` than a blind line-walker.

    py -3.12 eval_behaviour.py --episodes 20
    py -3.12 eval_behaviour.py --episodes 20 --random
"""

from __future__ import annotations

import argparse
import math
import random

import torch

import config_rl
from ats_env import ATSEnvironment
from model_rl import RLPolicy


def _load_policy(fresh: bool) -> RLPolicy | None:
    if fresh:
        return None
    for ckpt in (config_rl.BEST_MODEL_PATH, config_rl.MODEL_PATH):
        if not ckpt.exists():
            continue
        data = torch.load(ckpt, map_location="cpu")
        if data.get("reward_scheme_version") != config_rl.REWARD_SCHEME_VERSION:
            print(f"[EVAL] skipping {ckpt.name}: reward_scheme_version mismatch")
            continue
        policy = RLPolicy(hidden_size=data.get("hidden_size", config_rl.MIND_HIDDEN_SIZE))
        policy.load_state_dict(data.get("model_state_dict", data["state_dict"]))
        print(f"[EVAL] loaded {ckpt.name} (hidden={policy.hidden_size})")
        return policy
    print("[EVAL] no compatible checkpoint found — evaluating an untrained policy")
    return RLPolicy()


def run(episodes: int, max_ticks: int, use_random: bool) -> None:
    policy = None if use_random else _load_policy(fresh=False)
    env = ATSEnvironment()

    straightness: list[float] = []
    approach_hits = approach_total = 0

    for _ in range(episodes):
        state = env.reset()
        start = (env.agent.x, env.agent.y)
        path_len = 0
        prev = start

        for _ in range(max_ticks):
            mask = env.agent.get_action_mask(env.world, env.day_night)
            _, _, obj_pos = env.world.nearest_object(env.agent.x, env.agent.y)
            before = (abs(obj_pos[0] - env.agent.x) + abs(obj_pos[1] - env.agent.y)) if obj_pos else None

            if use_random:
                valid = [i for i, m in enumerate(mask) if m]
                action = random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    logits, _ = policy(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                        action_mask=torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
                    )
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())

            state, _, done, _ = env.step(action)
            now = (env.agent.x, env.agent.y)

            if now != prev:
                path_len += 1
                if before is not None and obj_pos is not None:
                    after = abs(obj_pos[0] - now[0]) + abs(obj_pos[1] - now[1])
                    approach_total += 1
                    if after < before:
                        approach_hits += 1
            prev = now

            if done:
                break

        if path_len:
            net = math.hypot(env.agent.x - start[0], env.agent.y - start[1])
            straightness.append(net / path_len)

    label = "RANDOM" if use_random else "POLICY"
    mean_straight = sum(straightness) / len(straightness) if straightness else float("nan")
    approach_rate = 100.0 * approach_hits / approach_total if approach_total else float("nan")
    print(f"\n[EVAL:{label}] episodes={episodes}")
    print(f"[EVAL:{label}] straightness  : {mean_straight:.3f}   (1.0 = perfectly straight line)")
    print(f"[EVAL:{label}] approach rate : {approach_rate:.1f}%  ({approach_hits}/{approach_total} moves closed distance)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Behavioural probe for the ATS policy")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=1200, help="max ticks per episode")
    ap.add_argument("--random", action="store_true", help="evaluate a random policy baseline instead")
    args = ap.parse_args()
    run(args.episodes, args.ticks, args.random)
