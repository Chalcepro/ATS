"""Smoke test: does the PPO learner actually learn?

Trains a fresh ``RLPolicy`` on ``SanityEnv`` through the *same*
``ContinualLearner`` that ``main.py`` uses, and checks that behaviour
improves — higher reward, and a higher goal-reach rate than both its own
early episodes and a random policy.

    py -3 sanity_train.py        # ~45s on CPU, deterministic

Run it after any change to model_rl.py / continual_learner.py /
config_rl.py before committing hours to a full ATS training run. A pass
means the RL plumbing is sound; a fail means the learner is broken and no
amount of world-tuning will help.
"""

from __future__ import annotations

import random
import statistics
import sys

import torch

from mind.continual_learner import ContinualLearner
from model_rl import RLPolicy
from sanity_env import SanityEnv

EPISODES = 600
WINDOW = 100
SEED = 0


def _random_reach_rate(n: int = 200) -> float:
    rng = random.Random(SEED)
    env = SanityEnv(seed=SEED)
    reached = 0
    info: dict = {}
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(rng.choice([0, 1, 2, 3]))
        reached += int(info.get("dist", 1) == 0)
    return reached / n


def _train() -> list[tuple[float, bool]]:
    torch.manual_seed(SEED)
    policy = RLPolicy()
    env = SanityEnv(seed=SEED)
    learner = ContinualLearner(policy)
    out: list[tuple[float, bool]] = []
    for _ in range(EPISODES):
        state = env.reset()
        done = False
        total = 0.0
        info: dict = {}
        while not done:
            mask = env.action_mask()
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value, _ = policy(st, action_mask=mt)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            action = int(a.item())
            state, reward, done, info = env.step(action)
            total += reward
            learner.collect(
                state=state, action=action, reward=reward,
                log_prob=logp, value=value.squeeze(0),
                action_mask=mask, done=done,
            )
            learner.maybe_update()
        out.append((total, info.get("dist", 1) == 0))
    return out


def main() -> int:
    base_reach = _random_reach_rate()
    hist = _train()

    first, last = hist[:WINDOW], hist[-WINDOW:]
    first_r = statistics.mean(x[0] for x in first)
    last_r = statistics.mean(x[0] for x in last)
    first_reach = sum(x[1] for x in first) / WINDOW
    last_reach = sum(x[1] for x in last) / WINDOW

    print(f"\n[SANITY] random policy  reach rate      : {base_reach:.1%}")
    print(f"[SANITY] learner first {WINDOW}  mean reward : {first_r:+.3f}   reach {first_reach:.1%}")
    print(f"[SANITY] learner last  {WINDOW}  mean reward : {last_r:+.3f}   reach {last_reach:.1%}")

    checks = {
        "reward climbed":            last_r > first_r + 0.30,
        "final reward positive":     last_r > 0.15,
        "beats random reach rate":   last_reach > base_reach + 0.12,
        "reach rate improved":       last_reach > first_reach + 0.12,
    }
    for name, ok in checks.items():
        print(f"[SANITY]   {'ok ' if ok else 'FAIL'} {name}")

    passed = all(checks.values())
    print(f"\n[SANITY] {'PASS' if passed else 'FAIL'} — PPO learner "
          f"{'learns' if passed else 'is NOT learning properly'}.\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
