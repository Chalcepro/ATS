"""Train one policy up the ladder: nursery -> primary -> junior.

    py -3 train_curriculum.py                 # the whole ladder
    py -3 train_curriculum.py --stage nursery # just one rung
    py -3 train_curriculum.py --max-grid 15

One network the whole way. A stage is passed when the policy succeeds in
`pass_rate` of the last `window` episodes - promotion is earned, never
scheduled, because a stage still being failed is a stage that still has
something to teach.

Before any training happens it runs `best_case_return` on every stage and
refuses to start if a competent agent would score negative. That check is
here because the full world did not have it: an ATS episode that survived to
max ticks scored -3281 while dying outright cost -10, so the policy correctly
learned to end episodes early. The reward function was the bug, and no amount
of training would have found it. The same check caught two stages of this
file's own ladder on its first run.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque

import torch

import brain
import config_rl
from curriculum import CurriculumEnv, best_case_return, default_ladder
from mind.continual_learner import ContinualLearner
from model_rl import RLPolicy


def rung_key(stage):
    """A stable name for one rung, so 'already passed' survives a restart."""
    return "%s %dx%d" % (stage.name, stage.grid, stage.grid)


def run_episode(env, policy, learner):
    state = env.reset()
    done = False
    total = 0.0
    info = {}
    while not done:
        mask = env.action_mask()
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(st, action_mask=mt)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        action = int(a.item())
        state, reward, done, info = env.step(action)
        total += reward
        learner.collect(state=state, action=action, reward=reward,
                        log_prob=logp, value=value.squeeze(0),
                        action_mask=mask, done=done)
        learner.maybe_update()
    return total, info


def train_stage(stage, policy, learner, max_episodes, seed=0, quiet=False,
                on_tick=None):
    """Returns (passed, episodes_used, last_window_stats)."""
    env = CurriculumEnv(stage, seed=seed)
    wins = deque(maxlen=stage.window)
    rewards = deque(maxlen=stage.window)
    got = deque(maxlen=stage.window)
    died = deque(maxlen=stage.window)
    started = time.time()

    for ep in range(1, max_episodes + 1):
        total, info = run_episode(env, policy, learner)
        wins.append(1 if info.get("success") else 0)
        rewards.append(total)
        got.append(info.get("collected", 0))
        died.append(1 if info.get("dead") else 0)

        if not quiet and ep % 50 == 0:
            # Over the window, not one episode: a single episode says nothing,
            # and "how is it failing" - starving for goals or dying - is the
            # thing that tells you which rung is wrong.
            print("    ep %-5d  reward %+7.2f  success %3.0f%%  "
                  "collected %.2f  died %3.0f%%"
                  % (ep, statistics.mean(rewards),
                     100.0 * sum(wins) / len(wins),
                     statistics.mean(got), 100.0 * sum(died) / len(died)))

        # Saved during the rung, not only at the end of it. A rung can take
        # an hour; losing an hour to a closed terminal is the thing this whole
        # file exists to stop.
        if on_tick and ep % 200 == 0:
            on_tick(ep)

        if len(wins) == stage.window and sum(wins) / len(wins) >= stage.pass_rate:
            return True, ep, (statistics.mean(rewards), sum(wins) / len(wins),
                              time.time() - started)

    stats = (statistics.mean(rewards) if rewards else 0.0,
             (sum(wins) / len(wins)) if wins else 0.0, time.time() - started)
    return False, max_episodes, stats


def preflight(ladder):
    """Refuse to train a ladder that cannot be climbed."""
    print("preflight - can a competent agent score positive on each rung?")
    bad = []
    for st in ladder:
        chain, g = [st], st.grown()
        while g:
            chain.append(g)
            g = g.grown()
        for k in chain:
            v = best_case_return(k, trials=20)
            flag = "ok " if v > 0 else "BAD"
            print("   %s %-8s %2dx%-2d goals=%-3d greedy %+7.2f"
                  % (flag, k.name, k.grid, k.grid, k.goals, v))
            if v <= 0:
                bad.append(k)
    if bad:
        print("\nRefusing to train: %d stage(s) where playing well still loses."
              % len(bad))
        print("Fix the reward before spending hours on it - that is the bug that")
        print("cost the full world 725 episodes.")
        return False
    print()
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="nursery | primary | junior")
    ap.add_argument("--max-grid", type=int, default=13)
    ap.add_argument("--episodes", type=int, default=4000,
                    help="cap per rung before giving up on it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore any saved weights")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--redo", action="store_true",
                    help="re-train rungs already marked passed")
    a = ap.parse_args(argv)

    ladder = default_ladder(max_grid=a.max_grid)
    if a.stage:
        ladder = [s for s in ladder if s.name == a.stage]
        if not ladder:
            print("no such stage:", a.stage)
            return 2

    if not a.skip_preflight and not preflight(ladder):
        return 1

    torch.manual_seed(a.seed)
    policy = RLPolicy()
    learner = ContinualLearner(policy)

    # Weights, optimiser moments, the adapted entropy coefficient, and which
    # rungs are already behind it - all of it, or a week-old brain re-learns
    # the nursery to find out it already knew it. See brain.py.
    progress = {} if a.fresh else brain.load(policy, learner)
    if not a.fresh and not progress and not brain.is_loadable(policy):
        # Either there is no brain yet or the one on disk belongs to an older
        # network. Either way the GUI's old checkpoints may still fit, and
        # starting from them beats starting from noise.
        brain.adopt_legacy(policy)
    passed = list(progress.get("passed") or [])
    episodes_total = int(progress.get("episodes") or 0)

    def stash():
        brain.save(policy, learner,
                   {"passed": passed, "episodes": episodes_total})

    passed_all = True
    for stage in ladder:
        rung = stage
        while rung is not None:
            key = rung_key(rung)
            if key in passed and not a.redo:
                print("\n=== %s   already passed, skipping" % key)
                rung = rung.grown()
                continue

            print("\n=== %s   goals=%d hazards=%d hostiles=%d  pass %.0f%% of %d"
                  % (key, rung.goals, rung.hazards, rung.hostiles,
                     rung.pass_rate * 100, rung.window))
            ok, used, (mr, wr, secs) = train_stage(
                rung, policy, learner, a.episodes, seed=a.seed,
                on_tick=lambda ep: stash())
            episodes_total += used
            print("    %s after %d episodes   reward %+.2f   success %.0f%%   %.0fs"
                  % ("PASSED" if ok else "gave up", used, mr, wr * 100, secs))

            if ok and key not in passed:
                passed.append(key)
            stash()

            if not ok:
                passed_all = False
                break
            rung = rung.grown()

        if not passed_all:
            break

    print("\nbrain -> %s   (%d rungs passed, %d episodes lived)"
          % (brain.DEFAULT_PATH, len(passed), episodes_total))
    print("ladder %s" % ("complete" if passed_all else "stopped early"))
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
