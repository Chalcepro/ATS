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
import random
import statistics
import sys
import time
from collections import deque

import torch

import brain
import trace as pathtrace
import config_rl
from curriculum import CurriculumEnv, best_case_return, default_ladder
from mind.continual_learner import ContinualLearner
from model_rl import RLPolicy


def rung_key(stage):
    """A stable name for one rung, so 'already passed' survives a restart."""
    return "%s %dx%d" % (stage.name, stage.grid, stage.grid)


def run_episode(env, policy, learner, tracer=None, episode=0):
    state = env.reset()
    if tracer:
        tracer.begin(env, episode)
    done = False
    total = 0.0
    info = {}
    # Memory starts empty every episode.  Carrying it across is how an agent
    # learns to be confused about which room it is in.
    hx = policy.initial_hidden(1)
    while not done:
        mask = env.action_mask()
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            hx_in = hx
            logits, value, hx = policy(st, action_mask=mt, hx=hx)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        action = int(a.item())
        # The state the action was CHOSEN IN, before env.step overwrites it.
        # PPO re-evaluates log pi(a|s) from the stored state; handing it the
        # state the action led TO makes ratio = exp(new - old) compare two
        # different distributions, and the gradient then pushes pi(a|s') for
        # an advantage earned in s. In a grid that reads as "one tile west of
        # here, prefer west" - a constant bias instead of a mapping.
        prev_state = state
        state, reward, done, info = env.step(action)
        if tracer:
            tracer.record(env, action, reward, info)
        total += reward
        learner.collect(state=prev_state, action=action, reward=reward,
                        log_prob=logp, value=value.squeeze(0),
                        action_mask=mask, done=done,
                        hidden=hx_in.squeeze(0).tolist())
        learner.maybe_update()
    if tracer:
        tracer.end(env, info)
    return total, info


def greedy_pass_rate(env, policy, stage, episodes=40, seed0=999000):
    """Success rate with argmax actions - no sampling, no luck.

    Separate seeds from training so this is not scored on rooms the agent
    has just been walked through.
    """
    policy.eval()
    try:
        succ = 0
        for k in range(episodes):
            e = type(env)(stage, seed=seed0 + k)
            state = e._state()
            hx = policy.initial_hidden(1)
            done = False
            n = 0
            info = {}
            while not done and n < stage.max_steps:
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mt = torch.tensor(e.action_mask(), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _, hx = policy(st, action_mask=mt, hx=hx)
                state, _, done, info = e.step(int(torch.argmax(logits)))
                n += 1
            succ += bool(info.get("success"))
        return succ / episodes
    finally:
        policy.train()


def train_stage(stage, policy, learner, max_episodes, seed=0, quiet=False,
                on_tick=None, tracer=None, rehearse=(), rehearse_rate=0.40):
    """Returns (passed, episodes_used, last_window_stats).

    `rehearse` is the rungs already behind this one. Two episodes in five
    are drawn from them, weighted toward the nearest, because nothing else
    in this file protects them.

    Why that is needed: ContinualLearner is online PPO with a rolling buffer.
    There is no replay, no EWC, no penalty for moving away from the old
    weights - so training a new rung overwrites the last one and the "passed"
    list becomes a historical claim rather than a current one. Measured after
    training the senior tier with no rehearsal, on rungs that had all passed:

        primary  9x9   70%  ok
        junior   9x9   18%  against a 60% bar   (had passed)
        junior  11x11  30%  against a 60% bar   (had passed)

    Junior was traded for senior and neither was kept. The rehearsed episodes
    do not count toward promotion - only the rung being trained does - they
    are there to stop the weights drifting off everything underneath.
    """
    env = CurriculumEnv(stage, seed=seed)
    rehearsal_envs = [CurriculumEnv(s, seed=seed + 7000 + i)
                      for i, s in enumerate(rehearse)]
    # Weighted toward the rungs nearest this one, not uniform.
    #
    # Uniform was the bug. With 25% of episodes spread evenly over thirteen
    # rungs, each one is rehearsed in about 2% of episodes, which is not
    # rehearsal - it is a rounding error. Measured after training the hand
    # and combat rungs, on a brain claiming fifteen rungs passed:
    #
    #     avoid     7x7    65% against a 75% bar
    #     junior    9x9    55% against a 60% bar
    #     senior  15x15    35% against a 50% bar
    #     senior  19x19    15% against a 50% bar
    #     satchel   5x5    35% against an 85% bar
    #     satchel7  7x7     5% against a 75% bar
    #
    # Six of the fifteen were gone, including the two that had been passed
    # twice that same hour. senior 15x15 at 35% is the one that mattered:
    # `armed` IS senior 15x15 plus a sword, so every combat rung was being
    # trained on a brain that could no longer cross the room, and no amount
    # of tuning the combat would have shown up as anything but a flat line.
    #
    # Linear weights: the rung directly below gets the most, the nursery the
    # least, and nothing gets zero.
    rehearsal_weights = [i + 1 for i in range(len(rehearsal_envs))]
    rng = random.Random(seed + 991)
    wins = deque(maxlen=stage.window)
    rewards = deque(maxlen=stage.window)
    got = deque(maxlen=stage.window)
    died = deque(maxlen=stage.window)
    started = time.time()

    for ep in range(1, max_episodes + 1):
        if rehearsal_envs and rng.random() < rehearse_rate:
            # Trained on, not scored on: an old rung the policy still clears
            # would otherwise inflate the window and promote it off this one.
            pick = rng.choices(rehearsal_envs, weights=rehearsal_weights)[0]
            run_episode(pick, policy, learner, None, ep)

        total, info = run_episode(env, policy, learner, tracer, ep)
        wins.append(1 if info.get("success") else 0)
        # The growth check cannot tell "flat because solved" from "flat
        # because impossible" without this.
        learner.note_outcome(bool(info.get("success")))
        rewards.append(total)
        got.append(info.get("collected", 0))
        died.append(1 if info.get("dead") else 0)

        if not quiet and ep % 50 == 0:
            # Over the window, not one episode: a single episode says nothing,
            # and "how is it failing" - starving for goals or dying - is the
            # thing that tells you which rung is wrong.
            walk = ""
            if tracer and tracer.last_row.get("detour_ratio") not in ("", None):
                walk = "  walk x%.2f" % tracer.last_row["detour_ratio"]
            print("    ep %-5d  reward %+7.2f  success %3.0f%%  "
                  "collected %.2f  died %3.0f%%%s"
                  % (ep, statistics.mean(rewards),
                     100.0 * sum(wins) / len(wins),
                     statistics.mean(got), 100.0 * sum(died) / len(died), walk))

        # Saved during the rung, not only at the end of it. A rung can take
        # an hour; losing an hour to a closed terminal is the thing this whole
        # file exists to stop.
        if on_tick and ep % 200 == 0:
            on_tick(ep)

        if len(wins) == stage.window and sum(wins) / len(wins) >= stage.pass_rate:
            # The rolling window measures the SAMPLING policy, and sampling
            # is exploration.  In a small room a near-uniform policy stumbles
            # onto the goal most episodes: a freshly initialised net scored
            # 98% on the nursery window at entropy 1.37 while scoring 14%
            # under argmax and walking 1.3 cells.  That is how corridors7 got
            # marked passed and then played at 12%.
            #
            # So the window only nominates.  Promotion is decided by greedy
            # play, where exploration cannot help.
            gr = greedy_pass_rate(env, policy, stage)
            if gr < stage.pass_rate * config_rl.GREEDY_GATE_FRACTION:
                if not quiet:
                    print('    window says %.0f%% but greedy play gives %.0f%% '
                          '- not promoting' % (100.0 * sum(wins) / len(wins), 100 * gr))
                wins.clear()
                continue
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
    ap.add_argument("--max-grid", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=4000,
                    help="cap per rung before giving up on it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rehearse", type=float, default=0.40,
                    help="fraction of episodes drawn from earlier rungs, "
                         "so climbing does not undo what is below. 0 for "
                         "the old behaviour")
    ap.add_argument("--no-trace", action="store_true",
                    help="skip the per-tick path log (keeps the summaries)")
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
    # The walk, not just the score. logs/trace/last_path.txt is the map of the
    # most recent episode - the thing to look at when the reward is climbing
    # and you want to know whether the route makes sense yet.
    tracer = pathtrace.PathTracer(per_tick=not a.no_trace)
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
                # Check before trusting it. `passed` records what was true
                # when the rung was cleared, and training anything above it
                # moves the same weights. junior sat in this list at 18%
                # against a 60% bar while the trainer skipped it on every
                # run - so the one rung that needed the work was the one rung
                # that could never get it.
                still = greedy_pass_rate(CurriculumEnv(rung, seed=a.seed),
                                         policy, rung)
                if still >= rung.pass_rate:
                    print("\n=== %s   still plays at %.0f%%, skipping"
                          % (key, still * 100))
                    rung = rung.grown()
                    continue
                print("\n=== %s   was passed, now plays at %.0f%% - retraining"
                      % (key, still * 100))
                passed.remove(key)

            print("\n=== %s   goals=%d hazards=%d hostiles=%d  pass %.0f%% of %d"
                  % (key, rung.goals, rung.hazards, rung.hostiles,
                     rung.pass_rate * 100, rung.window))
            # Everything underneath this rung, so climbing does not cost
            # what was already climbed. Built from the ladder rather than
            # from `passed`, so a --stage run rehearses too.
            behind = []
            for earlier in default_ladder(max_grid=a.max_grid):
                e = earlier
                while e is not None:
                    if rung_key(e) == key:
                        break
                    behind.append(e)
                    e = e.grown()
                else:
                    continue
                break

            ok, used, (mr, wr, secs) = train_stage(
                rung, policy, learner, a.episodes, seed=a.seed,
                on_tick=lambda ep: stash(), tracer=tracer,
                rehearse=behind, rehearse_rate=a.rehearse)
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
