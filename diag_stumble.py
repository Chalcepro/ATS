"""How big does a room have to be before luck stops finding the goal?

Bob's requirement for the senior tier is that "chances of stumbling upon the
final goal should be very slim". That is a measurable property of the room,
not an opinion about it, so measure it before building anything on top.

A random walker is the floor. If a policy that knows nothing can clear the
rung by wandering, the rung cannot teach anything - the reward arrives
whether or not the bearing was used, which is exactly the failure the whole
ladder exists to avoid.

    python diag_stumble.py
"""
from __future__ import annotations

import random

import numpy as np

import curriculum as C


def random_walk(stage, episodes=200, seed0=900000):
    """Success rate and mean steps for a policy that picks legal moves at random."""
    wins, steps, deaths = 0, [], 0
    for k in range(episodes):
        rng = random.Random(seed0 + k)
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        info, done, n = {}, False, 0
        while not done and n < stage.max_steps:
            mask = env.action_mask()
            legal = [i for i, m in enumerate(mask) if m]
            if not legal:
                break
            _s, _r, done, info = env.step(rng.choice(legal))
            n += 1
        if info.get("success"):
            wins += 1
            steps.append(n)
        deaths += bool(info.get("dead"))
    return wins / episodes, (sum(steps) / len(steps) if steps else 0.0), deaths / episodes


def shortest(stage, episodes=60, seed0=900000):
    """Mean optimal route length, so max_steps can be set honestly."""
    lens = []
    for k in range(episodes):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        try:
            n = env._route_len(set())
        except Exception:
            n = 0
        if n:
            lens.append(n)
    return sum(lens) / len(lens) if lens else 0.0


def trial(name, grid, max_steps, goals=1, respawn=False, walls=True,
          sequence=0, episodes=200):
    st = C.Stage(name, grid, walls=walls, hazards=0, goals=goals,
                 respawn=respawn, damage=False, max_steps=max_steps,
                 target=sequence or 1, sequence=sequence,
                 pass_rate=0.7, window=50)
    rate, mean_steps, _died = random_walk(st, episodes=episodes)
    opt = shortest(st)
    print("  %-22s %3dx%-3d steps %4d   route %5.1f   random finds it %5.1f%%"
          % (name, grid, grid, max_steps, opt, rate * 100))
    return rate, opt


print("\nHow often does a RANDOM walker reach a single goal?")
print("(the ladder's existing rungs, for calibration)")
print("-" * 74)
trial("nursery-like", 5, 120)
trial("corridors-like", 5, 160)
trial("corridors7-like", 7, 200)

print("\nCandidate senior sizes, one goal, episode ends on reaching it")
print("-" * 74)
for grid, steps in ((9, 300), (11, 340), (13, 380), (15, 420), (17, 460),
                    (19, 500), (21, 540)):
    trial("senior-%d" % grid, grid, steps)

print("\nThe same rooms with a trail of 5 legs instead of one goal")
print("(a leg is short, so reward arrives often even when the room is huge)")
print("-" * 74)
for grid, steps in ((15, 420), (17, 460), (19, 500)):
    trial("trail-%d" % grid, grid, steps, sequence=5)

print("""
Read it like this: a rung is only teaching something if the random floor is
near zero. Where the floor is still high, the reward is being handed out for
wandering and the bearing is optional.""")


def senior_floors(episodes=200, seed0=920000):
    """The random floor on the senior rungs exactly as the ladder defines them,
    including every size they grow to.

    This is the check that caught the tier growing the wrong way: goals_scale
    was adding goals as the room grew, so the 19x19 compass rung had four of
    them and the nearest was six tiles off, while a 1.5x-per-step clock had
    reached forty-seven times the route. A random walker won half of them.
    """
    print("")
    print("%-16s %5s %6s %7s %8s %8s %7s"
          % ("rung", "grid", "steps", "route", "clock", "random", "died"))
    print("-" * 64)
    for st in C.default_ladder():
        if not st.name.startswith("senior"):
            continue
        g = st
        while g is not None:
            rate, _steps, died = random_walk(g, episodes=episodes, seed0=seed0)
            r = shortest(g)
            print("%-16s %5d %6d %7.1f %7.1fx %7.1f%% %6.0f%%"
                  % (g.name, g.grid, g.max_steps, r,
                     g.max_steps / max(r, 1), rate * 100, died * 100))
            g = g.grown()


senior_floors()
