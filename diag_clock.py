"""How tight can the clock be before it stops a good player, not a lucky one?

The senior tier wants two things that pull against each other: a room where
luck does not find the goal, and a room where competence does. The clock
decides both, so it has to be sized against a player that actually plays well
- not against the greedy reference, which has no pathfinding and manages 32%
even on junior 9x9.

The oracle here walks the BFS shortest route, re-planned every step so it
follows the goal when the trail moves it. That is the ceiling. The random
walker is the floor. A clock is right when the ceiling is near 100% and the
floor is near zero, and the gap between them is what the bearing has to earn.

    python diag_clock.py
"""
from __future__ import annotations

import random
from collections import deque

import config_rl
import curriculum as C

ACT_FOR = {d: a for a, d in C.MOVES.items()}


def _step_towards(env):
    """One step along a shortest route to the nearest goal, avoiding hazards.

    Re-planned every step rather than followed as a fixed path, because a
    trail moves the goal the moment the current one is taken.
    """
    n = env.stage.grid
    start = (env.ax, env.ay)
    goals = set(env.goals)
    if not goals:
        return config_rl.ACT_WAIT

    for avoid_hazards in (True, False):
        prev = {start: None}
        q = deque([start])
        end = None
        while q and end is None:
            x, y = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in prev:
                    continue
                t = env.grid[ny][nx]
                if t == C.T_WALL:
                    continue
                if avoid_hazards and t == C.T_HAZARD:
                    continue
                prev[(nx, ny)] = (x, y)
                if (nx, ny) in goals:
                    end = (nx, ny)
                    break
                q.append((nx, ny))
        if end is not None:
            cur = end
            while prev[cur] != start:
                cur = prev[cur]
            return ACT_FOR.get((cur[0] - start[0], cur[1] - start[1]),
                               config_rl.ACT_WAIT)
    return config_rl.ACT_WAIT


def _at_clock(stage, steps):
    """The same rung with a different clock.

    Passing a longer loop bound is not enough and that mistake made this file
    lie: the environment ends the episode at its OWN stage.max_steps, so every
    multiple above the rung's own clock measured the rung's own clock. The
    sweep reported a flat 69% from 6x to 24x and looked like a hard ceiling
    when nothing had actually been varied.
    """
    return C.Stage(stage.name, stage.grid, walls=stage.walls,
                   hazards=stage.hazards, hostiles=stage.hostiles,
                   goals=stage.goals, respawn=stage.respawn,
                   can_pick=stage.can_pick, can_attack=stage.can_attack,
                   damage=stage.damage, hazard_damage=stage.hazard_damage,
                   max_steps=int(steps), target=stage.target,
                   sequence=stage.sequence, pass_rate=stage.pass_rate,
                   window=stage.window)


def play(stage, steps, oracle, episodes=120, seed0=880000):
    stage = _at_clock(stage, steps)
    wins = 0
    used = []
    for k in range(episodes):
        rng = random.Random(seed0 + k)
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        info, done, n = {}, False, 0
        while not done and n < steps:
            if oracle:
                act = _step_towards(env)
            else:
                legal = [i for i, m in enumerate(env.action_mask()) if m]
                if not legal:
                    break
                act = rng.choice(legal)
            _s, _r, done, info = env.step(act)
            n += 1
        if info.get("success"):
            wins += 1
            used.append(n)
    return wins / episodes, (sum(used) / len(used) if used else 0.0)


def route(stage, episodes=40, seed0=880000):
    lens = []
    for k in range(episodes):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        env.reset()
        try:
            n = env._route_len(set())
        except Exception:
            n = None
        if n:
            lens.append(n)
    return sum(lens) / len(lens) if lens else 0.0


def sweep(stage, multiples=(4, 6, 8, 12, 16, 24)):
    r = route(stage)
    print("\n%s %dx%d   one-goal route %.1f" % (stage.name, stage.grid,
                                                stage.grid, r))
    print("  %-8s %-9s %-9s %-9s" % ("clock", "steps", "oracle", "random"))
    for m in multiples:
        steps = max(30, int(r * m))
        o, o_used = play(stage, steps, oracle=True)
        w, _ = play(stage, steps, oracle=False)
        print("  %-8s %-9d %-9s %-9s%s"
              % ("%dx" % m, steps, "%.0f%%" % (o * 100), "%.0f%%" % (w * 100),
                 "   <- oracle clears it, luck does not"
                 if o >= 0.9 and w <= 0.12 else ""))


for st in C.default_ladder():
    if st.name.startswith("senior"):
        sweep(st)

print("""
A clock is right where the oracle is near 100% and the random walker is near
zero. Too tight and a good player is punished for the maze rather than for
its own mistakes; too loose and wandering is a strategy.""")
