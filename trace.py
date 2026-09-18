# -*- coding: utf-8 -*-
"""What the agent actually did, tick by tick, and whether it made sense.

The question this answers
------------------------
Episode return says an agent scored 47. It does not say whether it walked two
tiles to the goal or wandered sixty and arrived by accident. Those are the
same number and completely different animals, and only one of them has
learned anything.

So this records the path and compares it to the shortest one that existed.
A goal two tiles away that took sixty moves is a detour ratio of 30, and that
is the number to watch: it falls as the policy forms an opinion, and it moves
long before return does - return is dominated by whether it arrived at all.

Three outputs, because three different readers
----------------------------------------------
* ``action_trace.csv``   one row per tick. For plotting, and for anything
                         that wants the raw thing.
* ``episode_paths.csv``  one row per episode. Detour ratio, action counts,
                         how much of the room it bothered to visit.
* ``last_path.txt``      the most recent episode drawn as a map, with the
                         route on it. This is the one to paste at another
                         model: it shows the procedure rather than describing
                         it, and it fits in a message.

The optimal path is a breadth-first search over floor tiles, so walls count.
Manhattan distance would flatter a maze badly - it ignores the wall you have
to walk around, and would report a competent agent as a wasteful one.
"""

from __future__ import annotations

import csv
from collections import Counter, deque
from pathlib import Path

import config_rl

try:
    from agent import ACTION_NAMES
except Exception:                                  # pragma: no cover
    ACTION_NAMES = {}

T_WALL = 1                                          # curriculum.T_WALL

_STEP = {config_rl.ACT_MOVE_FORWARD:  (0, -1),
         config_rl.ACT_MOVE_BACKWARD: (0,  1),
         config_rl.ACT_MOVE_LEFT:     (-1, 0),
         config_rl.ACT_MOVE_RIGHT:    (1,  0)}


def _name(a) -> str:
    return ACTION_NAMES.get(int(a), str(a))


# ---------------------------------------------------------------------------
def shortest_path(grid, start, targets):
# ---------------------------------------------------------------------------
    """Steps from `start` to the nearest of `targets`, walking round walls.

    None when nothing is reachable, which is not the same as zero and must not
    be averaged in as though it were.
    """
    if not targets:
        return None
    goals = set(targets)
    n = len(grid)
    m = len(grid[0]) if n else 0
    seen = {tuple(start)}
    q = deque([(tuple(start), 0)])
    while q:
        (x, y), d = q.popleft()
        if (x, y) in goals:
            return d
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < m and 0 <= ny < n):
                continue
            if (nx, ny) in seen or grid[ny][nx] == T_WALL:
                continue
            seen.add((nx, ny))
            q.append(((nx, ny), d + 1))
    return None


# ---------------------------------------------------------------------------
def _curriculum_env(env):
# ---------------------------------------------------------------------------
    """The CurriculumEnv behind whatever was handed in, or None for the island.

    StageSession wears the full environment's face, so the room it is actually
    running lives one layer down in ._env.
    """
    if env is None:
        return None
    inner = getattr(env, "_env", None)
    if inner is not None and hasattr(inner, "grid") and hasattr(inner, "ax"):
        return inner
    if hasattr(env, "grid") and hasattr(env, "ax") and hasattr(env, "stage"):
        return env
    return None


_COLUMNS = ["episode", "stage", "grid", "ticks", "collected", "success",
            "return", "optimal_steps", "walk_steps", "detour_ratio",
            "revisit_ratio", "coverage", "wall_bumps", "still",
            "top_action", "action_counts"]


# ---------------------------------------------------------------------------
class PathTracer:
# ---------------------------------------------------------------------------
    """Records one run.

    Safe to point at the island, where it counts actions and leaves the path
    columns empty rather than inventing a shortest route through a world that
    has no grid to search.
    """

    def __init__(self, out_dir=None, per_tick: bool = True):
        self.dir = Path(out_dir or (config_rl.ROOT / "logs" / "trace"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.per_tick = per_tick

        self.tick_path = self.dir / "action_trace.csv"
        self.ep_path   = self.dir / "episode_paths.csv"
        self.map_path  = self.dir / "last_path.txt"

        self._tick_f = self._tick_w = None
        if per_tick:
            self._tick_f = open(self.tick_path, "w", newline="", encoding="utf-8")
            self._tick_w = csv.writer(self._tick_f)
            self._tick_w.writerow(["episode", "tick", "x", "y", "action",
                                   "action_name", "reward", "goal_dist",
                                   "collected"])

        self._ep_f = open(self.ep_path, "w", newline="", encoding="utf-8")
        self._ep_w = csv.writer(self._ep_f)
        self._ep_w.writerow(_COLUMNS)

        self.episode = 0
        self.last_row = {}
        self._reset_episode()

    # -- per episode ------------------------------------------------------

    def _reset_episode(self):
        self.path = []
        self.actions = Counter()
        self.ret = 0.0
        self.wall_bumps = 0
        self.still = 0
        self.legs = []                  # (optimal, walked) once per goal taken
        self._leg_start = 0
        self._leg_optimal = None
        self._grid = None
        self._stage = None
        self._spawn = None
        self._goals_now = ()
        self._goals_at_start = ()
        self._hazards = ()

    def begin(self, env, episode: int):
        """Call right after env.reset()."""
        self._reset_episode()
        self.episode = int(episode)
        ce = _curriculum_env(env)
        if ce is None:
            return
        self._stage   = ce.stage
        self._grid    = [row[:] for row in ce.grid]
        self._spawn   = (ce.ax, ce.ay)
        self._goals_now = tuple(ce.goals)
        self._goals_at_start = tuple(ce.goals)
        self._hazards = tuple(getattr(ce, "hazards", ()))
        self.path.append(self._spawn)
        self._leg_optimal = shortest_path(self._grid, self._spawn, ce.goals)

    def record(self, env, action, reward, info=None):
        """Call right after env.step()."""
        self.actions[int(action)] += 1
        self.ret += float(reward)

        ce = _curriculum_env(env)
        if ce is None:
            return

        pos = (ce.ax, ce.ay)
        moved = bool(self.path) and pos != self.path[-1]
        if int(action) in _STEP and not moved:
            self.wall_bumps += 1
        if not moved:
            self.still += 1
        self.path.append(pos)

        dist = shortest_path(self._grid, pos, ce.goals)
        if self.per_tick:
            self._tick_w.writerow([self.episode, ce.steps, pos[0], pos[1],
                                   int(action), _name(action),
                                   round(float(reward), 4),
                                   "" if dist is None else dist, ce.collected])

        # A goal was taken this tick, so the leg is over. Measuring per leg
        # rather than per episode is what keeps a respawning nursery honest:
        # three goals is three separate walks, and averaging them into one
        # number hides a clean first walk behind a hopeless third.
        if tuple(ce.goals) != self._goals_now:
            walked = ce.steps - self._leg_start
            if self._leg_optimal:
                self.legs.append((self._leg_optimal, walked))
            self._leg_start = ce.steps
            self._leg_optimal = shortest_path(self._grid, pos, ce.goals)
            self._goals_now = tuple(ce.goals)

    def end(self, env, info=None):
        """Call at the episode boundary. Returns the summary row."""
        info = info or {}
        ce = _curriculum_env(env)

        opt  = sum(o for o, _ in self.legs)
        walk = sum(w for _, w in self.legs)
        detour = (walk / opt) if opt else None

        uniq  = len(set(self.path))
        floor = sum(1 for row in (self._grid or []) for t in row
                    if t != T_WALL) or 1
        revisit = 1.0 - (uniq / len(self.path)) if self.path else 0.0
        top = self.actions.most_common(1)

        row = {
            "episode": self.episode,
            "stage": self._stage.name if self._stage else "island",
            "grid": self._stage.grid if self._stage else "",
            "ticks": getattr(ce, "steps", len(self.path)),
            "collected": info.get("collected", getattr(ce, "collected", 0)),
            "success": int(bool(info.get("success"))),
            "return": round(self.ret, 3),
            "optimal_steps": opt or "",
            "walk_steps": walk or "",
            "detour_ratio": "" if detour is None else round(detour, 3),
            "revisit_ratio": round(revisit, 3),
            "coverage": round(uniq / floor, 3) if self._grid else "",
            "wall_bumps": self.wall_bumps,
            "still": self.still,
            "top_action": _name(top[0][0]) if top else "",
            "action_counts": " ".join("%s=%d" % (_name(a), c)
                                      for a, c in self.actions.most_common()),
        }
        self._ep_w.writerow([row[k] for k in _COLUMNS])
        self._ep_f.flush()
        if self._tick_f:
            self._tick_f.flush()

        if self._grid:
            self.map_path.write_text(self.render(row), encoding="utf-8")
        self.last_row = row
        return row

    # -- the shareable picture --------------------------------------------

    def render(self, row) -> str:
        """The room, the route and the verdict, as text another model can read."""
        g = self._grid
        n = len(g)
        cells = [["#" if g[y][x] == T_WALL else "." for x in range(n)]
                 for y in range(n)]
        for (hx, hy) in self._hazards:
            cells[hy][hx] = "x"

        visits = Counter(self.path)
        for i in range(1, len(self.path)):
            px, py = self.path[i - 1]
            cx, cy = self.path[i]
            if (cx, cy) == (px, py):
                continue
            ch = {(0, -1): "^", (0, 1): "v",
                  (-1, 0): "<", (1, 0): ">"}.get((cx - px, cy - py), "o")
            # A tile walked more than once is drawn as a star instead of an
            # arrow: pacing the same corridor is the exact failure worth
            # seeing at a glance, and arrows hide it.
            cells[cy][cx] = ch if visits[(cx, cy)] <= 1 else "*"

        for (gx, gy) in self._goals_at_start:
            cells[gy][gx] = "G"
        if self._spawn:
            cells[self._spawn[1]][self._spawn[0]] = "S"

        d = row["detour_ratio"]
        verdict = ("no goal reached" if not self.legs else
                   "optimal" if d and d <= 1.05 else
                   "near-optimal" if d and d <= 1.5 else
                   "wandering" if d and d <= 4.0 else "lost")

        out = ["episode %s   %s %sx%s   %s"
               % (row["episode"], row["stage"], row["grid"], row["grid"], verdict),
               ""]
        out += ["  " + "".join(r) for r in cells]
        out += ["",
                "  S start   G goal   x hazard   # wall   * walked twice or more",
                "",
                "  ticks          %s" % row["ticks"],
                "  goals reached  %s" % row["collected"],
                "  return         %s" % row["return"]]
        if self.legs:
            out.append("  walk vs best   %s steps where %s would do  (x%s)"
                       % (row["walk_steps"], row["optimal_steps"], d))
            out.append("  per goal       %s"
                       % ", ".join("%d/%d" % (w, o) for o, w in self.legs))
        else:
            out.append("  walk vs best   never reached a goal")
        out += ["  revisited      %.0f%% of moves landed where it had already stood"
                % (row["revisit_ratio"] * 100),
                "  room seen      %.0f%%" % (float(row["coverage"] or 0) * 100),
                "  wall bumps     %s" % row["wall_bumps"],
                "  actions        %s" % row["action_counts"]]
        return "\n".join(out)

    def close(self):
        for f in (self._tick_f, self._ep_f):
            if f:
                try:
                    f.close()
                except Exception:
                    pass
