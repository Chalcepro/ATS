"""Minimal deterministic navigate-to-goal grid — a plumbing test for the RL stack.

This is **not** part of the ATS world. It exists so the learner can be
validated in seconds instead of hours: a tiny grid where the optimal policy
is obvious, exposed through the *same* interface as ``ATSEnvironment``
(``reset() -> state``, ``step(action) -> (state, reward, done, info)``,
``action_mask()``) and the *same* ``STATE_SIZE`` / ``ACTION_SIZE`` as the
real policy network.

If PPO cannot solve this, the bug is in the learner, not in the ATS world.
``sanity_train.py`` runs that check and exits non-zero on failure.

The friend's suggestion ("train it on something simpler like Snake") is the
right instinct — but the value of a simple environment here is *regression
testing the algorithm*, not replacing a world that already exists. This is
that simple environment, kept in-repo and cheap to run.
"""

from __future__ import annotations

import random

import config_rl

GRID = 7
MAX_STEPS = 60

# The first four action indices are the ATS movement actions; everything else
# is masked off so the policy only ever chooses a real move here.
_MOVES = {
    config_rl.ACT_MOVE_FORWARD:  (0, -1),
    config_rl.ACT_MOVE_BACKWARD: (0,  1),
    config_rl.ACT_MOVE_LEFT:     (-1, 0),
    config_rl.ACT_MOVE_RIGHT:    (1,  0),
}


class SanityEnv:
    """A 7x7 grid. Reach the goal tile. Random goal each episode."""

    def __init__(self, shaped: bool = True, seed: int | None = None):
        self.shaped = shaped
        self.rng = random.Random(seed)
        self.ax = self.ay = 0
        self.gx = self.gy = 0
        self.steps = 0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> list[float]:
        self.ax, self.ay = 0, 0
        while True:
            self.gx = self.rng.randint(0, GRID - 1)
            self.gy = self.rng.randint(0, GRID - 1)
            if (self.gx, self.gy) != (self.ax, self.ay):
                break
        self.steps = 0
        return self._state()

    # ------------------------------------------------------------------
    def action_mask(self) -> list[int]:
        m = [0] * config_rl.ACTION_SIZE
        for a in _MOVES:
            m[a] = 1
        return m

    def _dist(self) -> int:
        return abs(self.ax - self.gx) + abs(self.ay - self.gy)

    def _state(self) -> list[float]:
        s = [0.0] * config_rl.STATE_SIZE
        s[0] = self.ax / (GRID - 1)
        s[1] = self.ay / (GRID - 1)
        s[2] = self.gx / (GRID - 1)
        s[3] = self.gy / (GRID - 1)
        s[4] = (self.gx - self.ax) / (GRID - 1)   # signed dx
        s[5] = (self.gy - self.ay) / (GRID - 1)   # signed dy
        s[6] = self._dist() / (2 * (GRID - 1))
        # place the action mask in the tail, mirroring agent.build_state
        s[-config_rl.ACTION_SIZE:] = [float(v) for v in self.action_mask()]
        return s

    # ------------------------------------------------------------------
    def step(self, action: int):
        prev = self._dist()
        dx, dy = _MOVES.get(int(action), (0, 0))
        self.ax = min(GRID - 1, max(0, self.ax + dx))
        self.ay = min(GRID - 1, max(0, self.ay + dy))
        self.steps += 1

        d = self._dist()
        reward = -0.02                                   # time cost
        if self.shaped:
            reward += 0.10 * (prev - d)                  # + toward goal, - away

        done = False
        if d == 0:
            reward += 1.0
            done = True
        elif self.steps >= MAX_STEPS:
            done = True

        return self._state(), reward, done, {"dist": d}
