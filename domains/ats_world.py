"""ATS world domain adapter — wraps ATSEnvironment behind the Domain contract.

This is the first real domain adapter.  It translates the existing ATS
tile-world simulation into the state/action/reward/mask interface the
mind expects.
"""

from __future__ import annotations

import config_rl
from domains.base import Domain


class ATSWorldDomain(Domain):
    """Adapts *ATSEnvironment* to the domain contract."""

    def __init__(self):
        # Lazy import avoids circular dependency
        from ats_env import ATSEnvironment
        self.env = ATSEnvironment()
        self._last_reward = 0.0

    # ------------------------------------------------------------------
    # Domain interface
    # ------------------------------------------------------------------
    def encode_state(self) -> list[float]:
        return self.env._state()

    def apply_action(self, action_id: int):
        state, reward, done, info = self.env.step(action_id)
        self._last_reward = reward
        return state, reward, done, info

    def reward_signal(self) -> float:
        return self._last_reward

    def get_action_mask(self) -> list[int]:
        return self.env.get_action_mask()

    def problem_hints(self) -> list[float]:
        from mind.need_detector import NeedDetector
        nd = NeedDetector()
        return nd.detect(self.encode_state())

    def reset(self) -> list[float]:
        return self.env.reset()

    # ------------------------------------------------------------------
    # Convenience accessors for the observer GUI
    # ------------------------------------------------------------------
    @property
    def agent(self):
        return self.env.agent

    @property
    def world(self):
        return self.env.world

    @property
    def rewards(self):
        return self.env.rewards

    @property
    def day_night(self):
        return self.env.day_night

    @property
    def tick(self):
        return self.env.tick

    @property
    def episode(self):
        return self.env.episode

    @property
    def memory(self):
        return self.env.memory
