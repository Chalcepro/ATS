"""Abstract domain contract.

Every training universe — ATS tiles, English text, maths problems,
robotics sensors — must implement this interface so the same mind
code can attach without rewriting the learner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Domain(ABC):
    """Base class all domain adapters inherit from."""

    @abstractmethod
    def encode_state(self) -> list[float]:
        """Return the current domain state as a flat float vector."""
        ...

    @abstractmethod
    def apply_action(self, action_id: int) -> tuple[list[float], float, bool, dict]:
        """Execute *action_id*, return (new_state, reward, done, info)."""
        ...

    @abstractmethod
    def reward_signal(self) -> float:
        """Return the reward for the most recent tick."""
        ...

    @abstractmethod
    def get_action_mask(self) -> list[int]:
        """Return a list of 0/1 values (length = ACTION_SIZE).

        1 = action valid this tick, 0 = masked out.
        The policy will never select a masked action.
        """
        ...

    def problem_hints(self) -> list[float]:
        """Optional: return a need/problem vector derived from state.

        Domains that can identify problems (hunger, missing input, etc.)
        should override this.  Default returns empty list.
        """
        return []

    @abstractmethod
    def reset(self) -> list[float]:
        """Reset the domain for a new episode, return initial state."""
        ...
