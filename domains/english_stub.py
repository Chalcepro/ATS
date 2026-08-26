"""English stub domain — proves the mind is domain-agnostic.

Problem:  identify the missing word-category in a template sentence.
State:    bag-of-character-hashes (numeric; no actual English text to mind).
Actions:  select a token category id (0–N).
Reward:   +1 correct, -0.5 wrong.

This is a trivial placeholder.  The point is that ``mind/solution_loop``
and ``model_rl.RLPolicy`` run unchanged against this domain — proving
the mind architecture transfers without rewriting the learner.
"""

from __future__ import annotations

import hashlib
import random

import config_rl
from domains.base import Domain


# Small vocabulary of (category_id, words) — we only care about categories.
_CATEGORIES = {
    0: ["cat", "dog", "tree", "rock", "river"],     # noun
    1: ["run", "jump", "eat", "sleep", "build"],     # verb
    2: ["big", "small", "fast", "slow", "red"],      # adjective
    3: ["the", "a", "an", "this", "that"],           # determiner
}
_NUM_CATEGORIES = len(_CATEGORIES)

# Template: (present_categories, missing_category_id)
_TEMPLATES = [
    ([3, 2, 0], 1),   # "the big cat ___"  → verb
    ([3, 0, 1], 2),   # "the cat runs ___" → adjective
    ([2, 0, 1], 3),   # "big cat runs ___" → determiner
    ([3, 2, 1], 0),   # "the big runs ___" → noun
]


def _hash_word(word: str) -> float:
    """Convert a word to a float hash (no actual English leaks to model)."""
    h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0


class EnglishStubDomain(Domain):
    """Trivial text-category classification domain."""

    def __init__(self):
        self._template = None
        self._answer = -1
        self._present_hashes: list[float] = []
        self._done = False
        self._last_reward = 0.0
        self._episode = 0
        self._tick = 0

    def reset(self) -> list[float]:
        self._episode += 1
        self._tick = 0
        self._done = False
        self._last_reward = 0.0
        self._template = random.choice(_TEMPLATES)
        self._answer = self._template[1]

        # Build state: hash of each present word + padding
        present_cats = self._template[0]
        hashes = []
        for cat_id in present_cats:
            word = random.choice(_CATEGORIES[cat_id])
            hashes.append(_hash_word(word))
        self._present_hashes = hashes

        return self.encode_state()

    def encode_state(self) -> list[float]:
        # Pad to STATE_SIZE with zeros
        state = list(self._present_hashes)
        while len(state) < config_rl.STATE_SIZE:
            state.append(0.0)
        return state[: config_rl.STATE_SIZE]

    def apply_action(self, action_id: int):
        self._tick += 1
        # Only first _NUM_CATEGORIES actions are meaningful
        guess = action_id % _NUM_CATEGORIES
        if guess == self._answer:
            self._last_reward = 1.0
        else:
            self._last_reward = -0.5
        self._done = True  # one-shot per episode
        return self.encode_state(), self._last_reward, self._done, {}

    def reward_signal(self) -> float:
        return self._last_reward

    def get_action_mask(self) -> list[int]:
        mask = [0] * config_rl.ACTION_SIZE
        for i in range(_NUM_CATEGORIES):
            mask[i] = 1
        return mask

    def problem_hints(self) -> list[float]:
        # Single need: "answer needed"
        return [1.0] if not self._done else [0.0]
