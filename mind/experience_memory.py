"""Experience memory — links problems to solutions that worked.

Stores episodes of  (need_fingerprint, state_summary, action_sequence,
cumulative_reward, confidence).  On recall, the memory returns the
highest-confidence past entry whose need fingerprint is closest to the
current need vector.

This is *separate* from the cross-episode spatial ``memory.py`` which
tracks tile positions and entity locations.  Experience memory tracks
**what the mind did about a problem and whether it worked**.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import config_rl


@dataclass
class SolutionEntry:
    """One recorded attempt at solving a need."""
    need_fingerprint: str
    state_summary: list[float]       # compressed snapshot of state at detection
    action_sequence: list[int]       # ordered actions taken
    cumulative_reward: float         # total reward earned during the sequence
    confidence: float = 1.0          # decays per episode

    def __repr__(self):
        return (
            f"Solution(fp={self.need_fingerprint[:8]}, "
            f"actions={len(self.action_sequence)}, "
            f"reward={self.cumulative_reward:.2f}, "
            f"conf={self.confidence:.2f})"
        )


class ExperienceMemory:
    """Problem → solution store with confidence decay."""

    def __init__(self, max_entries: int = 500):
        self.entries: list[SolutionEntry] = []
        self.max_entries = max_entries
        # Track the currently-recording episode
        self._recording = False
        self._current_fp: str = ""
        self._current_state: list[float] = []
        self._current_actions: list[int] = []
        self._current_reward: float = 0.0

    # ------------------------------------------------------------------
    # Fingerprinting — hash top-2 active needs for fast matching
    # ------------------------------------------------------------------
    @staticmethod
    def fingerprint(need_vec: list[float]) -> str:
        """Create a short hash from the top-2 highest needs."""
        indexed = sorted(enumerate(need_vec), key=lambda p: p[1], reverse=True)
        top2 = tuple((idx, round(val, 1)) for idx, val in indexed[:2])
        raw = str(top2).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Recording — start/step/stop an episode of actions
    # ------------------------------------------------------------------
    def begin_recording(self, need_vec: list[float], state: list[float]):
        """Start recording a new solution attempt."""
        self._recording = True
        self._current_fp = self.fingerprint(need_vec)
        # Store a compressed state summary (first 20 values)
        self._current_state = list(state[:20])
        self._current_actions = []
        self._current_reward = 0.0

    def record_step(self, action: int, reward: float):
        """Record one action and its immediate reward."""
        if not self._recording:
            return
        self._current_actions.append(action)
        self._current_reward += reward

    def end_recording(self):
        """Finish recording and store the entry if non-trivial."""
        if not self._recording:
            return
        self._recording = False
        if not self._current_actions:
            return
        entry = SolutionEntry(
            need_fingerprint=self._current_fp,
            state_summary=self._current_state,
            action_sequence=list(self._current_actions),
            cumulative_reward=self._current_reward,
        )
        self.entries.append(entry)
        # Prune to max size (keep highest confidence)
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.confidence, reverse=True)
            self.entries = self.entries[: self.max_entries]

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Recall — find past solution that matches current need
    # ------------------------------------------------------------------
    def recall(self, need_vec: list[float]) -> SolutionEntry | None:
        """Return the highest-confidence entry matching *need_vec*, or None."""
        fp = self.fingerprint(need_vec)
        matches = [e for e in self.entries if e.need_fingerprint == fp and e.confidence >= 0.1]
        if not matches:
            return None
        return max(matches, key=lambda e: e.confidence * e.cumulative_reward)

    # ------------------------------------------------------------------
    # Episode decay — called once at the end of every episode
    # ------------------------------------------------------------------
    def decay(self, rate: float = 0.05):
        """Decay all confidences by *rate*, prune below 0.1."""
        for entry in self.entries:
            entry.confidence *= (1.0 - rate)
        self.entries = [e for e in self.entries if e.confidence >= 0.1]

    # ------------------------------------------------------------------
    # Persistence — Save & Load from disk
    # ------------------------------------------------------------------
    def save(self, path=None):
        """Save experience memory store to disk."""
        path = path or (config_rl.DATA_DIR / "experience_memory.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "need_fingerprint": e.need_fingerprint,
                "state_summary": e.state_summary,
                "action_sequence": e.action_sequence,
                "cumulative_reward": e.cumulative_reward,
                "confidence": e.confidence,
            }
            for e in self.entries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path=None):
        """Load experience memory store from disk."""
        path = path or (config_rl.DATA_DIR / "experience_memory.json")
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [
                SolutionEntry(
                    need_fingerprint=d["need_fingerprint"],
                    state_summary=d.get("state_summary", []),
                    action_sequence=d.get("action_sequence", []),
                    cumulative_reward=d.get("cumulative_reward", 0.0),
                    confidence=d.get("confidence", 1.0),
                )
                for d in data
            ]
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Snapshot for GUI
    # ------------------------------------------------------------------
    def last_entry(self) -> SolutionEntry | None:
        """Return the most recently stored entry (for debug display)."""
        return self.entries[-1] if self.entries else None
