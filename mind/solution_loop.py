"""Solution loop — the core cognitive cycle of the mind.

Every tick:
    1. Detect active needs from state
    2. Recall a past solution if one exists for this need
    3. Choose an action (policy + optional hint from memory)
    4. Record the action and its reward
    5. End a recording when the triggering need is resolved

This module is the bridge between the raw RL policy and the
higher-level problem-solving behaviour the system must learn.
"""

from __future__ import annotations

import torch

import config_rl
from mind.need_detector import NeedDetector, NUM_NEEDS
from mind.experience_memory import ExperienceMemory


# A need is considered "active" if its value exceeds this threshold.
_NEED_ACTIVE_THRESHOLD = 0.15

# Maximum ticks to record one solution attempt before auto-closing.
_MAX_RECORDING_TICKS = 80


class SolutionLoop:
    """Orchestrates  detect → recall → act → score → store."""

    def reset_memory(self):
        """Drop the recurrent state.  Call at every episode boundary."""
        self._hx = None
        self.last_hidden = None

    def __init__(self, policy, *, experience_memory: ExperienceMemory | None = None):
        self.policy = policy
        self.need_detector = NeedDetector()
        self.experience = experience_memory or ExperienceMemory()

        # Internal tracking
        self._active_need_idx: int | None = None
        self._recording_ticks: int = 0
        self._last_need_vec: list[float] = [0.0] * NUM_NEEDS
        self.last_log_prob = None
        self.last_value = None
        # Recurrent state, owned here because this is what drives the policy
        # in the GUI/world loop.  `last_hidden` is the state the action was
        # chosen FROM, which is what PPO has to re-evaluate against.
        self._hx = None
        self.last_hidden = None
        self._trace = {}

    # ------------------------------------------------------------------
    # Public API — called once per tick from ats_env / main
    # ------------------------------------------------------------------
    def choose_action(self, state: list[float], action_mask: list[int]) -> int:
        """Run the full cognitive cycle and return an action index."""
        need_vec = self.need_detector.detect(state)
        self._last_need_vec = need_vec
        self._trace = {}
        self._trace['detect'] = f"need_vec={need_vec}"


        # --- 1. Identify the most urgent active need ---
        urgent_idx, urgent_val = max(enumerate(need_vec), key=lambda p: p[1])

        # --- 2. Start a new recording if a need just became active ---
        if urgent_val >= _NEED_ACTIVE_THRESHOLD:
            if not self.experience.is_recording:
                self.experience.begin_recording(need_vec, state)
                self._active_need_idx = urgent_idx
                self._recording_ticks = 0
        else:
            # No active need — close any open recording (need resolved)
            if self.experience.is_recording:
                self.experience.end_recording()
                self._active_need_idx = None

        # --- 3. Recall past solution (optional hint) ---
        hint = self.experience.recall(need_vec)
        self._trace['recall'] = f"hint={hint}"

        # --- 4. Select action via policy (with mask and optional memory hint boost) ---
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if self._hx is None:
                self._hx = self.policy.initial_hidden(1)
            self.last_hidden = self._hx.squeeze(0).tolist()
            logits, value, self._hx = self.policy(state_t, action_mask=mask_t, hx=self._hx)

            # The *policy's own* distribution.  PPO's stored `old_log_prob`
            # MUST come from this one, not from the hint-boosted behaviour
            # distribution below: `_do_ppo_update` recomputes log-probs with
            # `policy.evaluate()` (which knows nothing about hints), so
            # storing a boosted log-prob made ratio = exp(new - old) compare
            # two different distributions and silently corrupted every
            # importance ratio on any tick where a hint fired.
            policy_dist = torch.distributions.Categorical(logits=logits)

            behaviour_logits = logits
            # If a high-confidence past solution was recalled with positive reward, boost its recommended action
            if (config_rl.MIND_HINT_ENABLED
                    and hint and hint.action_sequence and hint.cumulative_reward > 0):
                rec_act = hint.action_sequence[0]
                if 0 <= rec_act < len(action_mask) and action_mask[rec_act]:
                    boost = min(3.0, max(0.5, float(hint.confidence) * 2.0))
                    behaviour_logits = logits.clone()
                    behaviour_logits[0, rec_act] += boost
                    self._trace['recall'] = f"hint={hint} -> BOOST act_{rec_act}(+{boost:.1f})"

            # Sample from the (possibly boosted) behaviour distribution, but
            # score the chosen action under the policy distribution.
            action_t = torch.distributions.Categorical(logits=behaviour_logits).sample()
            log_prob = policy_dist.log_prob(action_t)

        self.last_log_prob = log_prob
        self.last_value = value.squeeze(0)
        action = int(action_t.item())
        self._trace['act'] = f"action={action}"

        # --- 5. (Recording will be stepped externally after env.step) ---
        return action

    def record_outcome(self, action: int, reward: float, state: list[float]):
        """Called after env.step() to record what happened."""
        self.experience.record_step(action, reward)
        self._recording_ticks += 1
        self._trace['record'] = f"action={action} reward={reward}"

        # Auto-close recording after too many ticks
        if self._recording_ticks >= _MAX_RECORDING_TICKS:
            self.experience.end_recording()
            self._active_need_idx = None

        # Check if the triggering need was resolved
        if self._active_need_idx is not None:
            need_vec = self.need_detector.detect(state)
            if need_vec[self._active_need_idx] < _NEED_ACTIVE_THRESHOLD:
                self.experience.end_recording()
                self._active_need_idx = None

    def on_episode_end(self):
        """Called at end of episode — close any open recording, decay memory."""
        if self.experience.is_recording:
            self.experience.end_recording()
            self._active_need_idx = None
        self.experience.decay()

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------
    @property
    def last_need_vec(self) -> list[float]:
        return list(self._last_need_vec)

    @property
    def last_solution_entry(self):
        return self.experience.last_entry()

    @property
    def trace(self) -> dict:
        """Expose internal trace dictionary for GUI rendering."""
        return self._trace
