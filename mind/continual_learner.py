"""Continual learner — online PPO updates while the simulation runs.

Instead of training only at the end of an episode (offline), this module
keeps a rolling buffer of transitions and performs mini PPO updates
every N ticks.  The same policy object is shared with the solution loop
so improvements are immediate.

``train_rl.py`` can still be used as an accelerated batch trainer —
it calls the same policy and the same PPO maths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.optim as optim

import config_rl
from model_rl import RLPolicy


@dataclass
class Transition:
    state: list[float]
    action: int
    log_prob: float
    reward: float
    value: float
    action_mask: list[int]
    done: bool = False


class ContinualLearner:
    """Online PPO trainer attached to a running simulation."""

    def __init__(self, policy: RLPolicy):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=config_rl.LEARNING_RATE)
        self.buffer: deque[Transition] = deque(maxlen=config_rl.CONTINUAL_BUFFER_SIZE)
        self._ticks_since_update = 0
        self.last_losses = None

        # Growth tracking
        self._total_ticks = 0
        self._last_growth_reward = 0.0
        self._reward_accumulator = 0.0
        self._reward_count = 0

    # ------------------------------------------------------------------
    # Collect — called once per tick
    # ------------------------------------------------------------------
    def collect(self, state, action, log_prob, reward, value, action_mask, done=False):
        """Store one transition in the rolling buffer."""
        self.buffer.append(Transition(
            state=list(state),
            action=action,
            log_prob=float(log_prob),
            reward=float(reward),
            value=float(value),
            action_mask=list(action_mask),
            done=bool(done),
        ))
        self._ticks_since_update += 1
        self._total_ticks += 1
        self._reward_accumulator += reward
        self._reward_count += 1

    # ------------------------------------------------------------------
    # Update — PPO on the rolling buffer
    # ------------------------------------------------------------------
    def maybe_update(self) -> bool:
        """Run a PPO update if enough ticks have elapsed.  Returns True if updated."""
        if self._ticks_since_update < config_rl.CONTINUAL_UPDATE_EVERY:
            return False
        if len(self.buffer) < config_rl.BATCH_SIZE:
            return False

        self._do_ppo_update()
        self.buffer.clear()          # keep updates on-policy — never re-train stale transitions
        self._ticks_since_update = 0
        self._maybe_grow()
        return True

    def flush(self) -> bool:
        """Force an update on whatever is buffered (call at an episode boundary).

        Returns True if an update ran.  Unlike :meth:`maybe_update` this
        ignores the tick counter so end-of-episode transitions are not
        carried, unlearned, into the next episode.
        """
        if len(self.buffer) < config_rl.BATCH_SIZE:
            return False
        self._do_ppo_update()
        self.buffer.clear()
        self._ticks_since_update = 0
        return True

    def _do_ppo_update(self):
        """PPO update on the current rollout buffer.

        Returns are the discounted reward-to-go, computed with **episode
        boundaries** (the running sum resets at every ``done``) and
        **bootstrapped at the buffer edge** from the last stored value.
        The critic regresses to these *raw* returns; only the *advantages*
        are normalised, and only for the policy-gradient term.

        The previous version normalised the returns themselves
        (``(returns - mean) / std`` per batch).  That made the critic's
        target non-stationary — it changed scale every update — so the
        value function never converged and ``advantage = return - value``
        was noise.  This is the single biggest reason the agent's reward
        curve never trended up.  See docs/architecture_and_reasoning_2026-09-04.md.
        """
        transitions = list(self.buffer)

        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
        masks = torch.tensor([t.action_mask for t in transitions], dtype=torch.float32)

        # --- discounted returns: episode-boundary aware + edge bootstrap ---
        last = transitions[-1]
        running = 0.0 if last.done else float(last.value)
        returns = []
        for t in reversed(transitions):
            if t.done:
                running = 0.0
            running = t.reward + config_rl.GAMMA * running
            returns.insert(0, running)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(config_rl.CONTINUAL_MINI_EPOCHS):
            log_probs, values, entropy = self.policy.evaluate(states, actions, action_masks=masks)

            advantages = returns - values.detach()
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_probs - old_log_probs.detach())
            clipped = torch.clamp(ratio, 1.0 - config_rl.CLIP_EPS, 1.0 + config_rl.CLIP_EPS)
            actor_loss = -torch.min(ratio * adv_norm, clipped * adv_norm).mean()

            critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
            entropy_bonus = -config_rl.ENTROPY_COEFF * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            self.last_losses = (float(loss.item()), float(actor_loss.item()), float(critic_loss.item()))

        print(f"[PPO] update {len(transitions)} steps | loss={self.last_losses[0]:.4f} "
              f"actor={self.last_losses[1]:.4f} critic={self.last_losses[2]:.4f}")

    # ------------------------------------------------------------------
    # Growth check
    # ------------------------------------------------------------------
    def _maybe_grow(self):
        """Check if the policy has plateaued and should expand.

        Disabled by default (``config_rl.MIND_GROWTH_ENABLED``).  The old
        trigger fired on *any* low reward-delta — including a converged or
        merely stuck policy — and doubling the hidden width resets Adam's
        moments and injects a zero block, which destabilises more than it
        helps.  Get the fixed-width policy learning first, then revisit.
        """
        if not getattr(config_rl, "MIND_GROWTH_ENABLED", False):
            return
        if self._total_ticks % config_rl.MIND_GROWTH_CHECK_EVERY != 0:
            return
        if self.policy.hidden_size >= config_rl.MIND_MAX_HIDDEN_SIZE:
            return

        avg_reward = self._reward_accumulator / max(self._reward_count, 1)
        delta = abs(avg_reward - self._last_growth_reward)

        if delta < config_rl.MIND_GROWTH_THRESHOLD:
            new_size = min(self.policy.hidden_size * 2, config_rl.MIND_MAX_HIDDEN_SIZE)
            print(f"[MIND GROWTH] plateau detected (delta={delta:.4f}).  "
                  f"Expanding {self.policy.hidden_size} → {new_size}")
            self.policy.expand(new_size)
            # Re-create optimizer to pick up new parameters and flush buffer
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config_rl.LEARNING_RATE)
            self.buffer.clear()
            self._ticks_since_update = 0

        self._last_growth_reward = avg_reward
        self._reward_accumulator = 0.0
        self._reward_count = 0

