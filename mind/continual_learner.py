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
    def collect(self, state, action, log_prob, reward, value, action_mask):
        """Store one transition in the rolling buffer."""
        self.buffer.append(Transition(
            state=list(state),
            action=action,
            log_prob=float(log_prob),
            reward=float(reward),
            value=float(value),
            action_mask=list(action_mask),
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
        self._ticks_since_update = 0
        self._maybe_grow()
        return True

    def _do_ppo_update(self):
        """Mini PPO update on the buffer contents."""
        transitions = list(self.buffer)

        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
        rewards_raw = [t.reward for t in transitions]
        masks = torch.tensor([t.action_mask for t in transitions], dtype=torch.float32)

        # Compute discounted returns
        returns = []
        running = 0.0
        for r in reversed(rewards_raw):
            running = r + config_rl.GAMMA * running
            returns.insert(0, running)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalise returns
        # Always normalise returns to keep advantages on a stable scale
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(config_rl.CONTINUAL_MINI_EPOCHS):
            log_probs, values, entropy = self.policy.evaluate(states, actions, action_masks=masks)
            advantages = returns - values.detach()

            ratio = torch.exp(log_probs - old_log_probs.detach())
            clipped = torch.clamp(ratio, 1.0 - config_rl.CLIP_EPS, 1.0 + config_rl.CLIP_EPS)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
            entropy_bonus = -config_rl.ENTROPY_COEFF * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            self.last_losses = (float(loss.item()), float(actor_loss.item()), float(critic_loss.item()))
            print(f"[PPO] loss={loss.item():.4f} actor={actor_loss.item():.4f} critic={critic_loss.item():.4f}")

    # ------------------------------------------------------------------
    # Growth check
    # ------------------------------------------------------------------
    def _maybe_grow(self):
        """Check if the policy has plateaued and should expand."""
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

