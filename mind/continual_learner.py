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
import math
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
        self.entropy_coeff = config_rl.ENTROPY_COEFF

        # Growth tracking
        self._total_ticks = 0
        self._last_growth_reward = 0.0
        self._reward_accumulator = 0.0
        self._reward_count = 0
        self._outcomes: deque[bool] = deque(maxlen=60)
        # None, not 0: a cooldown measured from tick zero would silently
        # forbid the FIRST growth for its whole duration, and report it as
        # "grew N ticks ago" about a growth that never happened.
        self._last_growth_tick = None
        self._last_growth_refusal = ''

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

        Advantages come from **GAE(lambda)** over the *old* (rollout-time)
        values stored on each transition; returns for the critic target are
        ``advantage + old_value``.  Only the advantages are normalised.

        Before this, advantages were raw n-step-to-go returns minus the
        (detached, current) value — effectively full Monte-Carlo advantage
        over up to a 256-step buffer.  That is unbiased but very
        high-variance, which is consistent with what the first long
        real-world run after the return-normalisation fix showed: reward
        climbed for ~150 episodes, dipped negative, climbed again to a
        peak, then declined — a noisy, non-monotonic trend rather than a
        broken one.  See updates/2026-09-04 first-long-run.md.

        The entropy bonus uses ``self.entropy_coeff``, an adaptive value
        (see ``__init__`` and the end of this method) rather than the flat
        ``config_rl.ENTROPY_COEFF`` — a fixed coefficient can't recover a
        policy that has already collapsed toward zero entropy, since the
        entropy gradient itself is tiny there too.  Confirmed directly on a
        2000-episode run: entropy decayed 0.0162->0.0001 within a single
        episode with actor loss pinned at 0.0000 throughout.  See
        updates/2026-09-04e ... and updates/2026-09-05 ... (the fix).
        """
        transitions = list(self.buffer)

        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
        masks = torch.tensor([t.action_mask for t in transitions], dtype=torch.float32)
        old_values = [t.value for t in transitions]

        # --- GAE(lambda): episode-boundary aware, bootstrapped at the buffer edge ---
        # `mask` zeroes both the bootstrap and the lambda-return propagation across
        # a `done` step, so one episode's advantage never leaks into the next.
        next_value = 0.0 if transitions[-1].done else old_values[-1]
        gae = 0.0
        advantages = [0.0] * len(transitions)
        for i in reversed(range(len(transitions))):
            mask = 0.0 if transitions[i].done else 1.0
            delta = transitions[i].reward + config_rl.GAMMA * next_value * mask - old_values[i]
            gae = delta + config_rl.GAMMA * config_rl.GAE_LAMBDA * mask * gae
            advantages[i] = gae
            next_value = old_values[i]
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(old_values, dtype=torch.float32)

        for _ in range(config_rl.CONTINUAL_MINI_EPOCHS):
            log_probs, values, entropy = self.policy.evaluate(states, actions, action_masks=masks)

            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_probs - old_log_probs.detach())
            clipped = torch.clamp(ratio, 1.0 - config_rl.CLIP_EPS, 1.0 + config_rl.CLIP_EPS)
            actor_loss = -torch.min(ratio * adv_norm, clipped * adv_norm).mean()

            critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
            entropy_bonus = -self.entropy_coeff * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            self.last_losses = (
                float(loss.item()), float(actor_loss.item()), float(critic_loss.item()),
                float(entropy.mean().item()),
            )

        # --- adapt the entropy coefficient for the *next* update, based on
        # the freshest entropy reading from this one. Ramps up while entropy
        # is below target (push harder before/while collapsing), relaxes
        # back toward the floor once entropy is healthy again. ---
        #
        # The target has to be read against how many actions were actually
        # *available*, not against the 25 the policy has in total. A fixed
        # 0.5 nats was set when the whole action set was live, where ln(25) =
        # 3.22 is the ceiling. Under a mask that leaves four moves the ceiling
        # is ln(4) = 1.39, so a healthy, well-spread policy sits around 1.3 -
        # always above 0.5, so the controller reads "plenty of exploration"
        # and walks the coefficient down to its floor every single update.
        # The collapse guard then only wakes at 0.5 nats, by which point a
        # four-action policy is already most of the way to deterministic.
        #
        # So: scale the target by the live action count. Same intent, read
        # against the space the policy is actually choosing in.
        # Applied only where the mask is narrow - the curriculum stages. The
        # full 25-action world keeps the target it was tuned with, because
        # raising it there is a real change to a system that has hours of runs
        # behind it and it deserves to be decided on its own evidence, not
        # inherited as a side effect of fixing the nursery.
        n_avail = float(masks.sum(dim=1).mean().item()) if masks.numel() else 0.0
        target = config_rl.ENTROPY_TARGET
        if 1.0 < n_avail <= config_rl.ENTROPY_NARROW_MASK:
            target = max(target, config_rl.ENTROPY_TARGET_FRAC * math.log(n_avail))

        measured_entropy = self.last_losses[3]
        if measured_entropy < target:
            self.entropy_coeff = min(self.entropy_coeff * config_rl.ENTROPY_ADAPT_UP,
                                      config_rl.ENTROPY_COEFF_MAX)
        else:
            self.entropy_coeff = max(self.entropy_coeff * config_rl.ENTROPY_ADAPT_DOWN,
                                      config_rl.ENTROPY_COEFF)

        print(f"[PPO] update {len(transitions)} steps | loss={self.last_losses[0]:.4f} "
              f"actor={self.last_losses[1]:.4f} critic={self.last_losses[2]:.4f} "
              f"entropy={self.last_losses[3]:.4f} coeff={self.entropy_coeff:.4f} "
              f"adv_mean={advantages.mean().item():.3f} adv_std={advantages.std().item():.3f}")

    # ------------------------------------------------------------------
    # Growth check
    # ------------------------------------------------------------------
    def note_outcome(self, success: bool):
        """Tell the learner whether an episode was won.

        The growth check needs this.  Reward alone cannot separate "flat
        because it solved the problem" from "flat because it cannot" - both
        are a small delta - and those two want opposite responses.  Callers
        that never report an outcome get no autonomous growth, on purpose:
        growing blind is how the old trigger doubled a healthy network.
        """
        self._outcomes.append(bool(success))

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return -1.0
        return sum(self._outcomes) / len(self._outcomes)

    def _maybe_grow(self):
        """Expand capacity only for the plateau that capacity can fix.

        A flat reward curve is three different animals:

          * mastered / converged and correct  -> growing resets Adam's
            moments and injects a zero block into a network that was doing
            fine.  This is what the original trigger did.
          * collapsed entropy, acting deterministically and wrong -> the
            policy has stopped exploring.  More neurons do not restore
            exploration; they just make a stuck policy bigger.
          * still exploring, still failing     -> this one, and only this
            one, is a plausible capacity problem.

        So all three conditions must hold together, and when they do not the
        reason is recorded rather than silently dropped - not knowing WHY the
        mind never grew is itself a bug worth fixing.
        """
        if not getattr(config_rl, "MIND_GROWTH_ENABLED", False):
            return
        if self._total_ticks % config_rl.MIND_GROWTH_CHECK_EVERY != 0:
            return
        if self.policy.hidden_size >= config_rl.MIND_MAX_HIDDEN_SIZE:
            return

        avg_reward = self._reward_accumulator / max(self._reward_count, 1)
        delta = abs(avg_reward - self._last_growth_reward)
        entropy = self.last_losses[3] if self.last_losses else None
        rate = self.success_rate
        since = (None if self._last_growth_tick is None
                 else self._total_ticks - self._last_growth_tick)

        why = None
        if delta >= config_rl.MIND_GROWTH_THRESHOLD:
            why = 'still improving (delta=%.4f)' % delta
        elif len(self._outcomes) < config_rl.MIND_GROWTH_MIN_EPISODES:
            why = ('only %d episodes reported (need %d)'
                   % (len(self._outcomes), config_rl.MIND_GROWTH_MIN_EPISODES))
        elif rate > config_rl.MIND_GROWTH_MAX_SUCCESS:
            why = 'succeeding %.0f%% - converged, not cramped' % (rate * 100)
        elif entropy is None:
            why = 'no entropy measurement yet'
        elif entropy < config_rl.MIND_GROWTH_MIN_ENTROPY:
            why = ('entropy %.3f collapsed below %.2f - stuck, not cramped'
                   % (entropy, config_rl.MIND_GROWTH_MIN_ENTROPY))
        elif since is not None and since < config_rl.MIND_GROWTH_COOLDOWN_TICKS:
            why = ('grew %d ticks ago, cooling down' % since)

        if why is not None:
            if why != self._last_growth_refusal:
                print('[MIND GROWTH] holding at %d wide: %s'
                      % (self.policy.hidden_size, why))
                self._last_growth_refusal = why
        else:
            new_size = min(self.policy.hidden_size * 2, config_rl.MIND_MAX_HIDDEN_SIZE)
            print('[MIND GROWTH] plateaued while still exploring and still '
                  'failing (delta=%.4f, entropy=%.3f, success=%.0f%%).  '
                  'Expanding %d -> %d'
                  % (delta, entropy, rate * 100, self.policy.hidden_size, new_size))
            self.policy.expand(new_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config_rl.LEARNING_RATE)
            self.buffer.clear()
            self._ticks_since_update = 0
            self._last_growth_tick = self._total_ticks
            self._last_growth_refusal = ''

        self._last_growth_reward = avg_reward
        self._reward_accumulator = 0.0
        self._reward_count = 0

