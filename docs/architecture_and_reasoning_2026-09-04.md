# ATS — RL Core: Diagnosis, Changes, Reasoning, and Intended Architecture

**Date:** 2026-09-04
**Author:** Claude (Sonnet 5), at KC's request
**Scope:** the reinforcement-learning core of ATS and its relationship to the
Mind layer and the Virus language model. No world/GUI mechanics were changed.

This document exists because the RL agent had been trained for hundreds of
episodes without its reward curve ever trending upward, and successive
AI-assisted edits had tuned symptoms rather than causes. It records what was
actually wrong, what changed, why, and where the project should go next.

---

## 1. TL;DR

- The agent wasn't learning for **several compounding reasons**, not one. The
  dominant one: the PPO update **normalised the returns**, which destroys the
  value function and turns the advantage signal into noise.
- Fixes in this pass: correct the PPO return/advantage maths, make rollouts
  on-policy, cap an oversized reward constant, fix a dead embedding lookup,
  disable the auto-growing network, repair a `NameError` that made
  `main.py --train` dead.
- New: **`sanity_train.py`** — a 40-second deterministic smoke test that proves
  the learner learns on a trivial task. Run it before every long training run.
- The Mind layer and the ATS→Virus pipeline were **left alone** on purpose:
  they can't be evaluated until the RL layer under them produces competent
  behaviour. Priorities for them are in §6.

---

## 2. Why the agent wasn't learning

### 2.1 Return normalisation broke the critic  *(the big one)*

`mind/continual_learner.py._do_ppo_update` did this:

```python
returns = discounted_reward_to_go(rewards)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # <-- per batch
...
advantages   = returns - values.detach()
critic_loss  = smooth_l1(values, returns)
```

PPO needs the critic to learn `V(s) ≈ E[return | s]` on a **stable scale**. If
you rescale the returns every batch by that batch's own mean and std, the
target the critic is chasing moves under it every update — a batch with a lucky
+50 island discovery and a batch without it define completely different "0" and
"1" points. The value head never converges, so `advantage = return − value`
is dominated by the critic's error rather than by which actions were actually
good. The policy gradient then points in a semi-random direction.

The project's own `ats_rl_training_issues_and_fixes.md` (entry #4) had moved in
exactly the wrong direction here — it made the normalisation *unconditional*.

**Standard practice, now implemented:** normalise the **advantage** (it's a
relative quantity, rescaling it only changes the effective learning rate), and
let the **critic regress to the raw returns**.

### 2.2 Returns ignored episode boundaries and had no bootstrap

The discounted sum ran straight through `done` flags (the online buffer is a
`deque` that spans episodes), and the earliest transition in the window was
treated as if the episode ended there. Both inject bias.

**Now:** the running return resets to 0 at every `done`, and if the buffer
doesn't end on a terminal step the tail is bootstrapped from the last stored
value estimate.

### 2.3 Rollouts were stale / off-policy

Buffer size 256, update every 64 ticks → every transition was used in ~4
updates, each time with the `old_log_prob` captured at collection. PPO's
importance ratio `π_new / π_old` assumes the data is *near* on-policy; after
several updates it isn't, and the clip just kills the gradient.

**Now:** `CONTINUAL_UPDATE_EVERY == CONTINUAL_BUFFER_SIZE` and the buffer is
cleared after each update. Each transition is trained exactly once (×
`CONTINUAL_MINI_EPOCHS`). `flush()` handles the episode boundary.

### 2.4 One reward constant dwarfed everything

`R_OCEAN_DEATH = -500.0`, applied by `ats_env` on **every** shark trigger —
including strikes the agent survives. Under γ=0.99 a single −500 tick dominates
the entire discounted return, and once advantages are normalised, every other
transition in that rollout is squashed to ≈0. The agent learns "ocean = doom"
(badly, since it's a single untelegraphed terminal tick) and nothing else from
those episodes.

**Now:** `-15.0` — clearly worse than a normal death (−10) but the same order
of magnitude, so per-tick signal survives.

The rest of the reward table is broad but not insane (`R_ISLAND_DISCOVERY = 50`
is large; consider `~15–25` later). Left as-is for now to keep this pass
focused.

### 2.5 The item embedding was dead

`model_rl.forward`:

```python
obj_idx = state_in[..., 19].long()          # index 19 = obj_id / 100  (0..1)
```

`.long()` on a value in `[0, 1)` is `0` for every object, so `item_embed` (a
100×32 table) was only ever read at row 0. The network could still see the raw
fraction through `fc1`, so it wasn't blind — but a whole learned-embedding
pathway was doing nothing.

**Now:** `(state_in[..., 19] * ITEM_VOCAB_SIZE).round().long()`. Entity index 15
is a raw int (`E001 → 1.0`) and was already correct.

### 2.6 The auto-growing network

`_maybe_grow` doubled the hidden width when the average-reward delta between
checks fell below `0.02`. That condition is true when the policy is **stuck**,
but also when it's **converged** or just **noisy around a mean**. Doubling the
width rebuilds every layer, throws away Adam's first/second-moment estimates,
and injects a zero block that the optimiser has to reactivate. On a policy that
isn't learning yet, this is pure disturbance — and the existing checkpoints are
already at hidden=1024 because of it.

**Now:** gated behind `MIND_GROWTH_ENABLED` (default `False`). Revisit width
*after* the fixed-256 policy demonstrably climbs.

### 2.7 `main.py --train` was dead

`train_rl.py` referenced `RewardEngine` without importing it → `NameError` on
episode 1. Fixed, plus the same return-normalisation correction.

### 2.8 Smaller issues noted but not fixed this pass

- **State feature scaling:** `float(self.level)`, `float(cur_island)`,
  `float(self.last_failed_action)` are raw unbounded values sitting next to
  0..1-normalised features feeding `fc1`. Should be normalised or embedded.
- **`agent.build_state` pads/truncates to `STATE_SIZE`**, so any future layout
  drift fails silently instead of loudly.
- **Documentation drift:** `ats_project_documentation.md` describes 42 actions /
  126–130 state dims; the code is **25 / 79** (`config_rl.py` is authoritative,
  `docs/system_overview.md` and `AGENTS.md` are correct). A correction banner
  was added to the stale doc.

---

## 3. Evidence the fix works

`sanity_env.py` is a 7×7 navigate-to-goal grid exposing the **same interface
and the same `STATE_SIZE`/`ACTION_SIZE`** as the real environment.
`sanity_train.py` trains a fresh `RLPolicy` through the **real
`ContinualLearner`** and checks that behaviour improves.

```
[SANITY] random policy  reach rate      : 43.0%
[SANITY] learner first 100  mean reward : -0.338   reach 33.0%
[SANITY] learner last  100  mean reward : +0.335   reach 65.0%
[SANITY] PASS — PPO learner learns.
```

An A/B run (fixed code vs. the old return-normalisation, same seed, 600
episodes) confirms the direction:

```
OLD (return-norm)  : first100 -0.553 -> last100 +0.093 | goal reach 52%
NEW (advantage-norm): first100 -0.338 -> last100 +0.335 | goal reach 65%
```

On this *deliberately trivial, densely-shaped* task the old code still limps
along. On the real ATS task — sparse, with large terminal spikes and
9000-tick episodes — the same pathology is far more destructive, which is
consistent with the flat real-world curves in the issues log.

**This does not yet prove the agent will master the ATS world.** It proves the
learning machinery is sound, so effort spent on the world and reward design
will actually compound. The next milestone is a real ATS run
(`python main.py --no-gui --fresh`) showing the 50-episode average climbing.

---

## 4. Intended architecture (the three layers)

```
  ┌─────────────────────────────────────────────────────────────┐
  │ LAYER 3  Virus LM  (language / narration / eventual dialogue)│
  │   trained on corpus exported from Layer 2's episodes         │
  └──────────────────────────▲──────────────────────────────────┘
                             │  narrations, grades, symbolic tokens
  ┌──────────────────────────┴──────────────────────────────────┐
  │ LAYER 2  Mind  (needs, episodic solution recall, hints)      │
  │   biases Layer 1's action choice; records what worked        │
  └──────────────────────────▲──────────────────────────────────┘
                             │  state (79) ↓   action (0..24) ↑
  ┌──────────────────────────┴──────────────────────────────────┐
  │ LAYER 1  Embodied RL  (world + PPO policy)   ← THIS PASS     │
  │   the agent's body and survival behaviour                    │
  └─────────────────────────────────────────────────────────────┘
```

**The layers must be validated bottom-up.** A "mind" that recalls past
solutions is meaningless if the underlying policy can't execute them; a
language model "narrating embodied experience" is meaningless if the
experience is a random walk. So:

1. **Layer 1 first.** Get the PPO agent climbing on the real world. Milestone:
   50-ep average trending up over a few hundred episodes, agent reliably
   surviving a day/night cycle and reaching island 1.
2. **Then Layer 2.** Right now the Mind is a hand-coded 4-need detector plus an
   exact-hash episodic cache that nudges a single action logit. It's a
   reasonable scaffold but oversold as a "cognitive architecture." Once Layer 1
   works, measure whether the hint mechanism actually improves need-resolution
   time; if not, redesign it (nearest-neighbour recall, multi-step hints,
   need-conditioned sub-policies).
3. **Then Layer 3.** The ATS→Virus bridge is currently mocked in practice —
   `generate_narration` shells out with a 0.8s timeout it can never meet, and
   `process_episode_grade` emits a byte-identical sentence for every BAD
   episode. Before this layer can do anything, narrations need to be real
   (generate them in a background process/thread and attach them next episode,
   or drop live narration and just export structured event logs) and the
   symbolic encoder needs a real hash (the current `sum(ord(c))` collides on
   every anagram).

### On the friend's "train it on something simpler" idea

The instinct is correct and now permanently useful — as a **regression guard
for the algorithm**, which is exactly what `sanity_env.py` / `sanity_train.py`
are. The point of a tiny environment here is: *if PPO can't solve a 7×7 grid in
40 seconds, don't waste an afternoon on a full ATS run.*

It is **not** a reason to abandon the ATS world. ATS is a genuine,
already-built survival environment with biomes, telegraphed combat, crafting
and day/night perception — that is the asset. Snake / Geometry Dash would be a
lateral move: another environment to integrate, with no more "human basics,
needs, wants" content than ATS already has. If a second real environment is
ever wanted, the clean way is to make `ATSEnvironment` and `SanityEnv` conform
to a tiny shared `Env` protocol (`reset`, `step`, `action_mask`,
`STATE_SIZE`, `ACTION_SIZE`) and drop new environments in behind it — the
learner already doesn't care which world it's in.

---

## 5. Files changed in this pass

| File | Change |
| :-- | :-- |
| `mind/continual_learner.py` | PPO returns/advantages fixed; on-policy rollouts; `Transition.done`; `flush()`; growth gated |
| `config_rl.py` | `LEARNING_RATE 3e-4`, `ENTROPY_COEFF 0.005`, `CONTINUAL_UPDATE_EVERY 256`, `MIND_GROWTH_ENABLED=False` |
| `main.py` | pass `done` to `collect`; `flush()` at episode boundary |
| `rewards.py` | `R_OCEAN_DEATH -500 → -15` |
| `model_rl.py` | de-normalise object id before the item-embedding lookup |
| `train_rl.py` | import `RewardEngine` (was `NameError`); advantage-norm fix |
| `sanity_env.py` | **new** — trivial grid env, same interface as `ATSEnvironment` |
| `sanity_train.py` | **new** — PPO smoke test |
| `docs/system_overview.md` | Continual Learner section updated |
| `ats_project_documentation.md` | correction banner (stale action/state numbers) |

All changes are on the `rl-core-fixes` branch, one concern per commit, on top
of a `checkpoint:` commit that snapshots the prior uncommitted working tree.

---

## 6. Recommended next steps (in order)

1. **Fresh real run:** `python main.py --no-gui --fresh --episodes 300` and
   watch the 50-ep average. If it climbs, the core is healthy.
2. If it *doesn't* climb: add GAE-λ, then normalise the state features in
   `build_state`, then reduce `R_ISLAND_DISCOVERY` / audit the reward table for
   any other >±25 constant.
3. Delete or archive `checkpoints/model_best.pt` and `model_rl.pt` (hidden=1024,
   pre-fix). Keep them only as a curiosity.
4. Only once (1) is green: evaluate the Mind hint mechanism empirically.
5. Only once the Mind earns its place: rebuild the Virus narration path for
   real (background generation) and give the symbolic encoder a real hash.
6. Reconcile `ats_project_documentation.md` fully with the code, or delete it in
   favour of `docs/`.
