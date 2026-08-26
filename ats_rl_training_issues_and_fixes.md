# ATS RL Training — Technical Challenges & Solutions Log

This document records the key issues, root-cause analyses, and implemented solutions encountered during the development and optimization of the ATS Reinforcement Learning (PPO/Policy Gradient) model.

---

## 1. Lack of Training Observability & Progress Tracking

### **Challenge**
Initial training runs produced minimal feedback. There was no real-time tracking of average performance trends, best-performing policies were overwritten or lost, and standard TensorBoard logging was absent.

### **Root Cause**
- `train_rl.py` only output raw single-episode rewards without a moving average.
- Single checkpoint (`model_rl.pt`) was overwritten periodically regardless of whether performance improved or regressed.
- No integration with PyTorch's `SummaryWriter`.

### **Fix Implemented**
- **TensorBoard Integration**: Added `torch.utils.tensorboard.SummaryWriter` logging `Reward/Episode`, `Reward/Avg50`, `Loss/Total`, `Loss/Policy`, and `Entropy` metrics to `logs/`.
- **Best Model Preservation**: Added `BEST_MODEL_PATH = "checkpoints/model_best.pt"` in `config_rl.py`. Updates are committed whenever the 50-episode moving average reward reaches a new high.
- **Periodic Checkpointing**: Periodic save interval set to every 100 episodes.

---

## 2. Vanishing Gradients & Near-Zero Loss Plateau

### **Challenge**
Training logs consistently showed loss stuck around `1e-8` to `1e-9` (e.g., `Loss=-1.77e-08`, `Loss=-2.62e-09`). The policy was stagnant and unable to optimize further.

### **Root Cause**
- The policy was purely optimizing policy gradient loss without an entropy regularizer.
- Once action probabilities converged or return variance centered around zero, the gradient $\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)$ effectively vanished.

### **Fix Implemented**
- Introduced an **Entropy Regularization Bonus**:
  $$\mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} - \beta \cdot \mathcal{H}(\pi)$$
- Added `ENTROPY_COEFF` parameter to force the policy to maintain exploration gradient signals.

---

## 3. Excessive Exploration & High Entropy Locking (`Entropy = 3.218`)

### **Challenge**
After adding entropy, loss remained locked around `-0.0031` and performance dropped sharply (`50-Ep Avg` fell to `-65.74`), while `Entropy` hovered at `~3.09 - 3.218` (near the theoretical maximum of $\ln(42) \approx 3.73$).

### **Root Cause**
1. **Hardcoded Entropy Coefficient**: `continual_learner.py` had `-0.01 * entropy.mean()` hardcoded rather than using `config_rl.ENTROPY_COEFF`.
2. **Coefficient Over-Dominance**: At `0.01`, the entropy bonus term ($-0.01 \times 3.1 \approx -0.031$) dominated actor-critic losses. The agent was heavily penalized for being confident, forcing random actions across all 42 discrete choices.

### **Fix Implemented**
- Connected `continual_learner.py` to `config_rl.ENTROPY_COEFF`.
- Reduced `ENTROPY_COEFF` to `0.001` in `config_rl.py`.
- Result: Entropy bonus is now $\sim 0.003$, allowing reward-driven policy updates to dominate while preserving exploration.

---

## 4. Return Scale Distortion & Normalization Bypassing

### **Challenge**
Large negative rewards (e.g., `-130.0` for dying/failing) under $\gamma = 0.99$ caused huge discounted returns. In some cases, conditional return normalization was bypassed.

### **Root Cause**
- Return normalization was wrapped inside an `if returns.std() > 1e-8:` guard. When standard deviation was low or near-zero across a mini-batch, raw un-normalized returns entered advantage computation, destroying gradient stability.

### **Fix Implemented**
- Removed conditional check and enforced unconditional return normalization across both `continual_learner.py` and `train_rl.py`:
  ```python
  returns = (returns - returns.mean()) / (returns.std() + 1e-8)
  ```

---

## 5. Indentation Error in Online PPO Loop

### **Challenge**
Launching `py -3 main.py` failed with:
`IndentationError: unexpected indent` at line 116 in `continual_learner.py`.

### **Root Cause**
- During code edits, `entropy_bonus = -config_rl.ENTROPY_COEFF * entropy.mean()` lost 4 spaces of indentation relative to the `for` loop scope.

### **Fix Implemented**
- Re-indented `entropy_bonus` and `loss` lines inside the mini-batch PPO epoch loop.
- Verified syntax across all core modules using Python `py_compile`.

---

## 6. Oversized Death Penalty Drowning Per-Tick Learning Signal

**Date:** 2026-07-31

### **Challenge**
Reward curves were not trending upward across extended training runs (~330–950 episodes). The 50-episode average hovered around `-100` to `-128` with no clear improvement, despite the loss and entropy corrections being in place.

### **Root Cause**
- `R_DEATH = -50.0` in `rewards.py` meant that any episode ending in agent death returned a single `-50.0` signal in one tick.
- Under `γ = 0.99` with episode lengths of 147–190 ticks, this single terminal penalty dominated the discounted return, making every dying episode look catastrophically and uniformly bad.
- The agent could not distinguish *how* it died (starvation vs. slime vs. wall-walking) because the cause-of-death signal was dwarfed by the identical cliff penalty, making the per-tick penalty gradients nearly meaningless.

### **Fix Implemented**
- Reduced `R_DEATH` from `-50.0` to `-10.0` in `rewards.py`:
  ```python
  # Before
  R_DEATH = -50.0

  # After
  R_DEATH = -10.0  # Death penalty lowered so per-tick rewards aren't drowned out
  ```
- At `-10.0`, the death event is still a clear negative signal but no longer dominates the entire return distribution. Per-tick penalties (`R_WALL_HIT`, `R_HUNGER_DRAIN`, `R_DAMAGE_PER_HIT`) now contribute meaningfully to gradient updates, allowing the agent to learn *why* episodes fail, not just *that* they failed.

---

## 7. Symbolic Dictionary Bloat from Numeric-Only Tokens

**Date:** 2026-07-31

### **Challenge**
The Auto-Translator (`ats_virus_translator.py`) is called once per episode to process the GRADE string (e.g. `"Automatic evaluation: Episode 330 performance was BAD with reward rate -0.47."`). After ~950 episodes, the `symbolic_dictionary.json` had grown to hundreds of entries, the majority of which were pure numeric tokens — episode numbers (`330`, `500`, `700`), reward magnitudes (`47`, `128`), etc. — encoded as meaningless symbol/word pairs.

### **Root Cause**
- `parse_external_sentence()` extracted *all* `\b\w+\b` word tokens from the sentence and passed every one to `encode_word()`.
- `encode_word()` had no filter for numeric-only strings; encountering `"330"` it would compute a hash-fallback symbol and call `register_mapping()`, permanently writing the entry to disk.
- Because episode numbers are unique integers that grow monotonically, each episode added new unrepeatable entries — the dictionary was functioning as an episode counter, not a vocabulary.

### **Fix Implemented**
Two coordinated changes in `ats_virus_translator.py`:

1. **`encode_word()` guard** — detect and short-circuit pure numeric tokens without registering them:
   ```python
   # Skip registration for pure numbers (e.g. '330', '500', '47')
   if w_lower.lstrip('-').isdigit():
       return f"NUM:{w_lower}"
   ```

2. **`parse_external_sentence()` pre-filter** — strip numeric words from the token list before encoding:
   ```python
   meaningful_words = [w for w in words if not w.lstrip('-').isdigit()]
   encoded_tokens = [self.mapper.encode_word(w) for w in meaningful_words]
   ```

- Numeric values are still represented in corpus lines (the raw sentence is unchanged), but they no longer pollute the symbol dictionary.
- The `NUM:<value>` passthrough token is used in the encoded output for downstream consumers that still need numeric context.

---

## Summary of Config Parameters (`config_rl.py` & `rewards.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LEARNING_RATE` | `5e-4` | Adam optimizer learning rate |
| `GAMMA` | `0.99` | Reward discount factor |
| `CLIP_EPS` | `0.2` | PPO ratio clipping epsilon |
| `ENTROPY_COEFF` | `0.001` | Exploration bonus scaling factor |
| `LOG_INTERVAL` | `10` | TensorBoard & terminal print frequency |
| `BEST_MODEL_PATH` | `"checkpoints/model_best.pt"` | High-watermark model checkpoint |
| `R_DEATH` | `-10.0` | Death terminal penalty *(was -50.0)* |

---
