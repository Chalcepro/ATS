# ATS System Architecture & Cognitive Overview

ATS integrates embodied reinforcement learning with modular cognitive architectures, procedural world generation, and language model grounding.

```mermaid
flowchart TD
    subgraph Mind ["Mind Architecture (mind/)"]
        ND[Need Detector] --> SL[Solution Loop]
        EM[Experience Memory] <--> SL
        SL --> ACT[Policy Forward (Hidden: 256, fixed)]
        CL[Continual Learner] -->|On-policy PPO, 256-step rollouts| ACT
    end

    subgraph Environment ["Environment (ats_env.py)"]
        W[World & Procedural Chunks]
        A[Agent Stats & Inventory]
        DN[Day / Night Cycle]
        RE[Reward Engine]
    end

    subgraph External ["External Integration"]
        GUI[Terminal CRT GUI]
        TR[Translator Agent]
        V[Virus LM Corpus]
    end

    Environment -->|State Vector (126-D)| Mind
    Mind -->|Action Index (0..41)| Environment
    Environment -->|Render Data| GUI
    Environment -->|Narrations & Evaluation| TR --> V
```

---

## 1. The Mind Architecture (`mind/`)

### A. Solution Loop (`mind/solution_loop.py`)
Executes the primary 5-stage cognitive cycle on every simulation tick:
1. **Perceive (Detect)**: `NeedDetector` converts the 126-dimensional state vector into a normalized 4-D urgency vector (`[Hunger, Injury, Threat, Tool]`).
2. **Need**: Determines if any need exceeds the active threshold ($> 0.15$).
3. **Recall**: Queries `ExperienceMemory` for previously successful action sequences addressing the urgent need.
4. **Act**: The Actor-Critic neural network samples an action constrained by the dynamic action mask.
5. **Record**: Logs action outcomes and reinforces sequences that resolve triggering needs.

### B. Continual Learner (`mind/continual_learner.py`)
- Collects a rollout of `CONTINUAL_BUFFER_SIZE` (256) transitions, runs a PPO
  update (`CONTINUAL_MINI_EPOCHS` epochs), then **clears the buffer** — every
  transition is trained exactly once, on-policy. `flush()` forces an update at
  an episode boundary so the tail of an episode isn't carried unlearned.
- Advantages are **GAE(`GAE_LAMBDA`=0.95)** over the rollout-time values
  stored per-transition, with a `done`-mask that zeroes bootstrap and
  lambda-return propagation across episode boundaries; critic target
  (`returns`) is `advantage + old_value`. Only the advantage is normalised.
  (Replaced raw discounted-reward-to-go advantage 2026-09-04 — that was
  unbiased but high-variance, matching a non-monotonic reward trend over a
  300-episode run; see `docs/architecture_and_reasoning_2026-09-04.md` and
  `updates/2026-09-04c ...` for why the earlier return-normalisation bug and
  this variance issue are separate problems. GAE fixed the variance — critic
  loss is confirmed low and stable across a full 2000-episode run.)
- **Entropy-collapse guard (fixed + verified 2026-09-05).** A flat
  `ENTROPY_COEFF` was found to let policy entropy decay toward zero and
  never recover — confirmed directly from per-update `entropy`/`adv_mean`
  logging (added 2026-09-04d): once entropy is near-zero, the entropy bonus
  (`ENTROPY_COEFF * entropy`) contributes a vanishingly small gradient
  regardless of the coefficient, so nothing pulls the policy back; a
  2000-episode run went from +50 avg reward to hundreds of consecutive
  byte-identical episodes (same reward, same tick count, `MaxTicks`
  termination). Fix: `ContinualLearner.entropy_coeff` (not
  `config_rl.ENTROPY_COEFF` directly) now *adapts* — ramps up toward
  `ENTROPY_COEFF_MAX` (0.10, 20x the floor) while measured entropy is below
  `ENTROPY_TARGET` (0.5 nats), relaxes back toward the floor
  (`config_rl.ENTROPY_COEFF`, kept live so the GUI's Hyperparameter Studio
  control still works) once healthy. **Verified** on a fresh 2000-episode
  run vs. the pre-fix run: mean reward -193 → +80, median -75 → +81,
  % positive episodes 40% → 98%, no sustained collapse (previously 170+
  exact-tick repeats and hundreds of identical dead-end episodes; after the
  fix the most-repeated tick count across 2000 episodes was 16, normal
  coincidence). Caveat: not full immunity — one single-episode dip still
  occurred (coefficient hit the ceiling but 15 mini-batches wasn't enough
  to arrest a fast in-episode entropy slide), but it self-corrected on the
  very next episode rather than compounding. See `updates/2026-09-04e ...`
  (diagnosis) and `updates/2026-09-05 ...` (fix + verification) for detail.
  `checkpoints/model_best.pt`/`model_rl.pt` from the verification run are a
  **healthy** policy, safe to resume from.
- Auto-growing the hidden width is **disabled by default**
  (`config_rl.MIND_GROWTH_ENABLED = False`).
- **Checkpoint Synchronization**: architecture metadata (`hidden_size`) is saved
  into `checkpoints/model_rl.pt` and `checkpoints/model_best.pt` so reloads pick
  the right width. Payloads also carry `reward_scheme_version`
  (`config_rl.REWARD_SCHEME_VERSION`) — `main.py`'s `_load_policy` **skips**
  (does not silently resume) any checkpoint whose version doesn't match the
  current one, since an old checkpoint's critic was calibrated under a
  different reward/return scheme and silently resuming it looks like a
  regression that isn't one (this happened once already — see
  `updates/2026-09-04b ...`). Bump `REWARD_SCHEME_VERSION` whenever the
  reward function or return/advantage computation changes.

---

## 3. Mission Control UI & Visual Telemetry (`debug_gui.py`)

### A. Authentic Retro Pixel Styling & Typography
- **Primary Font**: `04b_03 regular` (`C:\Windows\Fonts\04B_03B.TTF`, `04b03rev.ttf`, or `assets/fonts/04B_03.TTF`).
- **Retro Geometry**: Sharp 1px pixel rectangular geometry with zero rounded corners (`border_radius=0`) and no modern drop shadows.
- **Solid High-Contrast Action Buttons**:
  - `[⏹ STOP]`: Solid Vibrant Red (`#B91E1E`) with bold white text.
  - `[💾 SAVE]`: Solid Vibrant Green (`#148C3C`) with bold white text.
  - `[⚡ TURBO]`: Solid Vibrant Blue (`#146EB4`) with bold white text.
  - `[⏸ PAUSE]`: Solid Vibrant Amber (`#B48214`) with bold white text.

### B. Collapsible Navigation Sidebar Drawer
- Located at the top right of the viewport: `[ ◀ TABS / VIEWS ]`.
- Toggles a slide-out navigation drawer allowing instant switching between:
  - `1: 🎮 LIVE SIMULATION` (CRT survival viewport, 24-slot inventory, 7x7 grid, vitals, event log).
  - `2: 📊 TRAINING ANALYTICS` (Zoomable Reward & Loss telemetry graphs, timeframe presets, scrollable ascending log).
  - `3: ⚙️ HYPERPARAMETERS` (Pre-launch & paused parameter adjustment studio).
  - `4: 🧠 MIND & LLM` (Need vectors, cognitive cycle traces, and Virus language model bridge).

### C. Advanced Analytics & CSV Export Pipeline
- **Zoomable Timeframe Graphs**: Supports `Last 10`, `Last 25`, `Last 50`, `All Time`, and interactive `Zoom +/-` controls with dynamic Y-axis min/max scaling.
- **Ascending Episode History Table**: Renders episodes chronologically from top to bottom (latest episode at bottom), auto-scrolls, and supports mouse wheel / button scrolling (`▲ UP` / `▼ DOWN`).
- **Automatic CSV Export**: Automatically exports every completed episode to `data/ats_episode_history.csv` with fields:
  `episode, timestamp, total_reward, ticks_survived, termination_reason, speed_multiplier, learning_rate, entropy_coeff, hidden_size`


### B. Action Space (25 Discrete Actions)
- `0..3`: Movement (Forward/North, Backward/South, Left/West, Right/East)
- `4..5`: Sprint (Toggle On / Off)
- `6`: Jump
- `7`: Attack / Harvest (Attacks entity or harvests object directly in front)
- `8`: Pick Up (Collects loose resource or harvests bush)
- `9`: Interact (Interacts with NPC, chest, or water)
- `10..16`: Select Inventory Slot 0 to 6 (Only enabled for occupied slots; empty slots give disabled feedback)
- `17`: Use Selected Item (Secondary action for selected slot: eat food, apply bandage, ignite torch)
- `18..19`: Open / Close Inventory
- `20..22`: Open / Add / Close Crafting
- `23`: Sleep (Restores health if in bed at night)
- `24`: Wait

### C. State Vector Layout (79 Dimensions)
| Feature Index Range | Description |
| :--- | :--- |
| `0..4` | Vitals (Health, Hunger, Stamina, XP, Level) |
| `5..9` | Spatial Position & Elevation Profile ($Z$, $Z_{ahead}$, $Z_{left}$, $Z_{right}$, $Z_{behind}$) |
| `10..13` | Adjacent Tile Types (North, West, East, South) |
| `14..17` | Nearest Hostile Features (Distance, Type, State, Incoming Damage) |
| `18..19` | Nearest Object Features (Distance, Item ID / 100) |
| `20..33` | 7-Slot Inventory: 7 × (Normalized Item ID `id/100`, Normalized Count `cnt/99`) |
| `34..41` | Status Effects (Poisoned, Burning, Slowed, Blinded, Speed Boost, Leg Injured, Cold, Acid Burn) |
| `42..50` | Environment & Facing: Time of Day, Torch Active, Crafting Open, Inventory Open, Biome, Failed Action, Facing X, Facing Y, Ocean Ticks |
| `51..53` | Episode Memory Features (3 floats) |
| `54..78` | Action Validity Mask (25 binary flags — empty inventory slot actions are masked out = 0) |
