# ATS (AI Training System) & Virus Integration — Comprehensive Technical Documentation

## 1. Executive Summary & Project Vision

The **AI Training System (ATS)** is an interactive, 2D tile-world reinforcement learning simulator paired with a cognitive mind architecture and linked to **Virus**, a dual-profile Language Model.

The fundamental goal of ATS is to train an **embodied AI agent** that can navigate, perceive, interact, craft, fight, and solve complex multi-step survival and logic problems within an environment it actively inhabits. Rather than feeding an AI static text or isolated rewards, ATS gives the model a physical "shell" in a dynamic world governed by strict mechanical rules, survival needs (hunger, injury, threat, tool requirements), perception constraints (day/night, torch light), and combat mechanics.

By coupling this RL agent with the **Virus Language Model**, ATS translates physical world interactions into rich linguistic descriptions, allowing a language model to become acquainted with, narrate, and understand embodied experience.

---

## 2. Value Hierarchy: Most Valuable to Least Valuable Systems

Every component in ATS serves a specific purpose. The following hierarchy ranks all features, subsystems, and files from **Most Valuable (Tier 1 Core)** down to **Least Valuable (Tier 4 Utility & Stubs)** based on their contribution to intelligence, training efficacy, and system functionality.

```
+-----------------------------------------------------------------------------------+
| TIER 1: Core Embodied Intelligence & Cognitive Architecture                       |
|   1. Mind Architecture & Solution Loop (mind/solution_loop.py, need_detector.py)  |
|   2. Growing PPO Policy (model_rl.py)                                             |
|   3. Embodied Agent & Environment Core (agent.py, ats_env.py, world.py)            |
|   4. Dynamic Action Window & Masking (config_rl.py, agent.py)                     |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| TIER 2: Dual-Model Synergy & Online Learning                                      |
|   5. Virus Dual-Profile Adapter (ats_virus_adapter.py, export_virus_corpus.py)    |
|   6. Continual Online Learner (mind/continual_learner.py)                         |
|   7. Experience Memory & Solution Recall (mind/experience_memory.py)               |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| TIER 3: World Mechanics & Moddable Content                                       |
|   8. Hostility & Telegraph Combat State Machine (entities.py)                      |
|   9. Crafting & Recipe Engine (crafting.py)                                       |
|  10. Day/Night Perception Engine (day_night.py, biome.py)                         |
|  11. Data-Driven Asset Registry (assets/*.json, components/registry.py)           |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| TIER 4: Observer Interface & Extensibility Stubs                                 |
|  12. Pygame Monospaced CRT Terminal GUI (debug_gui.py)                             |
|  13. Cross-Episode Memory Decay (memory.py)                                       |
|  14. Event System Stubs (events.py)                                               |
+-----------------------------------------------------------------------------------+
```

---

### Tier 1: Core Embodied Intelligence & Cognitive Architecture (Most Valuable)

#### 1. Mind Architecture & Cognitive Solution Loop (`mind/solution_loop.py`, `mind/need_detector.py`)
- **Value**: ★★★★★ [Highest]
- **Role**: Bridges raw RL actions with high-level cognitive problem-solving.
- **Functionality**:
  - **Need Detection**: Continuously monitors agent vitals and environment states to quantify 4 active needs: `[Hunger, Injury, Threat, Tool]`.
  - **Solution Cycle**: Executes a 5-step cognitive cycle every tick: `Perceive -> Detect Need -> Recall Past Solution -> Act (Policy + Hint) -> Score & Store`.
  - **Trace Generation**: Produces a real-time inspection trace (`detect`, `recall`, `act`, `record`) rendered on the debug terminal and logged for analysis.

#### 2. Dynamic Growing PPO Policy (`model_rl.py`)
- **Value**: ★★★★★ [Highest]
- **Role**: Deep Reinforcement Learning network that expands capacity dynamically as world complexity scales.
- **Functionality**:
  - **Expandable Hidden Layers**: Starts with hidden width of 128 (`MIND_HIDDEN_SIZE`) and expands up to 1024+ at runtime when learning reward plateaus are detected.
  - **Weight Preservation**: Copying old weights into the top-left block of new expanded layers and zero-initializing new parameters ensures zero performance loss immediately following expansion.
  - **Checkpoint Compatibility**: Stores network width metadata in checkpoints (`model_rl.pt`), allowing seamless loading across varying hidden layer dimensions.

#### 3. Embodied Agent & Environment Core (`agent.py`, `ats_env.py`, `world.py`)
- **Value**: ★★★★★ [Highest]
- **Role**: Provides the physical embodiment, world grid, item placement, and step/reset loop.
- **Functionality**:
  - **Agent State Vector (126 floats)**: Encapsulates Vitals (HP, Hunger, Stamina, Leg Injury, Level, XP), Position & Height, 4 Adjacent Tile Biomes, Hostile Entity Proximity, Object Interactivity, 24-slot Inventory `(item_id, count)`, Status Effects, Day/Night Environment parameters, Memory features, and Action Masks.
  - **Movement & Navigation Engine**: Normal movement (forward, backward, left, right), Sprinting toggle (`ACT_SPRINT_ON` / `ACT_SPRINT_OFF` for high-speed evasion/travel), and Jumping.
  - **Tile World & Lockable Rooms**: Grid-based movement with biomes, obstacles, entity placement, and door unlocking triggers (e.g. requires sword acquisition + slime defeat).

#### 4. Context-Masked Dynamic Action Window (`config_rl.py`, `agent.py`)
- **Value**: ★★★★★ [Highest]
- **Role**: Hard enforcement of valid physical actions, preventing wasteful invalid choices.
- **Functionality**:
  - **42 Discrete Actions**: Encompasses movement (0–6), combat & picking up (7–9), inventory selection (10–33), item usage (34), inventory toggle (35–36), crafting toggle & item queueing (37–39), sleep (40), and wait (41).
  - **Action Masking Logic**: Computes a 42-element binary mask based on agent state, day/night, inventory state, and proximity. Masked-out actions receive logits set to `-1e8`, guaranteeing valid execution.

---

### Tier 2: Dual-Model Synergy & Online Learning (High Value)

#### 5. Virus Dual-Profile Adapter (`ats_virus_adapter.py`, `export_virus_corpus.py`)
- **Value**: ★★★★☆ [High]
- **Role**: Links the embodied RL simulator to the **Virus Language Model**.
- **Functionality**:
  - **Dual-Profile Architecture**: Virus maintains two separate model profiles:
    - `general`: Standard English text model trained on general text (`Virus/data/`).
    - `ats`: Specialized language model trained on ATS agent action narrations and world events (`Virus/data_ats/`).
  - **Runtime Corpus Export**: Streams episode summaries and narration prompts (`### INPUT: ... ### OUTPUT: ...`) into `ats_runtime_corpus.txt` to train Virus to articulate the agent's experiences.
  - **Profile Registry**: Exports `active_virus_profile.json` so downstream tools know which model/vocab is active.

#### 6. Continual Online Learner (`mind/continual_learner.py`)
- **Value**: ★★★★☆ [High]
- **Role**: Enables non-blocking, online PPO model updates while the simulation runs.
- **Functionality**:
  - Maintains a rolling transition buffer (`CONTINUAL_BUFFER_SIZE = 512`).
  - Triggers mini PPO update epochs every 256 ticks (`CONTINUAL_UPDATE_EVERY`), ensuring the model adapts continuously during live simulation without requiring full episode resets.

#### 7. Experience Memory & Solution Recall (`mind/experience_memory.py`)
- **Value**: ★★★★☆ [High]
- **Role**: Stores high-reward episode trajectories for need resolution and provides hints during action selection.
- **Functionality**:
  - Records step sequences when a need becomes active.
  - Matches current need vectors against historical successful solutions to supply hint biases during action selection.

#### 8. Auto-Translator & GRADE Feedback System (`ats_virus_translator.py`)
- **Value**: ★★★★☆ [High]
- **Role**: Converts end-of-episode performance grades into symbolic token packets and exports them to the Virus corpus, closing the evaluation-to-language feedback loop.
- **Functionality**:
  - **GRADE System**: Each episode is assigned a `MID` or `BAD` grade (based on per-tick reward rate). The grade + reward rate are formed into a natural language sentence: `"Automatic evaluation: Episode N performance was BAD with reward rate -0.47."`
  - **Symbolic Token Mapping**: Meaningful words in the grade sentence are mapped to deterministic symbolic codes (e.g. `"performance" → "1153$11"`), building a stable symbolic vocabulary for Virus.
  - **Numeric Filter**: Pure numeric tokens (episode counts, reward magnitudes) are excluded from dictionary registration via a `NUM:<value>` passthrough to prevent dictionary bloat.
  - **Polarity Signals**: Positive/negative English words in the grade string are parsed into a `+1.0 / -1.0` polarity signal appended to the Virus corpus line, giving the language model a direct sentiment tag.

---

### Tier 3: World Mechanics & Moddable Content (Medium Value)

#### 8. Hostility & Telegraph Combat State Machine (`entities.py`)
- **Value**: ★★★☆☆ [Medium]
- **Role**: Controls mob behavior, combat encounters, and entity interactions.
- **Functionality**:
  - **Hostility Rules**: Distinguishes between passive mobs, aggressive monsters, and **non-hostile until attacked** entities (allowing `INTERACT` or `ATTACK` based on agent choices).
  - **Telegraphing Attacks**: Monsters display warning states (e.g. charging up) before dealing damage, teaching the agent to dodge, sprint away, or time attacks.
  - **Entity Drops & Health**: Mobs drop items (meat, slime gel) upon defeat.

#### 9. Crafting & Recipe Engine (`crafting.py`)
- **Value**: ★★★☆☆ [Medium]
- **Role**: Governs item synthesis, recipe verification, and crafting menu interactions.
- **Functionality**:
  - Validates ingredient combinations queued from the inventory.
  - Auto-closes crafting interface upon recipe completion or cancellation.

#### 10. Day/Night Perception Engine (`day_night.py`, `biome.py`)
- **Value**: ★★★☆☆ [Medium]
- **Role**: Implements dynamic lighting, perception radius reduction, and environmental hazards.
- **Functionality**:
  - **Day/Night Cycle**: 300-tick diurnal cycle.
  - **Perception Radius**: Constricts vision from 5 tiles during daytime down to 2 tiles at night, unless a torch is equipped (`ACT_USE_ITEM`), which restores vision to 5 tiles.

#### 11. Data-Driven Asset Registry (`assets/items.json`, `assets/entities.json`, `components/registry.py`)
- **Value**: ★★★☆☆ [Medium]
- **Role**: Eliminates hardcoded items/mobs by parsing JSON component files.
- **Functionality**:
  - Decouples item attributes (`wood`, `iron_sword`, `torch`, `apple`, `book`) and entity stats (`weak_slime`, `zombie`, `villager`) into moddable files.

---

### Tier 4: Observer Interface & Extensibility Stubs (Least Valuable / Auxiliary)

#### 12. Pygame Monospaced CRT Terminal GUI (`debug_gui.py`)
- **Value**: ★★☆☆☆ [Auxiliary]
- **Role**: Provides visual debugging for human observers via a Pygame window styled like a retro green CRT terminal.
- **Functionality**:
  - Displays 5x5 World View, 24-slot Inventory grid, Needs Detector meters, Mind Trace output, Action Window (valid moves), and Agent Vitals.
  - *Note*: Operates independently of simulation logic (simulation runs headless with `--no-gui` for training speed).

#### 13. Cross-Episode Memory Decay (`memory.py`)
- **Value**: ★★☆☆☆ [Auxiliary]
- **Role**: Manages confidence decay of long-term memory over training episodes.

#### 14. Environmental Event Stubs (`events.py`)
- **Value**: ★☆☆☆☆ [Stub]
- **Role**: Provides event hook placeholders (`on_raid`, `on_fire`) for future expansion phases.

---

## 3. Detailed Architecture Breakdown

### 3.1 Neural Network & Growing Policy (`model_rl.py`)

```
   State (126 floats)
           │
           ▼
    Linear(126 → W)  [fc1]  + ReLU
           │
           ▼
    Linear(W → W)    [fc2]  + ReLU
        ┌──┴───────────────────────┐
        ▼                          ▼
 Actor Linear(W → 42)    Critic Linear(W → 1)
        │                          │
        ▼                          ▼
 Action Logits (42)        State Value (V)
   + Action Mask
```

- **Hidden Layer Expansion Algorithm**:
  ```python
  def expand(self, new_hidden_size):
      # Copies old weight matrices into top-left block of new linear layers.
      # Zeros out new neuron weights/biases to guarantee output identity at time of expansion.
  ```

---

### 3.2 The Mind Subsystem & Cognitive Cycle (`mind/`)

```
                        [ ATS Environment ]
                                 │
                         State Vector (126)
                                 │
                                 ▼
                     [ NeedDetector.detect() ]
                                 │
                        Need Vector (4 floats)
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
 [ Urgent Need Active? ]                     [ Experience Memory ]
   - Yes: Begin Recording                      - Recall past hint for need
   - No:  End Recording                                │
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 ▼
                     [ Policy Network + Hint ]
                                 │
                            Action (0-41)
                                 │
                                 ▼
                     [ Record Outcome / Step ]
```

---

### 3.3 Virus Dual-Model Pipeline (`ats_virus_adapter.py`)

```
  [ ATS Simulation Episode ]
              │
              ├──> Agent Actions & World Events
              │
              ▼
   [ VirusAdapter.export_runtime_corpus() ]
              │
              ▼
   Virus/data_ats/ats_runtime_corpus.txt
              │
              ▼
   [ py -3 auto_train.py --profile ats ]
              │
              ▼
   Virus/data_ats/model_ats_v3.pt (ATS Language Model)
```

---

### 3.4 Auto-Translator & GRADE Feedback Pipeline (`ats_virus_translator.py`)

```
  [ End of Episode ]
        │
        ▼
  GRADE assigned: MID / BAD
  (based on per-tick reward rate threshold)
        │
        ▼
  ExternalTranslatorAgent.process_episode_grade(episode, grade, rew_per_tick)
        │
        ▼
  Natural language sentence generated:
  "Automatic evaluation: Episode N performance was BAD with reward rate -0.47."
        │
        ├──▶ Polarity parsed (+1.0 GOOD / -1.0 BAD / 0.0 NEUTRAL)
        │
        ├──▶ Numeric tokens filtered out (episode number, reward value)
        │
        └──▶ Remaining words encoded to symbolic tokens via SymbolicTokenMapper
                │
                ▼
        TRANSLATOR_MSG corpus line exported to Virus corpus
        (POLARITY + TOKENS appended)
```

**Symbolic Dictionary** (`data/symbolic_dictionary.json`):
- Persists across runs as a JSON file with `symbol_to_word` and `word_to_symbol` maps.
- Grows only with meaningful vocabulary words; numeric-only tokens return `NUM:<value>` passthrough without a dictionary write.
- Deterministic hash-fallback ensures the same word always maps to the same symbol code across sessions.

---

## 4. Operational Workflow & Execution Commands

### 4.1 Quickstart & Interactive Simulation (GUI Mode)
To run the interactive simulation with the CRT debug terminal:

```powershell
cd C:\Users\Virus\Documents\github\ATS
py -3 create_rl.py
py -3 main.py --episodes 3
```

### 4.2 High-Speed Headless Training Mode
To run headless PPO training without graphics rendering:

```powershell
py -3 main.py --no-gui --episodes 50
py -3 train_rl.py
```

### 4.3 Training the Virus Language Model on ATS Experience
To train the Virus language model on narrations generated by the ATS agent:

```powershell
# 1. Run simulation and export narration corpus
py -3 main.py --virus-profile ats --no-gui --episodes 10

# 2. Train Virus ATS-profile LM
cd ..\Virus
py -3 auto_train.py --profile ats --epochs 30 --lr 0.0001

# 3. Generate narration from trained model
py -3 generate_v3.py --profile ats --prompt "### INPUT: action narration`n### OUTPUT:"
```

---

## 5. Summary Table of Key Project Artifacts & Files

| File / Component | Primary Responsibilities | Value Tier |
|------------------|--------------------------|------------|
| [mind/solution_loop.py](file:///c:/Users/Virus/Documents/github/ATS/mind/solution_loop.py) | Cognitive cycle orchestration (`detect -> recall -> act -> record`), GUI trace | Tier 1 (Highest) |
| [model_rl.py](file:///c:/Users/Virus/Documents/github/ATS/model_rl.py) | Dynamic Growing PPO Actor-Critic neural network with runtime layer expansion | Tier 1 (Highest) |
| [agent.py](file:///c:/Users/Virus/Documents/github/ATS/agent.py) | Embodied agent stats, inventory, movement, vitals, action mask generator | Tier 1 (Highest) |
| [ats_env.py](file:///c:/Users/Virus/Documents/github/ATS/ats_env.py) | Gym-style environment step/reset loop, state vector assembly | Tier 1 (Highest) |
| [world.py](file:///c:/Users/Virus/Documents/github/ATS/world.py) | Tile map grid, biome management, door unlock logic, entity/object tracking | Tier 1 (Highest) |
| [config_rl.py](file:///c:/Users/Virus/Documents/github/ATS/config_rl.py) | Central parameters: 42 actions, 126 state size, PPO hyperparams, Virus paths | Tier 1 (Highest) |
| [ats_virus_adapter.py](file:///c:/Users/Virus/Documents/github/ATS/ats_virus_adapter.py) | Bridge between ATS runtime events and Virus dual-model (`ats` vs `general`) | Tier 2 (High) |
| [ats_virus_translator.py](file:///c:/Users/Virus/Documents/github/ATS/ats_virus_translator.py) | Auto-Translator: GRADE feedback → symbolic token encoding → Virus corpus export | Tier 2 (High) |
| [mind/continual_learner.py](file:///c:/Users/Virus/Documents/github/ATS/mind/continual_learner.py) | Non-blocking online PPO learning buffer & update trigger | Tier 2 (High) |
| [mind/need_detector.py](file:///c:/Users/Virus/Documents/github/ATS/mind/need_detector.py) | Converts agent vitals and environment into 4 normalized need scores | Tier 2 (High) |
| [mind/experience_memory.py](file:///c:/Users/Virus/Documents/github/ATS/mind/experience_memory.py) | Trajectory recording, experience buffer, solution recall for hints | Tier 2 (High) |
| [rewards.py](file:///c:/Users/Virus/Documents/github/ATS/rewards.py) | Per-tick reward constants and terminal penalty definitions (`R_DEATH = -10.0`) | Tier 2 (High) |
| [entities.py](file:///c:/Users/Virus/Documents/github/ATS/entities.py) | Mob hostility states (passive, aggressive, non-hostile until attacked), attack telegraphing | Tier 3 (Medium) |
| [crafting.py](file:///c:/Users/Virus/Documents/github/ATS/crafting.py) | Multi-item crafting recipes, item queueing, menu auto-closing | Tier 3 (Medium) |
| [day_night.py](file:///c:/Users/Virus/Documents/github/ATS/day_night.py) | Diurnal cycle, perception radius constriction, torch illumination | Tier 3 (Medium) |
| [components/registry.py](file:///c:/Users/Virus/Documents/github/ATS/components/registry.py) | Data-driven JSON asset loader for items and entities | Tier 3 (Medium) |
| [debug_gui.py](file:///c:/Users/Virus/Documents/github/ATS/debug_gui.py) | Pygame CRT green-screen monospaced observer terminal | Tier 4 (Auxiliary) |
| [main.py](file:///c:/Users/Virus/Documents/github/ATS/main.py) | Main entrypoint for simulation, GUI toggle, and episode loops | Tier 4 (Auxiliary) |
