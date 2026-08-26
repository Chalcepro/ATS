# ATS System Architecture & Cognitive Overview

ATS integrates embodied reinforcement learning with modular cognitive architectures, procedural world generation, and language model grounding.

```mermaid
flowchart TD
    subgraph Mind ["Mind Architecture (mind/)"]
        ND[Need Detector] --> SL[Solution Loop]
        EM[Experience Memory] <--> SL
        SL --> ACT[Policy Forward (Hidden: 256)]
        CL[Continual Learner] -->|Online PPO| ACT
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
- Maintains a rolling ring buffer of recent transitions.
- Triggers online mini-PPO updates in the background every 64 ticks.
- Enables the agent to adapt and improve weights immediately while exploring without waiting for episode termination.
- **Checkpoint Synchronization**: Automatically saves matching architecture metadata (`hidden_size: 256`) to both `checkpoints/model_rl.pt` and `checkpoints/model_best.pt` for seamless warm-start reloading across sessions.

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


### B. Action Space (42 Discrete Actions)
- `0..3`: Movement (Forward/North, Backward/South, Left/West, Right/East)
- `4..5`: Sprint (Toggle On / Off)
- `6`: Jump
- `7`: Attack / Harvest (Attacks entity or harvests object directly in front)
- `8`: Pick Up (Collects loose resource or harvests bush)
- `9`: Interact (Interacts with NPC, chest, or water)
- `10..33`: Select Inventory Slot 0 to 23
- `34`: Use Selected Item (Eat food, apply bandage, ignite torch)
- `35..36`: Open / Close Inventory
- `37..39`: Open / Add / Close Crafting
- `40`: Sleep (Restores health if in bed at night)
- `41`: Wait

### C. State Vector Layout (126 Dimensions)
| Feature Index Range | Description |
| :--- | :--- |
| `0..4` | Vitals (Health, Hunger, Stamina, XP, Level) |
| `5..9` | Spatial Position & Elevation Profile ($Z$, $Z_{ahead}$, $Z_{left}$, $Z_{right}$, $Z_{behind}$) |
| `10..13` | Adjacent Tile Types (North, West, East, South) |
| `14..17` | Nearest Hostile Features (Distance, Type, State, Incoming Damage) |
| `18..19` | Nearest Object Features (Distance, Item ID) |
| `20..67` | 24-Slot Inventory: 24 × (Item ID, Item Count) |
| `68..73` | Status Effects (Poisoned, Burning, Slowed, Blinded, Speed Boost, Leg Injured) |
| `74..81` | Environment & Facing: Time of Day, Torch Active, Crafting Open, Inventory Open, Biome, Failed Action, **Facing X**, **Facing Y** |
| `82..84` | Episode Memory Features |
| `85..125` | Action Validity Mask (42 binary flags) |
