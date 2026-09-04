# 🤖 AGENTS.md — ATS AI Agent Onboarding & Contribution Guide

> **MANDATORY READ FOR ALL AI AGENTS.**
> Before writing a single line of code, read this file top to bottom.
> Your first action after reading this MUST be to update the relevant docs.

---

## 📌 What Is This Project?

**ATS (Adaptive Training System)** is a reinforcement learning simulation where a neural-network-driven agent learns to survive, explore, and craft in a procedurally generated island world.

- **Language**: Python 3.12
- **ML Framework**: PyTorch (PPO / Actor-Critic)
- **GUI**: Pygame terminal-style retro CRT GUI (`debug_gui.py`)
- **Entry Point**: `py -3.12 main.py`

---

## 🏗️ Project Structure (Key Files)

| File / Folder | Purpose |
| :--- | :--- |
| `main.py` | Entry point — GUI run loop, session orchestration |
| `ats_env.py` | Core simulation environment (step, reset, reward dispatch) |
| `agent.py` | Agent class — inventory, stats, state vector, action mask |
| `model_rl.py` | `RLPolicy` — Actor-Critic neural net, expansion logic |
| `config_rl.py` | **All constants** — action indices, state sizes, hyperparams |
| `rewards.py` | `RewardEngine` — all reward signals, grading, progression |
| `world.py` | `World` — tile map, island registry, lazy ocean generation |
| `island_gen.py` | `IslandRegistry` — 6 biome islands, tile layout, spacing |
| `debug_gui.py` | Mission Control terminal GUI — all rendering, HUD, tabs |
| `mind/solution_loop.py` | Cognitive cycle: detect → recall → act → record |
| `mind/experience_memory.py` | Problem/solution store — persists across sessions |
| `mind/need_detector.py` | Maps state vector to 4-D urgency needs |
| `mind/continual_learner.py` | Online PPO mini-updates between episodes |
| `memory.py` | Cross-episode spatial confidence decay |
| `docs/` | **Living documentation** — update these when you change things |
| `updates/` | **Chronological change logs** — add a new file for each session |

---

## ⚠️ Critical Architecture Facts

### State Vector — 79 Dimensions (`STATE_SIZE = 79`)
```
[0..4]    Vitals (Health, Hunger, Stamina, XP, Level)         — normalized 0..1
[5..9]    Spatial Z-elevation profile (5 directions)
[10..13]  Adjacent tile types (raw tile enum int)
[14..17]  Nearest hostile: dist, type, state, incoming damage
[18..19]  Nearest object: dist (norm), item_id (norm = id/100)
[20..33]  Inventory: 7 slots × (norm_item_id, norm_count)     — NORMALIZED!
[34..41]  Status effects: 8 binary/int flags
[42..50]  Environment: time, torch, crafting, inv_open, island, failed_action, facing_x, facing_y, ocean_ticks
[51..53]  Episode memory features (3 floats)
[54..78]  Action validity mask (25 binary)
```
> **⚡ All item IDs in the state are normalized as `id / ITEM_VOCAB_SIZE` (i.e. /100).**
> **An empty slot is always `[0.0, 0.0]`. Never pass raw integer IDs.**

### Action Space — 25 Actions (`ACTION_SIZE = 25`)
```
0..3   Move (F/B/L/R)      4..5  Sprint (on/off)    6   Jump
7      Attack/Harvest       8     Pick Up             9   Interact
10..16 Select Inventory Slot 0-6  (only enabled for OCCUPIED slots!)
17     Use Item (Secondary action for selected slot)
18..19 Open/Close Inventory
20..22 Open/Add/Close Crafting     23  Sleep    24  Wait
```
> **⚡ Action mask only enables slot-select actions for slots with `count > 0`.**
> If an empty slot is selected or used, direct "Disabled / Empty Slot" feedback (-1.0) is dispatched.

### Island IDs & Discovery
- Island 0 = **Starting island (Lush Grassland)** — pre-registered in `discovered_islands` on every `reset()`. Never gives +50 discovery reward.
- Islands 1–5 = remote biomes. Discovery reward fires **once only per episode** per island.

### Ocean / Shark Mechanics
- `OCEAN_SHARK_TICKS = 3` — 3 ticks in `TILE_OCEAN` → shark triggers
- If `HP >= 20`: respawn at island spawn with HP=15, run continues
- If `HP < 20`: harsh stop (`done=True`), episode graded `"BAD"`
- `TILE_WATER` (internal ponds) → `0.5×` speed; `TILE_OCEAN` → `0.5×` (entry) / `0.33×` (after 3 blocks)

### Experience Memory
- Located in `mind/experience_memory.py`
- Saves/loads from `data/experience_memory.json` across sessions
- Recalled hints inject soft logit boosts to matching past actions (not hard overrides)

---

## 📝 Your Contribution Rules (Follow Every Time)

### BEFORE making changes:
1. Read `docs/system_overview.md` and `docs/world_and_items.md`
2. Check the latest file in `updates/` to understand what was recently changed
3. Check `config_rl.py` for any constant you need — do not hardcode values

### AFTER making changes:
1. **Add an update file** in `updates/` following the naming format below
2. **Update `docs/system_overview.md`** if you changed the state vector, action space, or architecture
3. **Update `docs/world_and_items.md`** if you added/changed items, entities, or biomes
4. Run a quick sanity check: `python -c "from ats_env import ATSEnvironment; env = ATSEnvironment(); s = env.reset(); print('State len:', len(s))"`

### Update file naming format:
```
updates/YYYY-MM-DD <short description of what changed>
```
Examples:
- `updates/2026-08-28 inventory state normalization and action mask fix`
- `updates/2026-08-27 water mechanics and island spacing overhaul`

### Update file format (use this template):
```markdown
# ATS Update — YYYY-MM-DD
## Summary
One line summary of what this session changed.

## Changes
### [component name] — [file(s) changed]
- Bullet describing the specific change and why
- Include file links and line numbers where helpful

## Verified
- [ ] State vector length still == 130
- [ ] GUI renders without NameError or crash
- [ ] Training loop runs at least 1 episode cleanly
```

---

## 🔴 Known Gotchas — Don't Break These

| Trap | What to watch for |
| :--- | :--- |
| `embed_proj` expansion | When `RLPolicy.expand()` is called, `embed_proj` MUST be expanded alongside `fc1`. Missing this causes dimension mismatch RuntimeError. |
| State normalization | All item IDs must be `/100` normalized in state. Raw integers `1, 2, 8` will confuse the net. |
| `discovered_islands` init | Must `add(world.starting_island.island_id)` in **both** `__init__` and `reset()` of `ATSEnvironment`. |
| Action mask — empty slots | Never enable `ACT_SELECT_SLOT_BASE + i` for slots where `count == 0`. |
| `ckpt_exists` in `_draw_menu` | Must be defined locally inside `_draw_menu()` — it is not a class variable. |
| GUI freeze | `generate_narration()` must use `timeout=0.8` to avoid blocking the render loop. |
| Fresh reset | Must clear `experience_memory.json` AND reset policy weights. |

---

## 📂 Docs to Keep Updated

| Doc | When to update |
| :--- | :--- |
| [`docs/system_overview.md`](docs/system_overview.md) | State vector changes, action space changes, architecture changes |
| [`docs/world_and_items.md`](docs/world_and_items.md) | New items, entity changes, biome changes, crafting recipes |
| [`docs/how_to_use.md`](docs/how_to_use.md) | GUI flow changes, new buttons, startup flags |
| [`updates/`](updates/) | **Every session** — log what you changed |

---

*This file is maintained by AI agents and the project owner. Update it when the architecture evolves.*
