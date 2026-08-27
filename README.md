# ATS (AI Training System)

Phase 1 baseline: tile-world RL simulator with moddable asset components, linked to Virus dual-model profiles (`general` + `ats`).

```text
          _____                _____                    _____
         /\    \              /\    \                  /\    \
        /::\    \            /::\    \                /::\    \
        /::::\    \           \:::\    \              /::::\    \
       /::::::\    \           \:::\    \            /::::::\    \
      /:::/\:::\    \           \:::\    \          /:::/\:::\    \
     /:::/__\:::\    \           \:::\    \        /:::/__\:::\    \
    /::::\   \:::\    \          /::::\    \       \:::\   \:::\    \
   /::::::\   \:::\    \        /::::::\    \    ___\:::\   \:::\    \
  /:::/\:::\   \:::\    \      /:::/\:::\    \  /\   \:::\   \:::\    \
 /:::/  \:::\   \:::\____\    /:::/  \:::\____\/::\   \:::\   \:::\____\
 \::/    \:::\  /:::/    /   /:::/    \::/    /\:::\   \:::\   \::/    /
  \/____/ \:::\/:::/    /   /:::/    / \/____/  \:::\   \:::\   \/____/
           \::::::/    /   /:::/    /            \:::\   \:::\    \
            \::::/    /   /:::/    /              \:::\   \:::\____\
            /:::/    /    \::/    /                \:::\  /:::/    /
           /:::/    /      \/____/                  \:::\/:::/    /
          /:::/    /                                 \::::::/    /
         /:::/    /                                   \::::/    /
         \::/    /                                     \::/    /
          \/____/                                       \/____/

                           AGENT TRAINING SYSTEM
```

Source: [logo/logo_Style.txt](logo/logo_Style.txt)

## Quick start

```powershell
cd C:\Users\Virus\Documents\github\ATS
py -3 -m pip install -r requirements.txt
py -3 create_rl.py
py -3 main.py --episodes 2
```

This opens a **pygame window styled like a CRT terminal** (green monospace on black).
It is not the Windows console. Close with ESC or the window X.

Headless (no window):

```powershell
py -3 main.py --no-gui --episodes 1
py -3 train_rl.py
```

## Virus dual-model integration

ATS uses Virus profile `ats` by default. Virus keeps a separate `general` profile for normal English/text training.

| Profile | Data | Model | Vocab |
|---------|------|-------|-------|
| `general` | `Virus/data/` | `model_v3.pt` | `vocab.json` |
| `ats` | `Virus/data_ats/` | `model_ats_v3.pt` | `vocab_ats.json` |

### Train / swap models

```powershell
# Train ATS-specialized Virus LM
cd ..\Virus
py -3 auto_train.py --profile ats --epochs 30 --lr 0.0001

# Train general text model (unchanged default)
py -3 auto_train.py --profile general

# Create fresh ATS checkpoint
py -3 scripts\create_v3.py --profile ats

# Generate with ATS profile
py -3 generate_v3.py --profile ats --prompt "### INPUT: action narration`n### OUTPUT:"
```

### From ATS

```powershell
cd ..\ATS
py -3 main.py --virus-profile ats --no-gui --episodes 1
py -3 export_virus_corpus.py data\runtime_log.txt
```

Active profile metadata is written to `ATS/data/active_virus_profile.json`.
Runtime narration/corpus appends to `Virus/data_ats/ats_runtime_corpus.txt`.

## Modding assets (no hardcoding)

Add/edit JSON component files:

- `assets/items.json` — required: `name`, `type`
- `assets/entities.json` — required: `name`, `type`, `hp`, `damage`

World/entity/reward systems load through `components/registry.py` and `item_ids.py`.
New IDs (e.g. next item after `0014`) are data-only additions.

## Phase 1 systems

| Module | Role |
|--------|------|
| `ats_env.py` | Tick loop, episode reset |
| `world.py` | Starting room, tiles, door unlock |
| `agent.py` | Stats, actions 0–12, state array |
| `entities.py` | Telegraph attack state machine |
| `rewards.py` | Reward table |
| `crafting.py` | Recipes + auto-close menu |
| `day_night.py` | Cycle + perception radius |
| `memory.py` | Cross-episode confidence decay |
| `events.py` | Hooks for fire/raids (stubs) |
| `model_rl.py` / `train_rl.py` / `create_rl.py` | PPO-style policy scaffold |
| `debug_gui.py` | Pygame terminal-look window (not OS console) |
| `ats_virus_adapter.py` | Profile select / train / export |

## State / actions (GDD contract)

- State vector: ~40 floats (`config_rl.STATE_SIZE`)
- Actions: 0–12 (move, sprint, jump, attack, pick up, use, craft, sleep, interact, wait)
- Starting room: apple/sword/book/coal/wood + weak slime; door unlocks after sword + slime kill

## Notes

- Observer UI is a **pygame window that looks like a terminal** (no textures / fancy buttons).
  Close with ESC or the window X. Use `--no-gui` only for headless runs.
- Virus LM is for narration/personality; the RL agent never sees words.
- Full biome blending, villages, raids = later phases; hooks are in place.
