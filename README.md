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

## The training ladder

`curriculum.py` is the thing the agent is actually trained on. One network the
whole way; only the room changes, and it grows back up as the policy earns it.
A rung is passed when greedy play - not sampled play - clears its bar.

    python train_curriculum.py                    # the whole ladder
    python train_curriculum.py --stage warden     # one rung

Each rung adds **exactly one** new thing. That rule is the whole design: an
earlier version went from the nursery straight to "maze + hazards + damage +
items + goals that do not come back", trained to 32% and then decayed to 10%.

| rung | the one new thing |
|------|-------------------|
| `nursery` 5x5 | a goal exists and reaching it ends the episode |
| `corridors` 5x5 | walls |
| `corridors7` 7x7 | size |
| `avoid` 7x7 | something to avoid that cannot yet kill |
| `hazards` 7x7 | now it can kill |
| `foraging` 7x7 | scarcity - the room has to be cleared |
| `primary` 9x9 -> 11x11 | size, with everything above |
| `junior` 9x9 -> 11x11 | something in the room with you |
| `senior` 15x15 -> 19x19 | a room too big to find the goal by luck |
| `senior-short` | two goals, in order |
| `senior-trail` | five goals, in order |
| `armed` 15x15 -> 19x19 | a weapon, and a reason to carry it |
| `hunted` | they come to you |
| `warden` | more of them, and more again as the room grows |

Every rung is checked before a second of training is spent on it
(`best_case_return`): if a competent agent playing well would still score
negative, the reward is the bug and no amount of training will find it. That
check is here because the full world did not have it - an episode that
survived to max ticks scored -3281 while dying outright cost -10, so the
policy correctly learned to end episodes early.

### Where it got to

Greedy play, 40 episodes a rung, after consolidation (2026-09-26):

| rung | greedy | bar | | rung | greedy | bar |
|------|-------:|----:|-|------|-------:|----:|
| nursery 5x5 | 100% | 85% | | senior 15x15 | 72% | 50% |
| corridors 5x5 | 100% | 80% | | senior 17x17 | 75% | 50% |
| corridors7 7x7 | 100% | 75% | | senior 19x19 | 62% | 50% |
| avoid 7x7 | 90% | 75% | | satchel 5x5 | 98% | 85% |
| hazards 7x7 | 75% | 70% | | satchel7 7x7 | 92% | 75% |
| foraging 7x7 | 98% | 70% | | armed 15x15 | 62% | 50% |
| primary 9x9 | 95% | 65% | | armed 17x17 | 70% | 50% |
| primary 11x11 | 95% | 65% | | armed 19x19 | 57% | 50% |
| junior 9x9 | 98% | 60% | | hunted 15/17/19 | 78/72/68% | 50% |
| junior 11x11 | 90% | 60% | | warden 15/17/19 | 72/62/62% | 45% |

Every rung the brain claims, it plays. What the combat rungs do, 60
episodes each:

| rung | success | arms itself | kills/ep | meets a hostile |
|------|--------:|------------:|---------:|----------------:|
| armed | 73% | 68% | 1.28 | 95% |
| hunted | 73% | 60% | 1.48 | 100% |
| warden | 70% | 67% | 2.83 | 97% |

Still below bar: the `senior-short` and `senior-trail` rungs, which have
never been trained - they are the sequential-goal rungs and the ladder
reaches combat without them. They improved anyway while everything else was
being consolidated (senior-short 15x15 22% -> 45%, senior-trail 15x15 2% ->
18%), which is the clearest single piece of evidence that these rooms share
more than the one-new-thing framing suggests.

`senior 19x19` is worth calling out twice. This file used to record it as
the frontier - "plateaus around 38% over 900 episodes" - and it plays at
62%. What changed was not the rung. It was a crash (below) and then
consolidation.

### Climbing, then holding

Two different jobs, two commands:

    python train_curriculum.py     # climb: one rung at a time, promote, move on
    python consolidate.py          # hold: every rung interleaved, all scored

Climbing ratchets. Each pass gains the rung it is on and gives back a little
of the rungs below, because the thing being *scored* is always one room and
the rest are only defended by rehearsal. After enough passes `passed` is a
history rather than a description - six of nineteen rungs were below bar
when this was noticed.

Consolidating fixed all six and improved every rung in the ladder, several
by twenty points. The expected cost of interleaving - that no single rung
goes as high as training it alone - did not appear.

    python diag_ladder.py          # what it actually plays, with error bars

### Two bugs worth knowing about

**The network grows mid-episode, and that used to kill the run.**
`learner.maybe_update()` can widen the GRU (`RLPolicy.expand`), while
`run_episode` carries a hidden state made at the old width:

    RuntimeError: hidden0 has inconsistent hidden_size: got 256, expected 512

It had been doing this invisibly for a long time, because training was run
with stderr piped through a grep that only matched progress lines - the rung
ended with no PASSED and no "gave up", and the next one started as though
nothing had happened. `armed` looked like a rung that trains to a flat
8-17%; it was a rung that crashed every time.

**Rehearsal was uniform, which is the same as absent.** 25% of episodes
spread evenly over thirteen rungs rehearses each in about 2% of episodes.
It is weighted toward the nearest rungs now, at 0.40.

### Diagnostics

These measure the rungs, not the agent, and each exists because something
looked fine and was not:

| tool | question |
|------|----------|
| `diag_combat.py` | is the combat tier a fight, and a winnable one? |
| `diag_clock.py` | how much does the clock decide the rung? |
| `diag_stumble.py` | can a random walker stumble into the goal? |
| `diag_senior.py` | did training the top un-teach the bearing? |

`diag_combat.py` gates on four numbers a rung can fail while still running:
how often a hostile is actually met, what a competent agent scores, what a
random walker scores, and how often the competent one dies. It is what caught
the first combat tier clearing 100% of its rooms while picking up a sword in
under a third of them.

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
