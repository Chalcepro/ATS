# ATS Update — 2026-10-03
## Summary
Islands are close and joined by one guarded bridge per pair; sharks no longer walk on land; exploring the sea no longer pays.

## Why
Today's log, 1,024 full-world episodes: 855 died of "the Void" and 966 went
to sea. All 856 Void deaths followed a shark strike. The shark spawned at the
agent's ocean position, its movement allowed any non-wall tile, so it walked
ashore and chased the agent, killing on contact - logged as "the Void".
Meanwhile every first-visited ocean tile paid the exploration reward, so once
the island was explored the sea was the only thing left that paid. The other
islands sat ~250 tiles away behind a 3-tick shark: unreachable.

## Changes
### Layout and bridges — `island_gen.py`, `world.py`
- Island shapes use island-relative noise (`relative=True`) so an island can
  be slid into place without changing shape.
- Grassland in the middle; Dark Forest, Volcanic, Mushroom, Tundra around it
  in their old directions; Boss beyond the Tundra. Each is placed so every
  coast is >= `COAST_GAP` (8) tiles from every other coast - more ocean than
  the shark allows, so there is no lucky shortcut anywhere.
- `BRIDGES`: one per connected pair (0-1, 0-2, 0-3, 0-4, 4-5). A bridge is the
  ocean tiles on the line between the two middles, 4-connected (a diagonal
  step becomes two straight ones), one tile wide with sea on both sides.
  New tile `TILE_BRIDGE = 13` (fits the 16-row tile embedding).
- A guard (`Entity.guard`) stands mid-bridge: the first non-unkillable hostile
  of the destination island's pool. It never wanders; it fights within reach,
  and hostiles block movement, so it has to be dealt with.
- Far islands are materialised (tiles, resources, creatures) the first time
  the agent stands on a bridge to them, not at reset - every island's
  creatures ticking every step would slow training.
- `WORLD_BRIDGES = False` restores the old layout.

### Sharks — `entities.py`, `ats_env.py`
- `SHARKS_STAY_IN_WATER = True`: a shark moves only on `TILE_OCEAN`. At sea it
  is exactly as deadly as before. Bob: "sharks don't walk".
- Instant kills are named after the killer: "a shark", "the Void" (E014 only).

### Exploration — `ats_env.py`
- `EXPLORE_PAYS_OCEAN = False` (Bob's decision): a first visit to an ocean
  tile pays nothing. Bridge tiles are not ocean, so finding one still pays.

## Verified
- [x] State vector length still == 502 (`config_rl.STATE_SIZE`)
- [x] `test_bridges.py`: 5 seeds - 8.6+ tile coast gaps, 5 bridges each,
      4-connected, sea both sides, guard on each that blocks and holds its
      post, far islands built once on demand, sharks never leave the sea,
      old layout one switch away. World build 4.4 s vs 4.1 s before.
- [x] `test_bridges_env.py`: stepping on a bridge builds the far island,
      a bridge is not swimming, ocean tiles pay no exploration, 600 random
      steps run clean with no "Void" deaths on the start island.
- [ ] GUI renders bridges (`=BRDG=`) - not opened yet
- [ ] A real training run on the new world
