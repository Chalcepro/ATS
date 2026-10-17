# ATS Update — 2026-10-16
## Summary
W4, the Loop: dying no longer ends the episode. The agent lands in a stone
dungeon room it can't walk out of until the room's goal is met. Three rooms
later, it goes back into the world beside the nearest real version of the
last room. Design and decisions: [docs/W4-LOOP-FLOW.md](../docs/W4-LOOP-FLOW.md).

## Why
After `SEA_TIRES` (ed52ccd), 3 runs (~1,230 world episodes) lived ~550 ticks
instead of ~1,800, and reward fell from ~72 to ~50. It still swam out to sea
and now drowned there. 60% of deaths read "unattributed"; they were these
drownings (fixed in daylog). Chalce wants no land sense. The skills are to be
taught in rooms instead.

## Changes
### The Loop — `loop.py` (new), `ats_env.py`
- `LoopWorld`: a 15x15 room whose tiles repeat in every direction, so a
  doorway leads back into the same room until the doors open.
- Rooms: wolf (equip the sword), shore (out of the sea and stay out), hungry,
  lever (interact). Each is data: layout, kit, body, goal, real-world finder.
- Env: death → Loop (-20, plus the old 35% forfeit for a world death), 20
  ticks of bearings after every arrival, room change after 5 failures, half
  the rooms chosen by the cause of death, back to the world after 3 rooms.
### Held weapon — `agent.py`
- Damage comes from the selected slot only (`HELD_WEAPON_ONLY`). A sword
  anywhere else in the bag does nothing.
### Senses (no size change) — `agent.py`
- State index 6 = in the Loop, index 9 = held weapon damage / 20. Both slots
  were always 0 (`get_adjacent_z` is a stub).
- Route channel: the way to the lever, or to a doorway once the doors are open.
### World/items — `world.py`, `assets/items.json`
- `TILE_STONE` = 4 (was the never-drawn `TILE_NPC`). Item `0088` Lever, type `switch`.
### Logging — `main.py`, `daylog.py`
- "Survived" = reached the cap without dying. A `loop:` line per episode.
- "Died of drowning." is now named, not "unattributed".

## Verified
- [x] `test_loop.py` 17/17
- [x] State vector length unchanged (STATE_SIZE)
- [ ] Training run with the Loop: measure life, deaths per episode, rooms beaten

## 2026-10-17 — first run, and a per-room tally
Run `train_20261016_164640` (~280 world episodes, 13 h): flat from start to end.
First world death is a shark ~85% of the time; 4-5 Loop deaths and ~2.6 rooms
beaten per episode (3 needed to get back); ~8.5 time-outs per episode, i.e.
~5,100 of 8,000 ticks in unsolved rooms. Ends most often in lever rooms (in sight
as often as hidden), then lava; burning (338) ties sharks (337) as top cause.
- `ats_env.py`: `room_stats` counts how each try ended per room kind (lever split
  hidden / in sight): won, out of time, died, sent back.
- `main.py`: prints it per episode: `rooms won/out of time/died/sent back: ...`.
- No training change. Next: read the tally, then fix the rooms it can't solve.
