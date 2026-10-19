# ATS Update — 2026-10-19
## Summary
A lever gets its own tile and no pick-up, and two new ladder rungs (`switches`, `switches7`) teach interact.

## Why
Probed the live brain in 45 Loop lever rooms (15 each: at hand, in sight, hidden): **0 won**.
It walked onto the lever and pressed PICK_UP hundreds of times until the room timed out.
It chose INTERACT about 1 tick in 500 in the world and 1 in 5,000 in the Loop. The action
was not stuck (tools/revive_actions.py found nothing to revive). It had just never been taught:
no rung on the ladder used ACT_INTERACT.

The deeper cause: in its patch the lever was TILE_OBJECT, the same tile as an apple, and the
mask offered PICK_UP next to it. The brain did what its senses said.

## Changes
### Perception: `world.py`, `agent.py`, `config_rl.py`
- `TILE_SWITCH = 16`; `World.tile_type_at` draws a switch-type object (item 0088, the Lever) as it.
  `TILE_VOCAB_SIZE` 16 -> 17.
- The action mask no longer offers PICK_UP for a lever.
- Both behind `config_rl.LEVERS_ARE_LEVERS` (False = old rendering and mask).
- The remembered map's tile scale is pinned at the old 16 (`MEM_TILE_NORM`), so growing the
  vocabulary does not rescale every remembered tile. TILE_SWITCH reads as 1.0.

### Loading an older brain: `brain.py`
- `grow_tables()`: a saved brain with fewer embedding rows than the network gets the new rows
  (same init as `expand_vocab`), with zeroed Adam moments for them. Before, it was refused and
  main.py fell through to a FRESH policy. Measured on the live brain: grows tile_embed 16->17,
  keeps its 4 rungs, and gives exactly the same outputs (difference 0.0 over 300 world ticks).

### The ladder: `curriculum.py`, `config_rl.py`, `debug_gui.py`
- `Stage(switch=True)`: a lever (TILE_SWITCH) stands between the agent and the goal. Each room
  wires it to one thing it can see, never a random roll: half cage the goal (TILE_GATE over it,
  seen but out of reach), half keep it dark (the goal is not there until the lamp comes on).
- ACT_INTERACT pulls a lever anywhere in the 3x3 around the agent, as in the world. +0.5 once
  (R_PULL). Pressing with no lever in reach costs R_GRAB, like reaching for bare floor.
- Until it is pulled, the lever is what the object slots and compass describe (id 88, as the
  world's nearest-object slot would read it); the route channel leads to it.
- `switches` 5x5: nursery rules (no walls, no costs), pass 85%. `switches7` 7x7 maze, pass 75%.
- Placed after `avoid` and before `hazards` (Chalce's rule: the brain had lost hazards, 15 of 200
  rooms on 10-19, so the lever does not wait on it). Added to ROUTE_HINT_STAGES and the GUI menu.
- The oracle pulls first, then walks to the goal; it clears both rungs 100/100.

## Verified
- [x] State vector length still 502
- [x] tests/test_switches.py (10), test_loop (22), test_combat, test_revive_actions, test_look_and_stride, test_w3_land
- [x] Trial training on a copy of the live brain (10:36 save), each rung alone, no rehearsal:
  `switches` passed in 454 episodes (2% -> 100% by ep 200); `switches7` in 1,600 (98% sampled
  throughout; greedy took longer).
- [x] Same Loop lever probe as before (15 rooms each, sampled play):

  | lever      | before | after switches | after switches7 |
  | :--------- | -----: | -------------: | --------------: |
  | at hand    |   0/15 |          11/15 |           14/15 |
  | in sight   |   0/15 |           4/15 |           11/15 |
  | hidden     |   0/15 |           3/15 |            7/15 |

  Interact chosen in the world: 0.2% of ticks before, 8% after (it is also how it looks at
  things). Beside a lever it now presses interact, not pick-up.
