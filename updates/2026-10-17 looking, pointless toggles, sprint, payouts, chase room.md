# ATS Update — 2026-10-17
## Summary
Interact became a way of looking, toggles that undo themselves cost a wait,
sprinting is actually faster, farmable payouts stopped, and the Loop got a
lever-at-hand room and a chase room. Detail: docs/W4-LOOP-FLOW.md, last section.

## Why
Probe of the trained brain, 10 tries per room kind: shore 9/10, food 5/10,
lava 1/10, fight 0/10, lever 0/10. ~40% of its actions were free toggles,
it almost never pressed interact, and sprint drained stamina for no speed.

## Changes
### Looking — `agent.py`, `rewards.py`, `config_rl.py`
- `INSPECT`: interact always available; `Agent._inspect` looks at the best
  thing in reach. State 7 = kind code / 12, state 8 = freshness (were the
  z-left/right stubs, always 0; STATE_SIZE unchanged).
- `RewardEngine.on_inspect`: +0.3 the first look at each kind per episode.
  The same thing again straight away: `look_again`, a wait.
### Toggles — `agent.py`
- `POINTLESS_TOGGLE_COSTS`: sprint / inventory / crafting flipped straight
  back with nothing done between: `pointless_toggle`, a wait.
### Sprint — `agent.py`
- `SPRINT_DOUBLE_STEP`: movement split into `_stride`; a sprint step on dry
  footing takes a second stride (not into water, no wall-hit charge, no
  second stamina cost).
### Payouts — `rewards.py`
- `FARMABLE_PAYOUTS = False`: picking up or harvesting a kind it already
  has pays nothing. Discovery and eating unchanged.
### Loop — `loop.py`, `ats_env.py`
- Lever rooms: at hand / in sight / hidden, a third each.
- `chase_room`: one hunter, 6x hp, grudge forever, extra stride every other
  tick (`LoopWorld.tick_life`). Goal `_got_away`: 8 away round the loop.
  Room `stamina` 40-100. `pick_room`: a named killer -> fight or chase.
- `LoopWorld.near_image` / `loop_dist`; the env feeds creatures the
  agent's nearest copy, so they come the short way round the loop.

## Verified
- [x] `test_look_and_stride.py` 8/8 (new)
- [x] `test_loop.py` 22/22; shark, sea tires, combat, shark reach, W2 life, bridges suites pass
- [x] 18,000 random-action steps in the Loop, no errors
- [x] State vector length unchanged (502)
- [ ] Training run: room tally per kind, interact use, toggle share
