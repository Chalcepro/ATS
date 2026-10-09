# ATS Update — 2026-10-09 (b)

## Summary
Bigger islands so the agent has room to live without the coast a few steps away,
plus a log-only trend harness to watch a run's numbers without eyeballing logs.

## Changes
### Bigger land — island_gen.py, config_rl.py
- `BIG_LAND` (default True) + `LAND_SCALE` (2.0) scale the island radius
  (30-60 -> 60-120 tiles). Verified: coast ~2.2x farther from spawn, land ~4x.
- Islands sit ~250 tiles apart with bridges off, so no overlap. STATE_SIZE is
  unchanged (502) because state is agent-centric, so the SAME brain keeps
  loading - no forced fresh run. False = old size.
- Rationale: at episode ~544 the agent still drowned 99% of the time and reward
  plateaued in the mid-20s - the sea was simply too close on small islands.

### Trend harness — trend.py, Trend.bat
- `python trend.py` summarises the newest train log: episode #, reward by
  quarter, peak, avg life, survivals, and death-cause shares (watch
  shark/drowning fall). `--watch [secs]` loops; `--logs N` folds in the last N
  run logs. Double-click Trend.bat to watch live. Reads logs only - no torch,
  cannot slow training.

## Verified
- Land scaling 2.2x coast distance / 4x area; STATE_SIZE still 502.
- trend.py parsed the live run: "episode 544/700 ... reward -4.2 -> 23.3 ...
  deaths shark/drowning 99%".

## Takes effect
Bigger land applies on the NEXT Train.bat start (restart to pick it up). The
current brain loads straight into the bigger world.
