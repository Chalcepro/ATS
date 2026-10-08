# ATS Update — 2026-10-08

## Summary
Reverted the default world to the pre-bridge layout: the bridge/shark world had
driven survival to zero. Fixed a hardcoded benchmark output path.

## Changes
### World shape — config_rl.py
- `WORLD_BRIDGES` switched `True -> False`. The bridge world put the agent on
  small islands ringed by sea, where a shark is an instant kill the moment it
  steps into ocean. Measured regression over 10-04..10-06: avg reward
  500 -> 29, lifespan ~3500 -> ~800 ticks, survivors in a day many -> **0**.
  The agent respawned and walked back into the sea ~1.8x per episode.
- The bridge + shark code is untouched behind the flag, to return later as a
  gradual curriculum rung with a shark the agent can sense and flee.

### Benchmark path — run_benchmark.py
- `output_json` was hardcoded to `C:\Users\Virus\...`, so the summary write
  crashed on every run. Now repo-relative (`<repo>/data/baseline_benchmark.json`).

## Verified
- `config_rl.WORLD_BRIDGES == False`; `ATSEnvironment().reset()` returns state
  len 502, no crash.
- 8-episode eval at 1500 ticks on the current (shark-damaged) brain: **all 8
  survived the full cap**, reward 8-106, zero shark deaths (was dying ~800).
