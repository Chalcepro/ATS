# ATS Update — 2026-10-09

## Summary
The ocean is now survivable-but-brutal instead of a sudden teleport/one-shot,
so the agent can learn a boundary from it. Behind a flag.

## Changes
### Sea mechanic — ats_env.py, config_rl.py
- New `SEA_SURVIVABLE` (default True) + `SEA_DAMAGE_PER_TICK` (25).
- Past the grace ticks (`OCEAN_SHARK_TICKS`), a shark now deals heavy damage
  EVERY tick (cause "a shark"), but it's escapable: retreat to land and the
  damage stops (ocean_ticks resets), so the agent has a gradient to learn
  "water = edge, turn back" instead of an abrupt teleport it can't learn from.
- Death at HP<=0 is handled in the same tick by the normal death path.
- Old behaviour (teleport to spawn / drown) kept intact under `SEA_SURVIVABLE=False`.

## Verified (headless)
- In ocean: -25 HP/tick, cause "a shark", NOT teleported, survives one hit.
- Stays in ocean -> dies, death cause "a shark".
- On land: no sea damage, ocean_ticks resets (retreat works).
- Flag off: old teleport-to-spawn behaviour still fires.

## Note
Next: densify the map (trees/huts/stands/Z-axis/chests/interact), sharpen the
experience-memory fingerprint, and decouple a growable vocabulary/memory "cell".
