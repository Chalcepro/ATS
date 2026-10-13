# W3 flow: living land (food that grows where it should)

Status: **agreed with Chalce 2026-10-13 and built.** Answers at the end.

## What's there today

- Every tile rolls its own dice (`_populate_tile_objects`, a per-tile hash):
  Grassland is ~4% apples, ~7% berry bushes, ~43% of tiles have *some* object.
  Each tile is independent, so food is spread evenly everywhere: walk in a
  straight line and you keep bumping into it.
- Nothing regrows. Eat it and it's gone for the run.
- Bread exists as an item but isn't tied to any place people live.

## Chalce's rules (2026-10-13)

1. **Wild food grows where nature is**: apples by trees, berries on bushes, in
   forests and grassland.
2. **Man-made food only near man-made places**: bread etc. near a house, hut or
   village, along with other man-made items.
3. **Food is uneven**: it grows in **bunches**, or **sparse**, or **not at all**
   in some places. Patches are far enough apart that you can't walk a straight
   line and collect apples 5×, berries 17×, or bread by chance. After you find
   an apple you have to *look* for the next one.
4. **No lucky pop-in**: food never appears just because the agent walked into
   a new area. Growing is decided by the world, not by where the agent is.
5. **Regrowth is hidden by the agent's view**: things grow back where the agent
   can't currently see, so it never watches food appear in front of it.

## How

```
world gen:
  a low-frequency noise field per food kind  → "fertile here?" (0..1)
  pick patch centres where it's fertile, at least D tiles apart
  each patch: a source (tree / bush / hut) + a few food tiles around it
  some patches are bunches (5-8), some sparse (1-2); most land has none
  man-made food: only within R tiles of a settlement (huts/village, placed first)

every N ticks:
  for each eaten food spot whose patch still has its source (tree, bush, hut):
     if NOT inside the agent's view and its timer is up → it grows back
```

- Food regrows **at its own patch** (the tree grows a new apple), never at a
  random spot. So knowing *where* apples are pays off, which is the point of
  W3 (memory and planning start to matter).
- Grazers in W2 graze open ground; W3 can later make them eat real plants too.

## Proof

- Map check: for each food kind, the gap between patches (median, min) and how
  many you get walking straight lines across 5 worlds (should be ~1, not 5+).
- Regrowth test: eat a bunch, stand looking at it → nothing; walk away, come
  back later → it's there.
- Speed check as W1/W2; survival check with the kept brain (expect a dip: food
  is now scarce and must be found).

## Questions for Chalce

1. Your last sentence: from the AI's view it's "a simple radius", and "from our
   own point of view it's going to be as if..." got cut off. Did you mean the
   replay should show us the fog (what the AI can't see) where things regrow?
2. **Villages/huts**: there are none yet (people are W5). For bread, put simple
   huts down now as places (no people), or keep man-made food out until W5?
3. **Regrowth speed**: an apple tree regrows one apple every ~500 ticks?
   Berries faster, bread slowest? (Numbers go in the data, easy to tune.)
4. Should this also change the non-food scatter (sticks, stones, logs) to
   patches, or only food for now?

## Chalce's answers (2026-10-13), as built

1. **The "heat box"**: a second thinning pass so a patch of ~10 ends up 1-2,
   never 0. Built at world creation (each item kept with `KEEP_CHANCE` 0.4,
   at least one per patch), not at the moment the agent's view box arrives.
   Same result, but done at arrival the agent's long-range "nearest object"
   sense would point at food that then vanishes as it walks up, a mirage.
   Easy to move to arrival time if wanted.
2. **Villages now**, empty houses (TILE_HOUSE, 2x2, impassable). No house, no
   village; a first house is rare; each house makes the next likelier
   (`min(0.85, 0.35 + 0.1 n)`, max 10). Bread, water flasks, bandages, books
   only around houses.
3. **Regrowth**: `"regrow"` ticks per item in items.json (apple 500, berry bush
   300, mushrooms 400, bread 1500, stick 1000, flower 800, log 1500). It grows
   back at its own spot, only when the agent can't see that spot.
   (1000 ticks ~ "a day" as a feel, not a day-night setting.)
4. **Scatter too**: sticks, stones, logs, flowers, bushes in patches. Trees
   use their own grove noise.

Measured (5 worlds, all islands, 40-tile straight walks): apples on one walk
max 8 → 1, berries 9 → 2, sticks 7 → 3. Apples in the world 2336 → 379.
`W3_PATCHES = False` brings back the old scatter. Tests: `test_w3_land.py` 8/8.
Biomes spreading at their edges: not built, stays in W3's list.
