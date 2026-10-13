# W3 flow: living land (food that grows where it should)

Status: **flow written 2026-10-13, waiting for Chalce's OK. No code yet.**

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
