# W3b flow: biomes that spread at their edges

Status: **draft, waiting on Chalce's answers.** Last part of W3.

## What's there today

- Each island is **one** biome (grassland, dark forest, volcanic, mushroom,
  tundra, boss). Islands are 100+ tiles of sea apart, joined by single bridges.
- So two biomes never touch: there is no edge *between* biomes on land.
- Inside an island, the land is fixed for the whole run: hazard tiles (lava,
  acid, ice), spore zones, walls, ponds, groves (W3) never move.

So "biomes spread at their edges" can only mean something **inside** an island.

## Proposal: the island's own features grow and shrink, slowly

```
every SPREAD_EVERY ticks (e.g. 500, half a "day"):
  pick a few edge tiles of a feature (grove, spore zone, hazard, pond)
  roll: grow onto a neighbouring open tile, or recede from an edge tile
  never: on/next to the agent's view (same rule as regrowth, no pop-in)
         onto the spawn, a bridge, a house, or a tile with an item/creature
  each feature stays inside a min..max size (no island eaten by lava)
```

What that gives the brain:
- **Groves spread** → new apple trees appear (out of sight), old ones can die.
  Remembered food spots slowly go stale: memory helps, but must be refreshed.
- **Hazards breathe** (lava/acid/ice/spore edges move a tile or two): a path
  that was safe yesterday may not be today. It has to *look*, not autopilot.

## Proof

- Map test over a long run: features stay in their size limits, nothing grows
  inside the view, spawn/bridges/houses untouched, island never fills up.
- Speed check (W1-style); survival check with the kept brain.
- `W3_SPREAD = False` turns it all off.

## Note on the sea

This does **not** touch the sea or the shark. The brain going to sea is a
separate question (likely: scarce food → wanders farther → reaches the coast,
and it was never trained on the new visible shark). Training is what fixes
that; a short replay check can confirm the cause first.

## Questions for Chalce

1. Is "inside the island, features grow and shrink" what you meant by biome
   spread, or did you picture something else (e.g. islands changing type)?
2. Which features move? Groves (trees/food), hazards (lava/acid/ice),
   spore zones, ponds. All, or only some?
3. Speed: one small change per island every ~500 ticks, or slower?
4. Can a hazard ever cover a remembered food spot, or do they keep apart?
