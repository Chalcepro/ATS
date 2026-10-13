# W3b flow: biomes that spread at their edges

Status: **agreed with Chalce 2026-10-13 and built.** As built at the end.
Last part of W3.

## What's there today

- Each island is **one** biome (grassland, dark forest, volcanic, mushroom,
  tundra, boss). Islands are ~250 tiles of sea apart (bridges are off, see below).
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

## Chalce's answers (2026-10-13)

- **Within one episode only.** The world is rebuilt each episode
  (`ats_env.reset` makes a new World), so every change starts fresh. Already true.
- **Every 500 ticks**, one small change per island.
- **Lava / acid**: the hazard area does **not** grow. Its inside shifts:
  safe and unsafe spots move over time.
- **Volcanic island**: the island itself grows outward, sea turning to land, up
  to a max size.
- **Ice (tundra)**: ice grows out over the sea as a walkable shelf, limited by
  island size: a small island gets only a thin ring, a big one up to 10-20 tiles.
- **Hazard over things**: lava or acid moving onto a tile destroys any common
  item (apple, stick, sword...) and any creature, alive or dead. **Legendary**
  items (golden apple, legendary sword: none exist yet) survive.
  Acid can instead make food poisonous.
- **Ice destroys nothing.** No frozen food or frozen items.
- **Groves** (forest/grass) spread and shrink but never destroy anything.

Checked: the brain does see the sea as its own tile (8 in the 7x7 patch,
nothing else uses 8). The close-island world (8-tile gaps, bridges) exists but
is OFF since 2026-10-08 (`WORLD_BRIDGES = False`) because the invisible shark
killed it; W2b's visible shark was the condition for bringing it back.

## To confirm

1. Volcanic: the island grows into the sea (new ash/rock land) and the lava
   inside only shifts, it doesn't grow?
2. Acid on food: destroy it now, and a "poisoned apple" item later (eating it
   hurts)? Destroying is simpler and needs no new item.
3. Keys / gate items count as legendary (survive), so a run can never become
   unwinnable?
4. Turn the close islands + bridges back on before training, or keep the far
   islands for this round?

Chalce's confirmations: (1) yes, the volcanic island just grows land, lava only
moves inside; (2) acid and lava destroy food, poison-on-food is a later poison
biome; (3) keys survive but are not called legendary to the brain; (4) close
islands and bridges back on before training.

## As built

- `world._tick_spread`, from `tick_life` every `SPREAD_EVERY` (500) ticks, on
  every built island, `SPREAD_STEP` (4) tiles per change, nothing within the
  agent's view (PATCH_RADIUS).
- Groves: trees spread within 2 tiles of a tree, or one dies, staying within
  75-125% of the island's trees at the first change.
- Lava (volcanic) and acid (mushroom): one hazard tile moves one step, onto
  open ground or an item, never onto the ash safe path or the island middle.
  `world.cover()` destroys common items (and their regrowth) and kills
  creatures there; items marked `fireproof` (Key) or `gate_key` survive and
  the lava flows round them.
- Volcanic land (open ground) and tundra ice shelf grow into the sea from
  the coast, reach sized by the island's starting area (volcanic 3-8, ice
  3-20; measured 3-6 and 10-13), always keeping 3 tiles of sea to other
  islands and bridges.
- `W3_SPREAD = False` turns it off. Tests: `test_w3b_spread.py` 8/8; all other
  world tests still pass. Cost: ~0.04 s per change (every 500 ticks).
