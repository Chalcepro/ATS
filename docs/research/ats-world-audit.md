# ATS world: honest audit (2026-10-12)

Phase 0 of `docs/WORLD-BACKLOG.md`. Read from the code on `rl-core-fixes` at
`0e6860d`, not from older docs.

## What the world has (more than expected)

- **Islands**: 6, generated from noise (`island_gen.py`), with internal
  water, hazards (lava / acid / ice), safe paths, spore zones, walls, and
  **bridges** with a guard on each (`world.py`). Far islands are only built
  when the agent first steps onto a bridge, which keeps training fast.
- **Biomes**: 7 (forest, desert, volcanic, plains, cave, tundra, swamp) with
  elevation (`biome.py`).
- **Items**: 35 (food, tools, weapons, materials, curiosities), **crafting**
  recipes (`crafting.py`), chests and keys.
- **Creatures**: about 20 types from `assets/entities.json` (slime, zombie,
  sheep, wolf, villager, sharks...) with HP, damage, drops, wander/flee, and
  attack telegraphs.
- **Agent body**: HP, hunger, stamina, sprint, jump, leg injury, status
  effects, a torch that widens sight at night.
- **Day/night** (`day_night.py`), a small event system (`events.py`), and a
  survival ladder (`survival.py`).

## What the model actually senses (the real bottleneck)

From `config_rl.py` (state vector, 495 numbers at 25 actions):
- its own vitals, position, inventory (7 slots), status effects, environment
- the **4 tiles** next to it, plus a **7×7 patch** of tile types around it
- the **nearest one** creature (type + direction) and the **nearest one** object
- an **11×11 remembered map** (what was seen, where it walked, a route hint
  that fades out in later stages)

So it can't see **two** creatures at once, can't tell a sheep being chased by a
wolf from a sheep standing still, and knows nothing about what other beings
are doing. Chalce's goal ("it sees everything, feels everything, like a baby
among people") is blocked by **senses**, not by world content.

Every new sense changes the brain's input size, and the trained brain then has
to be **migrated** (`RLPolicy.expand_state`, `migrate_scheme.py`). So senses must
be planned in batches, not added one at a time.

## What the world is missing, compared with WorldBox

1. **Nobody lives their own life.** Every creature's AI is about the agent:
   `tick_ai(agent_x, agent_y)` returns damage dealt *to the agent*. Creatures
   don't eat, sleep, age, breed, or interact with each other. The world is a
   set of obstacles around the player, not a place.
2. **No people with needs.** "Villager" exists as a creature type, but with no
   needs, home, family or group.
3. **No relationships.** Hostile or not is fixed per type. Nothing like
   WorldBox's tags ("friendly with dragons") or kingdoms.
4. **No consequences that outlast a moment.** Biomes don't spread, food doesn't
   regrow on a cycle, nothing is remembered by the world.

These matter for the **charter**, not just for richness: "help who you can"
and "judge threat by actions toward innocents" can't be learned in a world
with no innocents and no actions between others.

## Keep / change

**Keep:** the islands, bridges, biomes, items, crafting, the agent body, the
survival ladder, and the lazy island building (it's what keeps training fast).
**Change:** creature AI (from "about the agent" to "living its own life"), and
the senses (a planned batch, with one migration).
