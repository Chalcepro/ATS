# WorldBox: notes for ATS

Started 2026-10-12 from Chalce's idea: WorldBox (a god-sim with a first-person
mod) already has the world ATS needs. Decision: **use it as a design reference
now, and later as a "visit" world, not as the training school.** Roadmap
entries: Phase 2c and Phase 3 below.

## Why not train inside it
- RL needs millions of tries, far faster than real time, plus thousands of clean
  resets a day. WorldBox is a Unity game that draws every frame, one copy at a
  time, at roughly game speed. Hours of training would become months.
- Its people are scripted. Feeding "everything" to the model means a C# mod that
  turns the game's tile map and units into numbers, written against code we
  can't see.
- Modding goes through community loaders: **NeoModLoader (NML)**, which succeeds
  NCMS; there's a DLL in the game's StreamingAssets mods folder and the game's
  experimental mode has to be on ([NML on GitHub](https://github.com/dwwite/ModLoader),
  [Steam Workshop page](https://steamcommunity.com/sharedfiles/filedetails/changelog/3080294469?p=1)).
  Some Steam comments call it unsafe. Unverified; check the source before installing.

## How WorldBox is built (from real mod source code, 2026-10-12)

The fandom wiki blocks automated reading, so this comes from mods on GitHub
that plug into the game's own classes (the "source information" route):
[WorldBoxOpenMods/ModExample](https://github.com/WorldBoxOpenMods/ModExample),
[PantheonsGodMod](https://github.com/MasonScarbro/PantheonsGodMod),
[Cultiway-Reborn](https://github.com/inmny/Cultiway-Reborn), and
[Wikipedia](https://en.wikipedia.org/wiki/WorldBox) for the overview.

1. **Everything is a data asset in a library.** `AssetManager.actor_library`,
   `.traits`, `.kingdoms`, `.biome_library`, `.job_actor`, `.tasks_actor`. A
   creature is an `ActorAsset`: `base_stats` (lifespan, health, damage, speed,
   attack speed, mass, accuracy, range...) plus many yes/no flags (`flying`,
   `die_in_lava`, `damaged_by_ocean`, `use_items`, `can_talk_with`, `has_soul`,
   `force_land_creature`...). New creatures are made by **cloning** an existing
   one and changing a few values.
2. **Traits modify stats and add behaviour.** An `ActorTrait` has `base_stats`
   changes (e.g. +100 health), a `birth` chance (how often a newborn has it), a
   group, and an optional special action that runs during play.
3. **Relationships are tags.** A `KingdomAsset` has tags and *friendly tags*
   ("friendly with dragons, undead"). Who fights whom comes from tags, not
   hard-coded pairs. Wild creatures belong to "wild kingdoms" too.
4. **Behaviour = job → tasks → behaviours.** An `ActorJob` is an ordered list
   of tasks. A `BehaviourTaskActor` is a chain of small reusable steps
   (`BehGoToTileTarget`, custom ones like "flee if in danger"), each returning a
   result (continue / stop / restart). That's how thousands of units live
   their own lives cheaply.
5. **Biomes are alive.** A `BiomeAsset` has high/low top tiles, a spawn list,
   `grow_vegetation_auto`, `spread_biome` (biomes push into neighbours), and a
   map-generation weight.
6. **Civilisation layer** (overview): humans, orcs, elves, dwarves found
   kingdoms → cities; alliances, wars, rebellions, clans, cultures, religions,
   languages, families; **ages** that change biomes and creatures.

## What each means for ATS

- (1)+(2) → ATS's `assets/entities.json` is already halfway there. Give
  creatures **stats + flags + traits** instead of special-cased code, so a new
  creature is data, not a code change.
- (3) → replace "hostile per type" with **tags**: predator/prey, herd,
  group membership. Threat then comes from what a being *does*, which is
  exactly what the charter asks the model to judge.
- (4) → this is the big one. ATS creatures act only toward the agent. A
  small **job/task/step** system lets every being live its own life (eat,
  rest, wander home, herd, hunt, flee) and costs little per tick.
- (5) → regrowing food and slowly spreading biomes make the world change
  over a run, so memory and planning pay off.
- (6) → people, families and groups come **last**, after beings can live
  on their own. Without them, "help who you can" has no one to help.

## The "visit" world (later, Phase 3)
Once the brain is trained, a WorldBox mod feeds it what's around one unit;
Chalce walks in first person and watches how it behaves among people. An exam
room and a place to meet it, not a school.
