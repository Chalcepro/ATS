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

## Systems to borrow, one at a time, into the fast terminal world
From general knowledge of the game. **Not yet checked against a wiki** (the
fandom wiki refused the fetch). Verify before using any as a spec.
- Terrain types / biomes with different danger and food (ATS has land, sea, bridges).
- Creatures: prey, predators, neutral; ATS has sharks.
- Units with needs and life stages (hunger, age, birth, death).
- Traits that change behaviour (brave, fast, weak...), as per-agent variation.
- Groups: villages / kingdoms that form, trade and fight.
  → the charter's "help who you can" and "judge threat by actions toward innocents"
  need other people in the world first.
- Disasters and weather as world events.

Order to add them: decide with Chalce, after Phase 2 (hearing). Each one gets its
user flow first (see the hex BACKLOG rule; the same process applies here).

## The "visit" world (later, Phase 3)
Once the brain is trained, a WorldBox mod feeds it what's around one unit;
Chalce walks in first person and watches how it behaves among people. An exam
room and a place to meet it, not a school.
