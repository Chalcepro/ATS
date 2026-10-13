# W1 flow: beings as data

Status: **flow written 2026-10-13, waiting for Chalce's OK. No code yet.**
Step W1 of `WORLD-BACKLOG.md`. The rule for this step is **no behaviour change**:
creatures act the same, the brain sees the same numbers, and the replay is the
same tick for tick.

## What I found reading the code (2026-10-13)

1. **The data mostly exists already.** `assets/entities.json` has 29 creatures
   with HP, damage, telegraph, drop, island. W1 reshapes it; it doesn't start it.
2. **16 of the creature flags do nothing.** Written in the JSON but nothing
   reads them: `ranged`, `range`, `pack`, `pack_size`, `wall_traverse`,
   `disturbs_sight`, `guards_ruins`, `can_trade`, `patrols`,
   `attacks_hostiles`, `night_only`, `flee_speed`, `drop_count`,
   `poison_on_hit`, `blind_on_hit`, `slow_on_hit`. So the Skeleton isn't ranged,
   wolves don't hunt in packs, the Toxic Toad doesn't poison. The data promises
   more than the world does.
3. **Creatures are still special-cased by ID in 5 places:**
   - `entities.py`: Void Echo (`E020`) hits with no warning; Magma Cube splits
     into `E027`s.
   - `ats_env.py`: the Void (`E014`) gets its own death message; killing the
     Slime (`E001`) sets a flag.
   - `agent.py`: "boss" = `E009`, `E023`, `E025` (Golem, Toxic Toad,
     Frost Wraith; the last two look like a mistake, see below).
   - `rewards.py`: kill rewards keyed by ID, **with wrong comments** (says
     E016 is a Pig, it's the Rabbit; E023 "Lava Golem" is the Toxic Toad).
   - `world.py`: which creatures live on which island, by ID list.
4. **The shark is invisible to the brain.** The brain gets a creature's type as
   the number in its ID (`E011` → 11). `E_SHARK` isn't a number, so it becomes 0,
   the same as "no creature". The brain can't tell "shark here" from "nothing
   here". (Not fixed in W1, since that changes what it senses. Goes into W4.)
5. **Today's hostile behaviour is already "defend your space"**: a hostile
   wanders, and only attacks when you're within 2 tiles. That's your Valheim
   boar with a space of 2. Nothing hunts the agent today except the shark.

## The new shape of one creature

```json
"E011": {
  "name": "Wolf",
  "brain_id": 11,
  "island": 1,
  "stats":  { "hp": 8, "damage": 4, "telegraph": 1, "move_every": 5 },
  "tags":   ["hostile", "beast", "pack"],
  "temper": { "kind": "defends_space", "space": 2 },
  "on_hit":   { },
  "on_death": { "drop": "0036", "spawn": [] },
  "reward":   { "kill": 2.5, "boss": false },
  "planned":  { "pack_size": 3 }
}
```

What each part is for:
- **brain_id**: the number the brain sees. Written down, so renaming an ID can
  never silently change what the brain sees. The shark gets `0` for now (today's
  truth); W4 gives it a real one.
- **stats**: numbers only. `move_every` is the "moves every 5 ticks" that is
  hard-coded today, now per creature (all 5 in W1).
- **tags**: what it *is*. In W1 nothing reads them except `hostile` /
  `passive` / `neutral` (today's `type`). W2 uses them for who-eats-whom.
- **temper**: why it acts toward the agent. W1 has exactly today's kinds:
  - `defends_space` (+ `space`): every hostile today, space 2
  - `flees_when_hit`: deer, rabbit, sheep
  - `ignores`: bats, villager, guardian
  - `hunts`: the shark
  - `lingering_death` (+ range, ticks): the Void
  - `strikes_instantly`: Void Echo (today's `E020` special case)
  - `explodes` (+ radius): Creeper

  W2 adds new kinds (`hunts` for some land creatures, `fights_back`) and
  changes who has which. That's where your boar/wolf/hunter mix gets made.
- **on_hit**: the status it gives you (`poisoned: 5`, `burning: 3`, ...).
  Replaces the separate `poison` / `burn` / `slow` / `blindness` flags.
- **on_death**: drop + what it splits into (Magma Cube → two `E027`).
- **reward**: kill reward and boss flag, moved here from `rewards.py`/`agent.py`.
- **planned**: the 16 flags that don't work yet go here, so the data stops
  pretending. Each one gets built in W2 or later, then moves out of `planned`.

Island pools (`world.py`) stay where they are for W1; a creature already says
its island, so W3 can read pools from data.

## Order of work

1. **Record the "before"**: a script runs fixed seeds (say 5 worlds × 2000
   ticks) with a fixed random action sequence, and writes a fingerprint of every
   tick (agent position/HP, every creature's position/HP/state, reward, the
   state vector). Commit the fingerprint file **before** touching anything.
2. **One loader** (`creature_spec.py`): reads the new JSON, checks every field
   (unknown key = error, so a typo can't silently do nothing), and gives the
   code one object per creature.
3. **Convert the JSON** to the new shape by script (not by hand), keeping every
   number.
4. **Swap each special case** for the data, one at a time, re-running the
   fingerprint after each: entities.py, then ats_env.py, agent.py, rewards.py.
5. **Prove it**: same seeds, fingerprint identical to step 1. Plus: training
   speed per tick and RAM before/after, and one short training run from the
   current brain to show survival isn't broken.
6. **Add a creature with data only**, as a test (not left in the world), to show
   a new creature needs zero code.

**Done** = fingerprint identical, no creature ID left in game logic, speed
unchanged, a short training run survives as before, pushed.

## Questions for Chalce

1. **Boss list.** Today "boss" = Golem, Toxic Toad and Frost Wraith. The Toad and
   Wraith look like leftovers from when the IDs meant different creatures (the
   comments call them Lava Golem and Cave Bat Queen). In W1 I keep it exactly as
   is (no behaviour change), and **fix it at the start of W2**: bosses = Ancient
   Golem + Lava Golem. OK?
2. **Rewards per kill** stay exactly as they are in W1 (same reason). Fine?
3. Anything about the shape above you'd name differently? (e.g. "temper" vs
   "attitude")
