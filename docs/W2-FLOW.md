# W2 flow: beings live their own lives

Status: **agreed with Chalce 2026-10-13 and built.** Decisions at the end;
they override the first draft above where they differ.
Step W2 of `WORLD-BACKLOG.md`. W1 made creatures data; W2 gives them a life
that doesn't revolve around the agent.

## What I found reading the code (2026-10-13, after W1)

1. **Every creature only relates to the agent.** `Entity.tick_ai(agent_x,
   agent_y)` is the whole brain: measure distance to the agent, then wander or
   attack. A wolf next to a rabbit does nothing.
2. **Nothing gets hungry or tired.** No need drives any creature; "wander" is
   a random step every `move_every` ticks.
3. **No creature is ever born.** Creatures appear when an island is built and
   only ever decrease. Fine today (only the agent kills); once wolves eat
   rabbits, an island empties out in one run.
4. **There's no food tile for animals.** `TILE_OBJECT` is the agent's
   harvestables. Grazers would have to eat the agent's food or "graze" plain
   ground. (Real regrowing food is W3.)
5. **Small bug**: `random.shuffle(directions[:2])` shuffles a copy, so fleeing
   creatures are fully predictable. Fixed in W2 (behaviour changes anyway).

## The idea in one picture (WorldBox's job → task → step)

```
every tick, per creature on a built island:
  needs tick:    hunger += 1, tiredness += 1 (slowly)
  if no job, or a more urgent one appears → pick a job:
      danger near (a predator by tags, or whoever just hit it) → FLEE
      hungry & food/prey in sight                              → EAT / HUNT
      tired                                                    → REST
      otherwise                                                → WANDER (herds keep together)
  a job is a short list of tasks:  HUNT = [find prey, go to it, attack, eat]
  a task does one step per tick:   move one tile, or hit once, or eat once
```

Urgency order: **flee > defend space > eat/hunt > rest > wander**. A job is only
dropped for a *more* urgent one, so creatures don't twitch between choices.

## What a creature gains in the data

```json
"E011": {
  "name": "Wolf",
  "tags":  ["hostile", "beast", "predator", "pack"],
  "diet":  { "eats": ["prey"], "hunger_max": 400 },
  "sight": 5,
  "temper": { "kind": "defends_space", "space": 2, "hunts_agent_when": "hungry" }
}
"E015": {
  "name": "Deer",
  "tags":  ["passive", "beast", "prey", "herd"],
  "diet":  { "eats": ["grass"], "hunger_max": 600 },
  "sight": 4,
  "temper": { "kind": "flees_when_hit", "fears": ["predator"] }
}
```

- **Relationships are tags, never IDs.** "Wolf eats prey" means *anything*
  tagged `prey`. A new prey animal is hunted by every predator automatically.
- **The agent is a being too**, with tags (`human`, `meat`). Whether a creature
  hunts it is the creature's own rule (`hunts_agent_when`: `never` / `hungry`
  / `always`), so the agent is neither always-hunted nor just-prey: a fed
  wolf leaves it alone, a starving one comes for it. Boars keep `defends_space`.
- **sight**: how far a creature notices food and danger. Keeps the search small.

## Who becomes what (first pass, yours to change)

| Group | Creatures | Life |
|---|---|---|
| Grazers (prey, herd) | Deer, Sheep, Rabbit | graze, keep near their herd, flee predators |
| Pack hunters | Wolf, Ice Wolf | hunt prey when hungry; hunt the agent only when hungry |
| Defend space | Zombie, Skeleton, Husk, Spider, Yeti, Golems, ... | today's behaviour + hunger, eat prey that wanders close |
| Guards | Guardian NPC, bridge guards | hold post, attack `hostile` beings that come near (the `attacks_hostiles` flag finally works) |
| Untouched | Void, Void Echo, Shark, bats, Villager | as today (W5 gives villagers lives) |

## Population (so islands don't empty)

Each island remembers how many of each creature it started with. When one is
below that, there's a small chance per tick that a new one is born **next to an
adult of its kind** (or at the island's spawn spots if none are left). No
unbounded growth, no extinction, cheap. Real breeding waits for W5.

## Cost (this laptop)

- Only creatures on built islands tick, as today.
- "What's near me" uses a per-island grid of who is on which tile, rebuilt once
  per tick, so a creature checks its `sight` square instead of every creature.
- **Measured** with the W1 bench (`ticks/s`, same seed): W1 baseline ~96
  ticks/s. **Target: no worse than -15%.** If it's worse, creatures far from
  the agent think every 4th tick instead of every tick.

## Proof it works

W2 *changes* behaviour, so the W1 fingerprint can't match. Instead:
1. **Small scene tests**: a hungry wolf next to a rabbit catches and eats it; a
   fed wolf ignores the agent; a deer runs from a wolf; a starving wolf comes
   for the agent; a guard attacks a zombie; a dead herd refills over time.
2. **A 5-world run** with counts: kills between creatures, births, how often
   each job runs. Every job must actually happen, or it's dead code.
3. **Speed** before/after as above.
4. **Training check** (the backlog's "done"): the trained brain plays a short
   run in the new world; survival time is compared with before. It's allowed to
   dip (the world got harder), not to collapse.

## Questions for Chalce

1. **Grass**: until W3 adds real regrowing food, grazers "eat" any open ground
   (no tile changes). OK, or should they compete with the agent for its
   harvestables now?
2. **Hungry wolves hunting the agent**: yes for both wolves, or start with
   only Ice Wolf (the tougher island) so the starting island stays gentle?
3. **Births**: the "refill to starting count" rule above, fine for now?
4. The table above: any creature you see differently?

## Chalce's answers (2026-10-13), as built

1. **Grazers eat any open ground.** No tile changes; W3 brings real food.
2. **Wolves never hunt the agent, hungry or not.** Only if the agent attacks
   one does it come after the agent (a *grudge*), and it gives up once the
   agent is far away (`grudge: 8` tiles). Every creature that defends its
   space got the same grudge. Starting point; one number per creature in the data.
3. **Refill, gradually**: births beside an adult of its kind, about one per
   100 ticks per kind that is short, so it reads as a life cycle, not an
   empty island suddenly full. Never above the starting count.
4. **Table as drafted.** New rule for later creatures: **small animals
   (chickens, rats, ...) are never a threat by default**: at most they fight
   back when attacked. Snakes and scorpions: decide when they're added.

## What else changed while building

- **Predators and prey lived on different islands**, so hunting would never
  happen. Added Deer to the Dark Forest pool and Rabbit to the Tundra pool
  (`world.py`, one line each). Both bats are now `prey`/`small` and run from
  predators; spiders, zombies, husks and the yeti eat prey that comes close.
- **Old wander always tried north first** (the direction list was never
  shuffled), so idle creatures drifted north. Now random.
- **Land creatures could walk into the sea** (`_passable` treated "no tile" as
  open). Now land creatures stay on land.
- **Dead creatures were never removed** from the world's list; births would
  have grown it forever. Cleared at each birth check.
- Not built yet, still under `planned`: pack hunting, guards attacking
  hostiles (the Guardian NPC isn't placed on any island), ranged attacks.
