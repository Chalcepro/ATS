# W4 flow: the Loop (rooms you can't walk out of)

Status: **built 2026-10-16** (`loop.py`, `ats_env.py`, `test_loop.py`), behind
`config_rl.LOOP`. See "As built" at the end; the sections in between are
the design history.

## Why

Since `SEA_TIRES` (ed52ccd) the agent still swims out to sea and drowns at about
550 ticks, where it used to starve at about 1,800. Chalce on 10-16: keep the
tiring sea, and **no land sense**. A person at sea has no land sense, only
tools to navigate with. The agent needs to learn the skills themselves, and
the curriculum rooms (nursery → warden) teach them in a toy grid that isn't
the real world.

## Chalce's rules (2026-10-16)

1. **Chunk rooms, like the Backrooms.** Rooms are hand-made chunks, each built
   for **one challenge** (a wolf with a sword in your hand, a river to cross,
   food you have to find while hungry ...). They are built from the real world
   (real tiles, items, creatures, inventory and combat), not the toy grid.
2. **Four doorways, and no way out.** You can walk as far as you like in any
   direction, but until the room's goal is met you keep walking into the
   same room again, with the same things on the floor. It's open, not a maze
   in the usual sense, and it loops.
3. **You start with tools.** A room can give you a kit (for example a wooden
   sword), so you have to learn that you have an inventory and use it.
4. **Goal met → cut straight to the world.** Teleport into a real seed, dropped
   next to **the nearest real version of what you just beat**. After the wolf
   room you land by the nearest wolf (or two, or three). If it survives, it
   carries on from there.
5. **Die → back in the Loop**, like Finn in the shrine (Adventure Time). It
   lands in a room with a **random** goal and has to beat it to get out again.
   Death in a room puts it back in that same room.
6. **A few seconds to get its bearings** (3-4 s) after every arrival, before
   anything can hurt it.
7. **Keep it going until we're happy with it.** Random goals, so it never
   settles into "walk to the shiny thing, get paid, next room".

## The worked example: the wolf room

```
arrive (Loop)  →  3-4 s to get its bearings (nothing moves, nothing hurts)
  kit: wooden sword in the inventory
  a wolf at 20-30% health, in front of it
  sword EQUIPPED  → one or two hits kill the wolf
  sword not equipped → it most likely loses
  walks out a doorway → comes back in through the opposite one, same wolf
wolf dies  →  cut  →  world (random seed), next to the nearest wolf
  survives → keeps living
  dies     →  Loop, random room
```

## How (proposed)

- **One room = one template**: a small tile chunk (about 15×15) with four
  doorways, plus a kit, the creatures and their health, the goal check, and a
  "where is this in the real world?" finder (nearest wolf, nearest river,
  nearest food patch).
- **Looping**: the room's tiles repeat in every direction, so whatever the
  agent can see past a doorway is the same room again. Walking out the east
  door puts you in at the west door. Creatures and items are the same objects,
  not copies: kill the wolf in one copy and it's dead in all of them.
- **One life, many deaths**: a world death no longer ends the episode. It
  sends the agent to the Loop, the death is still paid for (the death
  penalty), and the episode ends at its tick cap as it does now.
- **Bearings timer**: N ticks where the world is paused for the agent: no
  damage, nothing moves, and its actions do nothing.
- **First rooms**: wolf + sword, the shore (swim back to land before your
  stamina runs out), hungry with food somewhere in the loop, lava between you
  and the doorway. Each one is a template, so adding rooms is data, not code.

## Already there vs new

- Already there: the real World, Agent, combat, inventory, items, stamina,
  regrowth. The curriculum rehearsal swaps in toy-grid rooms on 40% of
  episodes.
- **Not there: equipping.** Damage today is the best weapon *anywhere in the
  inventory* (`agent._weapon_damage`). Rule 4 of the wolf room needs a held
  item: damage from the selected slot only. This changes the real world for
  the trained brain too, so it's a question below.
- New: the looping room, death sending it to the Loop instead of ending the
  episode, the drop next to the nearest matching thing, and the bearings timer.

## Proof

- `test_loop.py`: walk out every doorway and arrive back in the same room;
  goal met → in the world within the set distance of a wolf; die → in a room;
  no damage during the bearings ticks; sword in hand kills a 25% wolf in 1-2
  hits, bare hands mostly don't.
- Compare runs as before: life, reward, causes of death. Plus: rooms beaten
  per episode, and how many ticks each room takes.

## Answers so far (2026-10-16)

- Goals before the world drop: **a set number** (proposal: 3, in config).
- Equipping: **damage from the held slot only, everywhere.** The state has no
  "what am I holding" input; the height-behind input (index 9) is always 0
  (`get_adjacent_z` is a stub), so it becomes "held weapon damage".
- The Loop sits **alongside** the toy-grid rehearsal.
- Bearings: **20 ticks**.
- Chalce: a **shore room**: go into the sea → back to the start of the room;
  stay on land 3-4 s → done.

## My take (Claude, 10-16) - waiting on Chalce

Better than what we have: real-world rooms, the drop next to the real thing,
deaths that teach instead of ending the episode, and kits that force the
inventory. Four changes I'd make:

1. **Give-up timer**: a room it can't solve would trap it for the whole
   episode. After N ticks, a different room.
2. **Dying costs more than a room pays**, or dying becomes a shortcut to
   room rewards.
3. **Half the rooms match what killed it** (drowned → shore room), half random.
4. **A "you're in the Loop" input**, so it doesn't mix the Loop up with the world.

Note: a real wolf has only 8 HP and the sword does 5, so "25% health" would be
2 HP, and bare hands kill that too. Proposed instead: a full 8 HP wolf (2 sword
hits) and the agent arriving at 20 HP (bare hands take 8 hits, and it dies after ~5 bites).

## Questions for Chalce (first round)

1. How many goals before the world drop: **one** (the wolf example), or a set number?
2. **Equipping**: damage from the held slot only, everywhere (in the world as
   well)?
3. Does the Loop **replace** the toy-grid rehearsal (40% of episodes), or sit
   alongside it?
4. Bearings: 3-4 seconds is how many ticks? (proposal: 20)

## As built (2026-10-16, after Chalce's second round)

Chalce's second round: dungeon look (**stone floor, walls, a doorway in each
wall**); before the goal a doorway loops back into the room, **after the goal
the doors open** and a doorway leads out; a room whose way out is **an object
you interact with** (a lever, E); and the **route channel** from the primary
maze rooms showing the way to it. Stuck room: **change after 5 deaths** in it.
Death **-20**. Wolf at full health, agent arriving at 20 HP: agreed.

| Room | Arrives with | Goal | Back in the world by |
|---|---|---|---|
| wolf | 20 HP, apple held, sword in slot 1 | kill the 8 HP wolf, then a door | the nearest wolf |
| shore | in a pool of sea | 20 ticks on stone (back in the water = start over) | two tiles off the nearest beach |
| hungry | hunger 10 | eat (apples behind walls), then a door | the nearest apple |
| lever | - | pull the lever (that is the way out) | where it died |

- Floor is `TILE_STONE` = 4 (was `TILE_NPC`, never drawn, so the brain's row
  for it was unused; no migration). Lever = item `0088`, type `switch`.
- Senses reuse two state slots that were always 0 (`get_adjacent_z` is a
  stub): index 6 = in the Loop, index 9 = held weapon damage / 20. State size
  is unchanged and the brain loads as it is.
- Route channel: the way to the lever, or to the nearest doorway once the
  doors are open. Never the way to the room's own challenge. Zeros in the world.
- Death: -20 each (`LOOP_DEATH_PENALTY`); a world death also forfeits 35% of
  what was earned, as before. A room pays +10 (`LOOP_GOAL_REWARD`).
- Room after a world death: half the time the room about the cause
  (drowning/shark -> shore, wolf -> wolf, starvation -> hungry), else random.
- 3 rooms (`LOOP_GOALS_BEFORE_DROP`), then the world. Every arrival: 20
  ticks of bearings. Back in the world: full health and hunger, kit kept.
- An episode now always runs to its tick cap; "Survived" means it reached
  the cap **without dying**, so the survival ladder can't be cheated.
- Loop rooms don't pay exploration, and their creatures don't breed.

## Generated rooms (2026-10-16, second build)

Chalce: four fixed rooms were too little - "not just exactly what I said".
Rooms are now generated on every entry (`loop.KINDS`); a retry after dying is
the same room, a new entry is a new one. 5 kinds, ~50 variants before layout:

- **fight**: any of 15 ordinary hostiles (slime, zombie, skeleton, spiders,
  husk, fire imp, witch, wolves, yeti, toxic toad, spore fiend, frost wraith,
  magma cube), packs for wolves/slimes/spiders. Weapon = the weakest that
  kills one in <=3 hits, in a random slot 1-6. Slot 0 (in hand) is never it.
  Health = enough to win armed, not to brawl. Back in the world by the
  nearest of the same creature (its island is built if needed).
- **food**: apple, berry bush, raw meat or bread, 1-3 placed far from the start,
  plus 2-6 things that aren't food. Arrives hungry (5-25) and weak (25-40 HP).
- **lever**: in plain sight (no route), or in a walled corner nook with one
  gap (route channel shows the way in). Clutter around.
- **shore**: a pool of 5-9 x 5-9 placed anywhere, start in its middle.
- **lava**: a band of lava (1-2 wide) with 1-2 ash paths; a key or book across.
  (The loop is a second way round: out one door, in at the other side.)
- Layouts: 0-2 inner walls with gaps, 0-6 pillars. A room is rolled again
  if it's cramped or a doorway is walled off (checked over 2,000 rooms).
- After a death, half the time the room is about the cause: killed by a
  creature -> a fight with that creature; drowned/shark -> shore; starved ->
  food; lava/burning -> lava.

Note: an episode now always runs to the cap (8,000 ticks), so episodes take
longer in wall-clock time than when a death ended them at ~550.

## Room time limit (2026-10-16, Chalce: "add that")

The first Loop run (4 fixed rooms) got stuck: in the lever room nothing can
kill it, so 5 deaths to swap took a whole episode (starving is ~3,000 ticks).
Now `LOOP_ROOM_TIMEOUT` = 600 ticks: not solved in time = a failed try
(-5, `LOOP_TIMEOUT_PENALTY`, less than a death so it's no escape hatch), the
room starts again; 5 failed tries of any kind = a different room. A room it
has WON but can't find the door out of lets it go after another 600 ticks.
