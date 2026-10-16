# W4 flow: the Loop (rooms you can't walk out of)

Status: **draft, 2026-10-16. Not built.** Waiting on the questions at the end.

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

## Questions for Chalce

1. How many goals before the world drop: **one** (the wolf example), or a set number?
2. **Equipping**: damage from the held slot only, everywhere (in the world as
   well)?
3. Does the Loop **replace** the toy-grid rehearsal (40% of episodes), or sit
   alongside it?
4. Bearings: 3-4 seconds is how many ticks? (proposal: 20)
