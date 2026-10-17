# ATS world: the one ordered list

Started 2026-10-12, same rules as Hex's backlog:
- **Top to bottom.** A new idea gets a place by what it depends on; it isn't
  started just because it's newest.
- **No code before the flow is agreed** with Chalce (what happens, in order).
- **Every step is measured on training**: speed per tick and RAM, before and
  after. A richer world that trains 10× slower is a step back on this laptop.
- **"Done"** = in the world, visible in the replay viewer, and a training run
  shows it doesn't break survival.

Research behind this: `docs/research/ats-world-audit.md` (what ATS has, what
the model senses) and `docs/research/worldbox-notes.md` (how WorldBox is built).

---

## W0: Study and audit (DONE 2026-10-12)
- [x] WorldBox's design from mod source code: assets, traits, tags,
      job → task → step behaviour, living biomes, civilisation layer.
- [x] ATS audit: lots of world content already; the model senses little of it;
      creatures only act toward the agent.

## W1: Beings as data (DONE 2026-10-13)
Creatures get **stats + flags + traits** in `assets/entities.json` (WorldBox's
asset idea), instead of special cases in code. A new creature = new data.
Proof it's safe: same seed, same replay before and after.
Flow: `docs/W1-FLOW.md`. Proof: `tests/test_w1_fingerprint.py` matched every tick of
5 worlds × 29 creatures. Speed unchanged (~94 → ~96 ticks/s, within noise).
Bosses fixed to Ancient Golem + Lava Golem.

## W2: Beings live their own lives (DONE 2026-10-13)
A small **job → task → step** system (WorldBox's behaviour idea). Every
being eats, rests, wanders, flees and (if a predator) hunts **whatever is prey
by its tags**, not just the agent. Relationships by **tags** (predator / prey /
herd / guard). Only creatures on built islands tick, as today, so cost stays
bounded. Measure ticks per second.
Flow: `docs/W2-FLOW.md`. **Built 2026-10-13**: `tests/test_w2_life.py` 8/8; every job
happens in real play; speed unchanged (starting island and all-islands).
Replay viewer shows each creature's job (and now draws creatures in the right
place). Kept brain, no learning, 8 fixed worlds × 3000 ticks: W1 median 810 /
mean 947 ticks survived, W2 median 988 / mean 1067. No collapse. 15 of 16
deaths in both were **sharks**: the brain's real weakness, not the creatures.

## W2b: A shark you can see coming
Chalce's idea (2026-10-13): today the sea just hurts, with nothing visible
attacking, so the brain can't learn what kills it (15 of 16 deaths). The shark
surfaces in view, waits, then charges in a straight line; dodge and it waits
again. New SHARK tile + a real brain_id.
Flow: `docs/W2b-SHARK-FLOW.md`. **Built 2026-10-13.** Kept brain, 8 worlds:
W2 median 988 / mean 1067 ticks → W2b median 1338 / mean 1348, before any
training on the shark.

## W3: Living land (DONE 2026-10-13)
Kept brain, 8 worlds: W2b median 1338 / mean 1348 → W3 median 906 / mean 1008.
The expected dip (food is ~6x scarcer), no collapse. Still mostly sharks, not
hunger: searching for food likely sends it to sea more. Training should fix both.
Food grows in patches where it belongs (wild by trees and bushes, man-made by
huts), far apart, and regrows out of the agent's sight.
**W3b (built 2026-10-13, `W3b-BIOME-FLOW.md`):** every 500 ticks, out of
sight, groves spread/die back, lava and acid move without growing (burning
common items and creatures, not keys), the volcanic island grows land (3-6
tiles out) and the tundra an ice shelf (10-13+, up to 20). `W3_SPREAD`.
Same day: close islands + bridges back ON (`WORLD_BRIDGES`), world build
10 s -> 3 s (numpy coastlines, identical islands).

## W4: Senses, in one batch (one brain migration)
Design with Chalce, then add together so the brain migrates once:
- **several** nearby beings at once (type, direction, distance, health, and
  **what they're doing**: eating, fleeing, attacking whom)
- **hearing** (ROADMAP Phase 2: H1+): sounds from beings and events, with direction
- anything W1–W3 made worth sensing

## W5: People
Villagers with needs, homes and families, then groups. This is where the
charter becomes trainable: there are innocents, and actions between others
to judge.

## W6: Ages, events, disasters
World-wide changes over long runs (WorldBox's ages): seasons, droughts,
storms.

## Later ideas (Chalce, 2026-10-13)
- **Real volcanoes** (needs the fake Z axis): an eruption sends lava downhill
  from the crater, outward, as far as the volcano's size allows; a small
  volcano on big land never reaches the sea.
- **Poison biome**: poison flowing over food makes it poisonous (eating it
  poisons you). Acid and lava just destroy food.
- **Keys sit close to what they open**, outside it (not on, under or in):
  house, cave, doorway, gate. Keys never burn, but are not labelled
  "legendary" to the brain; legendary keys can come later as their own item.

## Parked
- **WorldBox "visit" mod** (ROADMAP Phase 3): the trained brain among people,
  Chalce walking in first person. For meeting it, not training it.

---

## Decisions (Chalce, 2026-10-12)
- **All of W1–W6, incrementally**, in order.
- **Senses come with the world, not after:** hearing **and** wider sight go into
  the W4 batch (one brain migration). Sight size is Claude's call: **15×15
  patch (radius 7, up from 7×7)** and a **21×21 remembered map (radius 10, up
  from 11×11)**. Measure ticks/sec and RAM at W4 and shrink if training slows badly.
- **The agent is neither "always hunted" nor "just prey".** Each creature has its
  own reason: some hunt it, some only fight back when attacked, some
  **defend their space** and charge if it comes too close but leave it alone
  at a distance (like Valheim's boars). Built as per-creature tags/traits in W1–W2.
- **Keep the trained brain** and migrate it. The point is that it copes with a
  changing world instead of collapsing back to zero.

## Next: train, then W4
W2, W2b, W3 and W3b are built. Chalce trains on them (Train.bat), then the W4
senses batch is designed with him (the tile vocab is full, 16/16).
