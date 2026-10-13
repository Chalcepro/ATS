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

## W1: Beings as data (no behaviour change)
Creatures get **stats + flags + traits** in `assets/entities.json` (WorldBox's
asset idea), instead of special cases in code. A new creature = new data.
Proof it's safe: same seed, same replay before and after.
Flow: `docs/W1-FLOW.md` (written 2026-10-13, waiting for Chalce's OK).

## W2: Beings live their own lives
A small **job → task → step** system (WorldBox's behaviour idea). Every
being eats, rests, wanders, flees and (if a predator) hunts **whatever is prey
by its tags**, not just the agent. Relationships by **tags** (predator / prey /
herd / guard). Only creatures on built islands tick, as today, so cost stays
bounded. Measure ticks per second.

## W3: Living land
Food regrows on a cycle; biomes spread slowly at their edges. The world
changes over a run, so memory and planning start paying off.

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

## Next session: start W1
Write W1's flow (entities.json stats/flags/traits/tags shape + the "same seed,
same replay" proof), get Chalce's OK, then build.
