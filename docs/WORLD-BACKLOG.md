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

## Open questions for Chalce
1. **Hearing vs world first?** The ROADMAP says hearing is next. Suggestion:
   W1–W3 first (they create things worth hearing), then hearing goes into the
   W4 senses batch, so the brain migrates once instead of twice.
2. **Is the agent just another being?** In W2, predators hunt *prey*, and the
   agent counts as prey to wolves and sharks the way a sheep does, rather
   than every creature targeting the agent. OK?
3. **Keep the trained brain or start fresh?** Once creatures behave differently,
   some of what the brain learned (e.g. "everything near me attacks") stops
   being true. Migrate the current brain (backed up in Chalcepro/ATS-brain), or
   train a new one on the new world and keep the old one as a reference?
