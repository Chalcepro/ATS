# W2b flow: a shark you can see coming

Status: **flow written 2026-10-13, waiting for Chalce's OK. No code yet.**
Chalce's idea (2026-10-13), slotted before W3 because sharks are 15 of 16
deaths in the W1/W2 survival check.

## Why: what the brain lives through today

1. After `OCEAN_SHARK_TICKS` (3) in the sea, the agent loses health **every
   tick** (`ats_env.py`, `SEA_SURVIVABLE` branch). Nothing approaches first.
2. Each of those ticks `spawn_shark` drops a new shark **on the agent's own
   tile**. Nothing is ever seen arriving, and sharks pile up, one per tick at sea.
3. A shark's brain_id is `0`, the same as "no creature".

So to the brain, *the water itself hurts*. It can see the ocean tile, but the
thing that kills it has no shape, no direction, no warning. Like dying by a
sulphur lake: you can't learn to avoid a cause you can never perceive.

## The new shark, as Chalce described it

```
agent in the sea past the grace ticks
  → SURFACE: a shark appears on a sea tile inside the agent's view
     (if it would be outside the view, it is placed at the view's edge instead).
     That tile now reads as SHARK, a new tile type in the patch the brain sees.
  → WAIT ~2-3 s (ticks to be set): it sits still. The agent can see it and can act.
  → CHARGE: it moves in a straight line toward where the agent WAS when it
     charged, fast, several tiles per tick or one per tick (to set).
       hits the agent's tile → bite (heavy damage / death as today)
       agent dodged         → shark stops, WAIT again, then charges again
  agent reaches land → shark gives up, sinks (gone).
```

- One shark per swimmer, not one per tick.
- The brain sees it two ways at once: the **tile** says SHARK, and the
  nearest-creature sense gets a real `brain_id` for the shark instead of 0.
- Dodging = stepping off the charge line. That's learnable: see the box,
  move sideways or toward land.

## What it costs

A new tile type value (the tile vocabulary has room: 16 slots, 14 used) and a
new brain_id. The trained brain has never seen either. It keeps working (unknown
values just start meaning nothing) and learns them in training. This is the
"keep the brain, migrate it" decision, in a small dose ahead of W4.

## Proof

- Scene tests: a shark surfaces in view; waits; charges in a straight line;
  a sideways step makes it miss; it waits and charges again; land makes it go.
- Replay viewer shows the shark and its job (`surface` / `wait` / `charge`).
- Survival check like W2 (kept brain, 8 worlds) before and after: expect a dip
  at first (new thing), then training on it.

## Questions for Chalce

1. **Wait before charging**: the brain thinks in ticks. Today ~1 tick per step.
   Is "3 ticks" OK as your "2-3 seconds", or do you want it longer (e.g. 10)?
2. **Charge speed**: 1 tile per tick (dodgeable, you see it coming) or faster?
3. **The bite**: keep today's heavy damage per bite (you can survive one and
   swim back), or one bite = death?
4. Should the grace (3 ticks in the sea before a shark comes) stay, or does a
   shark come as soon as you're in open sea?
