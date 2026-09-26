"""How long an episode is allowed to run, earned rather than fixed.

The problem with a big number
-----------------------------
"Survive as many days as possible" reads like a request to raise MAX_TICKS,
and that is the one change guaranteed not to work. Measured headless on this
machine, the full world runs at 63 ticks a second:

    cap  4,400 (1.1 days)    70 s/episode     51 episodes/hour
    cap 12,000 (3.0 days)   192 s/episode     19 episodes/hour
    cap 40,000 (10  days)   639 s/episode      6 episodes/hour

PPO needs thousands of episodes. At six an hour, five thousand episodes is
eight hundred and thirty hours. An agent cannot be taught to survive ten
days by being shown ten-day episodes; there is not enough wall clock in the
week to show it enough of them.

The other half is that the cap is not currently the binding constraint at
all. The agent dies around tick 670, which is 15% of the cap it already has.
Raising 4,400 to 40,000 would change nothing whatsoever about what it
learns - it would only make each failure take ten times longer to observe.

What this does instead
----------------------
The same thing curriculum.py does with room size, applied to time: start at
a cap the agent can nearly reach, and raise it when it has earned it. Short
episodes while it is bad, because that is when episodes-per-hour matters
most and when the lesson is "do not die in the first minute". Long episodes
once it is good, when the lesson is "keep going", and when each one is worth
the ten minutes it costs.

The ladder is in days because that is the unit the world runs on -
DAY_LENGTH_TICKS is 4000, and night is the dangerous half.
"""
from __future__ import annotations

import config_rl

# Ticks per rung. Roughly: half a day, one day, two, three, five, ten.
#
# The first rung is 2000 and not shorter, which rewards.audit decided rather
# than taste. At 1000 ticks the wage for being alive is +20 while the fixed
# nuisance budget is -25, so an agent that played badly and survived scored
# -11.7 against -10.0 for dying on the first tick: it would have been paid to
# die. That is the bug the audit exists to catch, and it catches it at every
# rung of this ladder before any of them are trained on.
#
# 2000 is still three times how long the agent currently lasts (~670 ticks),
# so reaching the cap is a stretch rather than a fantasy.
LADDER = [2000, 4400, 8000, 12000, 20000, 40000]

# Fraction of recent episodes that must reach the cap before it grows, and
# how many to judge over. Same shape as the curriculum's pass_rate/window,
# and for the same reason: promotion is earned, never scheduled.
REACH_RATE = 0.60
WINDOW = 40


def cap_for(progress: dict | None) -> int:
    """The tick cap this brain has earned."""
    rung = int((progress or {}).get("survival_rung", 0))
    rung = max(0, min(rung, len(LADDER) - 1))
    return LADDER[rung]


def days(ticks: float) -> float:
    return ticks / float(config_rl.DAY_LENGTH_TICKS)


def label(progress: dict | None) -> str:
    cap = cap_for(progress)
    return "%d ticks (%.1f days)" % (cap, days(cap))


def earned(progress: dict | None, reached: list | tuple) -> bool:
    """Has it reached the cap often enough to deserve a longer one?

    `reached` is the recent history of "did this episode hit the cap",
    newest last. Judged over a window rather than on one episode, because a
    single long run is luck and the point is a policy that does it reliably.
    """
    rung = int((progress or {}).get("survival_rung", 0))
    if rung >= len(LADDER) - 1:
        return False
    if len(reached) < WINDOW:
        return False
    recent = list(reached)[-WINDOW:]
    return (sum(1 for r in recent if r) / float(len(recent))) >= REACH_RATE


def promote(progress: dict) -> int:
    """Move up one rung, and return the new cap."""
    rung = int(progress.get("survival_rung", 0))
    progress["survival_rung"] = min(rung + 1, len(LADDER) - 1)
    return cap_for(progress)
