"""One file per day of training. What happened, in detail, without the sprawl.

    python daylog.py                    today, summarised
    python daylog.py --date 2026-07-27  a specific day
    python daylog.py --days             which days exist
    python daylog.py --tail 20          the last 20 episodes of that day
    python daylog.py --deaths           only the episodes that died, with cause
    python daylog.py --raw              the JSON lines themselves

Why one file per day
--------------------
A single append-only log of a run that produces two thousand episodes a day
is unreadable within a week and unopenable within a month, which is the
failure the old CSV had. The filename carries the date and is chosen at write
time, so `logs/2026-07-27.jsonl` holds that day and nothing else, and a day
with no training simply has no file. Nothing needs rotating, nothing needs
pruning, and finding a day means knowing its date.

One JSON object per line, which is what makes it both appendable and
readable: a crash mid-write costs the last line rather than the file, and
there is no header to keep in step with the fields.

What a line holds
-----------------
Enough to answer "what happened" without opening a replay: the outcome and
cause, the health and position it ended at, the full reward breakdown, and
the last few logged events in order. About 700 bytes, so a 2,000-episode day
is under two megabytes.

The replay (replay.py) is the other half and deliberately separate. This is
the record of every episode; that is a watchable reconstruction of a few.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"

# Events kept per line. The agent keeps 30; the last dozen are the ones that
# explain an ending, and keeping all of them would triple the file for detail
# that belongs in a replay.
KEEP_EVENTS = 12

CAUSE_HINTS = [
    ("void", "the Void"),
    ("shark", "shark / drowning"),
    ("drowned", "shark / drowning"),
    ("ocean", "shark / drowning"),
    ("lava", "lava"),
    ("poison", "poison"),
    ("starv", "starvation"),
    ("hunger", "starvation"),
    ("fell in battle", "killed in battle"),
    ("battle", "killed in battle"),
    ("creeper", "creeper blast"),
    ("ice", "ice / leg injury"),
]


def guess_cause(events, health=None, hunger=None, starved_ticks=None) -> str:
    """Name the death from the events that preceded it.

    The environment has no death-cause field - death is `health <= 0` in
    three different places - but it already writes a human-readable line at
    each of them ("Consumed by the Void!", "Harsh Stop: Drowned in ocean
    with critical HP", "Agent fell in battle / perished."). Reading those
    back is less invasive than threading a cause through the env, and it
    keeps working for any new one that gets logged.

    Returns "unattributed" rather than guessing when nothing matches, because
    a wrong cause in a log is worse than an absent one.
    """
    blow = None
    for text in reversed(list(events or ())):
        low = str(text).lower()
        for needle, name in CAUSE_HINTS:
            if needle in low:
                blow = name
                break
        if blow:
            break

    # Starvation outranks whatever landed the final blow. Measured: hunger
    # reached zero around tick 1,500-1,700 in every death sampled, health then
    # bled down 1 HP per 30 ticks, and a hostile finished a 2 HP agent - which
    # the last event calls "fell in battle". Reporting that as the cause sends
    # you tuning combat when the agent is starving to death, so the underlying
    # condition is named first and the blow is kept as context.
    starving = (hunger is not None and hunger <= 0) or bool(starved_ticks)
    if starving:
        since = (" since tick %d" % starved_ticks) if starved_ticks else ""
        return ("starvation%s (finished by %s)" % (since, blow)) if blow \
            else ("starvation%s" % since)
    if blow:
        return blow
    if health is not None and health > 0:
        return "survived to the cap"
    return "unattributed"


def cause_kind(cause: str | None) -> str:
    """The cause without its detail, for counting.

    `guess_cause` puts the tick in the string because that is the useful thing
    to read on one death - but it means "starvation since tick 1730" and
    "since tick 1500" count as two different causes, and a day of starvation
    deaths reports as a scatter of one-offs instead of the one pattern it is.
    """
    if not cause:
        return "survived"
    head = str(cause).split(" since ")[0].split(" (")[0]
    return head.strip() or "unattributed"


def path_for(day: str | None = None) -> Path:
    day = day or time.strftime("%Y-%m-%d")
    return LOG_DIR / ("%s.jsonl" % day)


def write(record: dict, day: str | None = None):
    """Append one episode. Never raises - a log must not end a run."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path_for(day).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        pass


def record_for(env, ep, where, total, grade, eff, cap, hit_cap,
               replay_path=None, starved_ticks=None) -> dict:
    """Build the line from whatever the env is willing to say.

    Defensive throughout: a StageSession exposes a different reward object and
    a thinner agent than the island does, and a logger that only works on one
    of them would go quiet exactly when a rehearsal rung started failing.
    """
    agent = getattr(env, "agent", None)
    rewards = getattr(env, "rewards", None)

    events = []
    try:
        events = [t for t, _ts in list(getattr(agent, "event_log", ()) or ())]
    except Exception:
        pass

    breakdown = []
    try:
        breakdown = [[r, round(v, 2), c]
                     for r, v, c, _p in rewards.breakdown(getattr(env, "tick", 0))]
    except Exception:
        pass

    health = getattr(agent, "health", None)
    rec = {
        "ts": time.strftime("%H:%M:%S"),
        "ep": ep,
        "where": where,
        "ticks": int(getattr(env, "tick", 0) or 0),
        "cap": int(cap or 0),
        "reward": round(float(total), 2),
        "grade": grade,
        "eff": eff,
        "outcome": "survived" if hit_cap else "died",
        "cause": None if hit_cap else guess_cause(
            events, health, _num(agent, "hunger"), starved_ticks),
        "cause_kind": None if hit_cap else cause_kind(guess_cause(
            events, health, _num(agent, "hunger"), starved_ticks)),
        "health": None if health is None else round(float(health), 1),
        "hunger": _num(agent, "hunger"),
        "stamina": _num(agent, "stamina"),
        "pos": [int(getattr(agent, "x", 0) or 0), int(getattr(agent, "y", 0) or 0)]
               if agent is not None else None,
        "caps": len(getattr(agent, "capabilities", ()) or ()),
        "islands": sorted(getattr(rewards, "discovered_islands", ()) or ()),
        "tiles_seen": len(getattr(rewards, "discovered_tiles", ()) or ()),
        "paid": breakdown,
        "events": events[-KEEP_EVENTS:],
    }
    if replay_path:
        rec["replay"] = str(replay_path)
    return rec


def _num(obj, name):
    v = getattr(obj, name, None)
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return None


# ---- reading --------------------------------------------------------------

def read(day: str | None = None):
    p = path_for(day)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue          # a torn last line costs that line, not the file
    return out


def days():
    if not LOG_DIR.is_dir():
        return []
    return sorted(p.stem for p in LOG_DIR.glob("*.jsonl"))


def summarise(rows, day):
    if not rows:
        print("  nothing logged for %s" % day)
        return
    world = [r for r in rows if r.get("where") == "WORLD"]
    reh = [r for r in rows if r.get("where") != "WORLD"]
    print("\n  %s - %d episodes (%d world, %d rehearsal)"
          % (day, len(rows), len(world), len(reh)))
    if world:
        tk = [r["ticks"] for r in world]
        rw = [r["reward"] for r in world]
        surv = sum(1 for r in world if r["outcome"] == "survived")
        print("    world ticks   mean %.0f  best %d  worst %d  cap %d"
              % (sum(tk) / len(tk), max(tk), min(tk), world[-1].get("cap", 0)))
        print("    world reward  mean %+.1f  best %+.1f"
              % (sum(rw) / len(rw), max(rw)))
        print("    reached cap   %d of %d (%.0f%%)"
              % (surv, len(world), 100.0 * surv / len(world)))
        causes = {}
        for r in world:
            if r["outcome"] == "died":
                # Grouped by kind, not by the detailed string - see cause_kind.
                k = r.get("cause_kind") or cause_kind(r.get("cause"))
                causes[k] = causes.get(k, 0) + 1
        if causes:
            print("    how it died:")
            for c, n in sorted(causes.items(), key=lambda kv: -kv[1]):
                print("      %-24s %4d  (%.0f%%)"
                      % (c, n, 100.0 * n / max(1, len(world))))
        paid = {}
        for r in world:
            for reason, total, count in r.get("paid", []):
                slot = paid.setdefault(reason, [0.0, 0])
                slot[0] += total
                slot[1] += count
        if paid:
            print("    what it was paid for, over the whole day:")
            for reason, (total, count) in sorted(paid.items(),
                                                 key=lambda kv: -abs(kv[1][0]))[:10]:
                print("      %-22s %+10.1f  over %6d awards" % (reason, total, count))
    if reh:
        by = {}
        for r in reh:
            by[r["where"]] = by.get(r["where"], 0) + 1
        print("    rehearsed: %s"
              % ", ".join("%s x%d" % (k.replace("rehearse ", ""), v)
                          for k, v in sorted(by.items(), key=lambda kv: -kv[1])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--days", action="store_true")
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--deaths", action="store_true")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args(argv)

    if a.days:
        ds = days()
        if not ds:
            print("  no logs yet - they appear once training runs")
            return 1
        print("  %d day(s) of training logged:" % len(ds))
        for d in ds:
            rows = read(d)
            size = path_for(d).stat().st_size / 1024.0
            print("    %s  %5d episodes  %7.1f KB" % (d, len(rows), size))
        return 0

    day = a.date or time.strftime("%Y-%m-%d")
    rows = read(day)
    if a.raw:
        for r in rows[-(a.tail or len(rows)):]:
            print(json.dumps(r))
        return 0
    if a.deaths:
        dead = [r for r in rows if r.get("outcome") == "died"]
        if not dead:
            print("  no deaths logged for %s" % day)
            return 0
        print("\n  %d deaths on %s" % (len(dead), day))
        for r in dead[-(a.tail or 20):]:
            print("\n    ep %-7s %s  tick %d/%d  health %s  at %s"
                  % (r["ep"], r["ts"], r["ticks"], r.get("cap", 0),
                     r.get("health"), r.get("pos")))
            print("      cause: %s" % r.get("cause"))
            for e in (r.get("events") or [])[-5:]:
                print("        %s" % e)
        return 0
    if a.tail:
        print("\n  last %d episodes of %s" % (a.tail, day))
        for r in rows[-a.tail:]:
            print("    %s ep %-7s %-22s %6d/%-6d %+9.1f %-4s %-9s %s"
                  % (r["ts"], r["ep"], r["where"], r["ticks"], r.get("cap", 0),
                     r["reward"], r.get("grade", ""), r["outcome"],
                     r.get("cause") or ""))
        return 0
    summarise(rows, day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
