"""The death screen. What the agent was doing in the moments before it died.

    python replay.py                 play back the most recent death
    python replay.py --list          what replays exist
    python replay.py <file>          play a specific one
    python replay.py --speed 0.05    slower or faster
    python replay.py --still         all frames at once, no animation

Why this exists
---------------
A training run printed "Died" and a tick count and nothing else. Every
question worth asking - was it already hurt, did it try to heal, did it walk
into the hazard while exploring, was something chasing it - had no answer in
the logs, only in the reward totals, which are sums and so cannot say what
happened in what order.

The reward tally was the first half of the fix and it is not enough: a sum
records that damage was taken and a set records which tiles were entered,
but neither is timestamped, so "explored, then died" and "died, then
explored" look identical. This is the other half. It keeps the last stretch
of ticks with the tick number on every one, so the order is the data.

Self-contained on purpose
-------------------------
Each frame carries its own small window of the map rather than a world seed
to regenerate from. Regenerating is fragile - it breaks the moment world
generation changes, and then old replays silently show the wrong terrain -
and a window is about 200 bytes, which is cheap enough not to matter.

That cheapness has a limit: a death every 43 seconds is roughly two thousand
a day, so writing all of them would be over a hundred megabytes daily. Only
the most recent few are kept; see `prune`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAY_DIR = HERE / "replays"

# Ticks kept before the end. 150 at 10 ticks a second of world time is the
# last fifteen seconds, which is where the cause of a death lives.
KEEP_TICKS = 150

# Half-width of the map window stored per frame. 5 gives an 11x11 view, which
# is enough to see what was adjacent and what was approaching.
VIEW = 5

# How many replay files to keep. Deaths are frequent; see the module note.
KEEP_FILES = 12

# Ticks between whole-episode health samples. See Recorder.arc.
ARC_EVERY = 10

ARROWS = {(0, -1): "^", (0, 1): "v", (-1, 0): "<", (1, 0): ">",
          (0, 0): "o"}


def _arrow(facing) -> str:
    try:
        return ARROWS.get((int(facing[0]), int(facing[1])), "?")
    except Exception:
        return "?"


class Recorder:
    """Keeps the last `keep` ticks of an episode, ready to save on death.

    Every method swallows its own errors. This runs inside the training loop
    and a diagnostic that can end a run is worse than no diagnostic.
    """

    def __init__(self, keep: int = KEEP_TICKS, view: int = VIEW):
        self.keep = keep
        self.view = view
        self.frames: list[dict] = []
        self.episode = 0
        self.where = "WORLD"
        self._last_seen = None
        # Health and hunger for the WHOLE episode, sampled. The frame window
        # is the last fifteen seconds, which showed a death arriving at 12 HP
        # without showing that it had been falling for two thousand ticks -
        # "was it already hurt" is the first question and the window cannot
        # answer it. One sample every ARC_EVERY ticks is ~300 numbers for a
        # 3,000-tick episode.
        self.arc: list[list] = []

    def begin(self, episode: int, where: str = "WORLD"):
        self.frames.clear()
        self.arc.clear()
        self.episode = episode
        self.where = where
        self._last_seen = None

    def capture(self, env, action_idx=None, reward=None):
        try:
            self._capture(env, action_idx, reward)
        except Exception:
            pass                      # never let the recorder end an episode

    def _capture(self, env, action_idx, reward):
        agent = env.agent
        ax, ay = int(agent.x), int(agent.y)

        # Only the events that are new this tick. agent.event_log carries
        # wall-clock time rather than a tick, so the tick has to be attached
        # here - that correlation is the whole point of this file.
        #
        # Found by the log itself: comparing LENGTHS does not work. The log is
        # capped at 30 and pops from the front, so once it is full its length
        # stays 30 forever and no event is ever seen as new again. Every
        # replay past the first 30 events read "no logged events in this
        # window", which looked like a quiet episode and was a blind recorder.
        #
        # Matched by identity against the last entry consumed, which survives
        # both the rolling and two events sharing a timestamp.
        events = []
        log = list(getattr(agent, "event_log", ()) or ())
        if log:
            if self._last_seen is None:
                events = [t for t, _ts in log]
            else:
                at = None
                for i in range(len(log) - 1, -1, -1):
                    if log[i] is self._last_seen:
                        at = i
                        break
                # None means the log rolled past it entirely; take what is
                # there rather than silently dropping the gap.
                events = [t for t, _ts in (log[at + 1:] if at is not None else log)]
            self._last_seen = log[-1]

        from agent import ACTION_NAMES
        name = ACTION_NAMES.get(action_idx, str(action_idx)) \
            if action_idx is not None else getattr(agent, "last_action_name", "?")

        tiles, ents = [], []
        w = env.world
        for dy in range(-self.view, self.view + 1):
            row = []
            for dx in range(-self.view, self.view + 1):
                x, y = ax + dx, ay + dy
                try:
                    row.append(int(w.tile_type_at(x, y)))
                except Exception:
                    row.append(-1)
            tiles.append(row)
        try:
            for e in list(w.entities):
                if not getattr(e, "alive", True):
                    continue
                ex, ey = int(getattr(e, "x", 1 << 30)), int(getattr(e, "y", 1 << 30))
                if abs(ex - ax) <= self.view and abs(ey - ay) <= self.view:
                    ents.append([ex - ax, ey - ay,
                                 str(getattr(e, "kind", "") or
                                     type(e).__name__)[:10]])
        except Exception:
            pass

        self.frames.append({
            "t": int(getattr(env, "tick", 0)),
            "x": ax, "y": ay, "z": int(getattr(agent, "z", 0) or 0),
            "f": [int(agent.facing[0]), int(agent.facing[1])],
            "a": name,
            "hp": round(float(getattr(agent, "health", 0)), 1),
            "hun": round(float(getattr(agent, "hunger", 0)), 1),
            "sta": round(float(getattr(agent, "stamina", 0)), 1),
            "r": None if reward is None else round(float(reward), 3),
            "tiles": tiles,
            "ents": ents,
            "ev": events,
        })
        if len(self.frames) > self.keep:
            del self.frames[0:len(self.frames) - self.keep]

        tick = int(getattr(env, "tick", 0))
        if tick % ARC_EVERY == 0:
            self.arc.append([tick, round(float(getattr(agent, "health", 0)), 1),
                             round(float(getattr(agent, "hunger", 0)), 1)])

    def save(self, env, outcome: str, extra: dict | None = None):
        """Write the replay. Returns the path, or None if nothing to write."""
        try:
            if not self.frames:
                return None
            REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            day = time.strftime("%Y-%m-%d")
            folder = REPLAY_DIR / day
            folder.mkdir(parents=True, exist_ok=True)
            blob = {
                "episode": self.episode,
                "where": self.where,
                "outcome": outcome,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "view": self.view,
                "ticks": int(getattr(env, "tick", 0)),
                "arc": self.arc,
                "frames": self.frames,
            }
            if extra:
                blob.update(extra)
            path = folder / ("ep%06d_%s.json"
                             % (self.episode, time.strftime("%H%M%S")))
            path.write_text(json.dumps(blob), encoding="utf-8")
            prune()
            return path
        except Exception:
            return None


def all_replays():
    if not REPLAY_DIR.is_dir():
        return []
    return sorted(REPLAY_DIR.glob("*/ep*.json"),
                  key=lambda p: p.stat().st_mtime)


def prune(keep: int = KEEP_FILES):
    """Keep only the newest few. A death every 43 seconds fills a disk."""
    try:
        files = all_replays()
        for p in files[:-keep] if len(files) > keep else []:
            p.unlink(missing_ok=True)
        # Tidy up day folders left empty by pruning.
        for d in REPLAY_DIR.iterdir() if REPLAY_DIR.is_dir() else []:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    except Exception:
        pass


# ---- playback -------------------------------------------------------------

# Tile ids are integers whose meaning lives in the world module. Rendering
# them by parity of "is it passable" rather than by name keeps this readable
# without importing world generation into a viewer.
def _glyph(tile: int) -> str:
    if tile < 0:
        return " "
    return ".:-=+*#%@"[tile % 9]


def render(frame, view, width=None):
    """One frame as text: the map window, the agent, and its numbers."""
    tiles, ents = frame["tiles"], {(e[0], e[1]): e[2] for e in frame["ents"]}
    lines = []
    for dy in range(-view, view + 1):
        row = []
        for dx in range(-view, view + 1):
            if dx == 0 and dy == 0:
                row.append(_arrow(frame["f"]))
            elif (dx, dy) in ents:
                row.append("E")
            else:
                row.append(_glyph(tiles[dy + view][dx + view]))
        lines.append(" ".join(row))
    bar = int(max(0.0, min(1.0, frame["hp"] / 100.0)) * 20)
    info = [
        "tick %-6d  (%d,%d) z%d  facing %s" % (frame["t"], frame["x"],
                                               frame["y"], frame["z"],
                                               _arrow(frame["f"])),
        "action  %-14s reward %s" % (frame["a"],
                                     "-" if frame["r"] is None else "%+.2f" % frame["r"]),
        "health  [%s%s] %5.1f" % ("#" * bar, "." * (20 - bar), frame["hp"]),
        "hunger  %5.1f   stamina %5.1f" % (frame["hun"], frame["sta"]),
    ]
    out = []
    for i, ln in enumerate(lines):
        right = info[i - 1] if 1 <= i <= len(info) else ""
        out.append("   %-*s   %s" % (view * 4 + 2, ln, right))
    for ev in frame["ev"]:
        out.append("   >> %s" % ev)
    return "\n".join(out)


def play(path, speed=0.12, still=False):
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    frames, view = blob["frames"], blob.get("view", VIEW)
    print("\n  %s" % Path(path).name)
    print("  episode %s  %s  %s  ended at tick %s  (%d frames kept)"
          % (blob.get("episode"), blob.get("where"), blob.get("outcome"),
             blob.get("ticks"), len(frames)))
    if blob.get("cause"):
        print("  cause: %s" % blob["cause"])
    print()
    for i, fr in enumerate(frames):
        if not still:
            # Redraw in place, like a killcam rather than a scrolling log.
            sys.stdout.write("\033[H\033[J" if i else "")
        print(render(fr, view))
        if not still:
            print("\n   frame %d/%d" % (i + 1, len(frames)))
            time.sleep(speed)
        else:
            print("   ---")
    last = frames[-1]
    print("\n  final: tick %d at (%d,%d), health %.1f, hunger %.1f"
          % (last["t"], last["x"], last["y"], last["hp"], last["hun"]))
    hp = [f["hp"] for f in frames]
    print("  health over these %d ticks: %.1f -> %.1f (worst %.1f)"
          % (len(hp), hp[0], hp[-1], min(hp)))
    arc = blob.get("arc") or []
    if arc:
        # The whole episode, not just the window. "Was it already hurt" is the
        # first question asked of any death and the window cannot answer it.
        print("\n  health across the WHOLE episode:")
        step = max(1, len(arc) // 12)
        cells = arc[::step]
        print("    tick   " + "".join("%7d" % c[0] for c in cells))
        print("    health " + "".join("%7.0f" % c[1] for c in cells))
        print("    hunger " + "".join("%7.0f" % c[2] for c in cells))
        starved = [c[0] for c in arc if c[2] <= 0]
        if starved:
            print("    hunger hit zero at tick %d and stayed there for %d of "
                  "%d samples" % (starved[0], len(starved), len(arc)))
    evs = [(f["t"], e) for f in frames for e in f["ev"]]
    if evs:
        print("\n  what happened, in order:")
        for t, e in evs[-14:]:
            print("    tick %-6d %s" % (t, e))
    else:
        print("\n  no logged events in this window")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--speed", type=float, default=0.12)
    ap.add_argument("--still", action="store_true",
                    help="print every frame instead of animating")
    a = ap.parse_args(argv)

    files = all_replays()
    if a.list:
        if not files:
            print("  no replays yet - they are written when an episode dies")
            return 1
        print("  %d replay(s), newest last:" % len(files))
        for p in files:
            try:
                b = json.loads(p.read_text(encoding="utf-8"))
                print("    %-34s ep %-7s %-22s %s"
                      % (p.parent.name + "/" + p.name, b.get("episode"),
                         b.get("where"), b.get("outcome")))
            except Exception as exc:
                print("    %-34s (unreadable: %s)" % (p.name, exc))
        return 0

    target = Path(a.file) if a.file else (files[-1] if files else None)
    if target is None:
        print("  no replays yet. They are written when an episode dies, so")
        print("  run training and then come back.")
        return 1
    if not Path(target).is_file():
        print("  no such replay: %s" % target)
        return 1
    play(target, speed=a.speed, still=a.still)
    return 0


if __name__ == "__main__":
    sys.exit(main())
