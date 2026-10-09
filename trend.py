"""Training trend at a glance - the 'put a number on it' view of an RL run.

    python trend.py            # summarise the newest training log once
    python trend.py --watch    # re-summarise every 60s while training runs
    python trend.py --watch 30 # ...every 30s
    python trend.py --logs 3   # fold in the last 3 run logs (Train.bat restarts)

It reads the console logs the trainer tees to logs/train_*.log (see main.py).
It never imports the trainer, torch, or the env, so it is light and cannot slow
training down. The two things worth watching:
  - reward by quarter   (is it climbing?)
  - shark/drowning share (is it learning to avoid the sea? should fall)
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

RE_WORLD = re.compile(
    r"Episode (\d+) done \| WORLD \| reward=\s*([-+]?[\d.]+).*?ticks=(\d+)/\d+ \| (\w+)")
RE_CAUSE = re.compile(r"cause: (.+?)\s+replay")
RE_TOTAL = re.compile(r"training (\d+) episodes")
RE_GROWTH = re.compile(r"Expanding (\d+) -> (\d+)")


def newest_logs(k: int):
    files = sorted(glob.glob(os.path.join(LOGS, "train_*.log")), key=os.path.getmtime)
    return files[-k:] if files else []


def summarise(k: int) -> str:
    files = newest_logs(k)
    if not files:
        return "No training logs yet (logs/train_*.log). Start Train.bat."
    rew, ticks, surv, last, total, growth = [], [], 0, 0, None, None
    causes = Counter()
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            m = RE_WORLD.search(line)
            if m:
                last = int(m.group(1)); rew.append(float(m.group(2)))
                ticks.append(int(m.group(3)))
                if m.group(4) == "Survived":
                    surv += 1
                continue
            c = RE_CAUSE.search(line)
            if c:
                k2 = c.group(1).strip().lower()
                k2 = ("shark/drowning" if ("shark" in k2 or "drown" in k2)
                      else "starvation" if "starv" in k2
                      else k2.split()[0] if k2 else "?")
                causes[k2] += 1
                continue
            t = RE_TOTAL.search(line)
            if t:
                total = int(t.group(1))
            g = RE_GROWTH.search(line)
            if g:
                growth = "%s->%s" % (g.group(1), g.group(2))
    n = len(rew)
    if not n:
        return "Log found but no WORLD episodes yet - it may still be warming up."
    q = max(1, n // 4)
    quarters = [sum(rew[i*q:(i+1)*q]) / max(1, len(rew[i*q:(i+1)*q])) for i in range(4)]
    tot_c = sum(causes.values()) or 1
    cause_str = "  ".join("%s %d%%" % (name, v * 100 // tot_c)
                          for name, v in causes.most_common(4))
    out = []
    out.append("episode %d%s   |   %d world-eps%s" % (
        last, ("/%d" % total) if total else "", n,
        ("   |   mind grew %s" % growth) if growth else ""))
    out.append("reward by quarter:  %6.1f -> %6.1f -> %6.1f -> %6.1f"
               % tuple(quarters))
    out.append("avg %.1f   peak %.1f   avg life %d ticks   survived cap %d"
               % (sum(rew) / n, max(rew), sum(ticks) / n, surv))
    out.append("deaths: " + cause_str)
    return "\n".join(out)


def main(argv):
    k = 1
    if "--logs" in argv:
        i = argv.index("--logs")
        if i + 1 < len(argv):
            k = max(1, int(argv[i + 1]))
    if "--watch" in argv:
        i = argv.index("--watch")
        every = 60
        if i + 1 < len(argv) and argv[i + 1].isdigit():
            every = int(argv[i + 1])
        print("Watching training trend every %ds (Ctrl+C to stop).\n" % every)
        try:
            while True:
                print("--- %s ---" % time.strftime("%H:%M:%S"))
                print(summarise(k) + "\n")
                time.sleep(every)
        except KeyboardInterrupt:
            print("stopped.")
    else:
        print(summarise(k))


if __name__ == "__main__":
    main(sys.argv[1:])
