"""Print a CoalOS recording as plain steps: what Chalce did, in which app, on what.

    py -3.12 tools/rec_show.py            # the newest session
    py -3.12 tools/rec_show.py --list     # every session, with its step count
    py -3.12 tools/rec_show.py <name>     # one session (part of its file name is enough)
    add --sees to also print what the screen showed whenever it changed
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from household.train import SAVE  # noqa: E402

RECS = os.path.join(os.path.dirname(SAVE), "recordings")


def load(path):
    with open(path, encoding="utf-8") as f:
        steps = [json.loads(line) for line in f if line.strip()]
    return sorted(steps, key=lambda e: e.get("n", 0))


def line(e):
    k, app, on = e["kind"], e.get("app", ""), e.get("on", "")
    where = f" in {app}" if app and app != "desktop" else ""
    if k in ("opened", "closed", "minimized", "maximized", "retitled"):
        return f"{k} {e.get('title') or e.get('app')}"
    if k in ("click", "double_click", "right_click"):
        return f"{k.replace('_', ' ')} {e.get('role', '')} '{on}'{where}"
    if k == "drag":
        return f"dragged '{on}'{where} onto '{e.get('dropped_on')}'"
    if k in ("typed", "ticked", "picked"):
        return f"{k} {json.dumps(e.get('text'))} into '{on}'{where}"
    if k == "key":
        return f"pressed {e['key']}{where}"
    if k in ("scroll", "swipe", "pinch"):
        return f"{k} {e.get('means')} on '{on}'{where}"
    return k + (f" ({e['why']})" if e.get("why") else "")


def main(argv):
    if not os.path.isdir(RECS):
        print("No recordings yet. Press the record button in the CoalOS tray to make one.")
        return
    files = sorted(f for f in os.listdir(RECS) if f.endswith(".jsonl"))
    if "--list" in argv:
        for f in files:
            print(f"{f[:-6]}   {len(load(os.path.join(RECS, f)))} steps")
        return
    names = [a for a in argv if not a.startswith("--")]
    pick = [f for f in files if names and names[0] in f] if names else files[-1:]
    if not pick:
        print("No session matches that.")
        return
    steps = load(os.path.join(RECS, pick[-1]))
    t0 = steps[0]["t"] if steps else 0
    print(f"Session {pick[-1][:-6]}, {len(steps)} steps, started {datetime.fromtimestamp(t0 / 1000):%Y-%m-%d %H:%M:%S}")
    for e in steps:
        print(f"  {(e['t'] - t0) / 1000:7.1f}s  {line(e)}")
        if "--sees" in argv and e.get("sees"):
            s = e["sees"]
            print("           sees windows:", ", ".join(w["title"] + (" (front)" if w["front"] else "") for w in s["open"]) or "none")
            if s["controls"]:
                print("           front window has:", "; ".join(f"{c['role']} '{c['label']}'" for c in s["controls"][:12]))


if __name__ == "__main__":
    main(sys.argv[1:])
