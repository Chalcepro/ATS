"""A long one-on-one class with Coal (Chalce, 11-07): one Mind kept alive, talked to through files.

    py -3.12 tools/class_session.py BRAIN.pt          (run in its own window; Ctrl+C or a line "!end" stops)

Write a line to logs/class/in.txt -> Coal's reply lands in logs/class/out.txt (and the whole talk in
logs/class/talk.txt). Everything he learns while talking (facts, the reader's live learning, corrections,
pages he looks up) stays in this one Mind. Every 10 lines, and at the end, his mind is saved into BRAIN.pt.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")
from household.model import load, save
from household.mind import Mind

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "class")
os.makedirs(HERE, exist_ok=True)
IN, OUT, TALK = (os.path.join(HERE, n) for n in ("in.txt", "out.txt", "talk.txt"))

path = sys.argv[1]
brain, rounds = load(path)
mind = Mind(brain.keep["mind"])


def keep():
    brain.keep["mind"] = mind.to_keep()
    save(brain, path, rounds)


n = 0
open(OUT, "w", encoding="utf-8").write("ready\n")
while True:
    if not os.path.exists(IN):
        time.sleep(0.3)
        continue
    line = open(IN, encoding="utf-8").read().strip()
    os.replace(IN, IN + ".done")
    if line == "!end":
        break
    try:
        said = mind.reply(line)
    except Exception as e:                                  # he stumbles; the class goes on
        said = f"[error: {type(e).__name__}: {e}]"
    open(OUT + ".tmp", "w", encoding="utf-8").write(said + "\n")
    os.replace(OUT + ".tmp", OUT)
    with open(TALK, "a", encoding="utf-8") as f:
        f.write(f"you > {line}\ncoal> {said}\n")
    print(f"you > {line}\ncoal> {said}", flush=True)
    n += 1
    if n % 10 == 0:
        keep()
keep()
open(OUT, "w", encoding="utf-8").write("ended, saved\n")
