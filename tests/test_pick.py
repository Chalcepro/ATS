"""Chooser vs the old answer rules, on chats neither was tuned on (Chalce, 11-07).

    py -3.12 tests/test_pick.py        (test copy of Coal; nothing is saved)

Each chat tells a few things, then asks. A reply counts if it has the words wanted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

CHATS = [
    (["my phone is on the table", "the table is in the parlour"], "where is my phone", ["table"]),
    (["i hid the remote under the pillow", "the pillow is on the couch"], "is the remote on the couch", ["yes", "pillow", "couch"]),
    (["my sister bought a red bike", "my brother has a blue car"], "who has a blue car", ["brother"]),
    (["mum went to the market yesterday"], "when did mum go to the market", ["yesterday"]),
    (["dad stayed home because he was sick"], "why did dad stay home", ["sick"]),
    (["the spare key is inside the blue box", "the blue box is behind the door"], "where is the spare key", ["box"]),
    (["tolu cooked rice on sunday"], "what did tolu cook", ["rice"]),
    (["my laptop charger is in my bag"], "where is my charger", ["bag"]),
    ([], "what is a closure", ["function"]),
    ([], "what is jsx", ["javascript", "syntax", "xml", "extension"]),
    ([], "what opens video files", ["capcut", "vlc"]),
    ([], "is capcut an app on this device", ["yes", "app"]),
]


def run(mode):
    os.environ["COAL_PICK"] = mode
    import importlib
    from household import mind
    importlib.reload(mind)
    from household.model import load
    brain, _ = load(os.environ.get("COAL_BRAIN") or os.path.join("checkpoints", "household_brain_trend_understand_test.pt"))
    right = 0
    for tell, q, want in CHATS:
        m = mind.Mind(brain.keep["mind"])
        for t in tell:
            m.reply(t)
        a = m.reply(q)
        ok = any(w in a.lower() for w in want)
        right += ok
        print(f"  [{'ok' if ok else '--'}] {q:34} -> {a[:110]}")
    print(f"{mode}: {right}/{len(CHATS)}\n", flush=True)


if __name__ == "__main__":
    for mode in sys.argv[1:] or ("rules", "chooser"):
        print(f"== {mode} ==")
        run(mode)
