"""Talk to Coal in a terminal: the phone version (Termux), or any machine with Python and torch.

    python -m household.chat

The same Coal as in CoalOS: one brain file, the same replies and the same learning.
Type a line, Enter. "quit" (or Ctrl+C) ends it; anything Coal learned is already saved.
"""
from __future__ import annotations

from household.app import Session


def main() -> None:
    print("Waking Coal...")
    s = Session()
    print("Coal is awake. Type to Coal; 'quit' to stop.\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "bye coal"):
            break
        if text:
            print("coal >", s.say(text)["reply"])
    print("Bye.")


if __name__ == "__main__":
    main()
