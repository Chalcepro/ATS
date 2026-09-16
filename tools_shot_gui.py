"""Screenshot the ATS GUI without a window.

    py -3 tools_shot_gui.py [out.png]

The UI is laid out in absolute pixels, and absolute pixels are exactly the
kind of thing that is faster to *look at* than to reason about. SDL's dummy
video driver renders to an offscreen surface, so a layout change can be
checked in a couple of seconds instead of by opening the app.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

import debug_gui  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "build/gui_menu.png"


def main():
    gui = debug_gui.TerminalGUI("ATS")
    gui.app_state = debug_gui.STATE_MENU
    gui.active_tab = 0
    gui._clicks = []
    gui.screen.fill(debug_gui.BG)
    gui._draw_menu()
    pygame.display.flip()

    os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
    pygame.image.save(gui.screen, OUT)
    print("wrote", OUT, "%dx%d" % gui.screen.get_size())


main()
