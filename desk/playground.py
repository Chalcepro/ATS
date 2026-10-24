"""The ATS playground: the trained brain lives on the desktop as a pixel
mini-Rewrite. Shift+A while the DESKTOP is focused (not in any app) opens the
menu: spawn food and objects under the cursor, remove them, call it back when
it has wandered off screen, pause, see how it is doing.

Run:  Playground.bat   (or  py -3.12 desk/playground.py)
The brain is a frozen copy of checkpoints/brain.pt; island training is untouched."""
import ctypes, os, sys, time, traceback
import ctypes.wintypes as wt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import window
from window import OverlayWindow, user32
from sprites import Sheet, paste, bar, SCALE
from desk_world import DeskMind, SPAWNABLE, TILE_PX, HIDDEN

LOG_DIR = os.path.join(os.environ.get("APPDATA", HERE), "ATSPlayground")
FPS = 15
BRAIN_HZ = 5                        # brain ticks a second at 1x speed
VK_SHIFT, VK_A = 0x10, 0x41
DESKTOP_CLASSES = {"Progman", "WorkerW"}

user32.GetForegroundWindow.restype = wt.HWND
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]


def desktop_focused():
    """True only when the focused window is the desktop itself (icons/wallpaper)."""
    buf = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(user32.GetForegroundWindow(), buf, 64)
    return buf.value in DESKTOP_CLASSES


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class Playground:
    def __init__(self):
        window.dpi_aware()
        self.left, self.top, right, bottom = window.work_area()
        self.W, self.H = right - self.left, bottom - self.top
        self.cols, self.rows = self.W // TILE_PX, self.H // TILE_PX
        self.mind = DeskMind(self.cols, self.rows)
        self.sheet = Sheet()
        self.win = OverlayWindow(self.W, self.H, self.on_event)
        self.canvas = np.zeros((self.H, self.W, 4), np.uint8)
        self.paused, self.speed = False, 1
        self.prev = self.now = (self.mind.agent.x, self.mind.agent.y)
        self.facing_left = False
        self.eat_until = 0.0
        self.tick_t = time.time()
        self.a_was_down = False
        self.dirty = True

    # ── tiles <-> pixels ──
    def tile_at(self, sx, sy):
        return (sx - self.left) // TILE_PX, (sy - self.top) // TILE_PX

    def px(self, tx, ty):
        return tx * TILE_PX, ty * TILE_PX

    def on_event(self, name, **info):
        pass        # click-through window: nothing to click

    # ── the brain ──
    def brain_tick(self):
        m = self.mind
        hunger0 = m.agent.hunger
        m.step()
        a = m.agent
        self.prev, self.now = self.now, (a.x, a.y)
        if self.now[0] != self.prev[0]:
            self.facing_left = self.now[0] < self.prev[0]
        if a.hunger > hunger0 + 0.5:
            self.eat_until = time.time() + 0.6
        self.dirty = True

    # ── Shift+A ──
    def check_keys(self):
        a_down = key_down(VK_A)
        pressed = a_down and not self.a_was_down
        self.a_was_down = a_down
        if pressed and key_down(VK_SHIFT) and desktop_focused():
            self.open_menu()

    def open_menu(self):
        m, a = self.mind, self.mind.agent
        cx, cy = window.cursor_pos()
        tx, ty = self.tile_at(cx, cy)
        on_screen = 0 <= a.x < self.cols
        has_thing = bool(m.world.tiles.get((tx, ty)) and m.world.tiles[(tx, ty)].object_id)
        spawn = [(100 + i, label, False) for i, label in enumerate(SPAWNABLE)]
        items = [
            (1, f"Health {a.health:.0f}  ·  Hunger {a.hunger:.0f}  ·  doing: {m.last_action}", False, False),
            None,
            ("Spawn here", spawn),
            (2, "Scatter 5 apples", False),
            (3, "Remove the thing under the cursor", False, has_thing),
            (4, "Clear everything", False),
            None,
            (5, "Call it here" + ("" if on_screen else "  (it's off screen)"), False),
            (6, "Resume" if self.paused else "Pause", False),
            ("Speed", [(10 + s, f"{s}x", self.speed == s) for s in (1, 2, 4)]),
            (7, "New life (reset)", False),
            None,
            (8, "Type a word to it (yes / no)  — next step", False, False),
            None,
            (9, "Quit playground", False),
        ]
        pick = self.win.menu(items)
        w = m.world
        if pick >= 100:
            w.put(tx, ty, list(SPAWNABLE.values())[pick - 100])
        elif pick == 2:
            rng = np.random.default_rng()
            for _ in range(5):
                w.put(int(rng.integers(0, self.cols)), int(rng.integers(0, self.rows)), "0001")
        elif pick == 3:
            w.remove(tx, ty)
        elif pick == 4:
            for x, y, _ in w.objects():
                w.remove(x, y)
        elif pick == 5:
            a.x, a.y = max(0, min(self.cols - 1, tx)), max(0, min(self.rows - 1, ty))
            self.prev = self.now = (a.x, a.y)
        elif pick == 6:
            self.paused = not self.paused
        elif pick in (11, 12, 14):
            self.speed = pick - 10
        elif pick == 7:
            m.reset()
            self.prev = self.now = (m.agent.x, m.agent.y)
        elif pick == 9:
            self.win.close()
        self.dirty = True

    # ── drawing ──
    def draw(self, t):
        c = self.canvas
        c[:] = 0
        s = self.sheet
        for x, y, item in self.mind.world.objects():
            img = s.item(item)
            px, py = self.px(x, y)
            paste(c, img, px + (TILE_PX - img.shape[1]) // 2, py + TILE_PX - img.shape[0])
        # it slides from the last tile to the new one between brain ticks
        f = min(1.0, (time.time() - self.tick_t) * BRAIN_HZ * self.speed)
        ax = self.prev[0] + (self.now[0] - self.prev[0]) * f
        ay = self.prev[1] + (self.now[1] - self.prev[1]) * f
        moving = self.prev != self.now and f < 1.0
        kind = "eat" if time.time() < self.eat_until else ("walk" if moving and int(t * 8) % 2 else "idle")
        img = (s.rewrite_left if self.facing_left else s.rewrite)[kind]
        bob = 0 if moving else int(np.sin(t * 3) * 1.5) * SCALE
        px = int(ax * TILE_PX) + (TILE_PX - img.shape[1]) // 2
        py = int(ay * TILE_PX) + TILE_PX - img.shape[0] + bob
        paste(c, img, px, py)
        a = self.mind.agent
        bar(c, px, py - 12, img.shape[1], a.health / 100, (220, 50, 50))
        bar(c, px, py - 7, img.shape[1], a.hunger / 100, (240, 200, 60))
        if not 0 <= a.x < self.cols:         # off screen: a marker on the edge it left by
            ex = 0 if a.x < 0 else self.W - 10
            ey = max(0, min(self.H - 30, int(ay * TILE_PX)))
            c[ey:ey + 30, ex:ex + 10] = (60, 200, 255, 255)
        if self.paused:
            c[8:40, 8:16] = c[8:40, 24:32] = (255, 255, 255, 230)
        self.win.show_pixels(c[::-1], self.left, self.top)
        return moving

    def run(self):
        frame, last_draw = 1.0 / FPS, 0.0
        while self.win.pump():
            t = time.time()
            self.check_keys()
            if not self.paused and t - self.tick_t >= 1.0 / (BRAIN_HZ * self.speed):
                self.tick_t = t
                self.brain_tick()
            # full frame rate only while something moves; otherwise twice a second for the idle bob
            if self.dirty or t < self.eat_until or t - last_draw >= 0.5:
                self.dirty = self.draw(t)
                last_draw = t
            self.win.keep_on_top()
            time.sleep(max(0.0, frame - (time.time() - t)))


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        Playground().run()
    except Exception:
        with open(os.path.join(LOG_DIR, "error.log"), "a", encoding="utf-8") as fh:
            fh.write(time.ctime() + "\n" + traceback.format_exc() + "\n")
        raise
