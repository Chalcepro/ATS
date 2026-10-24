"""The ATS playground: the trained brain lives on the desktop as a pixel
mini-Rewrite, next to a player Chalce drives. Shift+A while the DESKTOP is
focused (not in any app) opens the menu: spawn food and objects under the
cursor, remove them, call it back when it has wandered off screen, pause.

  drag        any item, the brain or the player, with the mouse
  arrows      move the player (only while the desktop is focused)
  E           player picks up the thing it stands on or faces / puts it down

Items stack in the order they were made (first made at the back). Characters
use the top-down trick: an item whose base is lower on screen is drawn over
them, so they walk behind a tree's crown; only the base tile blocks.

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
from desk_world import DeskMind, SPAWNABLE, TILE_PX

LOG_DIR = os.path.join(os.environ.get("APPDATA", HERE), "ATSPlayground")
FPS = 15
BRAIN_HZ = 5                        # brain ticks a second at 1x speed
PLAYER_HZ = 8                       # player steps a second while an arrow is held
VK_SHIFT, VK_A, VK_E = 0x10, 0x41, 0x45
ARROWS = {0x25: (-1, 0), 0x26: (0, -1), 0x27: (1, 0), 0x28: (0, 1)}
DESKTOP_CLASSES = {"Progman", "WorkerW"}

user32.GetForegroundWindow.restype = wt.HWND
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.FindWindowW.restype = wt.HWND
user32.SetForegroundWindow.argtypes = [wt.HWND]


def desktop_focused():
    """True only when the focused window is the desktop itself (icons/wallpaper)."""
    buf = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(user32.GetForegroundWindow(), buf, 64)
    return buf.value in DESKTOP_CLASSES


def focus_desktop():
    """The menu takes focus to work; hand it back so the keys keep working."""
    h = user32.FindWindowW("Progman", None)
    if h:
        user32.SetForegroundWindow(h)


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class Glide:
    """A character's tile position, sliding from the last tile to the new one."""

    def __init__(self, x, y, hz):
        self.prev = self.now = (x, y)
        self.t, self.hz, self.left = 0.0, hz, False

    def go(self, x, y, t):
        if (x, y) != self.now:
            if x != self.now[0]:
                self.left = x < self.now[0]
            self.prev, self.now, self.t = self.now, (x, y), t

    def jump(self, x, y):
        self.prev = self.now = (x, y)

    def at(self, t):
        f = min(1.0, (t - self.t) * self.hz)
        return (self.prev[0] + (self.now[0] - self.prev[0]) * f,
                self.prev[1] + (self.now[1] - self.prev[1]) * f, f < 1.0 and self.prev != self.now)


class Playground:
    def __init__(self):
        window.dpi_aware()
        self.left, self.top, right, bottom = window.work_area()
        self.W, self.H = right - self.left, bottom - self.top
        self.cols, self.rows = self.W // TILE_PX, self.H // TILE_PX
        self.mind = DeskMind(self.cols, self.rows)
        self.sheet = Sheet()
        self.win = OverlayWindow(self.W, self.H, self.on_event)
        self.setup()

    def setup(self):
        """Everything but the window and the brain (the draw test builds these itself)."""
        self.canvas = np.zeros((self.H, self.W, 4), np.uint8)
        self.paused, self.speed = False, 1
        a = self.mind.agent
        self.bot = Glide(a.x, a.y, BRAIN_HZ)
        self.player = Glide(max(1, a.x - 3), a.y, PLAYER_HZ)
        self.mind.world.blocked = {self.player.now}
        self.carry = ""                         # what the player holds over its head
        self.face = (1, 0)
        self.eat_until = self.tick_t = self.step_t = 0.0
        self.keys_was = set()
        self.drag = None                        # ("item", tile) | ("bot",) | ("player",), plus the cursor
        self.dirty = True

    # ── tiles <-> pixels ──
    def tile_at(self, cx, cy):
        return int(cx // TILE_PX), int(cy // TILE_PX)

    def sprite_xy(self, img, tx, ty):
        """Top-left pixel of a sprite standing on tile (tx, ty): centred, feet on the tile's floor."""
        return int(tx * TILE_PX) + (TILE_PX - img.shape[1]) // 2, int(ty * TILE_PX) + TILE_PX - img.shape[0]

    # ── mouse ──
    def on_event(self, name, pos=None, **info):
        if name == "press":
            hit = self.hit(*pos)
            if hit:
                self.drag = (hit, pos)
        elif name == "move" and self.drag:
            self.drag = (self.drag[0], pos)
            self.dirty = True
        elif name == "release" and self.drag:
            self.drop(self.drag[0], self.tile_at(*pos))
            self.drag = None
            self.dirty = True

    def hit(self, x, y):
        """The topmost thing under the cursor: characters first, then items front to back."""
        t = time.time()
        for who, g, img in (("player", self.player, self.sheet.player), ("bot", self.bot, self.sheet.rewrite["idle"])):
            gx, gy, _ = g.at(t)
            px, py = self.sprite_xy(img, gx, gy)
            if px <= x < px + img.shape[1] and py <= y < py + img.shape[0]:
                return (who,)
        for ox, oy, item in reversed(self.mind.world.objects()):
            img = self.sheet.item(item)
            px, py = self.sprite_xy(img, ox, oy)
            if px <= x < px + img.shape[1] and py <= y < py + img.shape[0]:
                return ("item", (ox, oy))
        return None

    def drop(self, what, tile):
        w, a = self.mind.world, self.mind.agent
        tx, ty = max(0, min(self.cols - 1, tile[0])), max(0, min(self.rows - 1, tile[1]))
        if what[0] == "item":
            w.move(what[1], (tx, ty))
        elif what[0] == "bot" and w.in_bounds_passable(tx, ty):
            a.x, a.y = tx, ty
            self.bot.jump(tx, ty)
        elif what[0] == "player" and w.in_bounds_passable(tx, ty) and (tx, ty) != (a.x, a.y):
            self.player.jump(tx, ty)
            w.blocked = {(tx, ty)}

    # ── the brain ──
    def brain_tick(self, t):
        m = self.mind
        hunger0 = m.agent.hunger
        m.step()
        a = m.agent
        self.bot.go(a.x, a.y, t)
        if a.hunger > hunger0 + 0.5:
            self.eat_until = t + 0.6
        self.dirty = True

    # ── keys: Shift+A, the player ──
    def check_keys(self, t):
        if not desktop_focused():
            self.keys_was = set()
            return
        down = {vk for vk in (VK_A, VK_E, *ARROWS) if key_down(vk)}
        pressed, self.keys_was = down - self.keys_was, down
        if VK_A in pressed and key_down(VK_SHIFT):
            self.open_menu()
            return
        if VK_E in pressed:
            self.player_use()
        held = [ARROWS[vk] for vk in ARROWS if vk in down]
        if held and (t - self.step_t >= 1.0 / PLAYER_HZ or any(vk in pressed for vk in ARROWS)):
            self.step_t = t
            dx, dy = held[-1]
            self.face = (dx, dy)
            x, y = self.player.now[0] + dx, self.player.now[1] + dy
            w, a = self.mind.world, self.mind.agent
            if 0 <= x < self.cols and 0 <= y < self.rows and w.in_bounds_passable(x, y) and (x, y) != (a.x, a.y):
                self.player.go(x, y, t)
                w.blocked = {(x, y)}
            self.dirty = True

    def player_use(self):
        w = self.mind.world
        here = self.player.now
        ahead = (here[0] + self.face[0], here[1] + self.face[1])
        if self.carry:
            for spot in (ahead, here):
                if w.on_land(*spot) and not w._tile(*spot).object_id and 0 <= spot[0] < self.cols:
                    w.put(*spot, self.carry)
                    self.carry = ""
                    break
        else:
            for spot in (here, ahead):
                t = w.tiles.get(spot)
                if t and t.object_id:
                    self.carry = t.object_id
                    w.remove(*spot)
                    break
        self.dirty = True

    def open_menu(self):
        m, a = self.mind, self.mind.agent
        cx, cy = window.cursor_pos()
        tx, ty = self.tile_at(cx - self.left, cy - self.top)
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
            (6, "Bring my player here", False),
            (7, "Resume" if self.paused else "Pause", False),
            ("Speed", [(10 + s, f"{s}x", self.speed == s) for s in (1, 2, 4)]),
            (8, "New life (reset)", False),
            None,
            (9, "Type a word to it (yes / no)  — next step", False, False),
            None,
            (99, "Quit playground", False),
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
            self.drop(("bot",), (tx, ty))
        elif pick == 6:
            self.drop(("player",), (tx, ty))
        elif pick == 7:
            self.paused = not self.paused
        elif pick in (11, 12, 14):
            self.speed = pick - 10
            self.bot.hz = BRAIN_HZ * self.speed
        elif pick == 8:
            m.reset()
            self.bot.jump(m.agent.x, m.agent.y)
            m.world.blocked = {self.player.now}
        if pick == 99:
            self.win.close()
        else:
            focus_desktop()
        self.dirty = True

    # ── drawing ──
    def draw(self, t):
        c, s = self.canvas, self.sheet
        c[:] = 0
        dragged = self.drag[0] if self.drag else None
        objs = [(x, y, i) for x, y, i in self.mind.world.objects() if dragged != ("item", (x, y))]

        def item_at(x, y, i):
            img = s.item(i)
            paste(c, img, *self.sprite_xy(img, x, y))

        for o in objs:                       # back to front: the order they were made in
            item_at(*o)

        chars, moving = [], False
        for who, g in (("bot", self.bot), ("player", self.player)):
            if dragged == (who,):
                continue
            gx, gy, mv = g.at(t)
            moving = moving or mv
            chars.append((gy, who, gx, mv, g.left))
        for gy, who, gx, mv, left in sorted(chars):     # higher on screen = further back
            if who == "bot":
                kind = "eat" if t < self.eat_until else ("walk" if mv and int(t * 8) % 2 else "idle")
                img = (s.rewrite_left if left else s.rewrite)[kind]
            else:
                img = s.player_left if left else s.player
            px, py = self.sprite_xy(img, gx, gy)
            py += 0 if mv else int(np.sin(t * 3 + (who == "player")) * 1.5) * SCALE
            paste(c, img, px, py)
            if who == "bot":
                a = self.mind.agent
                bar(c, px, py - 12, img.shape[1], a.health / 100, (220, 50, 50))
                bar(c, px, py - 7, img.shape[1], a.hunger / 100, (240, 200, 60))
            elif self.carry:
                held = s.item(self.carry)
                paste(c, held, px + (img.shape[1] - held.shape[1]) // 2, py - held.shape[0])
            # the top-down trick: items standing lower down cover the character
            x0, x1, y0 = px, px + img.shape[1], py
            for o in objs:
                if o[1] > gy:
                    oi = s.item(o[2])
                    ox, oy = self.sprite_xy(oi, o[0], o[1])
                    if ox < x1 and ox + oi.shape[1] > x0 and oy < py + img.shape[0] and oy + oi.shape[0] > y0:
                        paste(c, oi, ox, oy)

        if dragged:                          # what's being dragged rides on the cursor, on top
            img = {"bot": s.rewrite["idle"], "player": s.player}.get(dragged[0]) \
                if dragged[0] != "item" else s.item(self.mind.world.tiles[dragged[1]].object_id)
            mx, my = self.drag[1]
            paste(c, img, mx - img.shape[1] // 2, my - img.shape[0] // 2)

        a = self.mind.agent
        if not 0 <= a.x < self.cols:         # off screen: a marker on the edge it left by
            ex = 0 if a.x < 0 else self.W - 10
            ey = max(0, min(self.H - 30, int(self.bot.now[1] * TILE_PX)))
            c[ey:ey + 30, ex:ex + 10] = (60, 200, 255, 255)
        if self.paused:
            c[8:40, 8:16] = c[8:40, 24:32] = (255, 255, 255, 230)
        self.win.show_pixels(c[::-1], self.left, self.top)
        return moving or bool(self.drag)

    def run(self):
        frame, last_draw = 1.0 / FPS, 0.0
        while self.win.pump():
            t = time.time()
            self.check_keys(t)
            if not self.paused and t - self.tick_t >= 1.0 / (BRAIN_HZ * self.speed):
                self.tick_t = t
                self.brain_tick(t)
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
