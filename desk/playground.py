"""The ATS playground: the trained brain lives on the desktop as a pixel
mini-Rewrite, next to a player Chalce drives. Shift+A while the DESKTOP is
focused (not in any app) opens the menu: spawn food and objects under the
cursor, remove them, call it back when it has wandered off screen, pause.

  drag        any item, the brain or the player, with the mouse
  arrows      move the player (only while the desktop is focused)
  E           player picks up the thing it stands on or faces / puts it down
  F           player breaks the thing it stands on or faces (it's gone)
  right-click an item to delete it
  Ctrl+M      mute / unmute the ear (the microphone); starts muted

Yes / no (desk/voice.py): the menu has yes and no buttons, a box to type a
word, and a box to ask it a question; it answers thumbs up or down from its
voice head. Every word is logged for sleep to learn from; nothing is learned
while it's awake.

Open windows are rooms: their top and bottom edges are walls, their sides are
doorways. Drop the brain on a window to put it inside; move the window and it
rides along.

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

sys.path.insert(0, os.path.dirname(HERE))
from ear import Ear
import window
from window import OverlayWindow, user32
from sprites import Sheet, paste, bar, art_stamp, text, text_width, SCALE
from desk_world import DeskMind, SPAWNABLE, TILE_PX, brain_stamp
from window_rooms import Rooms
from voice import Voice
from describe import DescribeHead

LOG_DIR = os.path.join(os.environ.get("APPDATA", HERE), "ATSPlayground")
FPS = 15
BRAIN_HZ = 5                        # brain ticks a second at 1x speed
BRAIN_CHECK_S = 30                  # how often to look for a newer brain.pt from training
PLAYER_HZ = 8                       # player steps a second while an arrow is held
ROOMS_EVERY = 0.5                   # seconds between re-reading the open windows
VK_SHIFT, VK_CTRL, VK_A, VK_E, VK_F, VK_M = 0x10, 0x11, 0x41, 0x45, 0x46, 0x4D
THUMB_S = 3.0                       # how long its thumbs-up / down stays up
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


def type_box(prompt, x, y):
    """A small box at the cursor to type into. Enter sends, Esc cancels. Returns the text or None."""
    import tkinter as tk
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#121211")
    root.geometry(f"+{max(0, x - 140)}+{max(0, y - 70)}")
    tk.Label(root, text=prompt, fg="#efece6", bg="#121211", font=("Segoe UI", 10)).pack(padx=10, pady=(8, 2), anchor="w")
    e = tk.Entry(root, width=34, font=("Segoe UI", 12), bg="#1b1b19", fg="#efece6", insertbackground="#f26b1d", relief="flat")
    e.pack(padx=10, pady=(0, 10))
    out = []
    e.bind("<Return>", lambda _: (out.append(e.get().strip()), root.destroy()))
    e.bind("<Escape>", lambda _: root.destroy())
    root.after(50, lambda: (root.focus_force(), e.focus_set()))
    root.mainloop()
    return out[0] if out and out[0] else None


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
        self.rooms = Rooms(self.left, self.top, TILE_PX, self.cols, self.rows)
        self.ear = Ear()
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
        self.rooms_t = 0.0
        self.brain_t = time.time()
        self.voice = Voice(self.mind.policy.hidden_size)
        self.thumb = (0.0, 0)                   # (until, +1 up / -1 down): its answer
        self.heard = (0.0, 0)                   # (until, +1/-1/0): a flash that your word landed
        self.last_word = ""
        self.puffs = []                         # (until, x, y) little clouds where something broke
        self.show_words = True                  # the narrator's words above its head
        self.describe = DescribeHead.load(self.mind.policy.hidden_size)
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
        elif name == "right":
            hit = self.hit(*pos)
            if hit and hit[0] == "item":
                self.destroy(*hit[1])
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

    def destroy(self, x, y):
        if self.mind.world.remove(x, y):
            self.puffs.append((time.time() + 0.4, x, y))
            self.dirty = True

    def drop(self, what, tile):
        w, a = self.mind.world, self.mind.agent
        tx, ty = max(0, min(self.cols - 1, tile[0])), max(0, min(self.rows - 1, tile[1]))
        if what[0] == "item":
            w.move(what[1], (tx, ty))
        elif what[0] == "bot":
            self.place_bot(*w.free_near(tx, ty))
        elif what[0] == "player":
            self.place_player(*w.free_near(tx, ty))

    def place_bot(self, x, y):
        a = self.mind.agent
        if (x, y) != self.player.now:
            a.x, a.y = x, y
            self.bot.jump(x, y)

    def place_player(self, x, y):
        a = self.mind.agent
        if (x, y) != (a.x, a.y):
            self.player.jump(x, y)
            self.mind.world.blocked = {(x, y)}

    def reload_brain_if_newer(self):
        """Training saved a better brain: swap it in, keep the desk as it is."""
        stamp = brain_stamp()
        if stamp is None or stamp == self.mind.brain_stamp:
            return
        try:
            self.mind.load_brain()
        except Exception:
            return                          # caught it mid-save; try again next check
        h = self.mind.policy.hidden_size
        if self.voice.head.out.in_features != h:      # it grew: the side heads start over
            self.voice = Voice(h)
            self.describe = DescribeHead.load(h)
        self.brain_loaded = time.time()

    # ── windows as rooms ──
    def update_rooms(self):
        w, a = self.mind.world, self.mind.agent
        r = self.rooms
        bot_in, player_in = r.room_of(a.x, a.y), r.room_of(*self.player.now)
        moved = r.refresh()
        w.set_walls(r.walls())
        # whoever is inside a window that moved goes with it
        if bot_in in moved:
            dx, dy = moved[bot_in]
            self.place_bot(*w.free_near(a.x + dx, a.y + dy))
        if player_in in moved:
            dx, dy = moved[player_in]
            self.place_player(*w.free_near(self.player.now[0] + dx, self.player.now[1] + dy))
        # a window that opened on top of someone: step them off the wall
        if (a.x, a.y) in w.walls:
            self.place_bot(*w.free_near(a.x, a.y))
        if self.player.now in w.walls:
            self.place_player(*w.free_near(*self.player.now))
        self.dirty = self.dirty or bool(moved)

    # ── the brain ──
    def brain_tick(self, t):
        m = self.mind
        hunger0 = m.agent.hunger
        # its ears are your microphone (H2): the same 34 numbers the island's
        # sounds make, no direction (one mic); silence while muted
        m.world.hearing_override = np.concatenate([self.ear.vec, [0.0, 0.0]])
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
        down = {vk for vk in (VK_A, VK_E, VK_F, VK_M, *ARROWS) if key_down(vk)}
        pressed, self.keys_was = down - self.keys_was, down
        if VK_A in pressed and key_down(VK_SHIFT):
            self.open_menu()
            return
        if VK_M in pressed and key_down(VK_CTRL):
            self.ear.set_muted(not self.ear.muted)
            self.dirty = True
            return
        if VK_E in pressed:
            self.player_use()
        if VK_F in pressed:
            here = self.player.now
            for spot in (here, (here[0] + self.face[0], here[1] + self.face[1])):
                if self.mind.world.tiles.get(spot) and self.mind.world.tiles[spot].object_id:
                    self.destroy(*spot)
                    break
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

    # ── yes / no ──
    def say(self, text):
        """Chalce said something: a question gets a thumbs answer, anything else is a word."""
        t = time.time()
        self.last_word = text
        if text.endswith("?"):
            self.thumb = (t + THUMB_S, self.voice.ask(self.mind.hidden, text))
        else:
            self.heard = (t + 1.0, self.voice.hear_word(text, list(self.mind.recent)))
        self.dirty = True

    def open_menu(self):
        m, a = self.mind, self.mind.agent
        cx, cy = window.cursor_pos()
        tx, ty = self.tile_at(cx - self.left, cy - self.top)
        on_screen = 0 <= a.x < self.cols
        has_thing = bool(m.world.tiles.get((tx, ty)) and m.world.tiles[(tx, ty)].object_id)
        spawn = [(100 + i, label, False) for i, label in enumerate(SPAWNABLE)]
        items = [
            (1, f"Health {a.health:.0f}  ·  Hunger {a.hunger:.0f}  ·  ate {m.ate}×  ·  doing: {m.last_action}", False, False),
            (43, "Brain: " + time.strftime("%H:%M", time.localtime(m.brain_stamp or 0))
             + " (reloads itself when training saves)", False, False),
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
            (20, ("Unmute the ear" if self.ear.muted else "Mute the ear") + "   Ctrl+M", False),
            (40, "Show the words it's told", self.show_words),
            (41, "Told: " + (" ".join(getattr(a, "last_words", [])) or "-"), False, False),
            (42, "It would say: " + (" ".join(self.describe.say(m.hidden)) or "-")
             if self.describe.trained else "It would say: (run tools/train_describe.py)", False, False),
            (21, "Ear: " + (self.ear.error or "muted") if self.ear.muted else
             f"Ear: loudness {self.ear.loudness():.3f}  ·  pitch {self.ear.pitch_hz():.0f} Hz", False, False),
            None,
            (30, "👍  Yes  (good, keep doing that)", False),
            (31, "👎  No  (stop that)", False),
            (32, "Type a word to it…", False),
            (33, "Ask it a yes/no question…", False),
            (34, "Last word: " + (f'"{self.last_word}"' if self.last_word else "none yet"), False, False),
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
        elif pick == 20:
            self.ear.set_muted(not self.ear.muted)
        elif pick == 40:
            self.show_words = not self.show_words
        elif pick == 30:
            self.say("yes")
        elif pick == 31:
            self.say("no")
        elif pick in (32, 33):
            text = type_box("Say something to it (yes / no / anything):" if pick == 32 else
                            "Ask it a yes/no question:", cx, cy)
            if text:
                self.say(text + ("?" if pick == 33 and not text.endswith("?") else ""))
        if pick == 99:
            self.ear.close()
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

        chars, moving, words_at = [], False, None
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
                self.draw_ear(c, px + img.shape[1] + 4, py)
                a = self.mind.agent
                bar(c, px, py - 12, img.shape[1], a.health / 100, (220, 50, 50))
                # hunger is not shown (Chalce, 10-24): it's in the menu line if you need it
                self.draw_bubbles(c, px + img.shape[1] // 2, py - 9, t)
                words_at = (px + img.shape[1] // 2, py - 26)     # drawn last, over everything
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

        if self.show_words and words_at:
            self.draw_words(c, *words_at)

        if dragged:                          # what's being dragged rides on the cursor, on top
            img = {"bot": s.rewrite["idle"], "player": s.player}.get(dragged[0]) \
                if dragged[0] != "item" else s.item(self.mind.world.tiles[dragged[1]].object_id)
            mx, my = self.drag[1]
            paste(c, img, mx - img.shape[1] // 2, my - img.shape[0] // 2)

        self.puffs = [p for p in self.puffs if p[0] > t]
        for until, x, y in self.puffs:       # a dotted grey cloud that grows as it fades
            r = int(6 + (0.4 - (until - t)) * 30)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            cloud = np.zeros((2 * r + 1, 2 * r + 1, 4), np.uint8)
            cloud[(xx * xx + yy * yy <= r * r) & ((xx + yy) % 3 == 0)] = (200, 200, 200, 220)
            paste(c, cloud, x * TILE_PX + TILE_PX // 2 - r, y * TILE_PX + TILE_PX // 2 - r)
        a = self.mind.agent
        if not 0 <= a.x < self.cols:         # off screen: a marker on the edge it left by
            ex = 0 if a.x < 0 else self.W - 10
            ey = max(0, min(self.H - 30, int(self.bot.now[1] * TILE_PX)))
            c[ey:ey + 30, ex:ex + 10] = (60, 200, 255, 255)
        if self.paused:
            c[8:40, 8:16] = c[8:40, 24:32] = (255, 255, 255, 230)
        self.win.show_pixels(c[::-1], self.left, self.top)
        return moving or bool(self.drag) or bool(self.puffs) or t < max(self.thumb[0], self.heard[0])

    def draw_bubbles(self, c, cx, top, t):
        """Above its head: its thumbs answer, a flash when your word lands, an apple when it eats."""
        s = self.sheet
        if t < self.thumb[0]:
            img = s.thumb_up if self.thumb[1] > 0 else s.thumb_down
            paste(c, img, cx - img.shape[1] // 2, top - img.shape[0])
        elif t < self.heard[0]:
            col = {1: (80, 220, 100), -1: (230, 60, 60), 0: (160, 160, 160)}[self.heard[1]]
            y = top - 14
            if y >= 4 and cx >= 6:
                c[y:y + 4, cx - 6:cx + 6] = (*col, 255)              # a minus for no...
                if self.heard[1] > 0:
                    c[y - 4:y + 8, cx - 2:cx + 2] = (*col, 255)      # ...a plus for yes
        elif t < self.eat_until:
            img = s.item("0001")
            paste(c, img, cx - img.shape[1] // 2, top - img.shape[0] - int((self.eat_until - t) * 10))

    def draw_words(self, c, cx, y):
        """Above its head, small: the narrator's words, danger in red, food in green."""
        words = getattr(self.mind.agent, "last_words", [])
        if not words:
            return
        line = " ".join(words)
        x = cx - text_width(line) // 2
        c[max(0, y - 2):y + 12, max(0, x - 3):min(self.W, x + text_width(line) + 1)] = (18, 18, 17, 170)
        for w in words:
            col = {"DANGER": (240, 80, 70), "GROWL": (240, 80, 70), "HURT": (240, 80, 70),
                   "FOOD": (110, 220, 110), "HUNGRY": (240, 200, 60)}.get(w, (230, 230, 230))
            x += text(c, x, y, w, col) + 8

    def draw_ear(self, c, x, y):
        """Beside its head: grey with a red slash when muted, else a live loudness meter."""
        H, W = c.shape[:2]
        if not (0 <= x < W - 8 and 0 <= y < H - 24):
            return
        if self.ear.muted:
            c[y + 6:y + 18, x:x + 6] = (120, 120, 120, 230)
            for i in range(12):
                c[y + 6 + i, x + 6 - i // 2 - 1:x + 7 - i // 2] = (220, 40, 40, 255)
            return
        level = min(1.0, self.ear.loudness() * 8)
        c[y:y + 24, x:x + 6] = (20, 20, 20, 200)
        h = int(22 * level)
        if h:
            c[y + 23 - h:y + 23, x + 1:x + 5] = (90, 200, 255, 255)

    def run(self):
        frame, last_draw = 1.0 / FPS, 0.0
        while self.win.pump():
            t = time.time()
            self.check_keys(t)
            if not self.ear.muted:
                self.ear.update(t)
                self.dirty = True       # the meter moves
            if t - self.rooms_t >= ROOMS_EVERY:
                self.rooms_t = t
                self.update_rooms()
                if t - self.brain_t >= BRAIN_CHECK_S:
                    self.brain_t = t
                    self.reload_brain_if_newer()
                if art_stamp() != self.sheet.stamp:     # a sprite was saved in the editor
                    self.sheet = Sheet()
                    self.dirty = True
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
