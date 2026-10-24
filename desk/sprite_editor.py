"""Sprite editor for the playground. Every sprite is listed with its exact size;
draw it pixel by pixel and Save: it goes to desk/art/<name>.png and the running
playground swaps it in within a second or two.

  left click / drag     paint
  right click / drag    erase (see-through)
  middle click          pick the colour under the cursor
  Ctrl+S save   Ctrl+Z undo   M mirror left-right   F flip upside down

Run:  Sprite Editor.bat   (or  py -3.12 desk/sprite_editor.py)"""
import os, sys, tkinter as tk
from tkinter import colorchooser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image

from sprites import CATALOGUE, PALETTE, ART_DIR, MAX_W, MAX_H, bake, load, SCALE

CELL = 26               # editor pixel size on screen
BG = "#121211"
INK = "#efece6"
ACCENT = "#f26b1d"


def hexcol(rgb):
    return "#%02x%02x%02x" % tuple(rgb[:3])


def start_pixels(name):
    """What the editor opens with: your saved png, else the built-in art, else thumb_up flipped."""
    img = load(name)
    if img is None and name == "thumb_down":
        img = load("thumb_up")[::-1]
    return img[::SCALE, ::SCALE].copy()          # back to 1 pixel per art pixel


class Editor:
    def __init__(self, root):
        self.root = root
        root.title("ATS sprite editor")
        root.configure(bg=BG)
        self.color = (40, 70, 170)
        self.undo = []

        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="y", padx=10, pady=10)
        tk.Label(left, text="Sprites", fg=INK, bg=BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.list = tk.Listbox(left, width=34, height=16, bg="#1b1b19", fg=INK, selectbackground=ACCENT,
                               font=("Consolas", 10), relief="flat", activestyle="none", exportselection=False)
        self.list.pack(fill="y", expand=True)
        self.names = list(CATALOGUE)
        for n in self.names:
            w, h = start_pixels(n).shape[1], start_pixels(n).shape[0]
            mark = "*" if os.path.exists(os.path.join(ART_DIR, n + ".png")) else " "
            self.list.insert("end", f"{mark} {n:<16} {w}x{h}")
        self.list.bind("<<ListboxSelect>>", lambda e: self.open(self.names[self.list.curselection()[0]]))
        tk.Label(left, text="* = your own drawing saved", fg="#6d6a64", bg=BG, font=("Segoe UI", 9)).pack(anchor="w")

        mid = tk.Frame(root, bg=BG)
        mid.pack(side="left", padx=10, pady=10)
        self.title = tk.Label(mid, text="", fg=INK, bg=BG, font=("Segoe UI", 12, "bold"))
        self.title.pack(anchor="w")
        self.hint = tk.Label(mid, text="", fg="#a19d95", bg=BG, font=("Segoe UI", 9))
        self.hint.pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(mid, bg="#2c2b28", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        for b, fn in (("<Button-1>", self.paint), ("<B1-Motion>", self.paint),
                      ("<Button-3>", self.erase), ("<B3-Motion>", self.erase), ("<Button-2>", self.pick)):
            self.canvas.bind(b, fn)
        self.canvas.bind("<ButtonPress-1>", self.stroke_start, add="+")
        self.canvas.bind("<ButtonPress-3>", self.stroke_start, add="+")

        size = tk.Frame(mid, bg=BG)
        size.pack(anchor="w", pady=6)
        tk.Label(size, text="Size", fg=INK, bg=BG).pack(side="left")
        self.wv, self.hv = tk.IntVar(), tk.IntVar()
        for v, top in ((self.wv, MAX_W), (self.hv, MAX_H)):
            tk.Spinbox(size, from_=4, to=top, width=3, textvariable=v, command=self.resize).pack(side="left", padx=3)
        self.size_note = tk.Label(size, text="", fg="#a19d95", bg=BG, font=("Segoe UI", 9))
        self.size_note.pack(side="left", padx=6)

        right = tk.Frame(root, bg=BG)
        right.pack(side="left", fill="y", padx=10, pady=10)
        tk.Label(right, text="Colours", fg=INK, bg=BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        pal = tk.Frame(right, bg=BG)
        pal.pack(anchor="w")
        cols = [c for c in PALETTE.values() if c]
        for i, c in enumerate(cols):
            tk.Button(pal, bg=hexcol(c), activebackground=hexcol(c), width=2, relief="flat",
                      command=lambda c=c: self.set_color(c)).grid(row=i // 4, column=i % 4, padx=1, pady=1)
        tk.Button(right, text="Any colour…", command=self.choose).pack(anchor="w", pady=4)
        self.swatch = tk.Label(right, text="  current  ", fg="white")
        self.swatch.pack(anchor="w", pady=(0, 10))

        tk.Label(right, text="In the game (real size)", fg=INK, bg=BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.preview = tk.Canvas(right, width=140, height=90, bg="#3a3a3a", highlightthickness=0)
        self.preview.pack(anchor="w", pady=4)
        for label, fn in (("Save  (Ctrl+S)", self.save), ("Undo  (Ctrl+Z)", self.do_undo),
                          ("Mirror  (M)", lambda: self.edit(lambda p: p[:, ::-1])),
                          ("Flip  (F)", lambda: self.edit(lambda p: p[::-1])),
                          ("Clear", lambda: self.edit(lambda p: np.zeros_like(p))),
                          ("Back to the built-in one", self.revert)):
            tk.Button(right, text=label, width=24, command=fn).pack(anchor="w", pady=2)
        self.status = tk.Label(right, text="", fg=ACCENT, bg=BG, wraplength=200, justify="left")
        self.status.pack(anchor="w", pady=8)

        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-z>", lambda e: self.do_undo())
        root.bind("m", lambda e: self.edit(lambda p: p[:, ::-1]))
        root.bind("f", lambda e: self.edit(lambda p: p[::-1]))
        self.set_color(self.color)
        self.list.selection_set(0)
        self.open(self.names[0])

    # ── document ──
    def open(self, name):
        self.name, self.px, self.undo = name, start_pixels(name), []
        self.builtin_size = (bake(CATALOGUE[name][0]).shape[1] // SCALE, bake(CATALOGUE[name][0]).shape[0] // SCALE) \
            if CATALOGUE[name][0] else None
        self.title.config(text=name)
        self.hint.config(text=CATALOGUE[name][1])
        self.status.config(text="")
        self.sync_size()
        self.redraw()

    def sync_size(self):
        h, w = self.px.shape[:2]
        self.wv.set(w)
        self.hv.set(h)
        bw, bh = self.builtin_size or (w, h)
        self.size_note.config(text=f"{w} x {h} art pixels = {w * SCALE} x {h * SCALE} on screen"
                                   + ("" if (w, h) == (bw, bh) else f"   (built-in is {bw} x {bh})"))
        self.canvas.config(width=w * CELL, height=h * CELL)

    def resize(self):
        w, h = max(4, min(MAX_W, self.wv.get())), max(4, min(MAX_H, self.hv.get()))
        old = self.px
        new = np.zeros((h, w, 4), np.uint8)
        # keep the drawing standing on the floor, centred: sprites stand on their bottom row
        oh, ow = old.shape[:2]
        ch, cw = min(h, oh), min(w, ow)
        new[h - ch:, (w - cw) // 2:(w - cw) // 2 + cw] = old[oh - ch:, (ow - cw) // 2:(ow - cw) // 2 + cw]
        self.undo.append(self.px)
        self.px = new
        self.sync_size()
        self.redraw()

    def edit(self, fn):
        self.undo.append(self.px)
        self.px = np.ascontiguousarray(fn(self.px))
        self.redraw()

    def do_undo(self):
        if self.undo:
            self.px = self.undo.pop()
            self.sync_size()
            self.redraw()

    def revert(self):
        path = os.path.join(ART_DIR, self.name + ".png")
        if os.path.exists(path):
            os.replace(path, path + ".old")          # kept, not deleted: rename it back to undo
        self.open(self.name)
        self.status.config(text="Back to the built-in one. Your drawing was kept as " + self.name + ".png.old")
        self.refresh_list()

    def save(self):
        os.makedirs(ART_DIR, exist_ok=True)
        path = os.path.join(ART_DIR, self.name + ".png")
        tmp = path + ".tmp"
        Image.fromarray(self.px, "RGBA").save(tmp, "PNG")
        os.replace(tmp, path)                        # the playground never reads half a file
        self.status.config(text="Saved. The playground picks it up in a second or two.")
        self.refresh_list()

    def refresh_list(self):
        i = self.names.index(self.name)
        h, w = self.px.shape[:2]
        mark = "*" if os.path.exists(os.path.join(ART_DIR, self.name + ".png")) else " "
        self.list.delete(i)
        self.list.insert(i, f"{mark} {self.name:<16} {w}x{h}")
        self.list.selection_set(i)

    # ── painting ──
    def cell(self, e):
        x, y = e.x // CELL, e.y // CELL
        h, w = self.px.shape[:2]
        return (x, y) if 0 <= x < w and 0 <= y < h else None

    def stroke_start(self, e):
        self.undo.append(self.px.copy())
        self.undo = self.undo[-50:]

    def paint(self, e):
        c = self.cell(e)
        if c:
            self.px[c[1], c[0]] = (*self.color, 255)
            self.redraw()

    def erase(self, e):
        c = self.cell(e)
        if c:
            self.px[c[1], c[0]] = 0
            self.redraw()

    def pick(self, e):
        c = self.cell(e)
        if c and self.px[c[1], c[0], 3]:
            self.set_color(tuple(int(v) for v in self.px[c[1], c[0], :3]))

    def set_color(self, c):
        self.color = tuple(c)
        self.swatch.config(bg=hexcol(c))

    def choose(self):
        got = colorchooser.askcolor(color=hexcol(self.color))
        if got and got[0]:
            self.set_color(tuple(int(v) for v in got[0]))

    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        h, w = self.px.shape[:2]
        for y in range(h):
            for x in range(w):
                p = self.px[y, x]
                if p[3]:
                    fill = hexcol(p)
                else:                                # checkerboard = see-through
                    fill = "#3a3936" if (x + y) % 2 else "#2c2b28"
                cv.create_rectangle(x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL, fill=fill, outline="#1b1b19")
        # where the sprite meets the ground, and a 32 px game tile, in game pixels
        cv.create_line(0, h * CELL - 1, w * CELL, h * CELL - 1, fill=ACCENT)
        self.draw_preview()

    def draw_preview(self):
        """Real game size: SCALE screen pixels per art pixel, standing on a 32 px tile."""
        pv = self.preview
        pv.delete("all")
        h, w = self.px.shape[:2]
        ox, floor = 70 - w * SCALE // 2, 80
        pv.create_rectangle(70 - 16, floor - 32, 70 + 16, floor, outline="#555")      # one tile
        for y in range(h):
            for x in range(w):
                p = self.px[y, x]
                if p[3]:
                    X, Y = ox + x * SCALE, floor - (h - y) * SCALE
                    pv.create_rectangle(X, Y, X + SCALE, Y + SCALE, fill=hexcol(p), width=0)


if __name__ == "__main__":
    root = tk.Tk()
    Editor(root)
    root.mainloop()
