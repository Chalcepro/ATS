"""Pixel art for the playground, drawn from text maps: one character per pixel,
scaled up with no smoothing. Cheap to draw (a numpy paste), nothing on the GPU.

The mini-Rewrite here is a stand-in until Chalce draws his own frames."""
import numpy as np

SCALE = 3   # every art pixel is SCALE x SCALE screen pixels

PALETTE = {
    ".": None,
    "k": (12, 12, 20), "b": (40, 70, 170), "B": (70, 110, 220), "w": (240, 240, 240),
    "r": (210, 30, 40), "R": (255, 90, 90), "g": (60, 160, 70), "G": (30, 110, 45),
    "n": (130, 80, 40), "N": (90, 55, 25), "s": (140, 140, 150), "S": (95, 95, 105),
    "y": (240, 200, 60), "o": (180, 70, 15), "O": (242, 107, 29), "c": (40, 40, 45), "p": (255, 200, 170), "l": (110, 200, 255),
}

# mini-Rewrite: dark blue spikes, red eyes, the wide grin
REWRITE = {
    "idle": """
....kk..kk..
...kbbkkbbk.
..kbbbbbbbbk
.kbbBBBBBBbk
kbbBkrBBkrBk
.kbBkkBBkkBk
.kbBBBBBBBBk
..kbwkwkwkbk
..kbbwkwkbk.
...kbbbbbk..
...kbkkkbk..
..kbbk.kbbk.
..kkk...kkk.
""",
    "walk": """
....kk..kk..
...kbbkkbbk.
..kbbbbbbbbk
.kbbBBBBBBbk
kbbBkrBBkrBk
.kbBkkBBkkBk
.kbBBBBBBBBk
..kbwkwkwkbk
..kbbwkwkbk.
...kbbbbbk..
...kbk.kbk..
..kbbk..kbk.
..kkk...kkk.
""",
    "eat": """
....kk..kk..
...kbbkkbbk.
..kbbbbbbbbk
.kbbBBBBBBbk
kbbBkrBBkrBk
.kbBkkBBkkBk
.kbBkkkkkkBk
..kbkrrrrkbk
..kbbkkkkbk.
...kbbbbbk..
...kbkkkbk..
..kbbk.kbbk.
..kkk...kkk.
""",
}

ITEMS = {
    "0001": """
....N.....
....Ng....
..rrNrr...
.rRrrrrr..
.rRrrrrr..
.rrrrrrr..
..rrrrr...
...rrr....
""",                                   # apple
    "0023": """
..GGgGG...
.GgrGgrG..
GgGGgGgGG.
GrgGgrgGG.
.GgGgGgG..
..GGNGG...
....N.....
""",                                   # berry bush
    "0008": """
........N.
.......N..
......n...
.....N....
....n.....
...N......
..n.......
""",                                   # stick
    "0010": """
.nnnnnnn..
nNnNnNnNn.
nnnnnnnnn.
nNnNnNnNn.
.nnnnnnn..
""",                                   # wood
    "0012": """
...ssss...
..sSssss..
.sssssSss.
.sSsssss..
..ssSss...
""",                                   # stone
    "0021": """
....GGGG....
..GGgGGgGG..
.GgGGGGGGgG.
GGGgGGgGGGGG
GgGGGGGGgGgG
GGGgGgGGGGGG
.GgGGGGGgGG.
GGGGGgGGGGgG
.GgGGGGGgGG.
..GGGgGGGG..
....GnNG....
.....nN.....
.....nN.....
.....nN.....
....nnNN....
""",                                   # oak tree
    "0024": """
..........
.NNNNNNNN.
NnnnnnnnnN
NnNnnNnnnN
.NNNNNNNN.
""",                                   # fallen log
    "0009": """
...cc.....
..cccck...
.ckccccc..
.cccckcc..
..ccccc...
""",                                   # coal
}
UNKNOWN = """
.yyyy.
y....y
....y.
...y..
......
...y..
"""

# the player Chalce drives: orange hood, the portfolio's accent colour
PLAYER = """
...oooo...
..oOOOOo..
.oOppppOo.
.oOpkpkOo.
.oOppppOo.
..oOOOOo..
.oOOOOOOo.
oOOOlOOOOo
.pOOlOOOp.
..oOOOOo..
..kk..kk..
..kk..kk..
"""

THUMB_UP = """
....kk....
...kyyk...
...kyyk...
.kkkyykkk.
kyyyyyyyyk
kyyyyyyyyk
kyyyyyyyyk
.kyyyyyyk.
..kkkkkk..
"""


def bake(art, scale=SCALE):
    """Text map -> (h, w, 4) premultiplied RGBA, top row first."""
    rows = [r for r in art.strip("\n").split("\n")]
    h, w = len(rows), max(len(r) for r in rows)
    img = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            col = PALETTE.get(ch)
            if col:
                img[y, x] = (*col, 255)
    return img.repeat(scale, 0).repeat(scale, 1)


class Sheet:
    def __init__(self):
        self.rewrite = {k: bake(v) for k, v in REWRITE.items()}
        self.rewrite_left = {k: v[:, ::-1] for k, v in self.rewrite.items()}
        self.items = {k: bake(v) for k, v in ITEMS.items()}
        self.unknown = bake(UNKNOWN)
        self.player = bake(PLAYER)
        self.player_left = self.player[:, ::-1]
        self.thumb_up = bake(THUMB_UP)
        self.thumb_down = self.thumb_up[::-1]

    def item(self, item_id):
        return self.items.get(item_id, self.unknown)


def paste(canvas, img, x, y):
    """Alpha-paste img (top row first) with its top-left at x, y; clipped to the canvas."""
    H, W = canvas.shape[:2]
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] > 0
    np.copyto(canvas[y0:y1, x0:x1], src, where=a)


def bar(canvas, x, y, w, frac, col):
    """A small 2-row bar: dark back, coloured fill."""
    H, W = canvas.shape[:2]
    if not (0 <= y < H - 4):
        return
    x0, x1 = max(x, 0), min(x + w, W)
    if x0 >= x1:
        return
    canvas[y:y + 4, x0:x1] = (20, 20, 20, 200)
    fx = min(x0 + int(w * max(0.0, min(1.0, frac))), x1)
    canvas[y + 1:y + 3, x0:fx] = (*col, 255)
