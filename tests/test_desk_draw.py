"""Draws the playground off screen (no window): the walk-behind trick, stacking,
solid bases, dragging. Saves a picture to look at in %TEMP%/desk_draw.png."""
import os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "desk"))
import numpy as np
import playground
from PIL import Image

class FakeWin:
    def show_pixels(self, rgba, x, y): self.last = rgba[::-1].copy()

p = playground.Playground.__new__(playground.Playground)
p.left = p.top = 0; p.W, p.H = 640, 320; p.cols, p.rows = 20, 10
p.mind = playground.DeskMind(p.cols, p.rows); p.sheet = playground.Sheet(); p.win = FakeWin()
p.setup(); p.ear = playground.Ear()
w, a = p.mind.world, p.mind.agent
# a tree one row below the brain: its crown must cover the brain's head
a.x, a.y = 5, 5; p.bot.jump(5, 5)
w.put(5, 6, "0021")
assert not w.in_bounds_passable(5, 6), "a tree's base should block"
assert w.in_bounds_passable(5, 4), "the tile above the tree is free"
# stacking: the apple made after the bush is drawn over it
w.put(9, 5, "0023"); w.put(10, 5, "0001")
assert [o[2] for o in w.objects()][-2:] == ["0023", "0001"]
# dragging keeps the stack order
assert w.move((10, 5), (12, 5)) and w.objects()[-1] == (12, 5, "0001")
# the player can't share the brain's tile, and the brain can't walk into the player
p.drop(("player",), (3, 5)); assert (3, 5) in w.blocked and not w.in_bounds_passable(3, 5)
t0 = time.perf_counter()
for _ in range(20):
    p.draw(time.time())
print("draw ms", round((time.perf_counter() - t0) / 20 * 1000, 1))
img = p.win.last; bg = np.zeros_like(img); bg[..., :3] = 70; bg[..., 3] = 255
out = np.where(img[..., 3:4] > 0, img, bg)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "."), "desk_draw.png")
Image.fromarray(out[..., :3]).resize((1280, 640), Image.NEAREST).save(path)
print("ok")

# windows as rooms: top/bottom edges are walls, sides are doorways, a moved window carries the brain
p.rooms = playground.Rooms(0, 0, 32, p.cols, p.rows)
p.rooms.refresh([(1, 32 * 8, 32 * 1, 32 * 14, 32 * 5)])      # tiles x 8..13, walls on rows 1 and 4
w.set_walls(p.rooms.walls())
assert not w.in_bounds_passable(10, 1) and not w.in_bounds_passable(10, 4), "top/bottom are walls"
assert w.in_bounds_passable(7, 2) and w.in_bounds_passable(8, 2), "the side is a doorway"
p.drop(("bot",), (10, 1))                                     # dropped on the top edge: lands inside or next to it
p.place_bot(10, 2); assert p.rooms.room_of(10, 2) == 1
bot_in = p.rooms.room_of(a.x, a.y)
moved = p.rooms.refresh([(1, 32 * 10, 32 * 3, 32 * 16, 32 * 7)])  # window dragged 2 right, 2 down
w.set_walls(p.rooms.walls())
dx, dy = moved[bot_in]; p.place_bot(*w.free_near(a.x + dx, a.y + dy))
assert (a.x, a.y) == (12, 4) and p.rooms.room_of(12, 4) == 1, (a.x, a.y)
assert w.in_bounds_passable(10, 1), "the old wall is gone"
print("rooms ok")
