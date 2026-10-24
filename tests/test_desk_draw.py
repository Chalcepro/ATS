"""Draws the playground off screen (no window) and saves a picture to look at."""
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
p.canvas = np.zeros((p.H, p.W, 4), np.uint8); p.paused = False; p.speed = 1
p.prev = p.now = (p.mind.agent.x, p.mind.agent.y); p.facing_left = False; p.eat_until = 0; p.tick_t = time.time()
for i, item in enumerate(playground.SPAWNABLE.values()):
    p.mind.world.put(2 + i * 2, 3, item)
t0 = time.perf_counter()
for _ in range(20):
    p.brain_tick(); p.draw(time.time())
print("tick+draw ms", round((time.perf_counter() - t0) / 20 * 1000, 1))
img = p.win.last; bg = np.zeros_like(img); bg[..., :3] = 70; bg[..., 3] = 255
out = np.where(img[..., 3:4] > 0, img, bg)
Image.fromarray(out[..., :3]).resize((1280, 640), Image.NEAREST).save(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "."), "desk_draw.png"))
print("ok")
