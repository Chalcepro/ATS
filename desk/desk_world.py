"""The desktop as an island: Chalce's screen cut into tiles, every tile land,
water past a hidden strip on each side. The brain gets the inputs it was
trained on, so what it learned on the island (find food, eat, explore) shows up
on the desktop with nothing retrained.

It is a real World with the islands wiped, and it runs through the real
ATSEnvironment.step, so hunger, eating and movement are the island's own rules.
The brain is a frozen copy (Sleep plan: nothing is learned while awake)."""
import os, sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import brain, config_rl
from ats_env import ATSEnvironment
from model_rl import RLPolicy
from mind.solution_loop import SolutionLoop
from world import World, Tile, TILE_EMPTY, TILE_OCEAN, TILE_OBJECT, TILE_WALL

TILE_PX = 32        # one island tile is this many screen pixels
HIDDEN = 3          # tiles of walkable land past each side edge (off screen)

# what the Shift+A menu can spawn: label -> item id from the island's registry
SPAWNABLE = {"Apple": "0001", "Berry bush": "0023", "Stick": "0008", "Wood": "0010",
             "Stone": "0012", "Oak tree": "0021", "Fallen log": "0024", "Coal": "0009"}
# only the base tile of these blocks walking (the top-down trick: you walk
# behind the tall part). On the island every object can be stepped on.
SOLID = {"0021", "0024", "0012"}


class DeskWorld(World):
    def __init__(self, cols, rows):
        super().__init__(seed=1)            # registry, items, bookkeeping; the islands get wiped below
        self.cols, self.rows = cols, rows
        self.tiles.clear()
        self.entities.clear()
        self._grid.clear()
        self.villages.clear()
        self._regrow.clear()
        self.agent_spawn = (cols // 2, rows - 2)
        self.order = {}             # (x, y) -> when it was made; first made is drawn at the back
        self.blocked = set()        # tiles nobody may step on (where the player stands)
        self.walls = set()          # window edges (window_rooms.py)
        self._made = 0

    def on_land(self, x, y):
        return -HIDDEN <= x < self.cols + HIDDEN and 0 <= y < self.rows

    def _tile(self, x, y):
        t = self.tiles.get((x, y))
        if t is None:
            t = Tile(tile_type=TILE_EMPTY, island_id=0) if self.on_land(x, y) else Tile(tile_type=TILE_OCEAN)
            self.tiles[(x, y)] = t
        return t

    def set_walls(self, walls):
        """Window top/bottom edges become wall tiles; an object already there keeps its tile."""
        for p in self.walls - walls:
            t = self.tiles.get(p)
            if t and t.tile_type == TILE_WALL:
                t.tile_type = TILE_EMPTY
        for p in walls:
            t = self._tile(*p)
            if not t.object_id and t.tile_type == TILE_EMPTY:
                t.tile_type = TILE_WALL
        self.walls = walls

    def free_near(self, x, y):
        """The nearest tile to (x, y) that can be stood on, searching outward."""
        for r in range(0, 6):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and self.in_bounds_passable(x + dx, y + dy)                             and self.on_land(x + dx, y + dy):
                        return x + dx, y + dy
        return x, y

    def get_island_id_at(self, x, y):
        return 0 if self.on_land(x, y) else -1

    def get_island_tier_at(self, x, y):
        return 0

    def in_bounds_passable(self, x, y, door_unlocked=True):
        t = self.tiles.get((x, y))
        if (t and t.object_id in SOLID) or (x, y) in self.blocked:
            return False
        return super().in_bounds_passable(x, y, door_unlocked)

    def materialise(self, island_id):
        return False

    def tick_life(self, ax=1 << 30, ay=1 << 30):
        return []           # nothing grows back on its own; Chalce spawns things

    def put(self, x, y, item_id):
        if not self.on_land(x, y):
            return False
        t = self._tile(x, y)
        t.object_id, t.tile_type = item_id, TILE_OBJECT
        self._made += 1
        self.order[(x, y)] = self._made
        return True

    def move(self, src, dst):
        """Drag an object to another free land tile; it keeps its place in the stack."""
        t = self.tiles.get(src)
        if src == dst or not (t and t.object_id) or not self.on_land(*dst) or self._tile(*dst).object_id:
            return False
        seq, item = self.order.get(src, 0), t.object_id
        self.remove(*src)
        self.put(*dst, item)
        self.order[dst] = seq
        return True

    def remove(self, x, y):
        t = self.tiles.get((x, y))
        if t and t.object_id:
            t.object_id, t.tile_type = "", TILE_EMPTY
            self.order.pop((x, y), None)
            return True
        return False

    def objects(self):
        """Every object as (x, y, id), back to front: the order they were made in."""
        objs = [(x, y, t.object_id) for (x, y), t in self.tiles.items() if t.object_id]
        return sorted(objs, key=lambda o: self.order.get((o[0], o[1]), 0))


class DeskMind:
    """The trained brain living in a DeskWorld. step() is one brain tick."""

    def __init__(self, cols, rows, path=None):
        shape = brain.saved_shape()
        self.policy = RLPolicy(hidden_size=int(shape.get("hidden_size", config_rl.MIND_HIDDEN_SIZE)))
        brain.load(self.policy, None, path=path)
        self.policy.eval()
        torch.set_grad_enabled(False)
        self.sl = SolutionLoop(policy=self.policy)
        self.env = ATSEnvironment(max_ticks=10 ** 9, vary_world=False, use_loop=False)
        self.cols, self.rows = cols, rows
        self.recent = deque(maxlen=10)      # its last 10 steps (2 s at 1x), for yes/no feedback
        self.ate = 0
        self.reset()

    def reset(self):
        self.env.reset()
        w = DeskWorld(self.cols, self.rows)
        env = self.env
        env.world = w
        env.agent.world = w
        env.agent.x, env.agent.y = w.agent_spawn
        env.memory.__init__()
        env.rewards.discovered_tiles = {w.agent_spawn}
        self.state = env._state()
        self.sl.reset_memory()
        self.last_action = "wait"
        self.done = False

    @property
    def hidden(self):
        """Its memory state right now (what the voice head answers from)."""
        hx = self.sl._hx
        return hx.reshape(-1).tolist() if hx is not None else [0.0] * self.policy.hidden_size

    @property
    def world(self):
        return self.env.world

    @property
    def agent(self):
        return self.env.agent

    def step(self):
        if self.done:               # it died: a new life, same desk
            objs = self.world.objects()
            self.reset()
            for x, y, i in objs:         # objects() is back to front, so the stack order survives
                self.world.put(x, y, i)
        env = self.env
        mask = env.agent.get_action_mask(env.world, env.day_night)
        before, hunger0 = self.state, env.agent.hunger
        action = self.sl.choose_action(self.state, mask)
        self.state, _, self.done, info = env.step(action)
        self.recent.append({"state": [round(v, 4) for v in before], "action": int(action)})
        self.ate += env.agent.hunger > hunger0 + 0.5
        from agent import ACTION_NAMES
        self.last_action = ACTION_NAMES.get(action, str(action))
        return info
