"""Staged environments: nursery -> primary -> junior.

Why this exists
---------------
The full ATS world asks the policy to learn perception, navigation, hunger
management, danger avoidance and progression *at the same time, from a cold
start*. 725 episodes of the last run ended in death 724 times. That is not a
learner that is failing; that is a learner that has never once seen the thing
it is supposed to be optimising toward.

So: small rooms first. The same network, the same state layout, the same
action indices — only the world gets simpler, and it grows back up as the
policy earns it.

The state vector fills the **real** ATS slots (config_rl.IDX_*), so weights
trained here transfer into the full world rather than needing a fresh head.

The reward rule these stages are built around
---------------------------------------------
In the full world the per-tick penalties can out-earn every positive in the
game. Standing against a wall costs -0.3 (wall hit) and -0.2 (standing still)
per tick; the survival bonus for lasting a whole episode is +10. An episode
that reached max ticks scored **-3281**, while dying outright costs -10.

An agent that learns "end the episode early" under that arithmetic has learned
correctly. The reward function was the bug.

Every stage here therefore obeys one rule, and ``best_case_return`` checks it:
**an agent playing well must finish clearly positive.** Penalties exist to
break ties between good behaviours, never to out-weigh the goal.
"""

from __future__ import annotations

import random

import config_rl

# Tile ids, chosen to match world.py so the tile embedding means the same
# thing here as it does in the real world.
T_FLOOR = 0
T_WALL = 1
T_HAZARD = 6          # TILE_LAVA in world.py

MOVES = {
    config_rl.ACT_MOVE_FORWARD:  (0, -1),
    config_rl.ACT_MOVE_BACKWARD: (0,  1),
    config_rl.ACT_MOVE_LEFT:     (-1, 0),
    config_rl.ACT_MOVE_RIGHT:    (1,  0),
}


class Stage:
    """One rung of the ladder.

    `pass_rate` and `window` are the promotion test: the fraction of the last
    `window` episodes that must count as a success before the policy moves up.
    Promotion is earned, never scheduled - a stage that is still being failed
    is a stage that still has something to teach.
    """

    def __init__(self, name, grid, *, walls=False, hazards=0, hostiles=0,
                 goals=1, respawn=True, can_pick=False, can_attack=False,
                 damage=False, max_steps=150, target=3,
                 pass_rate=0.80, window=50, grow_to=None, goals_scale=False,
                 hazard_damage=10, punitive=True, shaped=True, sequence=0):
        self.name = name
        self.grid = grid
        self.walls = walls
        self.hazards = hazards
        self.hostiles = hostiles
        self.goals = goals
        self.respawn = respawn
        self.can_pick = can_pick
        self.can_attack = can_attack
        self.damage = damage
        self.max_steps = max_steps
        # A trail: one goal visible at a time, the next appearing only when
        # the current one is taken, `sequence` of them, and the episode ends
        # when the trail is finished. Distinct from respawn, which drops a
        # fresh goal in forever and never terminates on success.
        #
        # The point is time pressure. With `sequence` set, failing to finish
        # the trail inside max_steps costs R_INCOMPLETE - so wandering has a
        # price for the first time on this ladder, and the shortest path is
        # worth something rather than merely tidier.
        self.sequence = int(sequence)
        self.target = target          # collects that count as a success
        self.pass_rate = pass_rate
        self.window = window
        self.grow_to = grow_to        # primary grows its maze as it is beaten
        # How much a hazard takes. At 20 the agent is dead in five mistakes,
        # which is not enough episodes-worth of contact to learn that tile 6
        # is the thing hurting it - the first run died 25-63% of the time and
        # never got above 2% success.
        self.hazard_damage = hazard_damage
        # God mode. In the nursery there is nothing to be hurt by and nothing
        # to lose: no time cost, no wall cost, no penalty for standing still,
        # and no shaping. The only way to score is to reach the goal.
        #
        # This is deliberate and it is the whole point of a first rung. The
        # question a nursery answers is "can it learn that the goal is good" -
        # and every penalty added alongside is another thing that has to be
        # disentangled from that one association before it can form. If the
        # agent wants to stand in a corner for an hour, that costs nothing;
        # it simply also earns nothing, and an episode ends eventually.
        self.punitive = punitive
        self.shaped = shaped
        # A bigger room gets proportionally more to find, so the reward on
        # offer grows with the walking required.
        if goals_scale:
            self.goals = max(goals, round(goals * grid / 5.0))

    @property
    def step_cost(self):
        if not self.punitive:
            return 0.0
        return EPISODE_TIME_COST / self.max_steps

    @property
    def still_cost(self):
        """Standing still has to cost clearly more than walking, or parking
        somewhere safe and running the clock out stays a viable policy - which
        is what the full world's agent actually learned to do."""
        if not self.punitive:
            return 0.0
        return (R_STILL_EPISODE / self.max_steps) * STILL_EVERY

    def grown(self):
        """The next size up, or None when this stage has nothing left to give."""
        if not self.grow_to or self.grid + 2 > self.grow_to:
            return None
        nxt = Stage(self.name, self.grid + 2, walls=self.walls,
                    hazards=min(self.hazards + 1, 4),
                    hostiles=self.hostiles, goals=self.goals, respawn=self.respawn,
                    can_pick=self.can_pick, can_attack=self.can_attack,
                    damage=self.damage, hazard_damage=self.hazard_damage,
                    # Room for the longer walk a bigger maze needs, or the
                    # episode ends before a good policy could have finished.
                    max_steps=int(self.max_steps * 1.5),
                    goals_scale=True,
                    target=self.target, pass_rate=self.pass_rate,
                    window=self.window, grow_to=self.grow_to)
        return nxt

    def __repr__(self):
        return "Stage(%s %dx%d)" % (self.name, self.grid, self.grid)


# ---- rewards -------------------------------------------------------------
#
# Deliberately few, deliberately lopsided towards the positive. The shaping
# term is what makes a cold-start policy find the goal at all: without it the
# first reward is however many random steps it takes to stumble onto one tile,
# which in a 9x9 maze is almost never.

R_GOAL = 2.0            # reaching the thing you were sent for
R_TOWARDS = 0.08        # per tile of progress, signed
# The time cost is derived per stage, not fixed - see Stage.step_cost. A
# constant per-step cost is the trap that made every maze above 5x5
# unlearnable on the first run of this file: the reward on offer stayed at
# +7 while a bigger maze took more steps to cross, so growing the room made
# the best possible score *worse*. Anything charged per step has to be
# expressed as a fraction of an episode, never as an absolute.
EPISODE_TIME_COST = -0.5   # what a whole episode of dawdling costs, total
R_STILL_EPISODE = -2.0  # what a whole episode of standing still costs
R_WALL = -0.02          # walking into a wall. Small: it is a mistake, not a sin
R_HAZARD = -1.0         # touching something that hurts
R_TRASH = -0.15         # picking up the wrong thing
R_GOOD = 0.5            # picking up a useful thing
R_DEATH = -3.0
R_CLEAR = 3.0           # collecting everything the room had
R_INCOMPLETE = -2.0     # running out of steps with a trail unfinished

STILL_GRACE = 4         # steps of stillness allowed before the penalty starts
STILL_EVERY = 2         # ...then it lands this often


def default_ladder(max_grid=12):
    """The ladder. Each rung adds **exactly one** new thing.

    The first version of this file went from nursery straight to "maze, plus
    hazards, plus damage, plus items to pick up, plus goals that do not come
    back" - five new problems at once. It trained to 32% and then *decayed* to
    10% over 2500 episodes.

    Which is precisely the criticism this whole file was written to make of
    the full world. Compounding difficulty is the thing that stops a policy
    learning, and it does not stop being that when the room is small.

    So: walls, then danger, then scarcity, then size, then something that
    fights back. One at a time, and each rung keeps everything the last one
    taught.
    """
    L = []

    # 1. A box and ONE goal, and the episode ends when it is reached.
    #
    #    This used to respawn: collect a goal, another appears somewhere
    #    random, repeat for 120 steps. Measured on 2026-09-18, that nursery
    #    finished with a goal-sensitivity of 0.0155 - indistinguishable from
    #    an untrained network - and its argmax followed the goal at chance.
    #    The reason is that the bearing at IDX_DIR_START points at whatever
    #    is nearest, and respawn moves that target at the exact instant of
    #    reward, severing the association the signal exists to teach.
    #
    #    One goal, ending the episode, took the same network to sensitivity
    #    0.6311, argmax 4/4, and a detour ratio of 1.98 in 60k steps.
    L.append(Stage("nursery", 5, walls=False, hazards=0, goals=1, respawn=False,
                   damage=False, punitive=False, shaped=False,
                   max_steps=120, target=1,
                   pass_rate=0.85, window=50))

    # A five-goal trail was built and measured here on 2026-09-18 and is
    # deliberately NOT in the ladder. `sequence` remains supported by Stage
    # and CurriculumEnv so it can be revisited, but as a rung it did harm:
    #
    #   nursery only       detour 1.98  sensitivity 0.6311  argmax 4.0/4
    #   trail only         detour 8.12  sensitivity 0.0210  argmax 1.0/4
    #   nursery -> trail   detour 3.96  sensitivity 0.1297  argmax 1.5/4
    #
    # Training the trail AFTER the nursery drags goal-sensitivity from 0.63
    # back to 0.13 and the argmax from 4/4 to chance - it un-teaches the
    # bearing. The cause is not an ambiguous bearing (only one goal is on the
    # floor at a time) but that the episode does not end at the reward: the
    # return from any state carries four more legs whose bearings are
    # unrelated, so the value function learns "how many legs remain" instead
    # of "how far to this goal". Signalling a GAE terminal at each leg was
    # tried and changed nothing (0.1279 vs 0.1297), so the fix is not simply
    # where the credit is cut.

    # 2. New thing: walls. Same goal, same respawn, still nothing that hurts.
    #
    #    NOTE: this rung and the three after it still use the respawn pattern
    #    that rung 1 was just changed away from, and the same measurement says
    #    respawn leaves goal-sensitivity indistinguishable from an untrained
    #    network. They are left alone pending a decision, because changing
    #    them is the ladder redesign that is on hold.
    L.append(Stage("corridors", 5, walls=True, hazards=0, goals=1, respawn=True,
                   damage=False, max_steps=160, target=3,
                   pass_rate=0.80, window=50))

    # 3. New thing: size. Same maze, same rules, more of it.
    L.append(Stage("corridors7", 7, walls=True, hazards=0, goals=1, respawn=True,
                   damage=False, max_steps=200, target=3,
                   pass_rate=0.75, window=50))

    # 4. New thing: something to avoid - but it cannot kill yet. The lesson
    #    here is "tile 6 is bad", and an agent that dies while learning it
    #    never gets enough contact with the tile to make the association. The
    #    first version turned damage on at the same moment the hazard first
    #    appeared, and the run died 25-63% of the time and never cleared 2%.
    L.append(Stage("avoid", 7, walls=True, hazards=3, goals=1, respawn=True,
                   damage=False, max_steps=200, target=3,
                   pass_rate=0.75, window=60))

    # 5. New thing: now it can kill you.
    L.append(Stage("hazards", 7, walls=True, hazards=3, goals=1, respawn=True,
                   damage=True, hazard_damage=10, max_steps=200, target=3,
                   pass_rate=0.70, window=60))

    # 4. New thing: scarcity. The goals no longer come back, so the room has to
    #    be cleared, and picking the wrong thing up costs.
    L.append(Stage("foraging", 7, walls=True, hazards=2, goals=3, respawn=False,
                   can_pick=True, damage=True, max_steps=240, target=2,
                   pass_rate=0.70, window=60))

    # 5. New thing: size. Everything above, in a room that keeps growing.
    L.append(Stage("primary", 9, walls=True, hazards=2, goals=4, respawn=False,
                   can_pick=True, damage=True, max_steps=300, target=3,
                   pass_rate=0.65, window=60, grow_to=max_grid))

    # 6. New thing: something in here with you, and running is no longer
    #    always the answer.
    L.append(Stage("junior", 9, walls=True, hazards=2, hostiles=1, goals=4,
                   respawn=False, can_pick=True, can_attack=True, damage=True,
                   max_steps=320, target=3, pass_rate=0.60, window=60,
                   grow_to=max_grid))
    return L


class CurriculumEnv:
    """One room, at whatever difficulty the stage says.

    Same interface as the sanity env and the real one: reset() -> state,
    step(a) -> (state, reward, done, info), action_mask().
    """

    def __init__(self, stage: Stage, seed: int | None = None):
        self.stage = stage
        self.rng = random.Random(seed)
        self.reset()

    # -- world building ----------------------------------------------------

    def _blank(self):
        n = self.stage.grid
        # A wall ring always, so "the edge" is a thing the agent can see in its
        # patch rather than an invisible rule it discovers by bumping.
        g = [[T_FLOOR] * n for _ in range(n)]
        for i in range(n):
            g[0][i] = g[n - 1][i] = T_WALL
            g[i][0] = g[i][n - 1] = T_WALL
        return g

    def _carve_maze(self, g):
        """Recursive backtracker on the odd cells. Every floor tile stays reachable."""
        n = self.stage.grid
        for y in range(1, n - 1):
            for x in range(1, n - 1):
                g[y][x] = T_WALL
        sx = sy = 1
        g[sy][sx] = T_FLOOR
        stack = [(sx, sy)]
        while stack:
            x, y = stack[-1]
            nbrs = []
            for dx, dy in ((0, -2), (0, 2), (-2, 0), (2, 0)):
                nx, ny = x + dx, y + dy
                if 1 <= nx < n - 1 and 1 <= ny < n - 1 and g[ny][nx] == T_WALL:
                    nbrs.append((nx, ny, dx, dy))
            if not nbrs:
                stack.pop()
                continue
            nx, ny, dx, dy = self.rng.choice(nbrs)
            g[y + dy // 2][x + dx // 2] = T_FLOOR
            g[ny][nx] = T_FLOOR
            stack.append((nx, ny))

        # A perfect maze has exactly one route anywhere, which makes a wrong
        # turn expensive and the credit assignment long. A few extra openings
        # keep it a maze without making it a punishment.
        for _ in range(n // 2):
            x = self.rng.randrange(1, n - 1)
            y = self.rng.randrange(1, n - 1)
            g[y][x] = T_FLOOR

    def _free_cells(self):
        return [(x, y)
                for y in range(self.stage.grid)
                for x in range(self.stage.grid)
                if self.grid[y][x] == T_FLOOR]

    def _place(self, taken, n):
        out = []
        free = [c for c in self._free_cells() if c not in taken]
        self.rng.shuffle(free)
        for c in free[:n]:
            out.append(c)
            taken.add(c)
        return out

    # -- episode -----------------------------------------------------------

    def reset(self):
        s = self.stage
        self.grid = self._blank()
        if s.walls:
            self._carve_maze(self.grid)

        free = self._free_cells()
        self.ax, self.ay = self.rng.choice(free)
        taken = {(self.ax, self.ay)}

        # A trail starts with exactly one goal on the floor however many it
        # will eventually ask for; the rest arrive as it is walked.
        self.goals = self._place(taken, 1 if s.sequence else s.goals)
        self.hazards = self._place(taken, s.hazards)
        for (hx, hy) in self.hazards:
            self.grid[hy][hx] = T_HAZARD
        self.trash = self._place(taken, 2 if s.can_pick else 0)
        self.good = self._place(taken, 1 if s.can_pick else 0)
        self.hostiles = self._place(taken, s.hostiles)

        self.health = 100
        self.steps = 0
        self.collected = 0
        self.still = 0
        self.last_pos = (self.ax, self.ay)
        self.dead = False
        return self._state()

    def action_mask(self):
        """Only the actions this stage has taught.

        Masking rather than letting the policy flail at 25 options is most of
        why the early stages are learnable at all: a random move in the nursery
        is useful one time in four, not one time in twenty-five.
        """
        m = [0] * config_rl.ACTION_SIZE
        for a in MOVES:
            m[a] = 1
        if self.stage.can_pick:
            m[config_rl.ACT_PICK_UP] = 1
        if self.stage.can_attack:
            m[config_rl.ACT_ATTACK] = 1
        return m

    # -- state -------------------------------------------------------------

    def _tile(self, x, y):
        if 0 <= x < self.stage.grid and 0 <= y < self.stage.grid:
            return self.grid[y][x]
        return T_WALL

    def _nearest(self, items):
        if not items:
            return None, 999
        best, bd = None, 999
        for (x, y) in items:
            d = abs(x - self.ax) + abs(y - self.ay)
            if d < bd:
                best, bd = (x, y), d
        return best, bd

    def _state(self):
        n = self.stage.grid
        span = max(1, 2 * (n - 1))
        s = [0.0] * config_rl.STATE_SIZE

        # Vitals, in the real slots
        s[0] = self.health / 100.0
        s[1] = 1.0                       # hunger: not a thing until later stages
        s[2] = 1.0                       # stamina
        s[3] = 0.0                       # xp
        s[4] = 0.0                       # level

        # Adjacent tiles (indices 10..13), same order as agent.build_state
        s[10] = float(self._tile(self.ax, self.ay - 1))
        s[11] = float(self._tile(self.ax - 1, self.ay))
        s[12] = float(self._tile(self.ax + 1, self.ay))
        s[13] = float(self._tile(self.ax, self.ay + 1))

        host, hd = self._nearest(self.hostiles)
        s[14] = min(1.0, hd / span)
        s[config_rl.IDX_ENT_TYPE] = 1.0 if host else 0.0

        goal, gd = self._nearest(self.goals)
        s[18] = min(1.0, gd / span)
        s[config_rl.IDX_OBJ_ID] = 1.0 / config_rl.ITEM_VOCAB_SIZE if goal else 0.0

        # The direction block. This is the single most useful thing in the
        # vector: without a signed bearing to the goal the policy has to infer
        # direction from the patch alone, which it can only do once the goal is
        # already within two tiles.
        d = config_rl.IDX_DIR_START
        if goal:
            s[d + 0] = (goal[0] - self.ax) / (n - 1)
            s[d + 1] = (goal[1] - self.ay) / (n - 1)
        if host:
            s[d + 2] = (host[0] - self.ax) / (n - 1)
            s[d + 3] = (host[1] - self.ay) / (n - 1)

        # The 5x5 patch around the agent
        p = config_rl.IDX_PATCH_START
        r = config_rl.PATCH_RADIUS
        i = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                s[p + i] = float(self._tile(self.ax + dx, self.ay + dy))
                i += 1

        s[config_rl.IDX_MASK_START:] = [float(v) for v in self.action_mask()]
        return s

    # -- stepping ----------------------------------------------------------

    def step(self, action):
        s = self.stage
        action = int(action)
        _, before = self._nearest(self.goals)
        reward = s.step_cost
        self.steps += 1

        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = self.ax + dx, self.ay + dy
            if self._tile(nx, ny) == T_WALL:
                reward += R_WALL if s.punitive else 0.0
            else:
                self.ax, self.ay = nx, ny

        elif action == config_rl.ACT_PICK_UP and s.can_pick:
            here = (self.ax, self.ay)
            if here in self.good:
                self.good.remove(here)
                reward += R_GOOD
            elif here in self.trash:
                self.trash.remove(here)
                reward += R_TRASH

        elif action == config_rl.ACT_ATTACK and s.can_attack:
            here = (self.ax, self.ay)
            hit = [h for h in self.hostiles
                   if abs(h[0] - here[0]) + abs(h[1] - here[1]) <= 1]
            if hit:
                self.hostiles.remove(hit[0])
                reward += R_GOOD

        # Standing still. The point is to stop the policy parking itself
        # somewhere safe and running the clock out - which is exactly what the
        # full world's agent learned to do. Small and periodic, not a cliff:
        # it should make waiting a bad option, not make waiting fatal.
        if (self.ax, self.ay) == self.last_pos:
            self.still += 1
            if self.still > STILL_GRACE and (self.still - STILL_GRACE) % STILL_EVERY == 0:
                reward += s.still_cost
        else:
            self.still = 0
        self.last_pos = (self.ax, self.ay)

        # Shaping, after the move: progress towards the nearest goal.
        _, after = self._nearest(self.goals)
        if self.goals and s.shaped:
            reward += R_TOWARDS * (before - after)

        # Reaching a goal
        here = (self.ax, self.ay)
        if here in self.goals:
            self.goals.remove(here)
            self.collected += 1
            reward += R_GOAL
            if s.sequence:
                # Next link, unless the trail is finished - in which case the
                # floor stays empty and `done` below ends the episode.
                if self.collected < s.sequence:
                    taken = {here, (self.ax, self.ay)}
                    taken.update(self.hazards)
                    self.goals.extend(self._place(taken, 1))
                else:
                    reward += R_CLEAR
            elif s.respawn:
                taken = {here, (self.ax, self.ay)}
                taken.update(self.hazards)
                new = self._place(taken, 1)
                self.goals.extend(new)
            elif not self.goals:
                reward += R_CLEAR

        # Hazards and hostiles only bite once damage is on the syllabus.
        if s.damage:
            if self._tile(self.ax, self.ay) == T_HAZARD:
                self.health -= s.hazard_damage
                reward += R_HAZARD
            for h in self.hostiles:
                if abs(h[0] - self.ax) + abs(h[1] - self.ay) <= 1:
                    self.health -= max(1, s.hazard_damage // 2)
                    reward += R_HAZARD * 0.5
            if self.health <= 0:
                self.dead = True
                reward += R_DEATH

        if s.sequence:
            finished = self.collected >= s.sequence
            done = self.dead or finished or self.steps >= s.max_steps
            # The penalty lands only when the clock beat it, not when it died
            # - death already costs R_DEATH and charging twice for one failure
            # makes dying look worse than it is.
            if done and not finished and not self.dead:
                reward += R_INCOMPLETE
            success = finished and not self.dead
        else:
            done = self.dead or self.steps >= s.max_steps or (not s.respawn and not self.goals)
            success = self.collected >= s.target and not self.dead
        return self._state(), reward, done, {
            "collected": self.collected, "success": success,
            "dead": self.dead, "steps": self.steps, "health": self.health,
        }


# ---- the guard that stops the original bug coming back -------------------

def best_case_return(stage: Stage, trials: int = 40, seed: int = 0) -> float:
    """What a competent agent scores, by playing one greedily.

    Not an optimal policy - just one that walks towards the nearest goal and
    picks things up. If *that* comes out negative, the stage is unlearnable
    and no amount of training will fix it, because the reward function is
    telling the agent that the best thing it can do is stop playing.

    This is the check the full ATS world never had.
    """
    total = 0.0
    for t in range(trials):
        env = CurriculumEnv(stage, seed=seed + t)
        env.reset()
        done = False
        ep = 0.0
        while not done:
            goal, _ = env._nearest(env.goals)
            act = config_rl.ACT_WAIT
            if goal:
                dx, dy = goal[0] - env.ax, goal[1] - env.ay
                # Try the bigger axis first, then the other, then anything that
                # is not a wall - enough to get through a maze most of the time.
                order = []
                if abs(dx) >= abs(dy):
                    order = [(1, 0) if dx > 0 else (-1, 0), (0, 1) if dy > 0 else (0, -1)]
                else:
                    order = [(0, 1) if dy > 0 else (0, -1), (1, 0) if dx > 0 else (-1, 0)]
                order += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for want in order:
                    if env._tile(env.ax + want[0], env.ay + want[1]) == T_WALL:
                        continue
                    for a, mv in MOVES.items():
                        if mv == want:
                            act = a
                            break
                    break
            _, r, done, _ = env.step(act)
            ep += r
        total += ep
    return total / trials
