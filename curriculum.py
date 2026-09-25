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
                 hazard_damage=10, punitive=True, shaped=True, sequence=0,
                 grow_goals=True, clock_growth=1.5):
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
        # Growing a rung normally means more to find and more time to find it.
        # On a rung whose whole lesson is "one goal, a long way off, follow
        # the bearing", both of those undo the lesson. Measured on the senior
        # compass rung before these existed:
        #
        #   15x15  route 18.0  clock  7.0x  random walker wins 11%
        #   17x17  route 12.1  clock 15.7x  random walker wins 22%
        #   19x19  route  6.0  clock 47.3x  random walker wins 49%
        #
        # The room got bigger and the rung got easier: goals_scale put four
        # goals in the 19x19 room so the nearest was six tiles away, while
        # 1.5x per step had turned the clock into forty-seven times the route.
        self.grow_goals = grow_goals
        self.clock_growth = clock_growth
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
                    max_steps=int(self.max_steps * self.clock_growth),
                    goals_scale=self.grow_goals,
                    target=self.target, pass_rate=self.pass_rate,
                    window=self.window, grow_to=self.grow_to,
                    sequence=self.sequence,
                    grow_goals=self.grow_goals,
                    clock_growth=self.clock_growth)
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


def trail_rung(grid=5, sequence=5, max_steps=160):
    """Bob's trail: one goal at a time, the next appearing when the current
    is taken, `sequence` of them, and a penalty if the clock beats it.

    Kept out of the ladder by measurement, not by opinion. 2026-09-18,
    60k environment steps per arm, two seeds, matched on steps:

        nursery only       detour 1.98  sensitivity 0.6311  argmax 4.0/4
        trail only         detour 8.12  sensitivity 0.0210  argmax 1.0/4
        nursery -> trail   detour 3.96  sensitivity 0.1297  argmax 1.5/4

    (an untrained network probes at ~0.02-0.05)

    Training it after the nursery drags goal-sensitivity from 0.63 back to
    0.13 and the argmax from 4/4 to chance: it un-teaches the bearing. The
    cause is not an ambiguous bearing - only one goal is ever on the floor -
    but that the episode no longer ends at the reward, so the return from
    any state carries four more legs whose bearings are unrelated, and the
    critic learns "how many legs remain" rather than "how far to this goal".

    Signalling a GAE terminal at each leg was tried and changed nothing
    (0.1279 vs 0.1297), so it is not merely where the credit is cut.

    To put it back:  default_ladder(with_trail=True)
    """
    return Stage("trail", grid, walls=False, hazards=0, sequence=sequence,
                 respawn=False, damage=False, punitive=False, shaped=False,
                 max_steps=max_steps, target=sequence,
                 pass_rate=0.80, window=50)


def default_ladder(max_grid=12, with_trail=False, senior_grid=15,
                   senior_to=19):
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

    # The five-goal trail sits behind a flag rather than in the ladder. It
    # measured as actively harmful after the nursery - see trail_rung() for
    # the numbers and the reason. One switch to bring it back.
    if with_trail:
        L.append(trail_rung(grid=5))

    # 2. New thing: walls. One goal, no respawn, episode ends on reaching it,
    #    exactly as rung 1 - so the only new variable is that the straight
    #    line is now blocked.
    L.append(Stage("corridors", 5, walls=True, hazards=0, goals=1, respawn=False,
                   damage=False, max_steps=160, target=1,
                   pass_rate=0.80, window=50))

    # 3. New thing: size. Same maze, same rules, more of it.
    #
    #    Still respawning, and that is a measured decision rather than an
    #    oversight. Converting this rung to one-goal-then-end - the change
    #    that works at 5x5 - made every number worse at 7x7 (40k steps, two
    #    seeds, fresh policy):
    #
    #      corridors  5x5  respawn  detour 10.4  sens 0.014  turns 53.5%
    #      corridors  5x5  one goal detour  8.0  sens 0.050  turns 62.3%
    #      corridors7 7x7  respawn  detour 10.5  sens 0.023  turns 51.0%
    #      corridors7 7x7  one goal detour 14.1  sens 0.007  turns 49.0%
    #
    #    A single goal in a bigger maze is found by luck, so episodes ran
    #    90 steps against 27 and the reward got sparse enough to drown the
    #    signal the change exists to protect. Neither config teaches the
    #    bearing here (chance is ~46%); the 5x5 rungs have to.
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

    # 6. New thing: scarcity. The goals no longer come back, so the room has to
    #    be cleared, and picking the wrong thing up costs.
    #
    #    Left at three simultaneous goals on purpose: clearing a room IS the
    #    lesson, and it cannot be taught one goal at a time. Note that three
    #    at once is the `no respawn x3` arm, which probed at 0.0451 - so this
    #    rung is not expected to teach the bearing, only to rely on it. If
    #    the bearing degrades from here, this is the first place to look.
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

    L.extend(senior_tier(grid=senior_grid, grow_to=senior_to))
    return L


def senior_tier(grid=15, grow_to=19):
    """Senior: a room too big to find the goal in by accident.

    Everything below junior is small enough that luck is a strategy. Measured
    with a random walker, 160-200 episodes per cell:

        nursery-like   5x5    finds the goal 95.0%
        corridors7     7x7                   71.5%
        9x9                                  63.0%
        15x15                                20.0%
        21x21                                13.5%

    Size alone never gets that floor to zero, because max_steps was being
    scaled up with the room. The clock is the real lever:

        single goal, random success vs how generous the clock is
        grid    route     2x     3x     4x     6x    10x    20x
        13x13    16.8   6.9%   9.4%  11.9%  15.6%  23.1%  34.4%
        15x15    15.8   6.9%   8.8%  10.0%  13.1%  15.6%  21.2%
        19x19    23.6   3.1%   3.1%   4.4%   6.9%   8.1%  16.9%

    So the senior rungs are big AND on a short clock - about four times the
    optimal route, where the floor sits near 10% and a 60% bar still means
    something.

    The shape of the tier
    ---------------------
    Bob's design: the first round keeps the trail, the last round is the
    compass alone. The compass is what the real world map uses, so it is the
    thing that has to survive.

    The trail earns its place here for a reason it did not have lower down.
    Measured above: a random walker completes a five-leg trail **0.0%** of the
    time, at every size tried. It cannot be stumbled through at all, while
    handing out reward often enough that there is a gradient to climb in a
    room where a single goal is found once in five episodes. That is the
    opposite of how it behaved next to the nursery.

    The risk, stated plainly
    ------------------------
    The trail is still the thing that un-taught the bearing. See trail_rung():
    after the nursery it dragged goal-sensitivity from 0.6311 to 0.1297 and
    the argmax from 4/4 to chance, because the episode no longer ends at the
    reward and the critic learns "how many legs remain" instead of "how far to
    this goal".

    The middle rung exists to attack exactly that mechanism: the same room,
    the same everything, with the trail cut from five legs to two. Fewer legs
    remaining is less for the critic to confuse with distance, so the value
    function is walked back toward distance before the compass rung asks for
    it alone.

    What happened when it was trained
    ---------------------------------
    200 episodes a rung, from the junior brain:

        senior-trail  15x15  5 legs  success  7%  collected 1.23/5  died 3%
        senior-short  15x15  2 legs  success 28%  collected 0.73/2  died 0%
        senior        15x15  1 goal  success 50%  PASSED at 125 episodes
        senior        17x17  1 goal  success 42%  (needs 50%)

    The compass rung - the one this whole tier exists for - is the EASIEST of
    the three, not the hardest. It passed in 125 episodes from a brain that
    had never seen a 15x15 room.

    So the premise above is wrong, and it is left standing so the correction
    is legible. The trail was put first to provide a gradient in a room where
    a single goal is found once in five episodes, on the assumption that the
    compass rung would need the help. It does not: one goal with the episode
    ending at the reward is the structure the ladder already established
    works, and it goes on working at 15x15.

    What the trail rungs actually are is *harder rungs placed first* - five
    sequential finds, or two, with the critic's "how many legs remain" problem
    on top. That front-loads the difficulty, which is the one thing this file
    was written to stop doing. By the ladder's own rule, exactly one new thing
    per rung, `senior` is the correct first senior rung: the only thing it
    adds is size.

    The bearing did survive, which was the open question. argmax 3.23/4 before
    the tier and 3.24/4 after, sensitivity 0.0510 -> 0.0583. The trail did not
    un-teach it this time. (diag_senior.py prints the same figure for all
    three rungs because only the bearing varies in what the probe shows the
    policy - it is one measurement of the brain, not three independent ones.)

    On the numbers, `senior` should come first and the two trail rungs after
    it, if they are wanted at all. That reorder is not done here; this is the
    measurement, recorded before anything is moved on the strength of it.
    """
    # The clock tightens as the trail shortens, and each multiple is measured
    # (diag_clock.py, oracle = BFS shortest route, 120 episodes a cell):
    #
    #   rung            clock   oracle   random walker
    #   trail, 5 legs     12x     100%       0%
    #   short, 2 legs      6x     100%       1%
    #   compass, 1 goal    4x     100%      11%
    #
    # The trail rungs can afford a generous clock because five sequential legs
    # cannot be stumbled into at any clock - the floor is 0% even at 24x. Only
    # the compass rung has to be tight, and that is the point: the trail is
    # what keeps luck out on the first two, and by the last one the bearing
    # has to do it instead.
    return [
        # 1. The trail, on a room that cannot be crossed by luck.
        Stage("senior-trail", grid, walls=True, hazards=3, hostiles=1,
              sequence=5, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 12), target=5,
              pass_rate=0.55, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),

        # 2. New thing: less trail. Same room, two legs instead of five.
        Stage("senior-short", grid, walls=True, hazards=3, hostiles=1,
              sequence=2, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 6), target=2,
              pass_rate=0.55, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),

        # 3. New thing: nothing to follow. One goal, the episode ends on it,
        #    and the only thing pointing at it is the bearing. This is the
        #    rung the whole tier is for, and it is the one the real world
        #    resembles.
        Stage("senior", grid, walls=True, hazards=3, hostiles=1, goals=1,
              sequence=0, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 4), target=1,
              pass_rate=0.50, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),
    ]


def _senior_clock(grid, multiple):
    """`multiple` times the optimal route for a room this size.

    Route length measured at about 1.3 tiles per row of grid: 19.5 on 15x15,
    20.9 on 17x17, 26.8 on 19x19.
    """
    return int(multiple * 1.3 * grid)


def _unit(dx, dy):
    """(dx, dy) scaled to length 1.  Shared definition so the curriculum and
    the real world cannot drift apart again."""
    m = (dx * dx + dy * dy) ** 0.5
    return (0.0, 0.0) if m < 1e-9 else (dx / m, dy / m)


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
        #
        # A stage WITH hazards needs more than a few.  In a perfect maze only
        # 35% of the cells on the route have any way round them, so "route
        # around the lava" is usually not a thing the room permits - the
        # hazard is either a wall or irrelevant, and neither teaches it.
        # Braiding lifts that, and the detour rate with it - but openness is
        # not free: `avoid` respawns its goals and does no damage, so a more
        # open room just lets a competent agent farm more of them, and its
        # best-case return runs 6.2 -> 10.3 -> 16.0 -> 18.4 -> 32.4 as the
        # braid goes 0 -> n/3 -> n/2 -> n -> 1.5n.  That is the rung getting
        # easier, not better.  n/3 roughly doubles the detour rate (24% ->
        # 41%) for the smallest move in difficulty, so it is the setting.
        # Stages with no hazards are untouched: that is the topology they
        # were tuned on and already passed.
        extra = n // 2
        if self.stage.hazards and getattr(config_rl, 'BRAID_MAZE_FOR_HAZARDS', True):
            extra += n // 3
        for _ in range(extra):
            x = self.rng.randrange(1, n - 1)
            y = self.rng.randrange(1, n - 1)
            g[y][x] = T_FLOOR

    def _free_cells(self):
        return [(x, y)
                for y in range(self.stage.grid)
                for x in range(self.stage.grid)
                if self.grid[y][x] == T_FLOOR]

    def _safe_route_exists(self, hazards):
        """Can the agent reach every goal without stepping on lava?"""
        n = self.stage.grid
        block = set(hazards)
        seen = {(self.ax, self.ay)}
        stack = [(self.ax, self.ay)]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if (nx, ny) in seen or (nx, ny) in block:
                    continue
                if self.grid[ny][nx] == T_WALL:
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        return all(g in seen for g in self.goals)

    def _route_len(self, block):
        """Steps to the nearest goal, treating *block* as impassable."""
        from collections import deque
        n = self.stage.grid
        start = (self.ax, self.ay)
        goals = set(self.goals)
        seen = {start}
        q = deque([(start, 0)])
        while q:
            (x, y), d = q.popleft()
            if (x, y) in goals:
                return d
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in seen:
                    continue
                if self.grid[ny][nx] == T_WALL or (nx, ny) in block:
                    continue
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return None

    def _direct_path(self):
        """Cells on one shortest agent->goal route, excluding both ends."""
        from collections import deque
        n = self.stage.grid
        start = (self.ax, self.ay)
        goals = set(self.goals)
        prev = {start: None}
        q = deque([start])
        end = None
        while q and end is None:
            x, y = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in prev:
                    continue
                if self.grid[ny][nx] == T_WALL:
                    continue
                prev[(nx, ny)] = (x, y)
                if (nx, ny) in goals:
                    end = (nx, ny)
                    break
                q.append((nx, ny))
        if end is None:
            return []
        path = []
        cur = prev[end]
        while cur is not None and cur != start:
            path.append(cur)
            cur = prev[cur]
        return path

    def _detour_cost(self, hazards):
        """Extra steps the safe route costs over walking straight through.

        Zero means the hazards are decoration - the agent reaches the goal
        just as fast whether or not it cares about them.
        """
        direct = self._route_len(set())
        safe = self._route_len(set(hazards))
        if direct is None or safe is None:
            return 0
        return safe - direct

    def _place_hazards(self, taken, n):
        """Hazards that leave at least one lava-free route to every goal.

        The maze is a perfect DFS carve, so its corridors are one cell wide
        and a single hazard dropped at random owns the entire route through
        them.  Measured before this existed: on `avoid` and `hazards` - the
        two rungs whose whole lesson is that lava is bad - 47% of rooms made
        avoiding it impossible, rising to 82% on primary grown to 11x11.
        That is not a harder room, it is a room that contradicts its own
        lesson, and it punishes the agent for the one behaviour the rung is
        trying to reward.

        So: propose a placement, keep it only if a safe route survives.  If
        the room genuinely has no room for that many hazards, place fewer -
        a rung with two avoidable hazards teaches avoidance; a rung with
        three unavoidable ones teaches that avoidance does not work.

        But "avoidable" is not enough on its own.  Merely rejecting blocking
        placements left the lava somewhere irrelevant 97% of the time, which
        fails the rung from the other side: an agent that never meets a
        hazard on its way anywhere learns nothing about hazards and then
        walks into the first one that matters.  So among the placements that
        DO leave a way round, prefer one that sits on the direct route and
        makes the safe way round longer.  Costly to avoid, never impossible
        to avoid - that is the whole lesson in one room.
        """
        if n <= 0:
            return []
        # NOTE: an earlier version deliberately seated one hazard ON the
        # route so avoiding it cost something.  Removed on request: while the
        # lesson is still "cross the room", a hazard on the only line to the
        # goal is not a harder lesson, it is a different one the agent has
        # not been taught yet.  ROUTE_MUST_BE_HAZARD_FREE brings it back.
        best = None
        for _ in range(24):
            free = [c for c in self._free_cells() if c not in taken]
            if len(free) < n:
                break
            self.rng.shuffle(free)
            pick = free[:n]
            if not self._safe_route_exists(pick):
                continue
            detour = self._detour_cost(pick)
            if detour > 0:
                taken.update(pick)
                return pick
            if best is None:
                best = pick
        if best is not None:
            taken.update(best)
            return best
        # Could not fit n avoidable hazards; add them one at a time and stop
        # at the last one that still leaves a way round.
        out = []
        for _ in range(n):
            free = [c for c in self._free_cells() if c not in taken]
            self.rng.shuffle(free)
            for c in free:
                if self._safe_route_exists(out + [c]):
                    out.append(c)
                    taken.add(c)
                    break
            else:
                break
        return out

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
        """Build a room, and keep building until it is actually playable.

        A map is only a training ground if a route from the agent to every
        goal exists, is four-connected (the agent cannot move diagonally, so
        a corner-to-corner gap is a wall), and is free of hazards.  Anything
        else is a room the agent cannot solve, and an episode it can only
        lose - which teaches it that trying does not work.

        Measured before this: 1 room in 1000 on primary/junior had a region
        cut off by a diagonal-only gap.  Rare, but it is exactly the case
        that is invisible from the screen and impossible from inside.
        """
        for _ in range(40):
            self._build()
            if not getattr(config_rl, 'VALIDATE_MAPS', True):
                return self._state()
            if self._playable():
                return self._state()
        # Could not generate a playable room at this difficulty; fall back to
        # one with no hazards at all rather than hand back an unwinnable map.
        self._build(force_no_hazards=True)
        return self._state()

    def _playable(self):
        """Every goal reachable from the agent, 4-connected, without lava."""
        from collections import deque
        n = self.stage.grid
        block = set(self.hazards)
        seen = {(self.ax, self.ay)}
        q = deque([(self.ax, self.ay)])
        while q:
            x, y = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                c = (x + dx, y + dy)
                if not (0 <= c[0] < n and 0 <= c[1] < n) or c in seen or c in block:
                    continue
                if self.grid[c[1]][c[0]] == T_WALL:
                    continue
                seen.add(c)
                q.append(c)
        return all(g in seen for g in self.goals)

    def _build(self, force_no_hazards=False):
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
        want = 0 if force_no_hazards else s.hazards
        self.hazards = (self._place_hazards(taken, want)
                        if getattr(config_rl, 'HAZARDS_LEAVE_A_SAFE_ROUTE', True)
                        else self._place(taken, want))
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
        # What the agent knows, as opposed to what it can currently see.
        self.seen = {}        # (x,y) -> tile id last observed
        self.visit = {}       # (x,y) -> recency, 1.0 = standing here now
        self._observe()

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

    def _observe(self):
        """Reveal what is within sight, and mark where we are standing.

        Sight is line-of-nothing - a plain radius, no occlusion.  The point
        is not realism, it is that a tile once seen stays known; the previous
        version re-encountered every cell as if for the first time.
        """
        r = config_rl.PATCH_RADIUS
        n = self.stage.grid
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = self.ax + dx, self.ay + dy
                if 0 <= x < n and 0 <= y < n:
                    self.seen[(x, y)] = self.grid[y][x]
        d = config_rl.MEM_VISIT_DECAY
        for k in list(self.visit):
            self.visit[k] *= d
            if self.visit[k] < 0.01:
                del self.visit[k]
        self.visit[(self.ax, self.ay)] = 1.0

    def _route_cells(self):
        """Cells on a shortest hazard-free route from the agent to a goal.

        The pathway, not the pointer.  A bearing is useless the moment a wall
        stands between you and the thing it points at; this says which way to
        turn.  Nothing obliges the agent to follow it - it is one more channel
        of observation, and the stages that get it are listed in config.
        """
        if self.stage.name not in getattr(config_rl, 'ROUTE_HINT_STAGES', ()):
            return set()
        from collections import deque
        n = self.stage.grid
        block = set(self.hazards)
        start = (self.ax, self.ay)
        goals = set(self.goals)
        prev = {start: None}
        q = deque([start])
        end = None
        while q and end is None:
            cur = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                c = (cur[0] + dx, cur[1] + dy)
                if not (0 <= c[0] < n and 0 <= c[1] < n) or c in prev or c in block:
                    continue
                if self.grid[c[1]][c[0]] == T_WALL:
                    continue
                prev[c] = cur
                if c in goals:
                    end = c
                    break
                q.append(c)
        if end is None:
            return set()
        out = set()
        cur = end
        while cur is not None:
            out.add(cur)
            cur = prev[cur]
        return out

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
        s[config_rl.IDX_OBJ_ID] = 1.0 / config_rl.ITEM_ID_SCALE if goal else 0.0

        # The direction block. This is the single most useful thing in the
        # vector: without a signed bearing to the goal the policy has to infer
        # direction from the patch alone, which it can only do once the goal is
        # already within two tiles.
        # UNIT vector, to match agent.build_state.  These two encoders write
        # the same four slots and used to disagree: the curriculum scaled the
        # delta by the grid span (so magnitude carried distance) while the
        # world normalised to length 1.  A policy trained on one read the
        # other wrong.  Distance already has its own slot above, so direction
        # alone is the honest content here.
        d = config_rl.IDX_DIR_START
        if goal:
            s[d + 0], s[d + 1] = _unit(goal[0] - self.ax, goal[1] - self.ay)
        if host:
            s[d + 2], s[d + 3] = _unit(host[0] - self.ax, host[1] - self.ay)

        # The 5x5 patch around the agent
        p = config_rl.IDX_PATCH_START
        r = config_rl.PATCH_RADIUS
        i = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                s[p + i] = float(self._tile(self.ax + dx, self.ay + dy))
                i += 1

        # Remembered map: three channels over one window, agent-centred.
        route = self._route_cells()
        mr = config_rl.MEM_RADIUS
        k0 = config_rl.IDX_MEM_KNOWN_START
        v0 = config_rl.IDX_MEM_VISIT_START
        r0 = config_rl.IDX_MEM_ROUTE_START
        i = 0
        for dy in range(-mr, mr + 1):
            for dx in range(-mr, mr + 1):
                c = (self.ax + dx, self.ay + dy)
                if c in self.seen:
                    # +1 so that 0 means "never seen", which is a different
                    # thing from "seen, and it was empty floor".
                    s[k0 + i] = (self.seen[c] + 1.0) / (config_rl.TILE_VOCAB_SIZE + 1.0)
                s[v0 + i] = self.visit.get(c, 0.0)
                s[r0 + i] = 1.0 if c in route else 0.0
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
        # The reward penalty is OUTSIDE the damage guard on purpose.
        #
        # It used to be inside, so on `avoid` - damage=False, the rung whose
        # entire stated lesson is "tile 6 is bad" - standing in lava cost
        # nothing whatsoever. No health, no reward. The rung did not teach
        # avoidance; it spent 737 episodes teaching that lava is a free
        # shortcut, and then `hazards` turned damage on and the agent died
        # 47% of the time and failed the rung at the 8000-episode cap.
        #
        # The comment on `avoid` says damage is off so the agent survives
        # long enough to form the association. That reasoning is right - it
        # just also removed the only signal the association could form from.
        if self._tile(self.ax, self.ay) == T_HAZARD:
            reward += R_HAZARD
        if s.damage:
            if self._tile(self.ax, self.ay) == T_HAZARD:
                self.health -= s.hazard_damage
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
        self._observe()          # memory updates once per tick, after moving
        return self._state(), reward, done, {
            "collected": self.collected, "success": success,
            "dead": self.dead, "steps": self.steps, "health": self.health,
        }


# ---- the guard that stops the original bug coming back -------------------

def oracle_step(env, stage):
    """One step along a shortest route to the nearest goal, avoiding harm.

    This replaced a greedy "walk the bigger axis first" chooser, which had no
    pathfinding and so measured the maze rather than the reward. It cleared
    only 32% of junior 9x9 and about 0% of a 15x15 room, so every senior rung
    came back negative and preflight refused to train rungs that were in fact
    fine.

    Re-planned every step rather than followed as a fixed path, because a
    trail moves the goal the instant the current one is taken.

    The two passes preserve what the greedy version got right: prefer a route
    that never touches lava or stands next to something that bites, and only
    accept one that does when there is no other - which happens in a narrow
    corridor, and is exactly when taking the hit is correct.
    """
    from collections import deque

    n = stage.grid
    start = (env.ax, env.ay)
    goals = set(env.goals)
    if not goals:
        return config_rl.ACT_WAIT

    bite = set()
    if stage.damage:
        for h in env.hostiles:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if abs(dx) + abs(dy) <= 1:
                        bite.add((h[0] + dx, h[1] + dy))

    for careful in (True, False):
        prev = {start: None}
        q = deque([start])
        end = None
        while q and end is None:
            x, y = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in prev:
                    continue
                if env.grid[ny][nx] == T_WALL:
                    continue
                if careful and (env.grid[ny][nx] == T_HAZARD
                                or (nx, ny) in bite):
                    continue
                prev[(nx, ny)] = (x, y)
                if (nx, ny) in goals:
                    end = (nx, ny)
                    break
                q.append((nx, ny))
        if end is not None:
            cur = end
            while prev[cur] != start:
                cur = prev[cur]
            want = (cur[0] - start[0], cur[1] - start[1])
            for act, d in MOVES.items():
                if d == want:
                    return act
    return config_rl.ACT_WAIT


def best_case_return(stage: Stage, trials: int = 40, seed: int = 0) -> float:
    """What a competent agent scores, by playing one greedily.

    Not an optimal policy - just one that walks towards the nearest goal,
    picks things up, and does not deliberately walk into things that hurt.
    If *that* comes out negative, the stage is unlearnable and no amount of
    training will fix it, because the reward function is telling the agent
    that the best thing it can do is stop playing.

    It avoids hazards, and that is not a detail. The first version only
    avoided walls, so on any rung with damage it measured a goal-seeker that
    strolls through fire - the worst possible player of exactly those rungs.
    It reported `hazards` at +0.12 and `junior` at +0.34 and both looked
    barely learnable, when what it had actually measured was recklessness.

    This is the check the full ATS world never had.
    """
    total = 0.0
    s = stage                      # `s` is what the reward code calls it
    for t in range(trials):
        env = CurriculumEnv(stage, seed=seed + t)
        env.reset()
        done = False
        ep = 0.0
        while not done:
            # A pathfinder, not a greedy axis-walker. The old chooser
            # had no route planning and cleared 32% of junior 9x9 and
            # almost none of a 15x15 room, so every senior rung scored
            # negative and preflight refused to train rungs that were
            # in fact fine. It was measuring the maze, not the reward.
            act = oracle_step(env, s)
            _, r, done, _ = env.step(act)
            ep += r
        total += ep
    return total / trials
