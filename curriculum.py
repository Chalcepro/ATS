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
T_HOSTILE = 2         # TILE_HOSTILE  - something standing there
T_OBJECT = 3          # TILE_OBJECT   - something lying there to be picked up
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
                 grow_goals=True, clock_growth=1.5,
                 swords=0, hostile_hp=1, hostile_damage=0, chase=0,
                 aggro=0, grow_hostiles=False, guards=False,
                 weapon_first=False, pick_goals=False):
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

        # ---- combat ------------------------------------------------------
        #
        # Everything below junior treats a hostile as furniture: it is placed
        # once, it never moves, one ACT_ATTACK deletes it, and there is no
        # weapon anywhere in the room. That is not something to fight, it is
        # a pot-hole that bites, and raising `hostiles` on such a rung raises
        # the number of pot-holes rather than teaching combat.
        #
        # These four fields are what turn it into a fight, and they are
        # deliberately separable so a rung can add exactly one of them:
        #
        #   hostile_hp      how many blows it takes. >1 is what makes a
        #                   weapon worth carrying; at 1 the sword is decor.
        #   swords          how many are on the floor to be picked up.
        #   chase           how often it steps toward the agent, as one move
        #                   every `chase` ticks. 0 is the old furniture.
        #   aggro           how close you must be before it notices. Outside
        #                   this radius it wanders, so the room is not one
        #                   long pursuit from tick zero.
        #
        # `hostile_damage` splits from hazard_damage because they are two
        # different lessons - lava is stood in, a hostile comes to you - and
        # tuning one used to silently move the other. 0 keeps the old
        # behaviour (half of hazard_damage).
        self.swords = int(swords)
        self.hostile_hp = int(hostile_hp)
        self.hostile_damage = int(hostile_damage)
        self.chase = int(chase)
        self.aggro = int(aggro)
        # Hostiles seated on the route to the goal rather than dropped
        # anywhere. See CurriculumEnv._place_guards for the measurement
        # that made this necessary.
        self.guards = bool(guards)
        # While unarmed, the object slots and the shaping point at the
        # nearest weapon instead of the goal. Without it ACT_PICK_UP has
        # no gradient anywhere on the ladder - see CurriculumEnv._aim for
        # the 0%-armed measurement that made this necessary.
        self.weapon_first = bool(weapon_first)
        # The goal is taken with ACT_PICK_UP while standing on it, not by
        # standing on it. This exists because no rung on this ladder ever
        # taught the hand: every one of them is solved by moving alone, and
        # a policy that has passed all of them chooses ACT_PICK_UP 2% of the
        # time and ACT_ATTACK *never* - measured 0 times in 3,495 ticks.
        self.pick_goals = bool(pick_goals)
        # Goals scaling with room size undoes a compass rung (see grow_goals).
        # Hostiles scaling is the opposite: on a combat rung the encounter
        # rate IS the lesson, and a bigger room with the same two hostiles is
        # a rung that quietly teaches less as it grows.
        self.grow_hostiles = grow_hostiles

        # A bigger room gets proportionally more to find, so the reward on
        # offer grows with the walking required.
        if goals_scale:
            self.goals = max(goals, round(goals * grid / 5.0))

    @property
    def bite(self):
        """What one blow from a hostile takes."""
        return self.hostile_damage or max(1, self.hazard_damage // 2)

    @property
    def tempo(self):
        """One hostile turn every this many agent ticks.

        A hostile's move and its blow are the same turn. Before this they
        were not: `chase` gated movement while damage was applied on every
        single tick a hostile stood adjacent, so a pursuer that could only
        step every other tick was still hitting twice as often as it could
        act. Three of them meant 24 health a tick against a 100-health
        agent, and the oracle - a competent player, by construction - died
        in up to 38% of rooms.
        """
        return max(1, self.chase)

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
        # More room, more of them. Without this a combat rung gets easier
        # every time it grows: the same two hostiles spread over 19x19 are
        # met half as often as over 15x15, so the rung that exists to teach
        # fighting would teach progressively less of it. Off by default, so
        # nothing below the combat tier changes.
        more = self.hostiles
        if self.grow_hostiles:
            more = max(self.hostiles,
                       round(self.hostiles * (self.grid + 2) / float(self.grid)))
        nxt = Stage(self.name, self.grid + 2, walls=self.walls,
                    hazards=min(self.hazards + 1, 4),
                    hostiles=more, goals=self.goals, respawn=self.respawn,
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
                    clock_growth=self.clock_growth,
                    # Carried explicitly. A field added to __init__ and not
                    # added here does not raise - it silently defaults, so a
                    # grown rung loses the one thing it was built to teach
                    # and still reports the same name.
                    swords=self.swords, hostile_hp=self.hostile_hp,
                    hostile_damage=self.hostile_damage, chase=self.chase,
                    aggro=self.aggro, grow_hostiles=self.grow_hostiles,
                    guards=self.guards, weapon_first=self.weapon_first,
                    pick_goals=self.pick_goals)
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

# ---- combat --------------------------------------------------------------
#
# Fighting is never the point. On every combat rung success is still
# "reach the goal", and the hostiles are in the way of it - so these
# numbers only have to make fighting the better answer when something is
# blocking the route, and never make hunting more profitable than walking.
#
# R_KILL under R_GOAL is that rule stated in arithmetic: clearing the room
# of hostiles and going home empty must score below simply going home.
R_SWORD = 0.5           # picking up a weapon
R_HIT = 0.15            # landing a blow. Shaping: at hostile_hp > 1 a kill
                        # is several ticks away, and without this the first
                        # two swings look identical to swinging at nothing.
R_KILL = 1.0            # finishing one off
R_SWING = -0.05         # swinging at empty air. Small, but ATTACK is free
                        # otherwise and a policy that mashes it loses nothing.

SWORD_DAMAGE = 3        # blows to kill: 1 at hp 3 with a sword, 3 without
FIST_DAMAGE = 1

# Item 0013 in the real registry - crafting.py makes it from Stick + Stone.
# The curriculum does not craft, but it writes the same id into the same
# inventory slot, so a policy that learns "slot 0 holds 0.13 -> my swing
# kills" reads the real world's inventory the same way.
STONE_SWORD = 13

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
                   senior_to=19, with_combat=True):
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
    # Last, because it is hardest and because it depends on everything the
    # senior rungs teach: a room this size has to be navigated before it is
    # worth learning to fight your way across one.
    if with_combat:
        L.extend(combat_tier(grid=senior_grid, grow_to=senior_to))
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

    Where it got to
    ---------------
    Reordered easiest-first, on the numbers above:

        senior 15x15   passed at 125 episodes, 50%
        senior 17x17   passed at  60 episodes, 52%
        senior 19x19   plateaus around 38% over 900 episodes (23-50%, noisy)

    19x19 is the frontier and it is not a bug in the rung. A BFS oracle clears
    it 100% at the clock it has, the reward is positive, and when the policy
    does succeed it walks near-optimal routes (walk ratio 1.0-1.5) - so it is
    not wandering, it simply fails to find the goal at all in about 60% of
    episodes.

    Nor is it the bearing degrading with distance: the bearing is a unit
    vector, direction only, so it reads the same at 32 tiles as at 5. What
    19x19 actually asks for is routing a 32-tile path through a maze the agent
    can only see locally, steering by compass alone. That is the limit worth
    knowing about, and it is the same wall as the rung-4 stall - perception
    range, not curriculum.

    The bearing survived the tier, measured before and after: argmax 3.19/4 ->
    3.25/4, sensitivity 0.0581 -> 0.0602.
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
        # 1. New thing: size, and nothing else. One goal, the episode ends on
        #    reaching it - exactly the structure of the nursery and corridors,
        #    in a room three times the width. Measured at 50% and passed in
        #    125 episodes from the junior brain, which is why it is first: by
        #    this ladder's own rule a rung adds one new thing, and size is all
        #    this one adds.
        Stage("senior", grid, walls=True, hazards=3, hostiles=1, goals=1,
              sequence=0, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 4), target=1,
              pass_rate=0.50, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),

        # 2. New thing: a second thing to find, and an order to find them in.
        #    Two legs, so the critic has little "how many legs remain" to
        #    confuse with distance. Measured at 28% after 200 episodes.
        Stage("senior-short", grid, walls=True, hazards=3, hostiles=1,
              sequence=2, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 6), target=2,
              pass_rate=0.55, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),

        # 3. New thing: a long route. Five legs, which is the hardest rung on
        #    the ladder and measured hardest - 7% after 200 episodes, 1.23 of
        #    5 legs collected. It goes last because it is hardest, which is
        #    the only reason any rung should go anywhere.
        Stage("senior-trail", grid, walls=True, hazards=3, hostiles=1,
              sequence=5, respawn=False, can_pick=True, can_attack=True,
              damage=True, max_steps=_senior_clock(grid, 12), target=5,
              pass_rate=0.55, window=60, grow_to=grow_to,
              grow_goals=False, clock_growth=1.2),
    ]


def combat_tier(grid=15, grow_to=19):
    """Senior, but the room fights back.

    Bob's ask: "it can pick up items, it can pick up swords, all this stuff,
    and then battle... the chances of encountering an enemy are higher... you
    should be able to fend for yourself."

    What was already there, and why it was not combat
    -------------------------------------------------
    `hostiles` has existed since junior, and junior, primary and all three
    senior rungs carry one. It was never a fight:

      * the hostile is placed at reset and never moves again
      * one ACT_ATTACK deletes it, whatever it is
      * there is no weapon anywhere in the curriculum
      * items on the floor are not in the observation at all, so the only
        way to find one is to walk over it by accident

    That is a pot-hole that bites, and turning `hostiles` up on such a rung
    adds pot-holes. Four things had to change in the environment before a
    combat rung could mean anything, and they are in Stage as `hostile_hp`,
    `swords`, `chase` and `aggro`, plus the TILE_OBJECT/TILE_HOSTILE overlay
    that makes both visible.

    The shape of the tier - one new thing per rung, as everywhere else
    -----------------------------------------------------------------
    1. `armed`   a weapon exists and is worth carrying. Hostiles still do
                 not move, so the ONLY new thing is the sword: three swings
                 bare-handed, one with it.
    2. `hunted`  they come to you. Same numbers, same sword, but avoidance
                 stops being free and disengaging becomes a decision.
    3. `warden`  more of them, and more again as the room grows. This is the
                 "higher chance of encountering an enemy" rung, and it is
                 last because it is hardest.

    Success is still reaching the goal
    ----------------------------------
    Not "clear the room". R_KILL is deliberately below R_GOAL so that
    hunting pays worse than walking home, and the hostiles are an obstacle
    between the agent and the thing it was sent for. This keeps the reward
    structure the ladder has already proved learnable, and avoids handing
    the critic a second objective to confuse with distance - which is
    exactly what the trail rungs did to the bearing (see trail_rung).

    The clock is looser than the compass rung's 4x because fighting costs
    ticks that walking does not: arming yourself is a detour, and every
    exchange is a tick not spent travelling.
    """
    # 0. Before any of it: the hand.
    #
    #    Every rung below this tier is solved by moving. Nothing on the
    #    ladder has ever required ACT_PICK_UP or ACT_ATTACK, and a policy
    #    that had passed all thirteen of them chose ACT_PICK_UP in 2.2% of
    #    ticks and ACT_ATTACK in **0 of 3,495**. Trained 600 episodes on
    #    `armed` it picked up a sword in 0% of episodes and killed 0.00
    #    hostiles, walking out the clock at 8%.
    #
    #    Giving the sword a bearing and shaping was necessary and still not
    #    enough: the agent then walked to the sword and STOOD on it for
    #    twenty-odd ticks without pressing anything, because reaching a
    #    thing has been the whole job for thirteen rungs. Shaping cannot
    #    teach an action; it can only teach where to stand.
    #
    #    So the same argument the nursery makes for the goal, made for the
    #    hand: one room, one thing, nothing to be hurt by, and the only way
    #    to score is the button. `satchel` IS the nursery - same size, same
    #    no-walls, same no-penalties, same single goal ending the episode -
    #    with exactly one thing added, which is that arriving is not enough.
    hand = dict(hazards=0, goals=1, sequence=0, respawn=False, damage=False,
                can_pick=True, pick_goals=True, punitive=False,
                target=1, window=50)
    tier = [
        Stage("satchel", 5, walls=False, shaped=False, max_steps=120,
              pass_rate=0.85, **hand),
        # Then the same lesson where the thing has to be found first, so
        # "press it when you arrive" survives a room it cannot see all of.
        Stage("satchel7", 7, walls=True, shaped=True, max_steps=200,
              pass_rate=0.75, **hand),
    ]

    common = dict(walls=True, hazards=3, goals=1, sequence=0, respawn=False,
                  can_pick=True, can_attack=True, damage=True,
                  target=1, window=60, grow_to=grow_to,
                  grow_goals=False, clock_growth=1.2,
                  hostile_hp=3, hostile_damage=8, aggro=6, guards=True,
                  weapon_first=True)
    return tier + [
        # 1. New thing: a weapon. Hostiles stand still exactly as they have
        #    since junior, so nothing about avoiding them has changed - but
        #    they now take three blows bare-handed and one with the sword,
        #    which is the first time in this curriculum that carrying
        #    something has altered what an action does.
        Stage("armed", grid, swords=2, hostiles=2, chase=0,
              max_steps=_senior_clock(grid, 5), pass_rate=0.50, **common),

        # 2. New thing: they move. Walking past is no longer a plan, and the
        #    sword stops being optional. `chase=2` is one hostile move per
        #    two agent ticks on purpose: at 1 it is a perfect pursuer and
        #    fleeing is impossible, which makes the rung a fight simulator
        #    rather than a choice between fighting and leaving.
        Stage("hunted", grid, swords=2, hostiles=2, chase=2,
              max_steps=_senior_clock(grid, 6), pass_rate=0.50, **common),

        # 3. New thing: numbers. Four in the room instead of two, and
        #    `grow_hostiles` keeps that density as the room grows to 19x19 -
        #    without it the rung would get easier every time it grew, since
        #    the same two hostiles in a room 1.6x the area are met far less
        #    often. This is the rung that answers "higher chance of
        #    encountering an enemy".
        Stage("warden", grid, swords=3, hostiles=4, chase=2,
              grow_hostiles=True,
              max_steps=_senior_clock(grid, 7), pass_rate=0.45, **common),
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

    def _place_guards(self, taken, n):
        """Hostiles that stand between the agent and where it is going.

        Measured before this existed, on the `armed` rung - two hostiles and
        two swords in a 15x15 maze, played by the oracle:

            armed 15x15   met 82%   armed 39%   kills 0.39   success 100%
            armed 17x17   met 59%   armed 32%   kills 0.34   success 100%
            armed 19x19   met 62%   armed 30%   kills 0.34   success 100%

        A competent player cleared every room while picking up a sword in
        under a third of them and killing almost nothing. That is not a rung
        about weapons with a hard bit; it is a rung about walking that has
        some swords lying in it, and the bigger the room got the more
        thoroughly the whole mechanic could be ignored.

        The mistake was assuming "a weapon" and "a reason to use it" are two
        separate things that could go on two rungs. They are not: nothing
        on a static-hostile rung ever forces an exchange, so the weapon has
        no effect that the agent can be rewarded for noticing.

        So a guard is placed the way a hazard is - preferring the direct
        route, because that is where it is a decision - and the same rule
        applies as for hazards: never make the goal unreachable. A hostile
        can always be fought through, so 'blocked' here means only that the
        agent cannot get by without an exchange, which is the point.
        """
        if n <= 0:
            return []
        path = [c for c in self._direct_path() if c not in taken]
        out = []
        if path:
            # Spread along the route, not clustered at its midpoint. The
            # first version seated every guard as near the middle as it
            # could, which on the four- and six-hostile rungs put the whole
            # garrison in one place: they then converged into a mob, landed
            # four bites a tick, and killed a competent player 31-38% of the
            # time. Spacing them is the difference between three fights and
            # one unwinnable one.
            #
            # Half-way along the first gap, so nothing stands on the agent's
            # doorstep before a sword could possibly have been found.
            step = len(path) / float(n + 1)
            for k in range(n):
                idx = min(len(path) - 1, int(step * (k + 1)))
                c = path[idx]
                if c not in taken:
                    out.append(c)
                    taken.add(c)
        if len(out) < n:
            out.extend(self._place(taken, n - len(out)))
        return out

    def _place(self, taken, n):
        out = []
        free = [c for c in self._free_cells() if c not in taken]
        self.rng.shuffle(free)
        for c in free[:n]:
            out.append(c)
            taken.add(c)
        return out

    def _step_hostiles(self):
        """Hostiles take a turn: chase if they have noticed you, else wander.

        This is the difference between "there are enemies in the room" and
        "you will be attacked". With static hostiles the encounter rate is
        whatever the agent's route happens to blunder into - measured at 15x15
        with two of them, a competent agent walking to the goal met one in
        under a third of episodes, and adding more just added more furniture
        to walk around.

        Two deliberate limits, because a perfect pursuer makes the rung
        unwinnable rather than hard:

        `chase` is a period, not a speed - one hostile move every `chase`
        agent ticks. At 2 the agent outruns it and disengaging is a real
        option, which is what makes choosing to fight a choice.

        `aggro` is a leash on noticing, not on following. Outside it they
        wander, so a room is not one unbroken pursuit from tick one; inside
        it they commit. They do not path around walls - they take the better
        of the two axis steps and stall on corners, which is the behaviour
        the full world's entities have and is also what makes a corridor a
        defensible place to stand.
        """
        s = self.stage
        if not s.chase or not self.hostiles:
            return
        if self.steps % s.tempo:
            return
        taken = {(h[0], h[1]) for h in self.hostiles}
        for i, (hx, hy) in enumerate(self.hostiles):
            d = abs(hx - self.ax) + abs(hy - self.ay)
            # Already in reach: hold and fight. It cannot step onto the
            # agent, so without this it took the next-best step - sideways,
            # out of reach - and a pursuer that had cornered the agent
            # wandered off instead of pressing. Measured: four ticks stood
            # next to one cost a single blow rather than two, because it
            # spent every other turn stepping away and back.
            if d <= 1:
                continue
            wandering = bool(s.aggro) and d > s.aggro
            if wandering:
                opts = [(hx + dx, hy + dy) for dx, dy in MOVES.values()]
                self.rng.shuffle(opts)
            else:
                # Close the bigger gap first, and fall back to the other axis
                # when a wall is in the way.
                opts = []
                if abs(self.ax - hx) >= abs(self.ay - hy):
                    opts.append((hx + (1 if self.ax > hx else -1), hy))
                    opts.append((hx, hy + (1 if self.ay > hy else -1)))
                else:
                    opts.append((hx, hy + (1 if self.ay > hy else -1)))
                    opts.append((hx + (1 if self.ax > hx else -1), hy))
                opts = [o for o in opts if o != (hx, hy)]
            for nx, ny in opts:
                if self._tile(nx, ny) in (T_WALL, T_HAZARD):
                    continue
                if (nx, ny) in taken or (nx, ny) == (self.ax, self.ay):
                    continue
                taken.discard((hx, hy))
                taken.add((nx, ny))
                self.hostiles[i] = (nx, ny)
                break

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
        # One sword early on the way, the rest scattered. A weapon that only
        # ever lies BEHIND the thing it is for is not a weapon the rung can
        # teach: the agent meets the guard bare-handed every time, and the
        # sword becomes a reward for having already won the fight.
        self.swords = []
        if s.swords:
            early = [c for c in self._direct_path() if c not in taken]
            if early:
                c = early[max(0, len(early) // 4)]
                self.swords.append(c)
                taken.add(c)
            self.swords.extend(self._place(taken, s.swords - len(self.swords)))
        # Health, not a position: a hostile is now a thing with a state
        # rather than a coordinate that is either there or deleted. Kept as
        # a parallel list so every existing `for h in self.hostiles` that
        # only wants the position goes on working unchanged.
        self.hostiles = (self._place_guards(taken, s.hostiles) if s.guards
                         else self._place(taken, s.hostiles))
        self.hostile_hp = [s.hostile_hp] * len(self.hostiles)
        self.armed = False
        self.kills = 0
        self.hits_taken = 0
        self.met = False          # did a hostile ever come within sight?

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

    def _view(self, x, y):
        """What the agent SEES at a cell, as opposed to what the map is.

        Swords and hostiles are kept out of `self.grid` on purpose - every
        route check in this file tests `== T_FLOOR` to find a free cell, and
        writing an item into the terrain would quietly make its tile
        unreachable to the maze validator, the hazard placer and the oracle
        at once.

        So the terrain stays terrain and this is the overlay. Both ids are
        the real world's: TILE_HOSTILE and TILE_OBJECT are what world.py
        renders an entity and a ground item as, which is the whole reason
        this is worth doing rather than adding curriculum-only ids - the
        tile embedding then means the same thing on both sides.

        Without it there is no way to learn to pick a sword up. Items have
        never been in this observation at all: `good` and `trash` are placed,
        rewarded on ACT_PICK_UP, and invisible, so every can_pick rung has
        been rewarding a lottery.
        """
        raw = self._tile(x, y)
        if raw == T_WALL:
            return raw
        if (x, y) in self.hostiles:
            return T_HOSTILE
        if (x, y) in self.swords or (x, y) in self.good or (x, y) in self.trash:
            return T_OBJECT
        return raw

    def _aim(self):
        """What the agent is currently trying to reach, and how far it is.

        Unarmed in a room that has a weapon in it, the thing to head for is
        the weapon; after that, the goal. This is one target at a time, never
        two, because the object slots hold one object.

        Why this is needed at all
        -------------------------
        The sword was visible in the patch and remembered on the map, and
        that was not enough. Trained 400 episodes on `armed` from a brain
        that had passed every senior rung, the policy picked up a sword in
        **0%** of episodes and killed **0.00** hostiles - it walked out the
        clock at 13% success while a competent agent cleared 100%.

        It was not losing fights. It never had one. ACT_PICK_UP had no
        gradient anywhere on the ladder: seeing a tile two squares away does
        not tell a policy to walk to it, and finding one exact tile in a
        15x15 maze by chance and then pressing the right button on it is the
        "single goal found by luck" problem this file already documents as
        unlearnable. The goal is learnable because it has a bearing and
        shaping; the sword had neither.

        Pointing the OBJECT slots at it is also the more faithful reading of
        the real world, not a curriculum-only hack: IDX_OBJ_ID and the first
        direction pair are `nearest object` in agent.build_state, and a
        sword lying on the floor is an object. The curriculum has been
        putting an abstract goal in them.
        """
        if (self.stage.weapon_first and self.swords and not self.armed
                and self.stage.hostiles):
            pos, d = self._nearest(self.swords)
            return pos, d, "sword"
        pos, d = self._nearest(self.goals)
        return pos, d, "goal"

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
                    # Terrain and items are remembered; hostiles are not.
                    # A hostile that walks away would otherwise leave a
                    # permanent ghost on the remembered map, and the agent
                    # would spend the rest of the episode routing around
                    # somewhere nothing is standing.
                    t = self.grid[y][x]
                    if t != T_WALL and ((x, y) in self.swords
                                        or (x, y) in self.good
                                        or (x, y) in self.trash):
                        t = T_OBJECT
                    self.seen[(x, y)] = t
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
        s[10] = float(self._view(self.ax, self.ay - 1))
        s[11] = float(self._view(self.ax - 1, self.ay))
        s[12] = float(self._view(self.ax + 1, self.ay))
        s[13] = float(self._view(self.ax, self.ay + 1))

        host, hd = self._nearest(self.hostiles)
        s[14] = min(1.0, hd / span)
        s[config_rl.IDX_ENT_TYPE] = 1.0 if host else 0.0

        aim, ad, kind = self._aim()
        s[18] = min(1.0, ad / span)
        if aim is None:
            s[config_rl.IDX_OBJ_ID] = 0.0
        elif kind == "sword":
            s[config_rl.IDX_OBJ_ID] = STONE_SWORD / float(config_rl.ITEM_ID_SCALE)
        else:
            s[config_rl.IDX_OBJ_ID] = 1.0 / config_rl.ITEM_ID_SCALE

        # What we are carrying, in the real inventory slots. Until now the
        # curriculum left all seven of them at zero, so a policy trained here
        # had never once seen the block it is supposed to read in the full
        # world - and, more immediately, "am I armed?" was not in the
        # observation at all. Without it the sword is not a decision: picking
        # it up changes nothing the policy can see, so the extra damage looks
        # like the same swing sometimes working and sometimes not.
        #
        # STONE_SWORD is item 0013 in the real registry, written the way
        # agent.build_state writes it, so the slot means the same thing on
        # both sides.
        if self.armed:
            inv = config_rl.IDX_INV_START
            s[inv + 0] = STONE_SWORD / float(config_rl.ITEM_ID_SCALE)
            s[inv + 1] = 1.0 / float(config_rl.ITEM_ID_SCALE)

        # Is it coming for me, and what does it land if it arrives? Slots 16
        # and 17 are `ent_state` and `incoming` in agent.build_state, and the
        # curriculum has always left both at zero. They are written here with
        # the same meaning the real world gives them: 2 is entities.py's
        # "attacking", and `incoming` is the damage one blow costs.
        #
        # Enemy hit points deliberately do NOT go in the vector. There is no
        # slot for them in the real world's 0..53 block, and inventing one
        # would change STATE_SIZE and strand every weight trained so far.
        # The decision the agent actually has to make is "am I armed", which
        # the inventory slots above now carry.
        if host and self.stage.chase:
            close = self.stage.aggro == 0 or hd <= self.stage.aggro
            s[16] = 2.0 if close else 0.0
            s[17] = (self.stage.bite / 100.0) if hd <= 1 else 0.0

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
        if aim:
            s[d + 0], s[d + 1] = _unit(aim[0] - self.ax, aim[1] - self.ay)
        if host:
            s[d + 2], s[d + 3] = _unit(host[0] - self.ax, host[1] - self.ay)

        # The 5x5 patch around the agent
        p = config_rl.IDX_PATCH_START
        r = config_rl.PATCH_RADIUS
        i = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                s[p + i] = float(self._view(self.ax + dx, self.ay + dy))
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
        _, before, aimed_at = self._aim()
        reward = s.step_cost
        self.steps += 1

        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = self.ax + dx, self.ay + dy
            if self._tile(nx, ny) == T_WALL:
                reward += R_WALL if s.punitive else 0.0
            else:
                self.ax, self.ay = nx, ny

        # On a pick_goals rung the goal is taken with the hand rather than by
        # standing on it. Noted here and settled with all the other goal
        # bookkeeping below, so trails and clear-the-room stay in one place.
        took = (action == config_rl.ACT_PICK_UP and s.can_pick and s.pick_goals
                and (self.ax, self.ay) in self.goals)

        if action == config_rl.ACT_PICK_UP and s.can_pick and not took:
            here = (self.ax, self.ay)
            if here in self.swords:
                self.swords.remove(here)
                # Only the first one is worth anything. Otherwise a room with
                # two swords pays twice for the same lesson, and on the grown
                # rungs that is a free score that has nothing to do with the
                # fight.
                reward += 0.0 if self.armed else R_SWORD
                self.armed = True
                self.seen.pop(here, None)
            elif here in self.good:
                self.good.remove(here)
                reward += R_GOOD
                self.seen.pop(here, None)
            elif here in self.trash:
                self.trash.remove(here)
                reward += R_TRASH
                self.seen.pop(here, None)

        elif action == config_rl.ACT_ATTACK and s.can_attack:
            here = (self.ax, self.ay)
            hit = [i for i, h in enumerate(self.hostiles)
                   if abs(h[0] - here[0]) + abs(h[1] - here[1]) <= 1]
            if hit:
                i = hit[0]
                dmg = SWORD_DAMAGE if self.armed else FIST_DAMAGE
                self.hostile_hp[i] -= dmg
                if self.hostile_hp[i] <= 0:
                    self.hostiles.pop(i)
                    self.hostile_hp.pop(i)
                    self.kills += 1
                    reward += R_KILL
                else:
                    reward += R_HIT
            else:
                # Swinging at nothing. Free otherwise, and an action that
                # costs nothing gets mashed.
                reward += R_SWING if s.punitive else 0.0

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

        # Shaping, after the move: progress towards whatever is being aimed
        # at - the weapon while unarmed, the goal after that.
        #
        # Skipped on the tick the aim CHANGES. Picking a sword up switches
        # the target from a tile underfoot to a goal across the room, and
        # shaping that as though it were movement would charge the agent
        # twenty tiles of "progress away" for doing exactly the right thing.
        # This is the same trap as respawning a goal at the instant of
        # reward, which is what made the first nursery unlearnable.
        _, after, now_aimed = self._aim()
        if s.shaped and now_aimed == aimed_at and (self.goals or self.swords):
            reward += R_TOWARDS * (before - after)

        # Reaching a goal - or, where the rung says so, reaching it AND
        # picking it up. `took` is set by the pick-up branch above.
        here = (self.ax, self.ay)
        if s.pick_goals and not took:
            pass
        elif here in self.goals:
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
        # The hostiles move AFTER the agent has acted, so an attack lands on
        # where the thing was when the agent could see it. Stepping them
        # first would mean swinging at a tile it had already left, which
        # reads to the policy as ATTACK working at random.
        self._step_hostiles()
        if self.hostiles:
            _, hostd = self._nearest(self.hostiles)
            if hostd <= config_rl.PATCH_RADIUS:
                self.met = True

        if self._tile(self.ax, self.ay) == T_HAZARD:
            reward += R_HAZARD
        if s.damage:
            if self._tile(self.ax, self.ay) == T_HAZARD:
                self.health -= s.hazard_damage
            # A blow is the hostile's turn, on the same clock as its move.
            if self.steps % s.tempo == 0:
                for h in self.hostiles:
                    if abs(h[0] - self.ax) + abs(h[1] - self.ay) <= 1:
                        self.health -= s.bite
                        self.hits_taken += 1
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
            "kills": self.kills, "armed": self.armed, "met": self.met,
            "hits_taken": self.hits_taken,
        }


# ---- the guard that stops the original bug coming back -------------------

def _first_step(env, stage, targets, block):
    """First move of a shortest route to any of `targets`, or None."""
    from collections import deque
    n = stage.grid
    start = (env.ax, env.ay)
    targets = set(targets)
    if not targets:
        return None
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
            if (nx, ny) in block and (nx, ny) not in targets:
                continue
            prev[(nx, ny)] = (x, y)
            if (nx, ny) in targets:
                end = (nx, ny)
                break
            q.append((nx, ny))
    if end is None:
        return None
    cur = end
    while prev[cur] != start:
        cur = prev[cur]
    want = (cur[0] - start[0], cur[1] - start[1])
    for act, d in MOVES.items():
        if d == want:
            return act
    return None


def oracle_step(env, stage):
    """One step along a shortest route to the nearest goal, avoiding harm.

    Combat, added 2026-09-25
    ------------------------
    The avoid-everything version below cannot play a rung whose hostiles
    chase: every cell adjacent to a pursuer is a `bite` cell, a pursuer
    follows, so the careful pass finds no route at all and the reckless pass
    walks the whole episode taking a blow a tick. It died in most rooms and
    reported the tier as unplayable, which would have had preflight refuse
    to train rungs that are in fact fine - the exact failure this function
    was last rewritten to fix, in a new costume.

    A competent player of a combat rung arms itself and kills what corners
    it, so that is what this does, in priority order.

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

    bite = set()
    if stage.damage:
        for h in env.hostiles:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if abs(dx) + abs(dy) <= 1:
                        bite.add((h[0] + dx, h[1] + dy))

    # 0. Standing on the thing we were sent for, on a rung where arriving is
    #    not enough. Nothing else can be worth a tick more than this.
    if stage.pick_goals and stage.can_pick and start in goals:
        return config_rl.ACT_PICK_UP

    # 1. Standing on a weapon with empty hands. Always worth one tick.
    if stage.can_pick and not env.armed and start in env.swords:
        return config_rl.ACT_PICK_UP

    # 2. Unarmed, with a weapon reachable without walking into anything.
    #    Arming first is what makes these rungs winnable and is the whole
    #    behaviour they exist to teach.
    #
    #    ORDER MATTERS, and getting it wrong cost a measurable 20% of
    #    episodes. The first version put "something is adjacent" above this
    #    and had it break off toward the sword; the next tick nothing was
    #    adjacent, so this rule routed back toward the sword THROUGH the
    #    guard, and the two rules traded the agent back and forth beside a
    #    hostile that hit it every tick. Static guards, which cannot even
    #    follow, killed a competent player 19% of the time - more often than
    #    the rung where they chase.
    #
    #    Asking for the sword first, and only fighting when it cannot be
    #    had safely, is a decision that cannot oscillate: the two branches
    #    are now mutually exclusive rather than alternating.
    if stage.can_pick and not env.armed and env.swords and env.hostiles:
        act = _first_step(env, stage, env.swords, bite | set(env.hazards))
        if act is not None:
            return act

    # 3. Something is in reach, and either we are armed or the weapon was
    #    not safely reachable. Either way the fight is the plan now.
    if stage.can_attack and env.hostiles:
        if any(abs(h[0] - start[0]) + abs(h[1] - start[1]) <= 1
               for h in env.hostiles):
            return config_rl.ACT_ATTACK

    if not goals:
        return config_rl.ACT_WAIT

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
