# -*- coding: utf-8 -*-
"""ATS main entrypoint — simulation runner wired to the Mission Control GUI.

App-state flow:
  GUI starts at MENU → user configures & clicks START
  → RUNNING (sim loop active)
  → PAUSED  (sim frozen, config editable)
  → STOPPED (episode target met or user stopped)
  → back to MENU via 🏠 button
"""

import argparse
import time
import torch

import brain
import config_rl
import survival
import trace as pathtrace
from ats_env import ATSEnvironment
from ats_virus_adapter import VirusAdapter
from debug_gui import (
    TerminalGUI, STATE_MENU, STATE_RUNNING, STATE_PAUSED, STATE_STOPPED,
    TAB_SIM, TAB_ANALYTICS, TAB_CONFIG, TAB_MIND
)
from mind.continual_learner import ContinualLearner
from mind.solution_loop import SolutionLoop
from model_rl import RLPolicy


# ---------------------------------------------------------------------------
def parse_args():
# ---------------------------------------------------------------------------
    p = argparse.ArgumentParser(description="ATS Mission Control")
    p.add_argument("--episodes",      type=int,   default=None,
                   help="Episodes to run (default: config_rl.EPISODES)")
    p.add_argument("--ticks",         type=int,   default=None,
                   help="Max ticks per episode (default: config_rl.MAX_TICKS)")
    p.add_argument("--train",         action="store_true",
                   help="Run batch PPO training (headless, uses train_rl.py)")
    p.add_argument("--no-gui",        action="store_true",
                   help="Headless mode — skips GUI entirely")
    p.add_argument("--gui",           action="store_true",
                   help="Explicit GUI mode (same as default)")
    p.add_argument("--fresh",         action="store_true",
                   help="Ignore existing checkpoint, start fresh")
    p.add_argument("--virus-profile", default=config_rl.VIRUS_PROFILE,
                   choices=["ats", "general"],
                   help="Virus LM profile (default: ats)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def _make_mind(fresh=False):
# ---------------------------------------------------------------------------
    """Policy, learner and ladder progress - all from brain.pt.

    Built together because they are one thing. The learner has to exist before
    the brain is opened: Adam's moments are half of what was saved, and a
    policy restored without them spends its first few hundred updates
    re-tuning the optimiser, which looks exactly like forgetting.

    Returns (policy, learner, progress).
    """
    shape  = {} if fresh else brain.saved_shape()
    hidden = int(shape.get("hidden_size", config_rl.MIND_HIDDEN_SIZE))

    if fresh:
        policy  = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
        learner = ContinualLearner(policy=policy)
        print(f"[ATS] Fresh policy (hidden={policy.hidden_size})")
        return policy, learner, {}

    policy  = RLPolicy(hidden_size=hidden)
    learner = ContinualLearner(policy=policy)

    if brain.DEFAULT_PATH.exists():
        if brain.is_loadable(policy):
            progress = brain.load(policy, learner)  # prints its own summary
            return policy, learner, progress
        # A brain from an older network shape. brain.load() prints exactly
        # which tensors do not fit; say it out loud, then fall through to the
        # legacy files rather than silently handing back a random policy.
        brain.load(policy, learner)
        print("[ATS] brain.pt belongs to an older network and cannot be "
              "resumed - looking for weights that fit.")

    # No usable brain - adopt whatever the old two-file scheme left behind,
    # so a run that predates this change keeps its weights instead of
    # restarting from noise. The ladder still starts at the nursery: those
    # weights never climbed one.
    if brain.adopt_legacy(policy):
        return policy, learner, {}

    print(f"[ATS] No brain to resume - fresh policy (hidden={policy.hidden_size})")
    return policy, learner, {}


# ---------------------------------------------------------------------------
def _progress(passed, episodes, rung=None):
# ---------------------------------------------------------------------------
    """What the ladder has to remember between sessions.

    Written whole on every save, so it must carry forward what was already
    there: handing brain.save() a bare {"episodes": n} would erase the list of
    rungs the curriculum runner had recorded - and, since survival.py,
    the tick cap the agent has earned. Leaving `survival_rung` out of a
    save resets the episode length to the shortest rung on the next load,
    silently, which is the same bug the paragraph above describes.
    """
    out = {"passed": list(passed), "episodes": int(episodes)}
    if rung is not None:
        out["survival_rung"] = int(rung)
    return out


# ---------------------------------------------------------------------------
def _save_brain(policy, learner=None, progress=None):
# ---------------------------------------------------------------------------
    """One authoritative file, plus a mirror for the older tools.

    brain.pt is what is loaded; model_rl.pt / model_best.pt are written so
    eval_behaviour, run_benchmark and the virus adapter keep working, but
    nothing reads them back into a training run any more.
    """
    path = brain.save(policy, learner, progress)

    payload = {"model_state_dict": policy.state_dict(),
               "state_dict": policy.state_dict(),
               "hidden_size": policy.hidden_size,
               "state_size": policy.state_size,
               "action_size": policy.action_size,
               "reward_scheme_version": config_rl.REWARD_SCHEME_VERSION}
    torch.save(payload, config_rl.MODEL_PATH)
    torch.save(payload, config_rl.BEST_MODEL_PATH)

    passed = len((progress or {}).get("passed") or [])
    print(f"[ATS] Brain saved -> {path.name} "
          f"(hidden={policy.hidden_size}, {passed} rung(s) passed)")


# ---------------------------------------------------------------------------
_REHEARSE_RATE = 0.40


def _rehearsal_env(gui, current):
    """Now and then, an earlier rung instead of the one being trained.

    Two episodes in five, weighted toward the rungs nearest this one - the
    same rate and the same weighting train_curriculum uses, for the same
    reason. Uniform over the whole ladder rehearses each rung about 2% of
    the time, which is a rounding error rather than rehearsal.

    Returns the environment to play next. Rehearsed episodes still train the
    policy; they are simply not the rung being scored for promotion, which
    is handled by `gui.active_stage` staying put.
    """
    import random

    from curriculum_adapter import StageSession, stage_by_name
    from debug_gui import TRACKS

    stage = getattr(gui, "active_stage", None)
    if stage is None:
        return current                    # the island, which has no ladder
    _name, rungs, _blurb, _open = TRACKS[gui.selected_track]
    here = [stage_by_name(r) for r in rungs]
    here = [s for s in here if s is not None]
    behind = []
    for s in here:
        if s.name == stage.name:
            break
        behind.append(s)
    if not behind or random.random() >= _REHEARSE_RATE:
        return current
    weights = [i + 1 for i in range(len(behind))]
    return StageSession(random.choices(behind, weights=weights)[0])


def _env_for_selection(gui):
    """The environment the menu's current selection means.

    A curriculum rung becomes a StageSession, which is a CurriculumEnv wearing
    the full environment's surface so the panel has what it draws. Full World,
    and anything whose rungs are not built yet, stays the island.

    Also tells the GUI which stage is live and what it permits, so the action
    toggles and the inventory grey out what this rung has not taught instead of
    showing twenty-five live buttons — which is what active_stage and
    active_mask were declared for and never given.
    """
    from debug_gui import TRACKS
    from curriculum_adapter import StageSession, stage_by_name

    name, rungs, _blurb, open_ = TRACKS[gui.selected_track]

    stage = None
    if open_ and rungs:
        # The first rung of the track that the brain has not already passed,
        # so PRIMARY resumes where it stopped rather than restarting at its
        # easiest room every time.
        # The brain records a rung as "name WxH", which is how the menu card
        # counts them too — matching on the bare name would never hit, and
        # every track would restart at its easiest room.
        from debug_gui import brain_progress
        passed = set(brain_progress().get("passed") or [])
        for r in rungs:
            st = stage_by_name(r)
            if st is None:
                continue
            if f"{st.name} {st.grid}x{st.grid}" not in passed:
                stage = st
                break
        if stage is None:
            stage = stage_by_name(rungs[-1])

    if stage is None:
        gui.active_stage = None
        gui.active_mask = None
        print(f"[ATS] {name}: full island")
        return ATSEnvironment(max_ticks=survival.cap_for(getattr(gui, "progress", None)))

    session = StageSession(stage)
    gui.active_stage = stage
    gui.active_mask = session.action_mask()
    print(f"[ATS] {name}: rung '{stage.name}', {stage.grid}x{stage.grid} room")
    return session


def run_gui(args):
# ---------------------------------------------------------------------------
    gui = TerminalGUI()
    gui.splash(duration=1.0)

    env     = None
    adapter = VirusAdapter(profile=args.virus_profile)
    policy  = None
    sl      = None     # SolutionLoop
    learner = None

    # We track a pending episode count override from menu
    target_episodes = args.episodes  # None = use config_rl.EPISODES

    # -------------------------------------------------------------------------
    # Outer loop: runs while the window is alive.
    # Each iteration of the outer loop = one complete session (menu → run → stop).
    # -------------------------------------------------------------------------
    while gui.alive:

        # ====== MENU PHASE ======
        gui.app_state  = STATE_MENU
        gui.active_tab = 0


        while gui.alive and gui.app_state == STATE_MENU:
            gui.render()   # draws home screen
            if gui.sig_fresh_reset:
                gui.sig_fresh_reset = False
                args.fresh = True
                print("[ATS] User requested fresh start from scratch.")
            if gui.sig_start:
                gui.sig_start = False
                gui.app_state = STATE_RUNNING
                break

        if not gui.alive:
            break

        # ====== SETUP for this session ======
        gui.clear_session_data()
        policy, learner, progress = _make_mind(args.fresh)
        policy.eval()
        sl      = SolutionLoop(policy=policy)
        if not args.fresh:
            sl.experience.load()

        # The ladder's own promotion test, run here as well as in
        # train_curriculum: a rung the GUI clears has to be recorded, or
        # _env_for_selection resumes at the same easiest room forever and the
        # ladder can only ever be climbed from the command line.
        from collections import deque
        passed_rungs  = list(progress.get("passed") or [])
        base_episodes = int(progress.get("episodes") or 0)
        window        = deque()
        # Did each of the recent island episodes reach its cap? That is what
        # earns a longer one - see survival.py.
        reached       = deque()

        # Which rung the menu was sitting on when LAUNCH was pressed.
        #
        # This used to be ignored: the session built ATSEnvironment() whatever
        # the button said, so picking NURSERY still dropped the agent on the
        # island. selected_track was read nowhere outside the GUI.
        env = _env_for_selection(gui)
        adapter.write_active_profile()

        # Where it walked, and whether the walk made sense. Return alone
        # cannot tell a two-tile solve from a sixty-tile accident that
        # happened to end on the goal; the detour ratio can, and it moves
        # long before return does. See trace.py.
        tracer = pathtrace.PathTracer(per_tick=config_rl.TRACE_PER_TICK)

        num_eps = target_episodes if target_episodes is not None else config_rl.EPISODES
        ep      = 0
        state   = env.reset()
        sl.reset_memory()
        tracer.begin(env, ep + 1)
        done    = False
        ep_rewards = []
        info    = {}

        # ====== RUN / PAUSE PHASE ======
        while gui.alive and gui.app_state in (STATE_RUNNING, STATE_PAUSED):

            # ---- Consume GUI signals ----------------------------------------
            if gui.sig_pause_toggle:
                gui.sig_pause_toggle = False
                gui.app_state = STATE_PAUSED if gui.app_state == STATE_RUNNING else STATE_RUNNING

            if gui.sig_save:
                gui.sig_save = False
                _save_brain(policy, learner, _progress(passed_rungs, base_episodes + ep))
                sl.experience.save()
                if env: env.agent.event_log.append(("Checkpoint & Memory saved!", time.time()))

            if gui.sig_fresh_reset and gui.app_state == STATE_PAUSED:
                gui.sig_fresh_reset = False
                policy, learner, _ = _make_mind(fresh=True)
                sl      = SolutionLoop(policy=policy)
                sl.experience.entries.clear()
                mem_file = config_rl.DATA_DIR / "experience_memory.json"
                if mem_file.exists():
                    mem_file.unlink(missing_ok=True)
                passed_rungs, base_episodes, window = [], 0, deque()
                reached = deque()
                if env: env.agent.event_log.append(("Policy & Memory reset to fresh!", time.time()))

            if gui.sig_stop:
                gui.sig_stop = False
                sl.experience.save()
                tracer.close()
                gui.app_state = STATE_STOPPED
                break

            # ---- PAUSED — just render, don't step --------------------------
            if gui.app_state == STATE_PAUSED:
                gui.render(env, needs=info.get("needs",[0]*4), trace=sl.trace)
                continue

            # ---- Episode boundary -------------------------------------------
            if done:
                sl.on_episode_end()
                # flush() is a real PPO update, and in a curriculum rung it is
                # the ONLY one: CONTINUAL_UPDATE_EVERY is 256 ticks, the rungs
                # run 120-200, and flush resets the tick counter at every
                # episode end — so maybe_update can never reach its threshold.
                # Its losses were never handed to the panel, which is why the
                # PPO history stayed empty while training was in fact running.
                if learner.flush() and learner.last_losses:
                    gui.record_loss(*learner.last_losses)
                walk = tracer.end(env, info)
                total_rew = sum(ep_rewards)
                from rewards import RewardEngine
                grade = RewardEngine.classify_score(total_rew)
                prog_eff = env.rewards.compute_progression_efficiency(env.tick)
                # None = this run has nothing to measure (a curriculum rung).
                # Kept as None through record_episode so the history says so
                # too, and rendered as "n/a" wherever it is printed.
                eff_str = " n/a" if prog_eff is None else f"{prog_eff:.3f}"
                caps_count = len(env.agent.capabilities)

                if env.agent.health <= 0:
                    reason = f"Perished (HP:{env.agent.health})"
                elif env.tick >= env.max_ticks:
                    reason = f"Survived the cap ({env.max_ticks})"
                else:
                    reason = "Episode complete"

                gui.record_episode(ep, total_rew, env.tick, reason, grade=grade, efficiency=prog_eff, capabilities=caps_count)

                # Earning a longer episode. Only on the island - a
                # curriculum rung has its own max_steps and is not what this
                # ladder is about.
                if getattr(gui, "active_stage", None) is None:
                    reached.append(env.tick >= env.max_ticks)
                    while len(reached) > survival.WINDOW:
                        reached.popleft()
                    if survival.earned(progress, reached):
                        cap = survival.promote(progress)
                        reached.clear()
                        print("[ATS] survived %d of the last %d - episode cap "
                              "is now %s" % (survival.WINDOW, survival.WINDOW,
                                             survival.label(progress)))
                        _save_brain(policy, learner,
                                    _progress(passed_rungs, base_episodes + ep,
                                              progress.get("survival_rung")))
                        env.max_ticks = cap

                narration = adapter.generate_narration(
                    f"### INPUT: ATS ep {ep} reward {total_rew:.2f} grade {grade}\n### OUTPUT:")
                gui.record_narration(narration)
                adapter.export_runtime_corpus([
                    f"episode {ep} summary|reward {total_rew:.2f}|grade {grade}|efficiency {eff_str}|caps {caps_count}|({reason})",
                    f"episode {ep} narration|{narration}",
                ])
                # The walk, next to the score. A rising return with a flat
                # detour ratio means it is collecting by accident; the two
                # moving together is what learning looks like.
                detour = walk.get("detour_ratio")
                walk_str = ("no goal" if detour in ("", None)
                            else f"x{detour:.2f} ({walk['walk_steps']}/{walk['optimal_steps']})")
                print(f"Episode {ep} done | reward={total_rew:.2f} | GRADE: {grade:4s} | "
                      f"Caps={caps_count} | Eff={eff_str} | walk={walk_str} | "
                      f"bumps={walk['wall_bumps']} | ticks={env.tick} | {reason}")

                # ---- Did this rung's own promotion test just pass? -------
                # The same rule train_curriculum uses: a fraction `pass_rate`
                # of the last `window` episodes counted as a success. Recorded
                # into the brain and then re-selected, so clearing the nursery
                # in the GUI actually moves the agent up a room instead of
                # replaying the same one until the episode budget runs out.
                learner.note_outcome(bool(info.get("success")))
                stage = getattr(gui, "active_stage", None)
                if stage is not None:
                    window.append(1.0 if info.get("success") else 0.0)
                    while len(window) > stage.window:
                        window.popleft()
                    rung = f"{stage.name} {stage.grid}x{stage.grid}"
                    rate = (sum(window) / len(window)) if window else 0.0
                    if (rung not in passed_rungs
                            and len(window) >= stage.window
                            and rate >= stage.pass_rate):
                        passed_rungs.append(rung)
                        window.clear()
                        # Written before re-selecting: _env_for_selection asks
                        # the brain on disk which rung comes next.
                        _save_brain(policy, learner,
                                    _progress(passed_rungs, base_episodes + ep))
                        print(f"[ATS] PASSED {rung} - {rate:.0%} over "
                              f"{stage.window} episodes (needed {stage.pass_rate:.0%})")
                        env = _env_for_selection(gui)
                        state, done, ep_rewards = env.reset(), False, []
                        sl.reset_memory()
                        tracer.begin(env, ep + 1)
                        continue

                # Check if target reached
                if ep >= num_eps:
                    _save_brain(policy, learner, _progress(passed_rungs, base_episodes + ep))
                    tracer.close()
                    sl.experience.save()
                    gui.app_state = STATE_STOPPED
                    break

                # Next episode, on this rung or on one below it.
                #
                # Rehearsal, which this loop did not have. train_curriculum
                # mixes earlier rungs into every rung's training because
                # ContinualLearner is online PPO with no replay - train one
                # room long enough and the weights leave every other room
                # behind. The GUI trained a single stage, episode after
                # episode, with nothing protecting the rest of the ladder.
                #
                # Measured, 600 GUI episodes on `avoid 7x7`:
                #
                #     rung        before   after
                #     satchel      98%      38%
                #     satchel7     92%      30%
                #     armed 15x15  62%      40%
                #     warden 17x17 62%      32%
                #     senior 19x19 62%      32%
                #     avoid 7x7    90%      48%   <- the rung being trained
                #
                # The last line is the one that matters: six hundred episodes
                # of training `avoid` made `avoid` worse. That is not
                # forgetting, it is thrashing, and rehearsal is what holds a
                # rung still long enough to converge.
                env = _rehearsal_env(gui, env)
                state      = env.reset()
                sl.reset_memory()
                done       = False
                ep_rewards = []
                tracer.begin(env, ep + 1)

            # ---- First tick of a new episode --------------------------------
            if not ep_rewards:      # fresh episode
                ep += 1

            # ---- Simulation step -------------------------------------------
            mask   = env.agent.get_action_mask(env.world, env.day_night)
            action = sl.choose_action(state, mask)

            # Kept before env.step overwrites it: PPO re-evaluates
            # log pi(a|s) from the state stored with the transition, so it
            # has to be the state the action was chosen in. See
            # train_curriculum.run_episode for the full note.
            prev_state = state
            state, reward, done, info = env.step(
                action,
                disabled_actions=gui.disabled_actions,
                reward_rules=gui.reward_rules,
            )
            ep_rewards.append(reward)
            tracer.record(env, action, reward, info)

            # Record step outcome in solution loop
            sl.record_outcome(action, reward, state)

            learner.collect(state=prev_state, action=action, reward=reward,
                            log_prob=sl.last_log_prob, value=sl.last_value,
                            action_mask=mask, done=done, hidden=sl.last_hidden)
            if learner.maybe_update() and learner.last_losses:
                gui.record_loss(*learner.last_losses)

            # ---- Render (skip some frames in turbo) -------------------------
            render_this = not gui.turbo_mode or (env.tick % 20 == 0)
            if render_this:
                if not gui.render(env,
                                  needs=info.get("needs",[0]*4),
                                  trace=sl.trace):
                    break
            else:
                gui.process_events()   # still pump events even if not rendering

        # ====== STOPPED PHASE ======
        # Stay in stopped state until user clicks 🏠 MENU or closes window
        gui.app_state = STATE_STOPPED
        while gui.alive and not gui.sig_return_home:
            gui.render(env, needs=info.get("needs",[0]*4) if info else [0]*4,
                       trace=sl.trace if sl else {})

        gui.sig_return_home = False
        # Loop back to MENU

    gui.close()


# ---------------------------------------------------------------------------
def run_headless(args):
# ---------------------------------------------------------------------------
    """Fast batch training with no window — for scripting/CI."""
    if args.ticks:
        config_rl.MAX_TICKS = args.ticks
    num_eps = args.episodes or config_rl.EPISODES

    policy, learner, progress = _make_mind(args.fresh)
    policy.eval()
    sl      = SolutionLoop(policy=policy)
    # The cap the agent has earned, not the constant. See survival.py: the
    # world runs at 63 ticks a second, so a ten-day episode is ten minutes
    # and six an hour - short episodes while it is bad, long ones once it
    # can use them.
    env     = ATSEnvironment(max_ticks=survival.cap_for(progress))
    print("[ATS] episode cap: %s" % survival.label(progress))
    adapter = VirusAdapter(profile=args.virus_profile)
    adapter.write_active_profile()

    from rewards import RewardEngine

    for ep in range(1, num_eps + 1):
        state      = env.reset()
        sl.reset_memory()
        done       = False
        ep_rewards = []

        while not done:
            mask   = env.agent.get_action_mask(env.world, env.day_night)
            action = sl.choose_action(state, mask)
            prev_state = state
            state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            learner.collect(state=prev_state, action=action, reward=reward,
                            log_prob=sl.last_log_prob, value=sl.last_value,
                            action_mask=mask, done=done, hidden=sl.last_hidden)
            learner.maybe_update()

        learner.flush()
        total = sum(ep_rewards)
        grade = RewardEngine.classify_score(total)
        prog_eff = env.rewards.compute_progression_efficiency(env.tick)
        eff_str = " n/a" if prog_eff is None else f"{prog_eff:.3f}"
        caps_count = len(env.agent.capabilities)
        reason = "MaxTicks" if env.tick >= config_rl.MAX_TICKS else "Died"
        print(f"Ep {ep:4d} | reward={total:+7.2f} | GRADE: {grade:4s} | Caps={caps_count} | Eff={eff_str} | ticks={env.tick} | {reason}")

    _save_brain(policy, learner, progress)



# ---------------------------------------------------------------------------
def main():
# ---------------------------------------------------------------------------
    args = parse_args()
    if args.train:
        from train_rl import train
        train(fresh=args.fresh)
        return
    if args.no_gui:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
