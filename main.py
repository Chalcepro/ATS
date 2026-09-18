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

import config_rl
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
def _load_policy(fresh=False):
# ---------------------------------------------------------------------------
    if fresh:
        policy = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
        print(f"[ATS] Fresh policy (hidden={policy.hidden_size})")
        return policy, False
    for ckpt in [config_rl.BEST_MODEL_PATH, config_rl.MODEL_PATH]:
        if ckpt.exists():
            try:
                data = torch.load(ckpt, map_location="cpu")
                version = data.get("reward_scheme_version") if isinstance(data, dict) else None
                if version != config_rl.REWARD_SCHEME_VERSION:
                    print(f"[ATS] SKIPPING {ckpt.name}: reward_scheme_version={version!r} "
                          f"does not match current {config_rl.REWARD_SCHEME_VERSION!r} — "
                          f"its critic was calibrated under a different reward/return scheme "
                          f"and resuming from it would silently reproduce old behaviour.")
                    continue
                hidden = data.get("hidden_size", config_rl.MIND_HIDDEN_SIZE) if isinstance(data, dict) else config_rl.MIND_HIDDEN_SIZE
                policy = RLPolicy(hidden_size=hidden)
                sd   = data.get("model_state_dict", data.get("state_dict", data)) if isinstance(data, dict) else data
                policy.load_state_dict(sd)
                print(f"[ATS] Loaded checkpoint {ckpt.name} (hidden={policy.hidden_size})")
                return policy, True
            except Exception as e:
                print(f"[ATS] Checkpoint load note: {e}")
                continue
    policy = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
    print(f"[ATS] No matching checkpoint — fresh policy (hidden={policy.hidden_size})")
    return policy, False


# ---------------------------------------------------------------------------
def _save_policy(policy):
# ---------------------------------------------------------------------------
    config_rl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": policy.state_dict(),
               "state_dict": policy.state_dict(),
               "hidden_size": policy.hidden_size,
               "state_size": policy.state_size,
               "action_size": policy.action_size,
               "reward_scheme_version": config_rl.REWARD_SCHEME_VERSION}
    torch.save(payload, config_rl.MODEL_PATH)
    torch.save(payload, config_rl.BEST_MODEL_PATH)
    print(f"[ATS] Checkpoint saved (hidden={policy.hidden_size})")


# ---------------------------------------------------------------------------
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
        return ATSEnvironment()

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
        policy, _ = _load_policy(args.fresh)
        policy.eval()
        sl      = SolutionLoop(policy=policy)
        if not args.fresh:
            sl.experience.load()
        learner = ContinualLearner(policy=policy)

        # Which rung the menu was sitting on when LAUNCH was pressed.
        #
        # This used to be ignored: the session built ATSEnvironment() whatever
        # the button said, so picking NURSERY still dropped the agent on the
        # island. selected_track was read nowhere outside the GUI.
        env = _env_for_selection(gui)
        adapter.write_active_profile()

        num_eps = target_episodes if target_episodes is not None else config_rl.EPISODES
        ep      = 0
        state   = env.reset()
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
                _save_policy(policy)
                sl.experience.save()
                if env: env.agent.event_log.append(("Checkpoint & Memory saved!", time.time()))

            if gui.sig_fresh_reset and gui.app_state == STATE_PAUSED:
                gui.sig_fresh_reset = False
                policy  = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
                sl      = SolutionLoop(policy=policy)
                sl.experience.entries.clear()
                mem_file = config_rl.DATA_DIR / "experience_memory.json"
                if mem_file.exists():
                    mem_file.unlink(missing_ok=True)
                learner = ContinualLearner(policy=policy)
                if env: env.agent.event_log.append(("Policy & Memory reset to fresh!", time.time()))

            if gui.sig_stop:
                gui.sig_stop = False
                sl.experience.save()
                gui.app_state = STATE_STOPPED
                break

            # ---- PAUSED — just render, don't step --------------------------
            if gui.app_state == STATE_PAUSED:
                gui.render(env, needs=info.get("needs",[0]*4), trace=sl.trace)
                continue

            # ---- Episode boundary -------------------------------------------
            if done:
                sl.on_episode_end()
                learner.flush()          # train on the tail of the episode before it rolls over
                total_rew = sum(ep_rewards)
                from rewards import RewardEngine
                grade = RewardEngine.classify_score(total_rew)
                prog_eff = env.rewards.compute_progression_efficiency(env.tick)
                caps_count = len(env.agent.capabilities)

                if env.agent.health <= 0:
                    reason = f"Perished (HP:{env.agent.health})"
                elif env.tick >= config_rl.MAX_TICKS:
                    reason = f"Max Ticks ({config_rl.MAX_TICKS})"
                else:
                    reason = "Episode complete"

                gui.record_episode(ep, total_rew, env.tick, reason, grade=grade, efficiency=prog_eff, capabilities=caps_count)

                narration = adapter.generate_narration(
                    f"### INPUT: ATS ep {ep} reward {total_rew:.2f} grade {grade}\n### OUTPUT:")
                gui.record_narration(narration)
                adapter.export_runtime_corpus([
                    f"episode {ep} summary|reward {total_rew:.2f}|grade {grade}|efficiency {prog_eff:.3f}|caps {caps_count}|({reason})",
                    f"episode {ep} narration|{narration}",
                ])
                print(f"Episode {ep} done | reward={total_rew:.2f} | GRADE: {grade:4s} | Caps={caps_count} | Eff={prog_eff:.3f} | ticks={env.tick} | {reason}")

                # Check if target reached
                if ep >= num_eps:
                    _save_policy(policy)
                    sl.experience.save()
                    gui.app_state = STATE_STOPPED
                    break

                # Next episode
                state      = env.reset()
                done       = False
                ep_rewards = []

            # ---- First tick of a new episode --------------------------------
            if not ep_rewards:      # fresh episode
                ep += 1

            # ---- Simulation step -------------------------------------------
            mask   = env.agent.get_action_mask(env.world, env.day_night)
            action = sl.choose_action(state, mask)

            state, reward, done, info = env.step(
                action,
                disabled_actions=gui.disabled_actions,
                reward_rules=gui.reward_rules,
            )
            ep_rewards.append(reward)

            # Record step outcome in solution loop
            sl.record_outcome(action, reward, state)

            learner.collect(state=state, action=action, reward=reward,
                            log_prob=sl.last_log_prob, value=sl.last_value,
                            action_mask=mask, done=done)
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

    policy, _ = _load_policy(args.fresh)
    policy.eval()
    sl      = SolutionLoop(policy=policy)
    learner = ContinualLearner(policy=policy)
    env     = ATSEnvironment()
    adapter = VirusAdapter(profile=args.virus_profile)
    adapter.write_active_profile()

    from rewards import RewardEngine

    for ep in range(1, num_eps + 1):
        state      = env.reset()
        done       = False
        ep_rewards = []

        while not done:
            mask   = env.agent.get_action_mask(env.world, env.day_night)
            action = sl.choose_action(state, mask)
            state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            learner.collect(state=state, action=action, reward=reward,
                            log_prob=sl.last_log_prob, value=sl.last_value,
                            action_mask=mask, done=done)
            learner.maybe_update()

        learner.flush()
        total = sum(ep_rewards)
        grade = RewardEngine.classify_score(total)
        prog_eff = env.rewards.compute_progression_efficiency(env.tick)
        caps_count = len(env.agent.capabilities)
        reason = "MaxTicks" if env.tick >= config_rl.MAX_TICKS else "Died"
        print(f"Ep {ep:4d} | reward={total:+7.2f} | GRADE: {grade:4s} | Caps={caps_count} | Eff={prog_eff:.3f} | ticks={env.tick} | {reason}")

    _save_policy(policy)



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
