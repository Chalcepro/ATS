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
               "action_size": policy.action_size}
    torch.save(payload, config_rl.MODEL_PATH)
    torch.save(payload, config_rl.BEST_MODEL_PATH)
    print(f"[ATS] Checkpoint saved (hidden={policy.hidden_size})")


# ---------------------------------------------------------------------------
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
        env     = ATSEnvironment()
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
                learner.maybe_update()
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
                            action_mask=mask)
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
                            action_mask=mask)
            learner.maybe_update()

        learner.maybe_update()
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
