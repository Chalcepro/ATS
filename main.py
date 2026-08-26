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
from debug_gui import TerminalGUI, STATE_MENU, STATE_RUNNING, STATE_PAUSED, STATE_STOPPED
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
    policy = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
    if fresh:
        print(f"[ATS] Fresh policy (hidden={policy.hidden_size})")
        return policy, False
    for ckpt in [config_rl.BEST_MODEL_PATH, config_rl.MODEL_PATH]:
        if ckpt.exists():
            try:
                data = torch.load(ckpt, map_location="cpu")
                sd   = data.get("model_state_dict", data) if isinstance(data, dict) else data
                policy.load_state_dict(sd)
                print(f"[ATS] Loaded checkpoint {ckpt.name} (hidden={policy.hidden_size})")
                return policy, True
            except Exception:
                continue
    print(f"[ATS] No matching checkpoint — fresh policy (hidden={policy.hidden_size})")
    return policy, False


# ---------------------------------------------------------------------------
def _save_policy(policy):
# ---------------------------------------------------------------------------
    config_rl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": policy.state_dict(),
               "hidden_size": policy.hidden_size}
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
        gui.active_tab = 3  # default to Hyperparams on menu

        while gui.alive and gui.app_state == STATE_MENU:
            gui.render()   # draws home screen
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
                if env: env.agent.event_log.append(("Checkpoint saved!", time.time()))

            if gui.sig_fresh_reset and gui.app_state == STATE_PAUSED:
                gui.sig_fresh_reset = False
                policy  = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
                sl      = SolutionLoop(policy=policy)
                learner = ContinualLearner(policy=policy)
                if env: env.agent.event_log.append(("Policy reset to fresh!", time.time()))

            if gui.sig_stop:
                gui.sig_stop = False
                gui.app_state = STATE_STOPPED
                break

            # ---- PAUSED — just render, don't step --------------------------
            if gui.app_state == STATE_PAUSED:
                gui.render(env, needs=info.get("needs",[0]*4), trace=sl.trace)
                continue

            # ---- Episode boundary -------------------------------------------
            if done:
                learner.maybe_update()
                total_rew = sum(ep_rewards)

                if env.agent.health <= 0:
                    reason = f"Perished (HP:{env.agent.health})"
                elif env.tick >= config_rl.MAX_TICKS:
                    reason = f"Max Ticks ({config_rl.MAX_TICKS})"
                else:
                    reason = "Episode complete"

                gui.record_episode(ep, total_rew, env.tick, reason)

                narration = adapter.generate_narration(
                    f"### INPUT: ATS ep {ep} reward {total_rew:.2f}\n### OUTPUT:")
                gui.record_narration(narration)
                adapter.export_runtime_corpus([
                    f"episode {ep} summary|reward {total_rew:.2f} ({reason})",
                    f"episode {ep} narration|{narration}",
                ])
                print(f"Episode {ep} done | reward={total_rew:.2f} | ticks={env.tick} | {reason}")

                # Check if target reached
                if ep >= num_eps:
                    _save_policy(policy)
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

            if config_rl.SPEED_MULTIPLIER < 50:
                time.sleep(0.001 / max(config_rl.SPEED_MULTIPLIER, 0.1))

        learner.maybe_update()
        total = sum(ep_rewards)
        reason = "MaxTicks" if env.tick >= config_rl.MAX_TICKS else "Died"
        print(f"Ep {ep:4d} | reward={total:+.2f} | ticks={env.tick} | {reason}")

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
