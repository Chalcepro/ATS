"""PPO-style training loop for ATS (phase 1 policy-gradient scaffold)."""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config_rl
from ats_env import ATSEnvironment
from ats_virus_translator import ExternalTranslatorAgent
from model_rl import RLPolicy
from rewards import RewardEngine   # was used un-imported -> NameError on episode 1


def compute_returns(rewards, gamma):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.insert(0, running)
    return returns


def train(fresh: bool = False):
    env = ATSEnvironment()
    model = RLPolicy(hidden_size=config_rl.MIND_HIDDEN_SIZE)
    translator = ExternalTranslatorAgent()
    target_ckpt = config_rl.MODEL_PATH
    if not fresh and target_ckpt.exists():
        try:
            ckpt = torch.load(target_ckpt, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt.get("model_stlte_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict)
            print(f"[ATS Train] Loaded existing checkpoint: {target_ckpt} (hidden_size={model.hidden_size})")
        except Exception as e:
            print(f"[ATS Train] Checkpoint architecture changed ({e}). Starting fresh {config_rl.MIND_HIDDEN_SIZE} policy.")
    else:
        print(f"[ATS Train] Starting from fresh weights (hidden_size={config_rl.MIND_HIDDEN_SIZE})")
    optimizer = optim.Adam(model.parameters(), lr=config_rl.LEARNING_RATE)

    # ---- TensorBoard writer ------------------------------------------------
    config_rl.LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config_rl.LOG_DIR))

    recent_rewards = []
    best_avg_reward = float('-inf')

    for episode in range(1, config_rl.EPISODES + 1):
        log_probs = []
        values = []
        entropies = []
        rewards = []
        state = env.reset()
        done = False

        while not done:
            mask = env.agent.get_action_mask(env.world, env.day_night)
            tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            logits, value = model(tensor_state, action_mask=tensor_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, done, _ = env.step(int(action.item()))
            log_probs.append(log_prob)
            values.append(value.squeeze())
            entropies.append(entropy)
            rewards.append(reward)

        returns = torch.tensor(compute_returns(rewards, config_rl.GAMMA), dtype=torch.float32)
        values_tensor = torch.stack(values)

        # Normalise the ADVANTAGE (policy term), not the returns. The critic
        # must regress to the real return scale or the value head is useless
        # and advantages collapse to noise — see continual_learner._do_ppo_update.
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        actor_loss = -(log_probs_tensor * advantages).mean()
        critic_loss = torch.nn.functional.smooth_l1_loss(values_tensor, returns)
        entropy_bonus = entropies_tensor.mean()

        loss = actor_loss + 0.5 * critic_loss - config_rl.ENTROPY_COEFF * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        ep_rew = sum(rewards)
        recent_rewards.append(ep_rew)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_50 = sum(recent_rewards) / len(recent_rewards)

        # ---- Logging & Peak-Relative Classification -----------------------
        ticks = len(rewards)
        rew_per_tick = ep_rew / max(ticks, 1)

        # Dynamic peak-relative grade classification (GOOD/MID/FAIR/BAD)
        grade = RewardEngine.classify_score(ep_rew)
        prog_eff = env.rewards.compute_progression_efficiency(ticks)
        caps_count = len(env.agent.capabilities)

        # Automatically process grade through Translator Agent
        packet = translator.process_episode_grade(episode, grade, rew_per_tick, ep_rew)

        if episode % config_rl.LOG_INTERVAL == 0:
            biomes_count = len(env.rewards.discovered_biomes)
            items_count = len(env.rewards.discovered_items)
            print(
                f"Episode {episode:4d}/{config_rl.EPISODES} | "
                f"Ticks={ticks:3d} | Rew/Tick={rew_per_tick:+6.2f} | "
                f"GRADE: {grade:4s} | Caps={caps_count} | Biomes={biomes_count} | Items={items_count} | "
                f"Eff={prog_eff:.3f} | Reward={ep_rew:7.2f} | 50-Ep Avg={avg_50:7.2f} | "
                f"Loss={loss.item():.4f}"
            )
            print(
                f"  [Auto-Translator] Grade '{grade}' parsed -> Polarity: {packet['feedback_polarity']:+1.1f} | Tokens: {' '.join(packet['encoded_symbols'][:4])}..."
            )
            writer.add_scalar('Reward/Episode', ep_rew, episode)
            writer.add_scalar('Reward/Avg50', avg_50, episode)
            writer.add_scalar('Score/Efficiency', prog_eff, episode)
            writer.add_scalar('Score/RewardPerTick', rew_per_tick, episode)
            writer.add_scalar('Progression/Capabilities', caps_count, episode)
            writer.add_scalar('Progression/Points', env.rewards.progression_points, episode)
            writer.add_scalar('Loss/Total', loss.item(), episode)
            writer.add_scalar('Loss/Actor', actor_loss.item(), episode)
            writer.add_scalar('Loss/Critic', critic_loss.item(), episode)
            writer.add_scalar('Entropy', entropy_bonus.item(), episode)


        # ---- Best-model checkpoint (based on 50-ep moving average) ---------
        if avg_50 > best_avg_reward:
            best_avg_reward = avg_50
            config_rl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, config_rl.BEST_MODEL_PATH)
            if episode % config_rl.LOG_INTERVAL == 0:
                print(f"  [Best Model Saved] -> {config_rl.BEST_MODEL_PATH} (Avg50={avg_50:.2f})")

        # ---- Periodic checkpoint every 100 episodes -----------------------
        if episode % 100 == 0 or episode == config_rl.EPISODES:
            config_rl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, config_rl.MODEL_PATH)
            print(f"  [Checkpoint Saved] -> {config_rl.MODEL_PATH} (Episode {episode})")

    writer.close()
    print(f"Training Complete! Final Model Saved -> {config_rl.MODEL_PATH}")


if __name__ == "__main__":
    train()
