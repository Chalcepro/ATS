"""Baseline benchmark runner for ATS model evaluation."""
import os, sys, json
import numpy as np
import torch

# Ensure ATS directory is in sys.path
ats_dir = r"c:\Users\Virus\Documents\github\ATS"
if ats_dir not in sys.path:
    sys.path.insert(0, ats_dir)

import config_rl
from ats_env import ATSEnvironment
from model_rl import RLPolicy
from mind.solution_loop import SolutionLoop

def run_benchmark(num_episodes=20, max_ticks=200):
    env = ATSEnvironment()
    policy = RLPolicy()
    
    if config_rl.MODEL_PATH.exists():
        ckpt = torch.load(config_rl.MODEL_PATH, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_stlte_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        policy.load_state_dict(state_dict)
        print(f"[BENCHMARK] Loaded policy checkpoint: {config_rl.MODEL_PATH}")
    else:
        print("[BENCHMARK] Warning: No checkpoint found, evaluating untrained baseline policy!")
        
    policy.eval()
    solution_loop = SolutionLoop(policy=policy)
    
    episode_rewards = []
    episode_ticks = []
    action_counts = {}
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ticks = 0
        
        while not done and ticks < max_ticks:
            mask = env.agent.get_action_mask(env.world, env.day_night)
            action = solution_loop.choose_action(state, mask)
            
            action_counts[action] = action_counts.get(action, 0) + 1
            state, reward, done, info = env.step(action)
            
            ep_reward += reward
            ticks += 1
            
        episode_rewards.append(ep_reward)
        episode_ticks.append(ticks)
        print(f"Episode {ep:2d}/{num_episodes} | Ticks: {ticks:3d} | Total Reward: {ep_reward:7.2f}")
        
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    min_reward = float(np.min(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    mean_ticks = float(np.mean(episode_ticks))
    
    summary = {
        "num_episodes": num_episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_ticks": mean_ticks,
        "action_distribution": action_counts
    }
    
    output_json = r"c:\Users\Virus\Documents\github\ATS\data\baseline_benchmark.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*50)
    print("BASELINE BENCHMARK SUMMARY")
    print("="*50)
    print(f"Mean Reward     : {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Reward Range    : [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"Mean Ticks/Ep   : {mean_ticks:.1f}")
    print(f"Saved benchmark to: {output_json}")

if __name__ == "__main__":
    run_benchmark(num_episodes=20, max_ticks=200)
