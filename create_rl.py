"""Create a fresh ATS RL checkpoint."""

from pathlib import Path

import torch

import config_rl
from model_rl import RLPolicy


def main():
    config_rl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model = RLPolicy()
    torch.save(model.state_dict(), config_rl.MODEL_PATH)
    print(f"Created RL model at {config_rl.MODEL_PATH}")


if __name__ == "__main__":
    main()
