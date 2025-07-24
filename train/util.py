# utils.py
import os
import random
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import wandb

from config import config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_writer(agent_name: str):
    log_dir = os.path.join(config["logging"]["tensorboard_dir"], agent_name)
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def init_wandb(agent_name: str):
    wandb.init(
        project=config["logging"]["wandb_project"], name=agent_name, config=config
    )


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Utility for computing Sharpe Ratio


def compute_sharpe(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    return_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)
    return return_ratio
