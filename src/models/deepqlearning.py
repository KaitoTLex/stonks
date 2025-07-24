# dqn_agent.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self.model = YourModel().to(device)
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        memory_size=100_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu",
        initial_balance=10_000.0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.device = device
        self.balance = initial_balance

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return torch.argmax(q_values).item()

    def store(self, transition):
        self.memory.append(transition)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device),
        )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return float(loss.item())

    def soft_update(self):
        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_weights(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_weights(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
