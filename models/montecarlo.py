import numpy as np
import random
from collections import defaultdict


class MCAgent:
    def __init__(
        self,
        state_size,
        action_size,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Use defaultdict for Q-values initialized to zero
        self.Q = defaultdict(lambda: np.zeros(self.action_size))
        self.returns_sum = defaultdict(lambda: np.zeros(self.action_size))
        self.returns_count = defaultdict(lambda: np.zeros(self.action_size))

        self.episode_memory = []

    def get_state_key(self, state):
        # Discretize or hash continuous states - for simplicity, round and convert to tuple
        return tuple(np.round(state, decimals=2))

    def select_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return int(np.argmax(self.Q[state_key]))

    def store_transition(self, state, action, reward):
        self.episode_memory.append((state, action, reward))

    def update(self):
        G = 0
        visited_state_actions = set()
        for state, action, reward in reversed(self.episode_memory):
            G = reward + G  # no discounting for simplicity, or add gamma if you want
            state_key = self.get_state_key(state)
            if (state_key, action) not in visited_state_actions:
                self.returns_sum[state_key][action] += G
                self.returns_count[state_key][action] += 1
                self.Q[state_key][action] = (
                    self.returns_sum[state_key][action]
                    / self.returns_count[state_key][action]
                )
                visited_state_actions.add((state_key, action))

        self.episode_memory = []
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
