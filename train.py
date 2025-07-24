import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from envs.stock_env import StockTradingEnv
from agents.dqn_agent import DQNAgent
from agents.mc_agent import MCAgent
from config import config


def train():
    env = StockTradingEnv()

    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = 3  # for each stock (buy, hold, sell)

    dqn_agent = DQNAgent(obs_dim, action_dim)
    mc_agent = MCAgent(obs_dim, action_dim)

    dqn_rewards, dqn_losses = [], []
    mc_rewards = []

    episodes = config["training"]["episodes"]
    max_steps = config["training"]["max_steps"]

    for episode in range(episodes):
        # --- Train DQN agent ---
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0
        losses = []

        for t in range(max_steps):
            action = dqn_agent.select_action(state)
            action_array = np.array([action] * env.action_space.n)
            next_state, reward, done, truncated, info = env.step(action_array)
            next_state = next_state.flatten()
            dqn_agent.store((state, action, reward, next_state, float(done)))
            loss = dqn_agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        dqn_rewards.append(total_reward)
        dqn_losses.append(np.mean(losses) if losses else 0)
        if (episode + 1) % config["training"]["save_every"] == 0:
            dqn_agent.save_weights(config["training"]["weights_path"])

        # --- Train MC agent ---
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0

        for t in range(max_steps):
            action = mc_agent.select_action(state)
            action_array = np.array([action] * env.action_space.n)
            next_state, reward, done, truncated, info = env.step(action_array)
            next_state = next_state.flatten()
            mc_agent.store_transition(state, action, reward)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        mc_agent.update()
        mc_rewards.append(total_reward)

        print(
            f"Episode {episode+1}: DQN Reward={dqn_rewards[-1]:.2f} Epsilon={dqn_agent.epsilon:.2f}, MC Reward={mc_rewards[-1]:.2f} Epsilon={mc_agent.epsilon:.2f}"
        )

    # Plot telemetry
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dqn_rewards, label="DQN Reward")
    plt.plot(mc_rewards, label="MC Reward")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dqn_losses)
    plt.title("DQN Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig("plots/training_telemetry_both.png")
    plt.close()


if __name__ == "__main__":
    train()
