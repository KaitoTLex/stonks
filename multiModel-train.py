import gymnasium as gym
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from envs.stock_env import StockTradingEnv
from agents.dqn_agent import DQNAgent
from agents.mc_agent import MCAgent
from config import config
import wandb
from torch.utils.tensorboard import SummaryWriter

# Lock for print sync
print_lock = threading.Lock()


def train_dqn():
    wandb.init(project="stock-trading", name="DQN_Agent", reinit=True)
    writer = SummaryWriter(log_dir="logs/dqn")

    env = StockTradingEnv()
    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = 3

    agent = DQNAgent(obs_dim, action_dim)

    episodes = config["training"]["episodes"]
    max_steps = config["training"]["max_steps"]

    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0
        losses = []

        for t in range(max_steps):
            action = agent.select_action(state)
            action_array = np.array([action] * env.action_space.n)
            next_state, reward, done, truncated, info = env.step(action_array)
            next_state = next_state.flatten()
            agent.store((state, action, reward, next_state, float(done)))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        avg_loss = np.mean(losses) if losses else 0

        wandb.log(
            {
                "episode": episode,
                "reward": total_reward,
                "loss": avg_loss,
                "epsilon": agent.epsilon,
            }
        )

        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Loss", avg_loss, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        with print_lock:
            print(
                f"[DQN] Episode {episode+1}: Reward={total_reward:.2f}, Loss={avg_loss:.4f}, Epsilon={agent.epsilon:.4f}"
            )

        if (episode + 1) % config["training"]["save_every"] == 0:
            agent.save_weights(config["training"]["weights_path"])

    writer.close()
    wandb.finish()


def train_mc():
    wandb.init(project="stock-trading", name="MC_Agent", reinit=True)
    writer = SummaryWriter(log_dir="logs/mc")

    env = StockTradingEnv()
    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = 3

    agent = MCAgent(obs_dim, action_dim)

    episodes = config["training"]["episodes"]
    max_steps = config["training"]["max_steps"]

    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            action_array = np.array([action] * env.action_space.n)
            next_state, reward, done, truncated, info = env.step(action_array)
            next_state = next_state.flatten()
            agent.store_transition(state, action, reward)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        agent.update()

        wandb.log(
            {"episode": episode, "reward": total_reward, "epsilon": agent.epsilon}
        )

        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        with print_lock:
            print(
                f"[MC] Episode {episode+1}: Reward={total_reward:.2f}, Epsilon={agent.epsilon:.4f}"
            )

    writer.close()
    wandb.finish()


def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(train_dqn)
        executor.submit(train_mc)


if __name__ == "__main__":
    main()
