import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from src.envs.trading_envs import StockTradingEnv
from src.models.deepqlearning import DQNAgent
from config import config
from utils import set_seed, get_writer, init_wandb, ReplayBuffer


def train():
    set_seed(42)
    writer = get_writer(config["logging"]["tensorboard_dir"])
    init_wandb(project=config["logging"]["wandb_project"], name="DQN-Agent")

    env = StockTradingEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config)
    replay_buffer = ReplayBuffer(config["agent"]["memory_size"])

    for episode in range(config["training"]["episodes"]):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < config["training"]["max_steps"]:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            agent.learn(replay_buffer)

            state = next_state
            total_reward += reward
            step += 1

        print(f"Episode {episode+1} - Total Reward: {total_reward:.2f}")
        writer.add_scalar("Reward/Episode", total_reward, episode)
        wandb.log({"episode": episode, "reward": total_reward})

        if (episode + 1) % config["training"]["save_every"] == 0:
            agent.save(config["training"]["weights_path_dqn"])

    env.close()
    writer.close()
    wandb.finish()


if __name__ == "__main__":
    train()
