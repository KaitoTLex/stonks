# train_parallel.py
import threading
from envs.trading_envs import StockTradingEnv
from models.deepqlearning import DQNAgent
from models.montecarlo import MCAgent
from config import config
from utils import set_seed, get_writer, init_wandb, ReplayBuffer
from tui.tui_dashboard import render_dual_agent_dashboard


def train_dqn_thread(dqn_history):
    set_seed(42)
    writer = get_writer("dqn")
    init_wandb("DQN-Agent")

    env = StockTradingEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config)
    replay_buffer = ReplayBuffer(config["agent"]["memory_size"])

    for episode in range(config["training"]["episodes"]):
        state = env.reset()
        env.history = {"portfolio_value": [], "price": []}  # Clear history
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

            # Log portfolio and price history
            env.history["portfolio_value"].append(info.get("portfolio_value", 0))
            env.history["price"].append(info.get("price", 0))

        print(f"[DQN] Episode {episode + 1} - Reward: {total_reward:.2f}")
        writer.add_scalar("Reward/Episode", total_reward, episode)

        if (episode + 1) % config["training"]["save_every"] == 0:
            agent.save(config["training"]["weights_path_dqn"])

    dqn_history.update(env.history)  # Pass history back
    writer.close()


def train_mc_thread(mc_history):
    set_seed(42)
    writer = get_writer("mc")
    init_wandb("MC-Agent")

    env = StockTradingEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MonteCarloAgent(state_dim, action_dim, config)

    for episode in range(config["training"]["episodes"]):
        state = env.reset()
        env.history = {"portfolio_value": [], "price": []}  # Clear history
        episode_data = []
        total_reward = 0
        done = False
        step = 0

        while not done and step < config["training"]["max_steps"]:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_data.append((state, action, reward))

            state = next_state
            total_reward += reward
            step += 1

            # Log portfolio and price history
            env.history["portfolio_value"].append(info.get("portfolio_value", 0))
            env.history["price"].append(info.get("price", 0))

        agent.learn(episode_data)

        print(f"[MC] Episode {episode + 1} - Reward: {total_reward:.2f}")
        writer.add_scalar("Reward/Episode", total_reward, episode)

        if (episode + 1) % config["training"]["save_every"] == 0:
            agent.save(config["training"]["weights_path_mc"])

    mc_history.update(env.history)  # Pass history back
    writer.close()


def main():
    dqn_history = {}
    mc_history = {}

    thread_dqn = threading.Thread(target=train_dqn_thread, args=(dqn_history,))
    thread_mc = threading.Thread(target=train_mc_thread, args=(mc_history,))

    thread_dqn.start()
    thread_mc.start()

    thread_dqn.join()
    thread_mc.join()

    # Render TUI dashboard comparing both agents
    render_dual_agent_dashboard(dqn_history, mc_history)


if __name__ == "__main__":
    main()
