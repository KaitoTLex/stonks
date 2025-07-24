import pytest
import numpy as np
from envs.trading_envs import StockTradingEnv
from models.deepqlearning import DQNAgent
from models.montecarlo import MCAgent
from config import NUM_STOCKS_PER_EPISODE


def test_env_reset_and_step():
    env = StockTradingEnv()
    obs, _ = env.reset()
    assert obs.shape == (NUM_STOCKS_PER_EPISODE, 15)
    assert env.cash_balance == env.initial_balance
    assert env.current_step == 0

    # Test one step with hold action
    actions = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=int)  # all hold
    obs, reward, done, truncated, info = env.step(actions)
    assert obs.shape == (NUM_STOCKS_PER_EPISODE, 15)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_dqn_agent_select_action():
    env = StockTradingEnv()
    obs, _ = env.reset()
    obs = obs.flatten()
    agent = DQNAgent(obs.shape[0], 3)
    action = agent.select_action(obs)
    assert action in [0, 1, 2]


def test_mc_agent_select_action_and_update():
    env = StockTradingEnv()
    obs, _ = env.reset()
    obs = obs.flatten()
    agent = MCAgent(obs.shape[0], 3)
    action = agent.select_action(obs)
    assert action in [0, 1, 2]

    # Simulate episode memory and update
    agent.store_transition(obs, action, 1.0)
    agent.update()
    assert agent.epsilon <= 1.0 and agent.epsilon >= 0.0
