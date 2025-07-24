import pytest
import numpy as np
from envs.trading_envs import StockTradingEnv
from  models.deepqlearning import DQNAgent
from  models.montecarlo import MCAgent
from config import NUM_STOCKS_PER_EPISODE, INITIAL_BALANCE


def test_env_reset_and_observation_shape():
    env = StockTradingEnv()
    obs, _ = env.reset()
    # Observation shape: (NUM_STOCKS_PER_EPISODE, 5 features)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (NUM_STOCKS_PER_EPISODE, 5)
    assert env.cash_balance == INITIAL_BALANCE
    assert env.current_step == 0


def test_env_step_and_reward():
    env = StockTradingEnv()
    obs, _ = env.reset()
    actions = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=int)  # hold for all stocks
    next_obs, reward, done, truncated, info = env.step(actions)

    # Check observation shape after step
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (NUM_STOCKS_PER_EPISODE, 5)

    # Reward should be float and typically close to zero when holding
    assert isinstance(reward, float)

    # Done and truncated flags are bool
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)

    # Info dict should include keys we expect
    assert "portfolio_value" in info
    assert "cash_balance" in info
    assert "holdings" in info
    assert "selected_tickers" in info
    assert "reward_components" in info
    rc = info["reward_components"]
    assert "delta_value" in rc and "sharpe" in rc and "trade_profit" in rc


def test_env_buy_and_sell_transitions():
    env = StockTradingEnv()
    obs, _ = env.reset()

    # Try buying one share in first stock, hold others
    actions = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=int)
    actions[0] = 1  # buy
    next_obs, reward, done, truncated, info = env.step(actions)

    assert env.holdings[0] == 1
    assert env.cash_balance < INITIAL_BALANCE  # cash decreased after buy

    # Now sell one share of first stock
    actions[0] = 2  # sell
    next_obs, reward, done, truncated, info = env.step(actions)

    assert env.holdings[0] == 0
    assert env.cash_balance > 0  # cash increased after sell


def test_dqn_agent_action_and_training():
    env = StockTradingEnv()
    obs, _ = env.reset()
    obs = obs.flatten()

    agent = DQNAgent(obs.shape[0], 3)
    action = agent.select_action(obs)
    assert action in [0, 1, 2]

    # Perform one training step with dummy transition
    agent.store((obs, action, 1.0, obs, False))
    loss = agent.train_step()
    # Loss can be None if batch not full; else float
    assert loss is None or isinstance(loss, float)


def test_mc_agent_action_and_update():
    env = StockTradingEnv()
    obs, _ = env.reset()
    obs = obs.flatten()

    agent = MCAgent(obs.shape[0], 3)
    action = agent.select_action(obs)
    assert action in [0, 1, 2]

    agent.store_transition(obs, action, 1.0)
    agent.update()
    assert 0.0 <= agent.epsilon <= 1.0
