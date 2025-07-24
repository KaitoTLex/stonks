import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import random
from datetime import datetime, timedelta
from scipy.stats import norm
from config import (
    CUSTOM_STOCK_LIST,
    NUM_STOCKS_PER_EPISODE,
    ALPHA_REWARD,
    BETA_REWARD,
    GAMMA_REWARD,
    INITIAL_BALANCE,
    EPISODE_HOURS,
)


def get_random_quarter_start(years_range=(2005, 2023)):
    year = random.randint(years_range[0], years_range[1])
    quarter = random.randint(1, 4)
    month = (quarter - 1) * 3 + 1
    return datetime(year, month, 1)


def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1h", progress=False)
    df = df.dropna()
    return df[["Open", "High", "Low", "Close", "Volume"]]


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Validate config
        if len(CUSTOM_STOCK_LIST) < NUM_STOCKS_PER_EPISODE:
            raise ValueError(
                f"CUSTOM_STOCK_LIST length ({len(CUSTOM_STOCK_LIST)}) < NUM_STOCKS_PER_EPISODE ({NUM_STOCKS_PER_EPISODE})"
            )

        # Action: for each stock, 0=hold, 1=buy, 2=sell
        self.action_space = spaces.MultiDiscrete([3] * NUM_STOCKS_PER_EPISODE)

        # Observation: window of 1 hour raw OHLCV per stock (shape: stocks x features)
        # For simplicity, 1-hour window (current hour only), 5 features per stock
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(NUM_STOCKS_PER_EPISODE, 5),
            dtype=np.float32,
        )

        self.initial_balance = INITIAL_BALANCE

        # Placeholder for loaded data for selected tickers (dict: ticker -> dataframe)
        self.price_data = {}
        self.selected_tickers = []
        self.current_step = 0

        # Portfolio state
        self.cash_balance = 0.0
        self.holdings = None  # shares held per stock (np.array)
        self.last_portfolio_value = None

        # Track trades in current step (for reward)
        self.trades_profit = 0.0

        # Keep history of portfolio values to compute Sharpe
        self.portfolio_values_history = []

        self.episode_length = (
            EPISODE_HOURS  # e.g. 3 months * ~30 days * 24 hrs = 2160 hours
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.selected_tickers = random.sample(CUSTOM_STOCK_LIST, NUM_STOCKS_PER_EPISODE)

        # Pick random quarter start date
        quarter_start = get_random_quarter_start()
        quarter_end = quarter_start + timedelta(days=90)

        # Download data for all selected tickers
        self.price_data = {}
        for ticker in self.selected_tickers:
            df = download_data(
                ticker,
                quarter_start.strftime("%Y-%m-%d"),
                quarter_end.strftime("%Y-%m-%d"),
            )
            if len(df) < self.episode_length:
                raise RuntimeError(f"Not enough data for ticker {ticker} in period")
            self.price_data[ticker] = df.reset_index(drop=True)

        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.holdings = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=np.float32)
        self.trades_profit = 0.0
        self.portfolio_values_history = []

        self.last_portfolio_value = self._calculate_portfolio_value()

        return self._get_obs(), {}

    def _get_obs(self):
        # Gather latest OHLCV for each ticker at current step
        obs = []
        for i, ticker in enumerate(self.selected_tickers):
            data = self.price_data[ticker].iloc[self.current_step]
            obs.append(
                [data["Open"], data["High"], data["Low"], data["Close"], data["Volume"]]
            )
        return np.array(obs, dtype=np.float32)

    def _calculate_portfolio_value(self):
        value = self.cash_balance
        for i, ticker in enumerate(self.selected_tickers):
            current_close = self.price_data[ticker].iloc[self.current_step]["Close"]
            value += self.holdings[i] * current_close
        return value

    def step(self, actions):
        """
        actions: array-like, length NUM_STOCKS_PER_EPISODE
        each element: 0=hold, 1=buy, 2=sell
        """

        assert len(actions) == NUM_STOCKS_PER_EPISODE

        self.trades_profit = 0.0

        # Execute trades sequentially
        for i, action in enumerate(actions):
            ticker = self.selected_tickers[i]
            current_price = self.price_data[ticker].iloc[self.current_step]["Close"]

            # Simple slippage model
            slippage_factor = np.random.uniform(0.998, 1.002)
            effective_price = current_price * slippage_factor

            if action == 1:  # Buy one share if possible
                if self.cash_balance >= effective_price:
                    self.holdings[i] += 1
                    self.cash_balance -= effective_price
                    # No immediate profit for buying
                else:
                    pass  # no buy if insufficient cash

            elif action == 2:  # Sell one share if holding any
                if self.holdings[i] >= 1:
                    self.holdings[i] -= 1
                    self.cash_balance += effective_price
                    # Profit is difference between sell price and average cost?
                    # Here we approximate profit as sell price - current_price (simplified)
                    self.trades_profit += (
                        effective_price  # count revenue here, profit calc below
                    )

            # else action == 0: hold, no trade

        # Update portfolio value and compute reward
        portfolio_value = self._calculate_portfolio_value()
        delta_value = portfolio_value - self.last_portfolio_value

        self.portfolio_values_history.append(portfolio_value)
        self.last_portfolio_value = portfolio_value

        # Compute Sharpe ratio on portfolio returns (if enough history)
        reward_sharpe = 0.0
        if len(self.portfolio_values_history) > 2:
            returns = (
                np.diff(self.portfolio_values_history)
                / self.portfolio_values_history[:-1]
            )
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) + 1e-9  # avoid div0
            reward_sharpe = mean_ret / std_ret

        # Normalize terms (roughly)
        norm_delta_value = delta_value / (self.initial_balance + 1e-9)
        norm_trades_profit = self.trades_profit / (self.initial_balance + 1e-9)
        norm_sharpe = reward_sharpe  # already ratio

        reward = (
            ALPHA_REWARD * norm_delta_value
            + BETA_REWARD * norm_sharpe
            + GAMMA_REWARD * norm_trades_profit
        )

        self.current_step += 1
        done = self.current_step >= (self.episode_length - 1)

        info = {
            "portfolio_value": portfolio_value,
            "cash_balance": self.cash_balance,
            "holdings": self.holdings.copy(),
            "selected_tickers": self.selected_tickers,
            "reward_components": {
                "delta_value": norm_delta_value,
                "sharpe": norm_sharpe,
                "trade_profit": norm_trades_profit,
            },
        }

        return self._get_obs(), reward, done, False, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Portfolio value: {self.last_portfolio_value:.2f}, "
            f"Cash: {self.cash_balance:.2f}, Holdings: {self.holdings}"
        )

    def close(self):
        pass
