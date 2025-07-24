import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import random
from datetime import datetime, timedelta

# Import configuration variables
from config import (
    CUSTOM_STOCK_LIST,
    NUM_STOCKS_PER_EPISODE,
    INITIAL_BALANCE,
    OBSERVATION_WINDOW_SIZE,
    EPISODE_HOURS,
    SLIPPAGE_PERCENT
)

class DataManager:
    """
    Handles downloading, storing, and providing historical stock data efficiently.
    Data is downloaded once and reused for all episodes.
    """
    def __init__(self, tickers: list, period: str = "730d", interval: str = "1h"):
        """
        Initializes the DataManager and downloads data for all specified tickers.
        
        Args:
            tickers (list): A list of stock tickers.
            period (str): The period to download data for (e.g., "730d").
            interval (str): The data interval (e.g., "1h").
        """
        self.data = {}
        self.tickers = tickers
        print("--- Downloading Historical Data ---")
        
        # Download data for all tickers and store it in a dictionary
        df = yf.download(tickers, period=period, interval=interval, progress=True)
        
        if df.empty:
            raise ValueError("No data downloaded. Check tickers and internet connection.")
            
        # Pre-process and store data for each ticker
        for ticker in tickers:
            # Extract columns for this ticker, handling multi-level columns
            ticker_df = df.xs(ticker, level=1, axis=1).copy()
            ticker_df.dropna(inplace=True)
            if not ticker_df.empty:
                self.data[ticker] = ticker_df
        
        print("--- Data Download Complete ---")
        if not self.data:
            raise ValueError("Could not retrieve valid data for any tickers.")

    def get_episode_data(self):
        """
        Selects a random one-month slice of data for a new episode.

        Returns:
            tuple: A tuple containing:
                - dict: DataFrames for the selected tickers and date range.
                - int: The total number of steps in the episode.
        """
        # Ensure there's enough data for a full episode
        min_length = OBSERVATION_WINDOW_SIZE + EPISODE_HOURS
        
        # Find a valid start index from a random ticker's data
        valid_tickers = [t for t in self.tickers if len(self.data.get(t, [])) > min_length]
        if not valid_tickers:
            raise ValueError("No tickers have enough data for a full episode.")
        
        reference_ticker = random.choice(valid_tickers)
        max_start_index = len(self.data[reference_ticker]) - min_length
        start_index = random.randint(0, max_start_index)
        end_index = start_index + min_length
        
        # Slice the data for all tickers for the chosen period
        episode_data = {}
        for ticker in self.tickers:
            if ticker in self.data:
                episode_data[ticker] = self.data[ticker].iloc[start_index:end_index]

        return episode_data, min_length


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning, compatible with Gymnasium.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        print("Initializing Trading Environment...")
        self.data_manager = DataManager(CUSTOM_STOCK_LIST)
        self.stock_list = self.data_manager.tickers

        # Action space: For each stock, 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.MultiDiscrete([3] * NUM_STOCKS_PER_EPISODE)

        # Observation space: For each stock, a window of past OHLCV data.
        # Shape: (Num Stocks, Window Size, Features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(NUM_STOCKS_PER_EPISODE, OBSERVATION_WINDOW_SIZE, 5),
            dtype=np.float32,
        )

        # Environment state variables
        self.selected_tickers = []
        self.episode_data = {}
        self.current_step = 0
        self.episode_length = 0
        
        # Portfolio state variables
        self.initial_balance = INITIAL_BALANCE
        self.cash_balance = 0.0
        self.holdings = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=np.float32)
        self.last_portfolio_value = 0.0
        print("Environment Initialized.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select a random subset of stocks for the episode
        self.selected_tickers = random.sample(self.stock_list, NUM_STOCKS_PER_EPISODE)
        
        # Get a new slice of data for the episode
        self.episode_data, self.episode_length = self.data_manager.get_episode_data()

        # Filter the data for the selected tickers
        self.current_episode_data = {t: self.episode_data[t] for t in self.selected_tickers}

        # Initialize portfolio
        self.current_step = OBSERVATION_WINDOW_SIZE
        self.cash_balance = self.initial_balance
        self.holdings = np.zeros(NUM_STOCKS_PER_EPISODE, dtype=np.float32)
        self.last_portfolio_value = self._calculate_portfolio_value()

        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def _get_obs(self):
        """
        Gets the observation for the current step.
        The observation is a window of the last N hours of data for each stock.
        """
        obs = []
        for ticker in self.selected_tickers:
            # Get the window of data up to the current step
            window = self.current_episode_data[ticker].iloc[
                self.current_step - OBSERVATION_WINDOW_SIZE : self.current_step
            ]
            # Select only the OHLCV columns
            ohlcv_window = window[['Open', 'High', 'Low', 'Close', 'Volume']].values
            obs.append(ohlcv_window)
        
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        return {
            "portfolio_value": self.last_portfolio_value,
            "cash_balance": self.cash_balance,
            "holdings": self.holdings.copy(),
            "selected_tickers": self.selected_tickers,
        }

    def _calculate_portfolio_value(self):
        """Calculates the total value of the portfolio (cash + holdings)."""
        value = self.cash_balance
        for i, ticker in enumerate(self.selected_tickers):
            current_price = self.current_episode_data[ticker].iloc[self.current_step]['Close']
            value += self.holdings[i] * current_price
        return value

    def step(self, actions):
        """
        Executes one time step within the environment.

        Args:
            actions (np.ndarray): An array of actions (0:Hold, 1:Buy, 2:Sell).
        """
        assert len(actions) == NUM_STOCKS_PER_EPISODE

        # Execute trades for each stock
        for i, action in enumerate(actions):
            ticker = self.selected_tickers[i]
            base_price = self.current_episode_data[ticker].iloc[self.current_step]['Close']

            # Simulate slippage for buy/sell actions
            if action == 1:  # Buy
                buy_price = base_price * (1 + SLIPPAGE_PERCENT * random.uniform(0, 1))
                if self.cash_balance >= buy_price:
                    self.holdings[i] += 1
                    self.cash_balance -= buy_price
            elif action == 2:  # Sell
                if self.holdings[i] > 0:
                    sell_price = base_price * (1 - SLIPPAGE_PERCENT * random.uniform(0, 1))
                    self.holdings[i] -= 1
                    self.cash_balance += sell_price
        
        # Update portfolio value and calculate reward
        portfolio_value = self._calculate_portfolio_value()
        reward = portfolio_value - self.last_portfolio_value
        self.last_portfolio_value = portfolio_value

        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step >= self.episode_length -1

        obs = self._get_obs()
        info = self._get_info()
        
        # The 'truncated' flag is not used here but is part of the standard API
        truncated = False

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        """Renders the environment's state."""
        if mode == "human":
            print(f"Step: {self.current_step - OBSERVATION_WINDOW_SIZE}/{EPISODE_HOURS}")
            print(f"Portfolio Value: {self.last_portfolio_value:,.2f}")
            print(f"Cash Balance: {self.cash_balance:,.2f}")
            holdings_str = [f"{ticker}: {shares:.2f}" for ticker, shares in zip(self.selected_tickers, self.holdings)]
            print(f"Holdings: {', '.join(holdings_str)}")
            print("-" * 30)

    def close(self):
        """Cleans up the environment."""
        print("Closing environment.")
        pass

