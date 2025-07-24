# config.py

CUSTOM_STOCK_LIST = [
"NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "BRK.B", "TSLA", "JPM", "WMT", "LLY", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ", "HD", "PG", "PLTR", "BAC", "ABBV","NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "META", "AVGO", "TSM", "TSLA", "BRK-B", "JPM", "WMT", "LLY", "V", "ORCL", "TCEHY", "MA", "NFLX", "XOM", "COST", "JNJ", "HD", "PG", "PLTR", "BAC","GS", "MSFT", "CAT", "HD", "V", "SHW", "AXP", "AMGN", "MCD", "JPM", "UNH", "CRM", "IBM", "TRV", "HON", "BA", "AMZN", "AAPL", "NVDA", "JNJ", "PG", "CVX", "MMM", "DIS", "WMT"
]

NUM_STOCKS_PER_EPISODE = 15  # Number of tickers agent trades simultaneously

INITIAL_BALANCE = 10000.0  # Starting cash for agent

# Reward weights
ALPHA_REWARD = 1.0  # portfolio value change weight
BETA_REWARD = 0.5  # sharpe ratio weight
GAMMA_REWARD = 0.2  # per-trade profit weight

# Episode length in hours (approx 3 months)
EPISODE_HOURS = 24 * 90

# Agent training params (keep your previous config or extend as needed)
config = {
    "env": {
        "window_size": 1,  # we only use 1-hour raw data per step here
        "initial_balance": INITIAL_BALANCE,
    },
    "agent": {
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "lr": 1e-3,
        "batch_size": 32,
        "memory_size": 10000,
        "target_update_freq": 50,
    },
    "training": {
        "episodes": 200,
        "max_steps": 500,
        "save_every": 25,
        "weights_path_dqn": "results/models/dqn_weights.pth",
        "weights_path_mc": "results/models/mc_weights.pth",
    },
    "logging": {"tensorboard_dir": "results/logs", "wandb_project": "stock-rl-agents"},
    "paths": {
        "utils_module": "utils.py",
        "train_dqn_script": "train/train_dqn.py",
        "train_mc_script": "train/train_mc.py",
    },
}
