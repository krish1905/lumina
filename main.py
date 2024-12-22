"""
lumina - Reinforcement Learning for Portfolio Management
Copyright (c) 2024 Krish
All rights reserved.

This script downloads market data, detects regimes via an HMM,
trains a PPO agent on a custom Gym environment, and backtests
the resulting policy. 

DISCLAIMER:
This code is provided for educational purposes only and does not
constitute financial advice. Use at your own risk.
"""

import logging
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

#########################
# User-Configurable Parameters
#########################
TICKERS = ["GS", "NVDA", "BRK-B", "C", "JPM", "^VIX"]
START_DATE = "2010-01-01"
END_DATE = "2024-12-15"
N_COMPONENTS_HMM = 3
WINDOW_SIZE = 20
TRANSACTION_COST_RATE = 0.001
TRAINING_TIMESTEPS = 500000  # Increased training timesteps
RISK_FREE_RATE = 0.02
MODEL_NAME = "ppo_portfolio_agent"

#########################
# Data Loading and Preprocessing
#########################
logging.info("Downloading historical data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Adj Close"].dropna(how='all')
if data.empty:
    raise ValueError("No data was downloaded. Please check your tickers or date ranges.")

logging.info("Calculating returns and scaling...")
returns = data.pct_change().dropna()
returns = returns.replace([np.inf, -np.inf], 0).fillna(0)  # Replace infinities and NaNs
scaler = MinMaxScaler()
scaled_returns = scaler.fit_transform(returns)  # scaled_returns is now a numpy array

# Convert back to DataFrame for convenience
scaled_returns_df = pd.DataFrame(scaled_returns, index=returns.index, columns=returns.columns)

#########################
# Hidden Markov Model (HMM) to Identify Regimes
#########################
logging.info("Fitting HMM to identify market regimes...")
hmm_model = hmm.GaussianHMM(n_components=N_COMPONENTS_HMM, covariance_type="full")
hmm_model.fit(scaled_returns)
hidden_states = hmm_model.predict(scaled_returns)

#########################
# Technical Indicators
#########################
def calculate_technical_indicators(returns, window=20):
    moving_avg = returns.rolling(window=window).mean()
    volatility = returns.rolling(window=window).std()
    return moving_avg, volatility

moving_avg, volatility = calculate_technical_indicators(returns, WINDOW_SIZE)

#########################
# Custom Gymnasium Environment
#########################
class TradingEnv(gym.Env):
    """
    A trading environment for RL agents.
    Observation:
        - Past scaled returns for a specified window.
        - Current market regime (from HMM).
        - Moving average and volatility indicators (unscaled).
        
    Action:
        - Continuous actions mapping to portfolio allocations across multiple assets.
        
    Reward:
        - Portfolio return minus transaction costs.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, scaled_returns_df, returns, hidden_states, window_size=20, transaction_cost_rate=0.001):
        super().__init__()
        self.returns = returns
        self.scaled_returns = scaled_returns_df
        self.hidden_states = hidden_states
        self.window_size = window_size
        self.n_assets = returns.shape[1]
        self.current_step = window_size
        self.transaction_cost_rate = transaction_cost_rate

        # Continuous action space: Portfolio weights for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation space: [past_scaled_returns, regime, ma, vol]
        observation_space_size = self.n_assets * window_size + 1 + (self.n_assets * 2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_space_size,),
            dtype=np.float32
        )
        self.portfolio = np.zeros(self.n_assets)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio = np.zeros(self.n_assets)
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action_vector = np.clip(action, 0, 1)  # Ensure actions are within bounds
        if np.sum(action_vector) == 0:
            action_vector = np.ones(self.n_assets) / self.n_assets  # Avoid division by zero
        else:
            action_vector /= np.sum(action_vector)  # Normalize weights

        day_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(day_returns, action_vector)

        transaction_cost = np.sum(np.abs(self.portfolio - action_vector)) * self.transaction_cost_rate

        reward = portfolio_return - transaction_cost

        self.portfolio = action_vector
        self.current_step += 1
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if self.current_step < self.window_size:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        past_scaled_returns = self.scaled_returns.iloc[self.current_step - self.window_size: self.current_step].values.flatten()
        current_regime = np.array([self.hidden_states[self.current_step]], dtype=np.float32)
        recent_returns = self.returns.iloc[self.current_step - self.window_size: self.current_step]
        ma = recent_returns.mean(axis=0).fillna(0).values  # Replace NaN with 0
        vol = recent_returns.std(axis=0).fillna(0).values  # Replace NaN with 0
        observation = np.concatenate((past_scaled_returns, current_regime, ma, vol))
        return observation.astype(np.float32)

#########################
# Train-Test Split
#########################
train_start, train_end = "2010-01-01", "2018-12-31"
test_start, test_end = "2019-01-01", "2024-12-15"

train_returns = returns[train_start:train_end]
test_returns = returns[test_start:test_end]

train_hidden_states = hidden_states[:len(train_returns)]
test_hidden_states = hidden_states[len(train_returns):]

train_scaled_returns = scaled_returns_df[train_start:train_end]
test_scaled_returns = scaled_returns_df[test_start:test_end]

#########################
# Training Environment
#########################
train_env = TradingEnv(
    train_scaled_returns,
    train_returns,
    train_hidden_states,
    window_size=WINDOW_SIZE,
    transaction_cost_rate=TRANSACTION_COST_RATE
)
train_env = DummyVecEnv([lambda: train_env])

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=1e-5,  # Smaller learning rate
    batch_size=256,
    n_steps=2048,
    gamma=0.99,
    ent_coef=0.001,
    vf_coef=0.5,
)

logging.info("Training the PPO model...")
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/")
model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=checkpoint_callback)
model.save(MODEL_NAME)

#########################
# Testing Environment
#########################
test_env = TradingEnv(
    test_scaled_returns,
    test_returns,
    test_hidden_states,
    window_size=WINDOW_SIZE,
    transaction_cost_rate=TRANSACTION_COST_RATE
)
test_env = DummyVecEnv([lambda: test_env])

#########################
# Backtesting the Trained Model
#########################
def backtest_ppo(model, env, returns, window_size):
    obs = env.reset()[0]
    portfolio_values = [1.0]
    portfolio_returns = []

    for _ in tqdm(range(len(returns) - window_size - 1), desc="Backtesting PPO Strategy"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs = obs[0]
        reward = reward[0]
        done = done[0]

        portfolio_values.append(portfolio_values[-1] * (1 + reward))
        portfolio_returns.append(reward)

        if done:
            break

    return np.array(portfolio_values), np.array(portfolio_returns)

# Run backtest
portfolio_values, portfolio_returns = backtest_ppo(model, test_env, test_returns, WINDOW_SIZE)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="PPO Strategy")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.title("PPO Strategy Portfolio Performance")
plt.legend()
plt.show()

#########################
# Evaluation Metrics
#########################
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02):
    portfolio_returns = np.array(portfolio_returns)
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    return (np.sqrt(252) * np.mean(excess_returns)) / (np.std(excess_returns) + 1e-8)

def calculate_max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / peak
    return np.max(drawdowns)

sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, RISK_FREE_RATE)
max_drawdown = calculate_max_drawdown(portfolio_values)

print(f"PPO Strategy - Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")
