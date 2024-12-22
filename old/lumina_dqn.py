import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
import sys
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
NUM_DISCRETE_ACTIONS = 5
TRANSACTION_COST_RATE = 0.001
TRAINING_TIMESTEPS = 500000
RISK_FREE_RATE = 0.02
MODEL_NAME = "dqn_portfolio_agent"

#########################
# Data Loading and Preprocessing
#########################
logging.info("Downloading historical data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Adj Close"].dropna(how='all')
if data.empty:
    raise ValueError("No data was downloaded. Please check your tickers or date ranges.")

logging.info("Calculating returns and scaling...")
returns = data.pct_change().dropna()
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
        - Discrete actions mapping to portfolio allocations across multiple assets.
        
    Reward:
        - Daily portfolio return minus transaction costs.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, scaled_returns_df, returns, hidden_states, window_size=20, num_discrete_actions=5, transaction_cost_rate=0.001):
        super().__init__()
        self.returns = returns
        self.scaled_returns = scaled_returns_df
        self.hidden_states = hidden_states
        self.window_size = window_size
        self.n_assets = returns.shape[1]
        self.current_step = window_size
        self.num_discrete_actions = num_discrete_actions
        self.transaction_cost_rate = transaction_cost_rate

        # Action space: Each action is a discrete number that maps to a vector of weights.
        self.action_space = spaces.Discrete(self.num_discrete_actions ** self.n_assets)

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
        # According to Gymnasium API, return (obs, info)
        return obs, {}

    def step(self, action):
        action_vector = self._discrete_action_to_weights(action)
        if np.sum(action_vector) == 0:
            # Avoid division by zero if all weights are zero
            action_vector = np.ones(self.n_assets) / self.n_assets
        else:
            action_vector /= np.sum(action_vector)

        # Calculate portfolio return
        day_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(day_returns, action_vector)

        # Calculate transaction cost
        transaction_cost = np.sum(np.abs(self.portfolio - action_vector)) * self.transaction_cost_rate

        # Update portfolio
        self.portfolio = action_vector

        # Reward: return minus transaction cost
        reward = portfolio_return - transaction_cost

        self.current_step += 1
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False  # We are not artificially truncating

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def _discrete_action_to_weights(self, action):
        # Convert a single integer action into a multi-asset weight vector
        digits = np.base_repr(action, base=self.num_discrete_actions, padding=self.n_assets)[-self.n_assets:]
        weights = np.array([int(d) / (self.num_discrete_actions - 1) for d in digits])
        return weights

    def _get_observation(self):
        if self.current_step < self.window_size:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Past scaled returns
        past_scaled_returns = self.scaled_returns.iloc[self.current_step - self.window_size: self.current_step].values.flatten()

        current_regime = np.array([self.hidden_states[self.current_step]], dtype=np.float32)

        # Compute moving average and volatility for the last step
        recent_returns = self.returns.iloc[self.current_step - self.window_size: self.current_step]
        ma = recent_returns.mean(axis=0).values
        vol = recent_returns.std(axis=0).values

        observation = np.concatenate((past_scaled_returns, current_regime, ma, vol))
        return observation.astype(np.float32)

#########################
# Instantiate and Train the Agent
#########################
logging.info("Initializing environment and training the DQN agent...")
env = TradingEnv(scaled_returns_df, returns, hidden_states, window_size=WINDOW_SIZE, num_discrete_actions=NUM_DISCRETE_ACTIONS, transaction_cost_rate=TRANSACTION_COST_RATE)
env = DummyVecEnv([lambda: env])

model = DQN("MlpPolicy", env, verbose=1)

logging.info("Starting training...")
model.learn(total_timesteps=TRAINING_TIMESTEPS)
model.save(MODEL_NAME)
logging.info("Training completed and model saved.")

#########################
# Backtesting and Evaluation
#########################
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02):
    portfolio_returns = np.array(portfolio_returns)
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    return (np.sqrt(252) * np.mean(excess_returns)) / (np.std(excess_returns) + 1e-8)

def calculate_max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / peak
    return np.max(drawdowns)

logging.info("Starting backtest...")
model = DQN.load(MODEL_NAME)

# Backtest the agent
obs = env.reset()  
portfolio_values = [1.0]
portfolio_returns = []
actions_taken = []

steps_to_backtest = len(returns) - WINDOW_SIZE - 1
for i in tqdm(range(steps_to_backtest), desc="Backtesting"):
    action, _states = model.predict(obs, deterministic=True)
    actions_taken.append(action)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(portfolio_values[-1] * (1 + reward[0]))
    portfolio_returns.append(reward[0])
    if done[0]:
        break

# Compute metrics
sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, RISK_FREE_RATE)
max_drawdown = calculate_max_drawdown(portfolio_values)

logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
logging.info(f"Maximum Drawdown: {max_drawdown:.2%}")

plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="DQN Portfolio")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.title("DQN Portfolio Performance")
plt.legend()
plt.show()

logging.info("Execution completed successfully.")
