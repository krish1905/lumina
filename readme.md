# Lumina - Reinforcement Learning for Portfolio Management

**Author:** Krish Garg
**Copyright:** © 2024 Krish. All rights reserved.

## Overview

**Lumina** is a Python-based project that leverages Reinforcement Learning (RL) for portfolio management. By integrating market data analysis, Hidden Markov Models (HMM), and a custom RL environment built with Gymnasium, this project trains and evaluates a Proximal Policy Optimization (PPO) agent to optimize portfolio performance.

---

## Features

1. **Market Regime Detection**: Utilizes Hidden Markov Models to identify market regimes based on historical returns.
2. **Reinforcement Learning Environment**: Custom Gymnasium environment tailored for portfolio management.
3. **Technical Indicators**: Includes moving averages and volatility for enhanced state representation.
4. **PPO Agent Training**: Trains a PPO model on a realistic simulation of market conditions.
5. **Backtesting**: Evaluates the trained strategy using historical data.
6. **Metrics**: Computes key metrics such as Sharpe Ratio and Maximum Drawdown.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Install required libraries:
  ```bash
  pip install -r requirements.txt

---

### Configuration

Update the user-configurable parameters in the script as required:

- Assets: TICKERS = ["GS", "NVDA", "BRK-B", "C", "JPM", "^VIX"]
- Date Range: START_DATE and END_DATE
- HMM Components: N_COMPONENTS_HMM = 3
- Transaction Costs: TRANSACTION_COST_RATE = 0.001
- Training Timesteps: TRAINING_TIMESTEPS = 500000

---

## Disclamer

This code is provided for educational purposes only and does not constitute financial advice. Use at your own risk.