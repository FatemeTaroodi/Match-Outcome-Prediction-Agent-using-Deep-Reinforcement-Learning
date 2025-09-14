## ⚽Match-Outcome-Prediction-Agent-using-Deep-Reinforcement-Learning
This project implements a Deep Reinforcement Learning (DRL) agent to predict the outcome of soccer matches by processing sequences of in-game events step by step.
Unlike traditional supervised learning approaches that rely on aggregated match statistics, this agent learns to anticipate the final result (win/draw/loss) dynamically as new events occur during the game.

##  Project Objectives

Perform exploratory data analysis (EDA) on the soccer event dataset.

Design and implement a custom simulation environment (SoccerMatchEnv) using Gymnasium.

Train and evaluate a DRL model (Deep Q-Network - DQN).

Analyze results and demonstrate how the agent’s accuracy improves as it observes more events.



## 📌 Repository Structure

RL-Code/ → Source code (data preparation, custom SoccerMatchEnv, DQN agent, training loop, results analysis).

SoccerMatchEnv.ipynb → Jupyter notebook for demonstration and testing.

README.md → Project documentation (this file).

training_results_final.csv → Saved training results (reward history, accuracy).

Sequential Prediction of Football Match Outcomes Using Deep Reinforcement Learning.docx → Code explanation and detailed report.

project.pdf → Exercise 3 solution (separate from the main project).

## 📊 Dataset

The project uses the Soccer Match Event Dataset, which includes:

matches_European_Championship.csv → 51 matches

events_European_Championship.csv → 78,125 in-game events

A new column match_outcome was added to each match:

0 → Draw

1 → Team 1 Win

2 → Team 2 Win

## 🏗️ Environment: SoccerMatchEnv

State Space: A 4D vector representing the normalized start and end positions of each event (x, y).

Action Space:

0 → Predict Draw

1 → Predict Team 1 Win

2 → Predict Team 2 Win

Reward Function:

Correct prediction → +1

Incorrect prediction → -1

Episode: A full match simulation, where the agent receives events sequentially until the game ends.

## 🤖 Model: Deep Q-Network (DQN)

Architecture:

Input: 4 features (pos_orig_x, pos_orig_y, pos_dest_x, pos_dest_y)

Two hidden layers (256 neurons, ReLU)

Output: Q-values for 3 possible actions (win/draw/loss)

Core Components:

Policy Network & Target Network

Replay Buffer

Epsilon-Greedy exploration strategy

Loss: Smooth L1 Loss (Huber Loss)

Optimizer: AdamW

## ⚙️ Experiment Results
Initial Test (default hyperparameters)

High fluctuations, unstable learning

Accuracy ~50% (close to random guessing)

Tuned Parameters

Learning rate = 1e-4

Batch size = 128

Replay buffer size = 100,000

Soft target update (tau = 0.005)

Episodes = 600 (analyzed up to episode 337)

## 📈 Results:

Convergence observed after ~260 episodes

Average reward ~500 (stable)

Prediction accuracy: 90–100% in later episodes


## 📚 Dependencies

Python 3.10+

Gymnasium

PyTorch

Pandas, Matplotlib, Seaborn

Kaggle API

## 📑 Report in persion

A detailed PDF report is included, covering:

Dataset EDA

Environment design

Model architecture

Training results & analysis

## 🔮 Future Work

Enrich state space with additional features (event type, player, time).

Explore advanced DRL algorithms (PPO, A2C, Transformer-based RL).

Extend to multi-agent setups for tactical analysis.

✍️ Author: [Fateme Taroodi]
