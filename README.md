# Reinforcement Learning CartPole with Q-Learning

A lightweight Reinforcement Learning project that teaches an agent
to balance a pole on a moving cart using the Q-Learning algorithm.

The project uses Gymnasium, NumPy, and Matplotlib.

---

## Project Overview

In this project, an RL agent interacts with the CartPole environment.

The agent:

1. Observes the current state.
2. Chooses an action.
3. Receives a reward.
4. Observes the next state.
5. Updates its Q-table.
6. Repeats this process over many episodes.
7. Uses the learned Q-table to make decisions during evaluation.

The goal is to keep the pole balanced for as many time steps as possible.

---

## Technologies Used

- Python
- Gymnasium
- NumPy
- Matplotlib
- Q-Learning
- Reinforcement Learning

No GPU or large deep-learning model is required.

---

## Why Q-Learning?

Q-Learning is a model-free Reinforcement Learning algorithm.

Instead of being given the correct action, the agent learns
which actions are better by interacting with the environment.

The agent stores learned values inside a Q-table.

The Q-table represents:

```text
State + Action → Q-Value
CartPole Environment

CartPole provides four main state values:

Cart position
Cart velocity
Pole angle
Pole angular velocity

The environment has two possible actions:

0 → Move cart left
1 → Move cart right

The agent receives a reward for keeping the pole balanced.

Continuous State to Discrete State

CartPole produces continuous values.

Basic Q-Learning works more easily with discrete states.

Therefore, the project divides each state variable into bins.

For example:

Continuous value
       ↓
   Discretization
       ↓
Discrete state
       ↓
Q-table lookup

This allows the Q-table to store learned values for different
combinations of states.

Q-Learning Process

The training process follows this loop:

Start Episode
      ↓
Observe State
      ↓
Choose Action
      ↓
Take Action
      ↓
Receive Reward
      ↓
Observe Next State
      ↓
Update Q-Value
      ↓
Repeat
      ↓
End Episode
      ↓
Start Next Episode
Exploration vs Exploitation

The agent uses epsilon-greedy action selection.

At the beginning:

epsilon = 1.0

This means the agent explores many random actions.

As training continues, epsilon decreases.

Eventually, the agent relies more heavily on its learned Q-values.

The project uses:

epsilon_decay = 0.995
epsilon_min = 0.01
Q-Learning Update

The Q-value is updated using:

Q(s,a) = Q(s,a) + α[
    r + γ max Q(s',a') - Q(s,a)
]

Where:

s = current state
a = current action
r = reward
s' = next state
α = learning rate
γ = discount factor

The implementation uses:

Learning rate = 0.1
Discount factor = 0.99
Project Structure
rl-cartpole-project/
│
├── cartpole_qlearning.py
├── q_learning.py
├── evaluate.py
├── q_table.npy
├── training_rewards.png
├── requirements.txt
├── .gitignore
└── README.md
File Explanation
cartpole_qlearning.py

A small initial test file used to verify that Gymnasium
and the CartPole environment are working correctly.

It checks:

Observation Space
Action Space
q_learning.py

This is the main training program.

It:

Creates the CartPole environment.
Converts continuous states into discrete states.
Creates the Q-table.
Uses epsilon-greedy exploration.
Trains the agent.
Updates Q-values.
Saves the trained Q-table.
Creates a training reward graph.

The trained model is saved as:

q_table.npy
evaluate.py

This file loads the trained Q-table.

It does not train the agent again.

Instead, it uses the learned policy to select actions.

q_table.npy
      ↓
evaluate.py
      ↓
Learned policy
      ↓
CartPole

It reports the score achieved by the agent.

q_table.npy

This file contains the trained Q-table.

It represents what the agent learned during training.

It can be loaded later without training the model again.

training_rewards.png

This graph shows the reward received by the agent during training.

It helps visualize the learning process.

requirements.txt

Contains the Python packages required to run the project:

gymnasium
numpy
matplotlib
.gitignore

Prevents unnecessary local files and folders from being uploaded
to GitHub.

Installation

Make sure Python is installed.

Install the required packages:

python -m pip install -r requirements.txt
Train the Agent

Run:

python q_learning.py

The program trains for:

500 episodes

Training progress is displayed every 10 episodes.

Example:

Episode: 100 | Average Reward: 36.60 | Epsilon: 0.606
Episode: 200 | Average Reward: 35.00 | Epsilon: 0.367
Episode: 300 | Average Reward: 51.20 | Epsilon: 0.222
Episode: 400 | Average Reward: 67.90 | Epsilon: 0.135
Episode: 500 | Average Reward: 54.60 | Epsilon: 0.082

The exact results can change because Reinforcement Learning
contains randomness.

Evaluate the Agent

After training:

python evaluate.py

The program loads:

q_table.npy

and evaluates the learned policy.

Example result:

Episode 1: Score = ...
Episode 2: Score = ...
Episode 3: Score = 37
Episode 4: Score = 28
Episode 5: Score = 32

Evaluation completed!
Average score: 30.20
Best score: 37

The exact scores may vary between runs.

Training Visualization

Training produces:

training_rewards.png

The graph shows how the reward changes across episodes.

The reward does not need to increase perfectly every episode.

Reinforcement Learning training can fluctuate because the agent
continues exploring different actions.

Hardware Requirements

This project is intentionally lightweight.

It does not require:

GPU
PyTorch
TensorFlow
Large AI models
Large datasets

A normal CPU-based computer can run it.

What This Project Demonstrates

This project demonstrates practical understanding of:

Reinforcement Learning
Agents and environments
States
Actions
Rewards
Q-tables
Q-Learning
Exploration
Exploitation
Epsilon-greedy strategy
State discretization
Model saving
Model evaluation
Training visualization
End-to-End Workflow
CartPole Environment
        ↓
Continuous State
        ↓
State Discretization
        ↓
Q-Table
        ↓
Action Selection
        ↓
Environment
        ↓
Reward
        ↓
Q-Table Update
        ↓
Training
        ↓
Saved Q-Table
        ↓
Evaluation
        ↓
Performance Score
Project Result

The final system successfully trains a Reinforcement Learning
agent to interact with the CartPole environment.

The trained Q-table is saved locally and can be reused for
evaluation without retraining.

This project is designed as a practical implementation of
Reinforcement Learning using a lightweight Q-Learning approach.

Future Improvements

Possible future improvements include:

More training episodes
Better state discretization
Reward smoothing
Moving-average reward visualization
Deep Q-Network (DQN)
Neural-network-based RL agent
Improved evaluation
Interactive visualization
Author

Built as a hands-on Reinforcement Learning project using Python
and Gymnasium.


Save it with **Ctrl + S**.

### Your folder should now contain

```text
rl-cartpole-project/
│
├── cartpole_qlearning.py
├── q_learning.py
├── evaluate.py
├── q_table.npy
├── training_rewards.png
├── requirements.txt
├── .gitignore
└── README.md