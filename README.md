# 🤖 CartPole Reinforcement Learning Lab

An end-to-end Reinforcement Learning project that uses **Q-Learning** to train an agent to balance a pole on a moving cart.

The project includes:

- Q-Learning training
- Continuous-state discretization
- Q-table learning
- Exploration vs exploitation
- Model saving
- Agent evaluation
- Training visualization
- Random Agent vs trained Agent comparison
- Interactive Streamlit frontend
- GitHub-ready project structure

---

## 🚀 Live Application

**Streamlit App:**  
Add your deployed Streamlit URL here after deployment.

**GitHub Repository:**  
https://github.com/kumarrajababu5454-lgtm/rl-cartpole-project

---

# 📌 Project Overview

CartPole is a classic Reinforcement Learning environment.

The objective is simple:

> Keep the pole balanced on the cart for as long as possible.

The agent can perform only two actions:

```text
0 → Move Left
1 → Move Right

🎯 What the Agent Learns

CartPole provides four values:

Cart position
Cart velocity
Pole angle
Pole angular velocity

These values are continuous.

For example:

Cart position = 0.12
Cart velocity = -0.43
Pole angle = 0.03
Angular velocity = 0.18

Q-Learning works more conveniently with discrete states.

Therefore, the project divides each continuous value into bins.

🧩 State Discretization

The project converts continuous values into discrete indexes.

For example:

Continuous state
        ↓
[0.12, -0.43, 0.03, 0.18]
        ↓
Discretization
        ↓
(5, 4, 6, 5)

The resulting tuple is used as an index in the Q-table.

📊 Q-Table

The Q-table stores the learned value of each possible action for each state.

The table has the structure:

State 1
State 2
State 3
...
        ↓
LEFT / RIGHT

The project uses:

q_table.shape

with:

11 × 11 × 11 × 11 × 2

The final dimension represents the two actions.

🔄 Exploration vs Exploitation

At the beginning of training, the agent does not know what to do.

Therefore, it explores.

Exploration
    ↓
Try random actions

Later, the agent increasingly uses what it has learned.

Exploitation
    ↓
Choose the action with the highest Q-value

This behavior is controlled using epsilon.

The project starts with:

epsilon = 1.0

and gradually reduces it.

📐 Q-Learning Update

The Q-value is updated using the standard Q-Learning formula:

Q(s,a) ← Q(s,a) + α[
    r + γ max Q(s',a') - Q(s,a)
]

Where:

Q(s,a) = current Q-value

α = learning rate

r = reward

γ = discount factor

s' = next state

a' = next action

The code implements this update directly.

🏋️ Training

The agent was trained for:

500 episodes

Important parameters:

Learning rate:       0.1
Discount factor:     0.99
Initial epsilon:     1.0
Epsilon decay:       0.995
Minimum epsilon:     0.01
Training episodes:   500

The training is intentionally lightweight.

No GPU is required.

No large neural network is used.

💾 Saved Model

After training, the learned Q-table is saved as:

q_table.npy

This allows the trained agent to be used later without retraining.

📈 Training Results

Training performance is stored in:

training_rewards.png

The graph shows how the agent's reward changes during training.

The later episodes generally show better performance than the early training episodes, although individual episodes can vary.

🧪 Evaluation

The trained agent is evaluated separately from training.

The evaluation process:

Load q_table.npy
        ↓
Start CartPole
        ↓
Choose best learned action
        ↓
Run episode
        ↓
Record score

The score represents how long the pole remained balanced.

🆚 Random Agent vs Q-Learning Agent

The Streamlit application allows users to compare:

Random Agent

The Random Agent selects actions randomly.

LEFT
RIGHT
LEFT
LEFT
RIGHT
...

It has no learned strategy.

Trained Q-Learning Agent

The trained agent uses the Q-table.

Observe state
     ↓
Look at Q-values
     ↓
Choose better action

This demonstrates the main idea of Reinforcement Learning:

An agent can improve its behavior by learning from interaction with an environment.

🖥️ Interactive Streamlit Frontend

The project includes an interactive web application.

The user can select:

Agent
├── Trained Q-Learning Agent
└── Random Agent

The user can also select the number of episodes.

After clicking:

▶ Run Agent

the application displays:

Average score
Best score
Worst score
Average steps
Episode results
Performance chart
Agent decisions
Training history
🖼️ Frontend Workflow
User
 ↓
Streamlit Interface
 ↓
Select Agent
 ↓
Select Number of Episodes
 ↓
Run Agent
 ↓
CartPole Environment
 ↓
Agent Makes Decisions
 ↓
Results
 ↓
Charts + Metrics
📁 Project Structure
rl-cartpole-project/
│
├── app.py
│
├── q_learning.py
│
├── evaluate.py
│
├── cartpole_qlearning.py
│
├── q_table.npy
│
├── training_rewards.png
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
📄 File-by-File Explanation
q_learning.py

Main training program.

It:

Creates the CartPole environment.
Discretizes the state.
Creates the Q-table.
Trains the agent.
Updates Q-values.
Stores rewards.
Saves the trained Q-table.

Output:

q_table.npy
evaluate.py

Evaluates the trained Q-Learning agent.

It loads the saved Q-table and runs the agent without training.

This separates:

Training

from:

Evaluation
app.py

The Streamlit frontend.

It loads the trained Q-table and provides an interactive interface.

Users can:

Select an agent.
Select number of episodes.
Run simulations.
Compare performance.
View results.
View agent decisions.
cartpole_qlearning.py

A lightweight/basic CartPole Q-Learning implementation created during the project development.

The main finalized training workflow is contained in q_learning.py.

q_table.npy

The trained Q-table.

This is the learned knowledge of the agent.

It is loaded by the evaluation script and Streamlit application.

training_rewards.png

Visualization of training performance.

requirements.txt

Contains the Python packages required to run the project.

.gitignore

Prevents unwanted files such as Python cache files and local environment files from being committed.

🛠️ Technologies Used
Python
NumPy
Pandas
Gymnasium
Streamlit
Matplotlib
Q-Learning
Git
GitHub
💻 Run Locally

Clone the repository:

git clone https://github.com/kumarrajababu5454-lgtm/rl-cartpole-project.git

Enter the project directory:

cd rl-cartpole-project

Install dependencies:

python -m pip install -r requirements.txt
🏋️ Train the Agent

Run:

python q_learning.py

This creates:

q_table.npy
training_rewards.png
🧪 Evaluate the Agent

Run:

python evaluate.py
🌐 Run Streamlit

Run:

streamlit run app.py

The application will open in your browser.

☁️ Deployment

The Streamlit frontend can be deployed using Streamlit Community Cloud.

The deployed application only needs the trained:

q_table.npy

It does not need to retrain the model every time a user opens the application.

This keeps the application lightweight.

⚡ Hardware Considerations

This project was designed to run on a normal CPU.

It does not require:

GPU
CUDA
TensorFlow
PyTorch
Large language models
Large datasets

The Q-table is relatively small and the training process is lightweight.

⚠️ Limitations

This implementation uses tabular Q-Learning with state discretization.

CartPole actually has continuous states.

Discretizing the states makes the problem manageable for a Q-table, but it also loses some information.

The agent is therefore not a production-level RL system.

It is primarily an educational and portfolio implementation demonstrating the fundamentals of Reinforcement Learning.

🚀 Future Improvements

Possible future improvements include:

More efficient state discretization
Reward analysis
Longer training
Better hyperparameter tuning
Q-Learning performance comparison
SARSA comparison
Deep Q-Network (DQN)
Live CartPole animation
More detailed agent analytics
🎓 What This Project Demonstrates

This project demonstrates practical understanding of:

Reinforcement Learning
        ↓
Environment
        ↓
State
        ↓
Action
        ↓
Reward
        ↓
Q-Table
        ↓
Exploration
        ↓
Exploitation
        ↓
Learning
        ↓
Evaluation
        ↓
Deployment

It is an end-to-end implementation rather than only a theoretical example.

👨‍💻 Project Status

Completed ✅

 CartPole environment
 State discretization
 Q-table
 Q-Learning training
 Exploration/exploitation
 Model saving
 Evaluation
 Training visualization
 Interactive Streamlit frontend
 GitHub repository
 Streamlit deployment URL
📌 Author

Raja

GitHub:

https://github.com/kumarrajababu5454-lgtm
