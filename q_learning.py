import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CREATE CARTPOLE ENVIRONMENT
# ============================================================

env = gym.make("CartPole-v1")


# ============================================================
# 2. DISCRETIZE THE CONTINUOUS STATE
# ============================================================

# CartPole gives 4 continuous values:
#
# 1. Cart position
# 2. Cart velocity
# 3. Pole angle
# 4. Pole angular velocity
#
# Q-learning uses discrete states.
# We divide each value into bins.

bins = [
    np.linspace(-2.4, 2.4, 10),
    np.linspace(-3.0, 3.0, 10),
    np.linspace(-0.2095, 0.2095, 10),
    np.linspace(-3.5, 3.5, 10)
]


def discretize_state(state):
    """
    Convert the continuous CartPole state
    into a discrete state.
    """

    discrete_state = []

    for i in range(len(state)):

        value = np.digitize(
            state[i],
            bins[i]
        )

        # Keep the state index inside
        # the Q-table range.
        value = min(value, 10)

        discrete_state.append(value)

    return tuple(discrete_state)


# ============================================================
# 3. CREATE Q-TABLE
# ============================================================

# 4 state dimensions
# Each dimension has 11 possible values.
#
# Final dimension represents the action:
#
# 0 = Move left
# 1 = Move right

q_table = np.zeros(
    (
        11,
        11,
        11,
        11,
        2
    )
)


# ============================================================
# 4. Q-LEARNING PARAMETERS
# ============================================================

learning_rate = 0.1

discount_factor = 0.99

# Exploration probability
epsilon = 1.0

# Gradually reduce exploration
epsilon_decay = 0.995

# Minimum exploration
epsilon_min = 0.01

# Number of training episodes
episodes = 500


# ============================================================
# 5. STORE TRAINING REWARDS
# ============================================================

episode_rewards = []


# ============================================================
# 6. TRAIN THE AGENT
# ============================================================

for episode in range(episodes):

    # Reset environment
    state, info = env.reset()

    # Convert continuous state
    # into discrete state.
    state = discretize_state(state)

    total_reward = 0

    done = False


    # ========================================================
    # RUN CURRENT EPISODE
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # CHOOSE ACTION
        # ----------------------------------------------------

        if np.random.random() < epsilon:

            # EXPLORATION
            # Choose a random action.

            action = env.action_space.sample()

        else:

            # EXPLOITATION
            # Choose the action with the
            # highest learned Q-value.

            action = np.argmax(
                q_table[state]
            )


        # ----------------------------------------------------
        # TAKE ACTION
        # ----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated


        # ----------------------------------------------------
        # CONVERT NEXT STATE
        # ----------------------------------------------------

        next_state = discretize_state(
            next_state
        )


        # ----------------------------------------------------
        # Q-LEARNING UPDATE
        # ----------------------------------------------------

        best_next_value = np.max(
            q_table[next_state]
        )

        current_q_value = q_table[
            state + (action,)
        ]

        new_q_value = (
            current_q_value
            + learning_rate
            * (
                reward
                + discount_factor * best_next_value
                - current_q_value
            )
        )

        q_table[
            state + (action,)
        ] = new_q_value


        # ----------------------------------------------------
        # MOVE TO NEXT STATE
        # ----------------------------------------------------

        state = next_state

        total_reward += reward


    # ========================================================
    # REDUCE EXPLORATION
    # ========================================================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    # ========================================================
    # STORE REWARD
    # ========================================================

    episode_rewards.append(
        total_reward
    )


    # ========================================================
    # DISPLAY TRAINING PROGRESS
    # ========================================================

    if (episode + 1) % 10 == 0:

        average_reward = np.mean(
            episode_rewards[-10:]
        )

        print(
            f"Episode: {episode + 1:3d} | "
            f"Average Reward: {average_reward:6.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# ============================================================
# 7. SAVE TRAINED Q-TABLE
# ============================================================

np.save(
    "q_table.npy",
    q_table
)


# ============================================================
# 8. FINAL RESULTS
# ============================================================

print("\nTraining completed!")

print(
    "Trained Q-table saved as q_table.npy"
)

print(
    f"Best episode reward: "
    f"{max(episode_rewards):.0f}"
)

print(
    f"Final 10-episode average: "
    f"{np.mean(episode_rewards[-10:]):.2f}"
)


# ============================================================
# 9. CREATE TRAINING REWARD GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    episode_rewards,
    label="Episode Reward"
)

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title(
    "CartPole Q-Learning Training Progress"
)

plt.legend()

plt.tight_layout()


# Save graph to the project folder.
plt.savefig(
    "training_rewards.png"
)

plt.show()


# ============================================================
# 10. CLOSE ENVIRONMENT
# ============================================================

env.close()

print(
    "Training graph saved as training_rewards.png"
)