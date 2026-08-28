import gymnasium as gym
import numpy as np


# ============================================================
# 1. LOAD TRAINED Q-TABLE
# ============================================================

q_table = np.load("q_table.npy")


# ============================================================
# 2. CREATE CARTPOLE ENVIRONMENT
# ============================================================

# render_mode="human" opens the CartPole game window.

env = gym.make(
    "CartPole-v1",
    render_mode="human"
)


# ============================================================
# 3. SAME BINS USED DURING TRAINING
# ============================================================

bins = [
    np.linspace(-2.4, 2.4, 10),
    np.linspace(-3.0, 3.0, 10),
    np.linspace(-0.2095, 0.2095, 10),
    np.linspace(-3.5, 3.5, 10)
]


def discretize_state(state):
    """
    Convert continuous state into
    the same discrete format used during training.
    """

    discrete_state = []

    for i in range(len(state)):

        value = np.digitize(
            state[i],
            bins[i]
        )

        value = min(value, 10)

        discrete_state.append(value)

    return tuple(discrete_state)


# ============================================================
# 4. EVALUATION
# ============================================================

episodes = 5

scores = []


for episode in range(episodes):

    state, info = env.reset()

    state = discretize_state(state)

    total_reward = 0

    done = False


    # ========================================================
    # RUN ONE EPISODE
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # USE LEARNED POLICY
        # ----------------------------------------------------
        #
        # During evaluation we do NOT randomly explore.
        #
        # The agent always chooses the action
        # with the highest learned Q-value.

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

        state = discretize_state(
            next_state
        )

        total_reward += reward


    # ========================================================
    # STORE SCORE
    # ========================================================

    scores.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Score = {total_reward:.0f}"
    )


# ============================================================
# 5. CLOSE ENVIRONMENT
# ============================================================

env.close()


# ============================================================
# 6. FINAL RESULTS
# ============================================================

print("\nEvaluation completed!")

print(
    f"Average score: {np.mean(scores):.2f}"
)

print(
    f"Best score: {max(scores):.0f}"
)