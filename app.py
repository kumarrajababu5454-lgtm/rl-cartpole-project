import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CartPole RL Lab",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 CartPole Reinforcement Learning Lab")

st.write(
    "Compare a random agent with a trained Q-Learning agent "
    "and see how reinforcement learning improves decision making."
)


# ============================================================
# LOAD TRAINED Q-TABLE
# ============================================================

@st.cache_resource
def load_q_table():

    return np.load("q_table.npy")


q_table = load_q_table()


# ============================================================
# STATE DISCRETIZATION
# ============================================================

bins = [
    np.linspace(-2.4, 2.4, 10),
    np.linspace(-3.0, 3.0, 10),
    np.linspace(-0.2095, 0.2095, 10),
    np.linspace(-3.5, 3.5, 10)
]


def discretize_state(state):

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
# RUN ONE EPISODE
# ============================================================

def run_episode(agent_type):

    env = gym.make("CartPole-v1")

    state, info = env.reset()

    state = discretize_state(state)

    total_reward = 0

    steps = 0

    actions = []

    done = False

    while not done:

        # ----------------------------------------------------
        # SELECT ACTION
        # ----------------------------------------------------

        if agent_type == "Random Agent":

            action = env.action_space.sample()

        else:

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
        # SAVE ACTION
        # ----------------------------------------------------

        actions.append(
            "LEFT" if action == 0 else "RIGHT"
        )

        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state = discretize_state(
            next_state
        )

        total_reward += reward

        steps += 1

    env.close()

    return total_reward, steps, actions


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

agent_type = st.sidebar.selectbox(
    "Choose Agent",
    [
        "Trained Q-Learning Agent",
        "Random Agent"
    ]
)

episodes = st.sidebar.slider(
    "Number of Episodes",
    min_value=1,
    max_value=20,
    value=5
)


# ============================================================
# AGENT INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Algorithm",
        "Q-Learning"
    )

with col2:

    st.metric(
        "Environment",
        "CartPole-v1"
    )

with col3:

    st.metric(
        "State Dimensions",
        "4"
    )

with col4:

    st.metric(
        "Actions",
        "2"
    )


st.divider()


# ============================================================
# RUN AGENT
# ============================================================

st.subheader("🎮 Run Agent")

if st.button(
    "▶ Run Agent",
    use_container_width=True
):

    rewards = []

    steps_list = []

    all_actions = []

    progress = st.progress(0)

    status = st.empty()

    for episode in range(episodes):

        status.write(
            f"Running episode {episode + 1} of {episodes}..."
        )

        reward, steps, actions = run_episode(
            agent_type
        )

        rewards.append(reward)

        steps_list.append(steps)

        all_actions.append(actions)

        progress.progress(
            (episode + 1) / episodes
        )

    status.success(
        "Simulation completed!"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    average_score = np.mean(
        rewards
    )

    best_score = np.max(
        rewards
    )

    worst_score = np.min(
        rewards
    )

    average_steps = np.mean(
        steps_list
    )


    st.subheader("📊 Results")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:

        st.metric(
            "Average Score",
            f"{average_score:.1f}"
        )

    with result_col2:

        st.metric(
            "Best Score",
            f"{best_score:.0f}"
        )

    with result_col3:

        st.metric(
            "Worst Score",
            f"{worst_score:.0f}"
        )

    with result_col4:

        st.metric(
            "Average Steps",
            f"{average_steps:.1f}"
        )


    # ========================================================
    # EPISODE TABLE
    # ========================================================

    st.subheader("📋 Episode Results")

    results_df = pd.DataFrame(
        {
            "Episode": range(
                1,
                episodes + 1
            ),
            "Score": rewards,
            "Steps": steps_list
        }
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PERFORMANCE GRAPH
    # ========================================================

    st.subheader("📈 Episode Performance")

    chart_df = pd.DataFrame(
        {
            "Score": rewards
        }
    )

    st.line_chart(
        chart_df
    )


    # ========================================================
    # ACTIONS
    # ========================================================

    st.subheader("🧠 Agent Decisions")

    selected_episode = st.selectbox(
        "Select an episode",
        range(
            1,
            episodes + 1
        )
    )

    selected_actions = all_actions[
        selected_episode - 1
    ]

    st.write(
        " → ".join(
            selected_actions[:50]
        )
    )

    if len(selected_actions) > 50:

        st.caption(
            f"Showing first 50 actions out of "
            f"{len(selected_actions)} actions."
        )


# ============================================================
# EXPLANATION
# ============================================================

st.divider()

st.subheader("🧠 What is happening?")

st.write(
    """
The CartPole environment gives the agent information about
the cart and pole. The agent must choose between two actions:
LEFT or RIGHT.

The Random Agent chooses actions randomly.

The trained Q-Learning Agent uses the knowledge stored in
q_table.npy to select the action with the highest learned
Q-value.

The score represents how long the pole stayed balanced.
"""
)


# ============================================================
# TRAINING GRAPH
# ============================================================

st.subheader("📈 Training History")

st.image(
    "training_rewards.png",
    caption="Q-Learning Training Performance"
)