import gymnasium as gym
import numpy as np


# Create the CartPole environment
env = gym.make("CartPole-v1")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

env.close()