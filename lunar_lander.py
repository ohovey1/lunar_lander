import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn import DQNAgent

# Initialize the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to get initial state
observation, info = env.reset(seed=42)

# Define action space for reference
# 0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine
action_space = env.action_space.n
state_dim = env.observation_space.shape[0]  # 8 dimensions for Lunar Lander

# Load the trained model or create a new one
model = DQNAgent(state_dim=state_dim, action_dim=action_space)

# Loading pre-trained model
model.load("lunar_lander_dqn.pth")

# Main game loop
done = False
total_reward = 0

for _ in range(1000):
    # Use the DQN model to predict the next action
    action = model.select_action(observation, evaluation=True)
    
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Update total reward
    total_reward += reward

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset(seed=42)
        print(f"Episode finished with total reward: {total_reward}")
        total_reward = 0

# Close the environment
env.close()

print(f"Episode finished with total reward: {total_reward}")


