import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to get initial state
observation, info = env.reset()

# Define action space for reference
# 0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine
action_space = env.action_space.n

# Main game loop
done = False
total_reward = 0

# TODO: Load the trained model here
# model = torch.load("path_to_model.pth")

while not done:
    # TODO: Replace this random action with model prediction
    # action = model.predict(observation)
    action = env.action_space.sample()  # Currently taking random actions
    
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Update total reward
    total_reward += reward
    
    # Check if episode is done
    done = terminated or truncated

# Close the environment
env.close()

print(f"Episode finished with total reward: {total_reward}")


