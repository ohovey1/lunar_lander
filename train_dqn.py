import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from dqn import DQNAgent, train_dqn, evaluate_dqn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 1000
EVAL_FREQUENCY = 100  # Evaluate every 100 episodes
SAVE_FREQUENCY = 100  # Save model every 100 episodes
MODEL_PATH = "lunar_lander_dqn.pth"

# Initialize the Lunar Lander environment
env = gym.make("LunarLander-v3")
eval_env = gym.make("LunarLander-v3", render_mode="human")  # For rendering during evaluation

# Get environment dimensions
state_dim = env.observation_space.shape[0]  # 8 dimensions for Lunar Lander
action_dim = env.action_space.n  # 4 actions

# Create DQN agent
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update=10
)

# Training loop
print("Starting training...")
start_time = time.time()

all_rewards = []
for episode in range(NUM_EPISODES):
    # Reset environment
    state, _ = env.reset(seed=episode)  # Different seed for each episode
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        # Select action
        action = agent.select_action(state)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Train agent
        agent.train(state, action, reward, next_state, done)
        
        # Update state and episode reward
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # Record episode reward
    all_rewards.append(episode_reward)
    
    # Print progress
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(all_rewards[-10:])
        print(f"Episode {episode+1}/{NUM_EPISODES}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Evaluate agent
    if (episode + 1) % EVAL_FREQUENCY == 0:
        print(f"\nEvaluating after {episode+1} episodes...")
        evaluate_dqn(eval_env, agent, num_episodes=3, render=True)
        print("Evaluation complete.\n")
    
    # Save model
    if (episode + 1) % SAVE_FREQUENCY == 0:
        agent.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

# Final evaluation
print("\nTraining complete! Final evaluation:")
evaluate_dqn(eval_env, agent, num_episodes=5, render=True)

# Save final model
agent.save(MODEL_PATH)
print(f"Final model saved to {MODEL_PATH}")

# Close environments
env.close()
eval_env.close()

# Calculate training time
training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f} seconds")

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(all_rewards)
plt.title("DQN Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("dqn_training_rewards.png")
plt.show() 