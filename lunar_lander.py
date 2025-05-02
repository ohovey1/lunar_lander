import gymnasium as gym
import numpy as np
import torch
import argparse
import re
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent

# Available agent types
AGENT_TYPES = {
    'dqn': DQNAgent,
    'double_dqn': DoubleDQNAgent,
    'dueling_dqn': DuelingDQNAgent,
    'ppo': PPOAgent
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run a trained agent on Lunar Lander')
    parser.add_argument('--agent', type=str, choices=list(AGENT_TYPES.keys()), default='dqn',
                      help='Type of agent to run')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to the model file to load (e.g., models/DQNAgent_final.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--hidden-dim', type=int, default=None,
                      help='Hidden dimension for neural networks (defaults to auto-detect or 64)')
    return parser.parse_args()

def detect_hidden_dim(error_msg):
    """Extract hidden dimension from size mismatch error message."""
    # Look for patterns like [128, 8] or [64, 8] in the error message
    matches = re.findall(r'\[(\d+),', error_msg)
    if matches and len(matches) > 0:
        return int(matches[0])
    return None

def create_agent_with_matching_architecture(agent_class, state_dim, action_dim, model_path, hidden_dim=None):
    """Create an agent with architecture matching the saved model."""
    if hidden_dim is not None:
        # If hidden_dim is explicitly provided, use it
        print(f"Creating agent with specified hidden_dim={hidden_dim}")
        return agent_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    
    # First try with default hidden_dim
    default_hidden_dim = 64
    agent = agent_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=default_hidden_dim)
    
    try:
        # Try to load the model with default architecture
        agent.load(model_path)
        print(f"Successfully loaded model with default hidden_dim={default_hidden_dim}")
        return agent
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            # If there's a size mismatch, try to detect the correct hidden_dim
            detected_dim = detect_hidden_dim(error_msg)
            if detected_dim and detected_dim != default_hidden_dim:
                print(f"Detected model trained with hidden_dim={detected_dim}, recreating agent with matching architecture")
                new_agent = agent_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=detected_dim)
                new_agent.load(model_path)
                return new_agent
        
        # If we couldn't handle the error, re-raise it
        raise

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize the Lunar Lander environment with rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create the agent
    AgentClass = AGENT_TYPES[args.agent]
    
    if args.model:
        print(f"Loading model from {args.model}")
        try:
            # Create agent with matching architecture and load the model
            agent = create_agent_with_matching_architecture(
                AgentClass, state_dim, action_dim, args.model, args.hidden_dim
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained agent with default parameters instead.")
            agent = AgentClass(state_dim=state_dim, action_dim=action_dim)
    else:
        print("No model specified. Using untrained agent.")
        agent = AgentClass(state_dim=state_dim, action_dim=action_dim)
    
    # Run the specified number of episodes
    total_reward = 0
    for episode in range(args.episodes):
        # Reset the environment to get initial state
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{args.episodes}")
        print("-" * 30)
        
        # Main game loop
        while not done:
            # Use the agent to select an action
            action = agent.select_action(state, evaluation=True)
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            # Print some info about the current step
            if step % 20 == 0 or done:
                status = "Landed!" if terminated and reward > 0 else "Crashed!" if terminated else "In progress"
                print(f"Step: {step}, Reward: {episode_reward:.2f}, Status: {status}")
        
        # Print episode summary
        print(f"Episode finished with total reward: {episode_reward:.2f}")
        total_reward += episode_reward
    
    # Print overall performance
    avg_reward = total_reward / args.episodes
    print(f"\nAverage reward over {args.episodes} episodes: {avg_reward:.2f}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()


