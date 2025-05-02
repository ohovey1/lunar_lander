import argparse
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import json

# Import agents
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent

# Available agent types
AGENT_TYPES = {
    'dqn': DQNAgent,
    'double_dqn': DoubleDQNAgent,
    'dueling_dqn': DuelingDQNAgent,
    'ppo': PPOAgent
}

# Default hyperparameters for each agent type
DEFAULT_HYPERPARAMS = {
    'dqn': {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_dim': 128
    },
    'double_dqn': {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_dim': 128
    },
    'dueling_dqn': {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update': 10,
        'hidden_dim': 128
    },
    'ppo': {
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'hidden_dim': 128,
        'update_epochs': 10,
        'batch_size': 64
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train reinforcement learning agents on Lunar Lander')
    
    parser.add_argument('--agent', type=str, choices=list(AGENT_TYPES.keys()), default='dqn',
                        help='Agent type to train')
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluate every n episodes')
    
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs')
    
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load model from')
    
    # Early stopping parameters
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    
    parser.add_argument('--target-reward', type=float, default=200.0,
                        help='Target reward threshold for early stopping')
    
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of evaluations with no improvement to wait before stopping')
    
    return parser.parse_args()

def create_agent(agent_type, state_dim, action_dim, hyperparams=None):
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        hyperparams: Optional hyperparameters for the agent
        
    Returns:
        Agent instance
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    AgentClass = AGENT_TYPES[agent_type]
    
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS[agent_type]
    
    return AgentClass(state_dim=state_dim, action_dim=action_dim, **hyperparams)

def train(agent, env, num_episodes, max_steps, eval_env=None, eval_episodes=10, eval_interval=100, 
          render=False, log_dir=None, model_dir=None, early_stopping=False, target_reward=200.0, patience=5):
    """
    Train an agent on the given environment.
    
    Args:
        agent: Agent to train
        env: Training environment
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        eval_env: Environment for evaluation
        eval_episodes: Number of episodes for evaluation
        eval_interval: Evaluate every n episodes
        render: Whether to render during evaluation
        log_dir: Directory to save logs
        model_dir: Directory to save models
        early_stopping: Whether to use early stopping
        target_reward: Target average reward for early stopping
        patience: Number of evaluations with no improvement to wait before stopping
        
    Returns:
        Dictionary of training metrics
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_timestamps = []
    
    # Early stopping variables
    best_eval_reward = float('-inf')
    evaluations_without_improvement = 0
    early_stop_triggered = False
    early_stop_episode = None
    
    # Training loop
    print(f"Starting training for {num_episodes} episodes...")
    if early_stopping:
        print(f"Early stopping enabled with target reward: {target_reward}, patience: {patience}")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Train agent
            metrics = agent.train(state, action, reward, next_state, done)
            if 'loss' in metrics:
                episode_loss += metrics['loss']
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # End of episode processing
        agent.episode_end(episode_reward)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = episode_loss / steps if steps > 0 else 0
            
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Avg Loss: {avg_loss:.4f}")
        
        # Evaluate agent
        if eval_env is not None and (episode + 1) % eval_interval == 0:
            print(f"\nEvaluating after {episode+1} episodes...")
            eval_reward = evaluate(agent, eval_env, eval_episodes, render)
            eval_rewards.append(eval_reward)
            eval_timestamps.append(episode + 1)
            print(f"Evaluation complete. Avg reward: {eval_reward:.2f}\n")
            
            # Save model after evaluation
            model_path = os.path.join(model_dir, f"{agent.__class__.__name__}_episode_{episode+1}.pth")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Check for early stopping
            if early_stopping:
                # Check if we've reached the target reward
                if eval_reward >= target_reward:
                    print(f"\n🎯 Target reward {target_reward} reached! Stopping training early.")
                    early_stop_triggered = True
                    early_stop_episode = episode + 1
                    break
                
                # Check for improvement
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    evaluations_without_improvement = 0
                    # Save best model
                    best_model_path = os.path.join(model_dir, f"{agent.__class__.__name__}_best.pth")
                    agent.save(best_model_path)
                    print(f"New best model saved to {best_model_path}")
                else:
                    evaluations_without_improvement += 1
                    print(f"No improvement for {evaluations_without_improvement} evaluations")
                    
                    # Check if we should stop due to no improvement
                    if evaluations_without_improvement >= patience:
                        print(f"\n⚠️ No improvement for {patience} evaluations. Stopping training early.")
                        early_stop_triggered = True
                        early_stop_episode = episode + 1
                        break
    
    # Final evaluation (unless we already stopped early)
    if not early_stop_triggered:
        print("\nTraining complete! Final evaluation:")
        final_eval_reward = evaluate(agent, eval_env, eval_episodes, render)
        eval_rewards.append(final_eval_reward)
        eval_timestamps.append(num_episodes)
        print(f"Final evaluation complete. Avg reward: {final_eval_reward:.2f}")
    
    # Save final model (or use best model if available)
    if early_stopping and os.path.exists(os.path.join(model_dir, f"{agent.__class__.__name__}_best.pth")):
        # Copy best model to final
        import shutil
        best_model_path = os.path.join(model_dir, f"{agent.__class__.__name__}_best.pth")
        final_model_path = os.path.join(model_dir, f"{agent.__class__.__name__}_final.pth")
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model copied to {final_model_path}")
    else:
        # Save current model as final
        final_model_path = os.path.join(model_dir, f"{agent.__class__.__name__}_final.pth")
        agent.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'eval_timestamps': eval_timestamps,
        'training_time': training_time,
        'agent_type': agent.__class__.__name__,
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'early_stopping': early_stopping,
        'early_stop_triggered': early_stop_triggered
    }
    
    if early_stop_triggered:
        metrics['early_stop_episode'] = early_stop_episode
        metrics['early_stop_reward'] = eval_rewards[-1]
        
    if early_stopping:
        metrics['target_reward'] = target_reward
        metrics['patience'] = patience
        metrics['best_eval_reward'] = best_eval_reward
    
    metrics_path = os.path.join(log_dir, f"{agent.__class__.__name__}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Plot training progress
    plot_training_progress(episode_rewards, eval_timestamps, eval_rewards, agent.__class__.__name__, log_dir, early_stop_triggered, early_stop_episode)
    
    return metrics

def evaluate(agent, env, num_episodes, render=False):
    """
    Evaluate the agent on the given environment.
    
    Args:
        agent: Agent to evaluate
        env: Environment for evaluation
        num_episodes: Number of episodes for evaluation
        render: Whether to render the environment
        
    Returns:
        Average reward over evaluation episodes
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluation=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / num_episodes
    
    return avg_reward

def plot_training_progress(episode_rewards, eval_timestamps, eval_rewards, agent_name, log_dir, early_stop_triggered=False, early_stop_episode=None):
    """
    Plot training progress and save the plot.
    
    Args:
        episode_rewards: List of episode rewards
        eval_timestamps: List of evaluation timestamps
        eval_rewards: List of evaluation rewards
        agent_name: Name of the agent
        log_dir: Directory to save the plot
        early_stop_triggered: Whether early stopping was triggered
        early_stop_episode: Episode at which early stopping was triggered
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training rewards
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.6)
    plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), label='Moving Average (10 episodes)')
    
    # Mark early stopping point if applicable
    if early_stop_triggered and early_stop_episode is not None:
        plt.axvline(x=early_stop_episode, color='r', linestyle='--', 
                   label=f'Early Stopping (Episode {early_stop_episode})')
    
    plt.title(f'{agent_name} Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot evaluation rewards
    plt.subplot(2, 1, 2)
    plt.plot(eval_timestamps, eval_rewards, 'o-', label='Evaluation Reward')
    
    # Mark early stopping point if applicable
    if early_stop_triggered and early_stop_episode is not None:
        plt.axvline(x=early_stop_episode, color='r', linestyle='--', 
                   label=f'Early Stopping (Episode {early_stop_episode})')
    
    plt.title(f'{agent_name} Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{agent_name}_training_progress.png"))
    
    # Also save a version focusing just on the evaluation
    plt.figure(figsize=(10, 6))
    plt.plot(eval_timestamps, eval_rewards, 'o-', label='Evaluation Reward')
    
    # Mark early stopping point if applicable
    if early_stop_triggered and early_stop_episode is not None:
        plt.axvline(x=early_stop_episode, color='r', linestyle='--', 
                   label=f'Early Stopping (Episode {early_stop_episode})')
    
    plt.title(f'{agent_name} Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"{agent_name}_evaluation.png"))

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environments
    env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3", render_mode="human" if args.render else None)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = create_agent(args.agent, state_dim, action_dim)
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        agent.load(args.load_model)
    
    # Train agent
    train(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_env=eval_env,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        render=args.render,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        early_stopping=args.early_stopping,
        target_reward=args.target_reward,
        patience=args.patience
    )
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 