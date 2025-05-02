import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize and compare RL agent performance')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory containing log files')
    
    parser.add_argument('--agents', nargs='+', default=None,
                        help='Specific agents to compare (e.g., DQNAgent DoubleDQNAgent)')
    
    parser.add_argument('--save-dir', type=str, default='plots',
                        help='Directory to save output plots')
    
    parser.add_argument('--smoothing', type=int, default=10,
                        help='Window size for moving average smoothing')
    
    return parser.parse_args()

def load_metrics(log_dir, agent_filter=None):
    """
    Load metrics from JSON files in the log directory.
    
    Args:
        log_dir: Directory containing log files
        agent_filter: Optional list of agent names to filter by
        
    Returns:
        Dictionary of agent metrics
    """
    metrics = {}
    
    for filename in os.listdir(log_dir):
        if filename.endswith('_metrics.json'):
            filepath = os.path.join(log_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            agent_name = data.get('agent_type', filename.split('_metrics.json')[0])
            
            if agent_filter is None or agent_name in agent_filter:
                metrics[agent_name] = data
    
    return metrics

def moving_average(data, window_size):
    """
    Calculate the moving average of a data series.
    
    Args:
        data: Data series
        window_size: Window size for the moving average
        
    Returns:
        Moving average of the data
    """
    if window_size <= 1:
        return data
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_curves(metrics, save_dir, smoothing=10):
    """
    Plot training curves for each agent.
    
    Args:
        metrics: Dictionary of agent metrics
        save_dir: Directory to save plots
        smoothing: Window size for moving average smoothing
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for training rewards
    plt.figure(figsize=(12, 8))
    
    for agent_name, data in metrics.items():
        rewards = data.get('episode_rewards', [])
        
        if len(rewards) > smoothing:
            # Plot raw rewards with low alpha
            episodes = range(1, len(rewards) + 1)
            plt.plot(episodes, rewards, alpha=0.2)
            
            # Plot smoothed rewards
            smoothed_rewards = moving_average(rewards, smoothing)
            smoothed_episodes = range(smoothing, len(rewards) + 1)
            plt.plot(smoothed_episodes, smoothed_rewards, linewidth=2, label=agent_name)
            
            # Mark early stopping point if applicable
            if data.get('early_stop_triggered', False) and 'early_stop_episode' in data:
                es_episode = data['early_stop_episode']
                plt.axvline(x=es_episode, color='r', linestyle='--', alpha=0.5)
                plt.text(es_episode, plt.ylim()[1]*0.9, f"{agent_name} Early Stop",
                        rotation=90, verticalalignment='top')
        else:
            # If we have too few episodes, just plot the raw rewards
            episodes = range(1, len(rewards) + 1)
            plt.plot(episodes, rewards, linewidth=2, label=agent_name)
    
    plt.title('Training Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_rewards_comparison.png'))
    
    # Create figure for evaluation rewards
    plt.figure(figsize=(12, 8))
    
    for agent_name, data in metrics.items():
        eval_rewards = data.get('eval_rewards', [])
        eval_timestamps = data.get('eval_timestamps', range(1, len(eval_rewards) + 1))
        
        if eval_rewards:
            plt.plot(eval_timestamps, eval_rewards, 'o-', linewidth=2, label=agent_name)
            
            # Mark early stopping point if applicable
            if data.get('early_stop_triggered', False) and 'early_stop_episode' in data:
                es_episode = data['early_stop_episode']
                es_reward = data.get('early_stop_reward', eval_rewards[-1])
                plt.plot(es_episode, es_reward, 'r*', markersize=15)
                plt.annotate(f"{agent_name} Early Stop",
                           xy=(es_episode, es_reward),
                           xytext=(es_episode, es_reward*0.9),
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                           horizontalalignment='center')
    
    plt.title('Evaluation Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Evaluation Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_rewards_comparison.png'))

def plot_comparison_bar_chart(metrics, save_dir):
    """
    Plot a bar chart comparing performance metrics.
    
    Args:
        metrics: Dictionary of agent metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    agent_names = list(metrics.keys())
    max_eval_rewards = []
    final_eval_rewards = []
    training_times = []
    episodes_to_target = []
    has_early_stopping_data = any('early_stop_episode' in data for data in metrics.values())
    
    for agent_name, data in metrics.items():
        eval_rewards = data.get('eval_rewards', [0])
        
        max_eval_rewards.append(max(eval_rewards) if eval_rewards else 0)
        final_eval_rewards.append(eval_rewards[-1] if eval_rewards else 0)
        training_times.append(data.get('training_time', 0))
        
        # Calculate episodes to reach target if early stopping was used
        if data.get('early_stop_triggered', False) and 'early_stop_episode' in data:
            episodes_to_target.append(data['early_stop_episode'])
        else:
            # Use max episodes if no early stopping or target not reached
            episodes_to_target.append(data.get('num_episodes', 0))
    
    # Bar chart for maximum evaluation rewards
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agent_names, max_eval_rewards)
    
    # Add values above bars
    for bar, value in zip(bars, max_eval_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.1f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.title('Maximum Evaluation Reward')
    plt.xlabel('Agent')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'max_reward_comparison.png'))
    
    # Bar chart for final evaluation rewards
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agent_names, final_eval_rewards)
    
    # Add values above bars
    for bar, value in zip(bars, final_eval_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.1f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.title('Final Evaluation Reward')
    plt.xlabel('Agent')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_reward_comparison.png'))
    
    # Bar chart for training time
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agent_names, training_times)
    
    # Add values above bars
    for bar, value in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.1f}s', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.title('Training Time Comparison')
    plt.xlabel('Agent')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'))
    
    # If early stopping data is available, plot episodes to target
    if has_early_stopping_data:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(agent_names, episodes_to_target)
        
        # Add values above bars
        for bar, value in zip(bars, episodes_to_target):
            plt.text(bar.get_x() + bar.get_width()/2, value + 5, f'{int(value)}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Episodes to Reach Target Reward')
        plt.xlabel('Agent')
        plt.ylabel('Number of Episodes')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'episodes_to_target_comparison.png'))

def generate_metrics_table(metrics, save_dir):
    """
    Generate a markdown table with performance metrics.
    
    Args:
        metrics: Dictionary of agent metrics
        save_dir: Directory to save the table
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare table data
    table_data = []
    for agent_name, data in metrics.items():
        eval_rewards = data.get('eval_rewards', [0])
        max_reward = max(eval_rewards) if eval_rewards else 0
        final_reward = eval_rewards[-1] if eval_rewards else 0
        training_time = data.get('training_time', 0)
        
        episodes = data.get('num_episodes', 0)
        early_stopped = data.get('early_stop_triggered', False)
        early_stop_episode = data.get('early_stop_episode', episodes if early_stopped else None)
        
        # Calculate sample efficiency (final reward / episodes trained)
        episodes_trained = early_stop_episode if early_stopped else episodes
        sample_efficiency = final_reward / episodes_trained if episodes_trained > 0 else 0
        
        # Get the first episode where the agent achieves 90% of max reward
        success_threshold = 0.9 * max_reward
        if max_reward > 0 and 'eval_rewards' in data and 'eval_timestamps' in data:
            for reward, episode in zip(data['eval_rewards'], data['eval_timestamps']):
                if reward >= success_threshold:
                    convergence_episode = episode
                    break
            else:
                convergence_episode = episodes  # Never reached threshold
        else:
            convergence_episode = episodes
        
        # Add row to table data
        table_data.append({
            'Agent': agent_name,
            'Max Reward': max_reward,
            'Final Reward': final_reward,
            'Training Time (s)': training_time,
            'Episodes to 90% Max': convergence_episode,
            'Early Stopped': "Yes" if early_stopped else "No",
            'Episodes Trained': episodes_trained,
            'Sample Efficiency': sample_efficiency
        })
    
    # Generate markdown table
    with open(os.path.join(save_dir, 'metrics_table.md'), 'w') as f:
        # Write table header
        f.write('| Agent | Max Reward | Final Reward | Training Time (s) | Episodes to 90% Max | Early Stopped | Episodes Trained | Sample Efficiency |\n')
        f.write('|-------|------------|--------------|-------------------|---------------------|---------------|------------------|-------------------|\n')
        
        # Write table rows
        for row in table_data:
            f.write(f"| {row['Agent']} | {row['Max Reward']:.2f} | {row['Final Reward']:.2f} | "
                    f"{row['Training Time (s)']:2f} | {row['Episodes to 90% Max']} | {row['Early Stopped']} | "
                    f"{row['Episodes Trained']} | {row['Sample Efficiency']:.4f} |\n")
    
    # Also generate CSV for easier analysis
    with open(os.path.join(save_dir, 'metrics_table.csv'), 'w') as f:
        # Write CSV header
        f.write('Agent,Max Reward,Final Reward,Training Time (s),Episodes to 90% Max,Early Stopped,Episodes Trained,Sample Efficiency\n')
        
        # Write CSV rows
        for row in table_data:
            f.write(f"{row['Agent']},{row['Max Reward']:.2f},{row['Final Reward']:.2f},"
                    f"{row['Training Time (s)']:.2f},{row['Episodes to 90% Max']},{row['Early Stopped']},"
                    f"{row['Episodes Trained']},{row['Sample Efficiency']:.4f}\n")
    
    # Generate a specific early stopping metrics table if any agent used early stopping
    if any(data.get('early_stop_triggered', False) for data in metrics.values()):
        with open(os.path.join(save_dir, 'early_stopping_metrics.md'), 'w') as f:
            # Write table header
            f.write('| Agent | Target Reward | Reward at Stop | Episodes to Target | % of Max Episodes | Time to Target (s) |\n')
            f.write('|-------|---------------|----------------|--------------------|--------------------|--------------------|\n')
            
            # Write table rows
            for agent_name, data in metrics.items():
                if data.get('early_stop_triggered', False):
                    target_reward = data.get('target_reward', 'N/A')
                    early_stop_reward = data.get('early_stop_reward', 'N/A')
                    early_stop_episode = data.get('early_stop_episode', 'N/A')
                    max_episodes = data.get('num_episodes', 1)
                    percent_of_max = (early_stop_episode / max_episodes) * 100 if early_stop_episode != 'N/A' and max_episodes > 0 else 'N/A'
                    
                    # Estimate time to target based on training time and episode ratio
                    training_time = data.get('training_time', 0)
                    time_to_target = (early_stop_episode / max_episodes) * training_time if early_stop_episode != 'N/A' and max_episodes > 0 else 'N/A'
                    
                    f.write(f"| {agent_name} | {target_reward} | {early_stop_reward} | {early_stop_episode} | "
                           f"{percent_of_max:.1f}% | {time_to_target:.2f} |\n")

def main():
    """Main function."""
    args = parse_args()
    
    # Load metrics from log files
    metrics = load_metrics(args.log_dir, args.agents)
    
    if not metrics:
        print(f"No metrics found in {args.log_dir}")
        return
    
    print(f"Found metrics for {len(metrics)} agents: {', '.join(metrics.keys())}")
    
    # Plot training curves
    plot_training_curves(metrics, args.save_dir, args.smoothing)
    
    # Plot comparison bar charts
    plot_comparison_bar_chart(metrics, args.save_dir)
    
    # Generate metrics table
    generate_metrics_table(metrics, args.save_dir)
    
    print(f"Visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main() 