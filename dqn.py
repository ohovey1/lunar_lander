import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer parameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        
        # Network parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target network update frequency
        self.target_update = target_update
        self.update_counter = 0
    
    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def train(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Start training when enough samples are available
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values for current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Q values for next states with target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update Q network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update counter and epsilon
        self.update_counter += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network if needed
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

def train_dqn(env, agent, num_episodes, max_steps=1000):
    """
    Train the DQN agent on the given environment.
    
    Args:
        env: Gymnasium environment
        agent: DQNAgent instance
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        
    Returns:
        List of rewards per episode
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Train agent
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return episode_rewards

def evaluate_dqn(env, agent, num_episodes=10, render=False):
    """
    Evaluate the trained DQN agent.
    
    Args:
        env: Gymnasium environment
        agent: Trained DQNAgent instance
        num_episodes: Number of episodes to evaluate
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
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    return avg_reward
