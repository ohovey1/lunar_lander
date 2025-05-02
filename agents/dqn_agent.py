import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from agents.utils import ReplayBuffer, EpsilonGreedy
from agents.networks import QNetwork

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent implementation.
    """
    
    def __init__(self, state_dim, action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 target_update=10,
                 hidden_dim=64,
                 device='auto'):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon value for exploration
            epsilon_end: Final epsilon value for exploration
            epsilon_decay: Decay rate for epsilon
            buffer_size: Maximum size of the replay buffer
            batch_size: Batch size for training
            target_update: Number of steps between target network updates
            hidden_dim: Dimension of the hidden layers in the Q-network
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(state_dim, action_dim, device)
        
        # Agent parameters
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy exploration
        self.exploration = EpsilonGreedy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target network update frequency
        self.target_update = target_update
        self.update_counter = 0
        
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            evaluation: Whether we're in evaluation mode (no exploration)
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Use greedy policy if evaluating
        if evaluation:
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()
        
        # Otherwise use epsilon-greedy exploration
        with torch.no_grad():
            q_values = self.q_network(state)
        return self.exploration.select_action(q_values, self.action_dim)
    
    def train(self, state, action, reward, next_state, done):
        """
        Train the agent on a single transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Update training steps
        self.train_steps += 1
        
        # Start training when enough samples are available
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.exploration.get_epsilon()}
        
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
        
        # Update exploration rate
        self.exploration.update()
        
        # Update target network if needed
        if self.train_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            "loss": loss.item(),
            "epsilon": self.exploration.get_epsilon()
        }
    
    def save(self, path):
        """
        Save the agent's model and parameters.
        
        Args:
            path: Path to save the agent to
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration': self.exploration.get_epsilon(),
            'train_steps': self.train_steps,
            'episodes_seen': self.episodes_seen
        }, path)
    
    def load(self, path):
        """
        Load the agent's model and parameters.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load exploration rate
        epsilon = checkpoint.get('exploration', self.exploration.get_epsilon())
        self.exploration = EpsilonGreedy(
            epsilon_start=epsilon,
            epsilon_end=self.exploration.epsilon_end,
            epsilon_decay=self.exploration.epsilon_decay
        )
        
        # Load training progress
        self.train_steps = checkpoint.get('train_steps', 0)
        self.episodes_seen = checkpoint.get('episodes_seen', 0) 