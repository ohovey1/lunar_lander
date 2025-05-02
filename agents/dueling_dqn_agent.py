import torch
import torch.nn as nn
import numpy as np
from agents.dqn_agent import DQNAgent
from agents.networks import DuelingQNetwork

class DuelingDQNAgent(DQNAgent):
    """
    Dueling Deep Q-Network agent implementation.
    
    This implementation uses a dueling network architecture that separates
    the representation of state value and action advantages.
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
        Initialize the Dueling DQN agent.
        
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
        # Initialize base class without creating networks
        super(DQNAgent, self).__init__(state_dim, action_dim, device)
        
        # Agent parameters
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy exploration (reuse from DQNAgent)
        self.exploration = self._create_exploration(
            epsilon_start, epsilon_end, epsilon_decay
        )
        
        # Replay buffer (reuse from DQNAgent)
        self.memory = self._create_replay_buffer(buffer_size)
        
        # Networks - use dueling architecture
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target network update frequency
        self.target_update = target_update
        self.update_counter = 0
    
    def _create_exploration(self, epsilon_start, epsilon_end, epsilon_decay):
        """
        Create the exploration strategy.
        
        This is factored out to enable easier subclassing.
        
        Args:
            epsilon_start: Initial epsilon value
            epsilon_end: Final epsilon value
            epsilon_decay: Decay rate for epsilon
            
        Returns:
            Exploration strategy object
        """
        from agents.utils import EpsilonGreedy
        return EpsilonGreedy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )
    
    def _create_replay_buffer(self, buffer_size):
        """
        Create the replay buffer.
        
        This is factored out to enable easier subclassing.
        
        Args:
            buffer_size: Maximum size of the replay buffer
            
        Returns:
            Replay buffer object
        """
        from agents.utils import ReplayBuffer
        return ReplayBuffer(buffer_size) 