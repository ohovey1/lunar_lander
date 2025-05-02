import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    
    This class defines the common interface that all agents must implement.
    """
    
    def __init__(self, state_dim, action_dim, device='auto'):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Determine device (CPU or GPU)
        if device == 'auto':
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize training metrics
        self.train_steps = 0
        self.episodes_seen = 0
        
    @abstractmethod
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            evaluation: Whether we're in evaluation mode (typically less exploration)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the agent's model and parameters.
        
        Args:
            path: Path to save the agent to
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the agent's model and parameters.
        
        Args:
            path: Path to load the agent from
        """
        pass
    
    def episode_end(self, total_reward):
        """
        Called at the end of an episode.
        
        Args:
            total_reward: Total reward obtained in the episode
        """
        self.episodes_seen += 1
        
    def preprocess_state(self, state):
        """
        Preprocess the state before using it for action selection or training.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            Preprocessed state
        """
        return state 