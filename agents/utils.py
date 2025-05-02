import numpy as np
import torch
import random
from collections import deque, namedtuple

class ReplayBuffer:
    """
    Replay buffer for experience replay in off-policy algorithms.
    """
    
    def __init__(self, capacity):
        """
        Initialize a replay buffer with the given capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                                    ('state', 'action', 'reward', 'next_state', 'done'))
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        transition = self.Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as numpy arrays
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = self.Transition(*zip(*transitions))
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        dones = np.array(batch.done)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            Current number of transitions in the buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer for prioritized experience replay.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        """
        Initialize a prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 - no prioritization, 1 - full prioritization)
            beta_start: Initial value of beta for importance sampling
            beta_frames: Number of frames over which to anneal beta to 1
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to ensure non-zero priority
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        super().add(state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions, importance sampling weights, and sample indices
        """
        self.beta = min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)
        self.frame += 1
        
        if len(self.buffer) == len(self.priorities):
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            samples = [self.buffer[idx] for idx in indices]
            
            # Importance sampling weights
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()
            
            batch = self.Transition(*zip(*samples))
            states = np.array(batch.state)
            actions = np.array(batch.action)
            rewards = np.array(batch.reward)
            next_states = np.array(batch.next_state)
            dones = np.array(batch.done)
            
            return (states, actions, rewards, next_states, dones), weights, indices
        else:
            # Fall back to uniform sampling if buffer and priorities are out of sync
            states, actions, rewards, next_states, dones = super().sample(batch_size)
            weights = np.ones(batch_size)
            indices = np.random.choice(len(self.buffer), batch_size)
            return (states, actions, rewards, next_states, dones), weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for the given transitions.
        
        Args:
            indices: Indices of the transitions
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + self.epsilon


# Network initialization utilities
def init_weights(m):
    """
    Initialize network weights using Xavier initialization.
    
    Args:
        m: PyTorch module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


# Exploration Strategies
class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy.
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            epsilon_start: Initial epsilon value
            epsilon_end: Final epsilon value
            epsilon_decay: Decay rate for epsilon
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, q_values, action_dim):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            q_values: Q-values from the model
            action_dim: Dimension of the action space
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(action_dim)
        else:
            return q_values.argmax().item()
    
    def update(self):
        """
        Update epsilon value.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_epsilon(self):
        """
        Get current epsilon value.
        
        Returns:
            Current epsilon value
        """
        return self.epsilon 