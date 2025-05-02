import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import init_weights

class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.layers(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling network architecture for Q-value approximation.
    
    This network separates the representation of state value and action advantages.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize the dueling Q-network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(DuelingQNetwork, self).__init__()
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class PolicyNetwork(nn.Module):
    """
    Neural network for policy approximation in policy gradient methods.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Logits for each action
        """
        return self.layers(x)
    
    def get_action_probs(self, x):
        """
        Get action probabilities using softmax.
        
        Args:
            x: Input state tensor
            
        Returns:
            Action probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_action(self, x):
        """
        Sample an action from the policy.
        
        Args:
            x: Input state tensor
            
        Returns:
            Sampled action and its log probability
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob


class ActorCriticNetwork(nn.Module):
    """
    Network that outputs both a policy (actor) and value function (critic).
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        features = self.feature_layer(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def get_action(self, x):
        """
        Sample an action from the policy and get its value.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, state_value)
        """
        features = self.feature_layer(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        # Calculate action probabilities and sample
        probs = F.softmax(action_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, state_value 