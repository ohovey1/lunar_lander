import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from agents.networks import ActorCriticNetwork

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    This is an actor-critic algorithm that uses a clipped surrogate objective
    to update the policy network.
    """
    
    def __init__(self, state_dim, action_dim,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 hidden_dim=64,
                 update_epochs=10,
                 batch_size=64,
                 device='auto'):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            actor_lr: Learning rate for the actor (policy) network
            critic_lr: Learning rate for the critic (value) network
            gamma: Discount factor
            gae_lambda: Lambda parameter for Generalized Advantage Estimation
            clip_ratio: Clip parameter for the surrogate objective
            value_coef: Coefficient for the value loss
            entropy_coef: Coefficient for the entropy bonus
            max_grad_norm: Maximum norm for gradient clipping
            hidden_dim: Dimension of the hidden layers
            update_epochs: Number of epochs to update the policy for each batch
            batch_size: Batch size for training
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(state_dim, action_dim, device)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Actor-critic network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': actor_lr},
            {'params': self.network.critic.parameters(), 'lr': critic_lr}
        ])
        
        # Buffer for storing trajectories
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []
        
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            evaluation: Whether we're in evaluation mode
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.network(state)
            
            if evaluation:
                # Use greedy action selection during evaluation
                action = torch.argmax(action_logits, dim=1).item()
                return action
            
            # Sample action from the distribution
            probs = torch.softmax(action_logits, dim=1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Store the values for training
            self.rollout_states.append(state)
            self.rollout_actions.append(action)
            self.rollout_log_probs.append(log_prob)
            self.rollout_values.append(value)
            
            return action.item()
    
    def train(self, state, action, reward, next_state, done):
        """
        Store transition for later training.
        
        Unlike DQN, PPO doesn't train on individual transitions but on entire trajectories.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Store the reward and done flag
        self.rollout_rewards.append(reward)
        self.rollout_dones.append(done)
        
        # Return empty metrics since we're not actually training yet
        return {"loss": 0.0}
    
    def episode_end(self, total_reward):
        """
        Process the episode and train the agent.
        
        This is called at the end of an episode to compute advantages and
        update the policy and value networks.
        
        Args:
            total_reward: Total reward obtained in the episode
        """
        super().episode_end(total_reward)
        
        # Skip if we don't have enough data
        if len(self.rollout_rewards) < 2:
            self._clear_rollout()
            return
        
        # Compute returns and advantages
        returns, advantages = self._compute_advantages()
        
        # Convert lists to tensors
        states = torch.cat(self.rollout_states).detach()
        actions = torch.cat(self.rollout_actions).detach()
        old_log_probs = torch.cat(self.rollout_log_probs).detach()
        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.update_epochs):
            # Create random batches
            indices = torch.randperm(states.size(0))
            for i in range(0, states.size(0), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_logits, values = self.network(batch_states)
                
                # Calculate action probabilities and log probabilities
                probs = torch.softmax(action_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio and surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear rollout buffer
        self._clear_rollout()
        
        self.train_steps += 1
    
    def _compute_advantages(self):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            Tuple of (returns, advantages)
        """
        # Convert lists to tensors
        values = torch.cat(self.rollout_values).detach()
        rewards = torch.FloatTensor(self.rollout_rewards).to(self.device)
        dones = torch.FloatTensor(self.rollout_dones).to(self.device)
        
        # Compute the next value for the last state
        if len(self.rollout_states) > len(self.rollout_rewards):
            last_state = self.rollout_states[-1]
            with torch.no_grad():
                _, next_value = self.network(last_state)
                values = torch.cat([values, next_value])
        else:
            # If we don't have the next state, just assume value 0
            values = torch.cat([values, torch.zeros(1, 1).to(self.device)])
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[i]
            delta = rewards[i] + self.gamma * values[i+1] * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Insert in front (since we're going backwards)
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def _clear_rollout(self):
        """
        Clear the rollout buffer.
        """
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []
    
    def save(self, path):
        """
        Save the agent's model and parameters.
        
        Args:
            path: Path to save the agent to
        """
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = checkpoint.get('train_steps', 0)
        self.episodes_seen = checkpoint.get('episodes_seen', 0) 