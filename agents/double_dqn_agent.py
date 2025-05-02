import torch
import torch.nn as nn
import numpy as np
from agents.dqn_agent import DQNAgent

class DoubleDQNAgent(DQNAgent):
    """
    Double Deep Q-Network (DDQN) agent implementation.
    
    This implementation addresses the overestimation bias of DQN by using
    the online network to select actions and the target network to evaluate them.
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
        Initialize the Double DQN agent.
        
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
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update=target_update,
            hidden_dim=hidden_dim,
            device=device
        )
    
    def train(self, state, action, reward, next_state, done):
        """
        Train the agent on a single transition.
        
        The key difference from DQN is how the target Q-values are computed.
        In Double DQN, we use the online network to select actions and the
        target network to evaluate them.
        
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
            # Use online network to select actions
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            
            # Compute target Q-values
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