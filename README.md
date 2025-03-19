# Lunar Lander - A Reinforcement Approach

## Overview:

This project is a reinforcement learning approach to the Lunar Lander environment from OpenAI Gym. The goal of this project is to train an agent to land a lunar lander on the moon -- simulating real world autonomous vehicle control. We compare the performance of various reinforcement learning algorithms in this project and
consider the tradeoffs between them.

## Environment:

The Lunar Lander environment is a continuous action space environment. The state space is 8-dimensional and the action space is 4-dimensional. The state and action spaces are defined below:

State Space:
1. x position (horizontal position of the lander)
2. y position (vertical position of the lander)
3. x velocity (horizontal velocity of the lander)
4. y velocity (vertical velocity of the lander)
5. angle of the lander (orientation of the lander)
6. angular velocity of the lander
7. left leg contact (boolean: 1 if in contact w/ the ground, 0 otherwise)
8. right leg contact (boolean: 1 if in contact w/ the ground, 0 otherwise)

Action Space:
1. fire left engine (boolean: 1 if firing, 0 otherwise)
2. fire main engine (boolean: 1 if firing, 0 otherwise)
3. fire right engine (boolean: 1 if firing, 0 otherwise)
4. do nothing (boolean: 1 if doing nothing, 0 otherwise)

The agent receives a reward for each time step, and the goal is to maximize the reward. The agent receives a reward of +100 for landing successfully, a reward of -100 for crashing, and a smaller reward/penalty for moving toward/away from the landing pad. Each time step consumes fuel, which also incurs a small penalty. 

The episode ends when the lander either crashes, successfully lands, flies out of bounds, or the maximum number of time steps is reached (fuel runs out).

## Models Trained:

Using PyTorch, we have trained and compared the performance of the following models:

*TODO: Add explanations of each model.*

1. **Deep Q-Network (DQN) - Baseline Model**: A value-based method that uses a neural network to approximate the Q-function. DQN is sample-efficient but may struggle with the continuous action space of Lunar Lander.

2. **Proximal Policy Optimization (PPO)**:

3. **Deep Deterministic Policy Gradient (DDPG)**: 

4. **Soft Actor-Critic (SAC)**: 

5. **Advantage Actor-Critic (A2C)**: 

### Performance Comparison:

*TODO: The "X" values will be filled in after training and evaluation.*

| Algorithm | Average Reward | Training Time | Sample Efficiency | Stability |
|-----------|----------------|---------------|-------------------|-----------|
| DQN       | X              | X             | Medium            | Medium    |
| PPO       | X              | X             | Medium            | High      |
| DDPG      | X              | X             | High              | Low       |
| SAC       | X              | X             | High              | High      |
| A2C       | X              | X             | Low               | Medium    |


### Tradeoffs:

*TODO: Fill out this section after training and evaluation.*

- **DQN**: 
- **PPO**: 
- **DDPG**: 
- **SAC**: 
- **A2C**: 


## Installation

```bash
pip install requirements.txt
```

## Usage

```bash
python lunar_lander.py
```

*TODO: Finish filling out README. More sections to come.*