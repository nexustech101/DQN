import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
import random
from collections import deque

# Hyperparameters
LEARNING_RATE = 0.01
GAMMA = 0.90  # Discount factor for future rewards
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay factor for exploration rate
MEMORY_SIZE = 50000  # Replay buffer size
BATCH_SIZE = 64  # Minibatch size
TARGET_UPDATE = 10  # Update target network every N episodes

# Environment setup
env = gym.make('MountainCar-v0', render_mode="human")
state_size = env.observation_space.shape[0]  # 2 features: position & velocity
action_size = env.action_space.n  # 3 possible actions: left, no-op, right

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on last layer (raw Q-values)


# Replay Memory for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Main network & target network
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # Sync weights

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON

    def get_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                # Best Q-value action
                return torch.argmax(self.model(state)).item()

    def train(self):
        """Train the network using experience replay"""
        if self.memory.size() < BATCH_SIZE:
            return  # Don't train until enough samples are collected

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, done = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (GAMMA * max_next_q_values * (1 - done))

        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions)

        # Compute loss and update model
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Update target network with the main model's weights"""
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate over time"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# Train the DQN Agent
agent = DQNAgent(state_size, action_size)

try:
    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Adjust reward to encourage reaching the goal
            if next_state[0] >= 0.5:  # If car reaches the flag
                reward = 5
            elif next_state[0] > state[0]:  # If car is moving forward
                reward = 1
            else:  # If car moves backward
                reward = -1

            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        # Decay exploration rate
        agent.decay_epsilon()

        # Update target network every `TARGET_UPDATE` episodes
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    # Save the model after training
    torch.save(agent.model.state_dict(), "model/dqn_model.pth")
    print("Model saved.")
    env.close()