#!/usr/bin/env python3
"""Train RL agents to play Minigrid environments"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle
from datetime import datetime
import os

# ============ Simple DQN Agent ============
class DQN(nn.Module):
    """Simple DQN network for Minigrid"""
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255.0  # Normalize
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    """DQN Agent for Minigrid"""
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon=1.0):
        self.env = env
        self.n_actions = env.action_space.n
        
        # Get observation shape
        obs, _ = env.reset()
        self.obs_shape = obs['image'].shape if isinstance(obs, dict) else obs.shape
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.obs_shape, self.n_actions).to(self.device)
        self.target_network = DQN(self.obs_shape, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

# ============ Simple A2C Agent ============
class ActorCritic(nn.Module):
    """Actor-Critic network for A2C"""
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size
        def conv_out_size(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        h = conv_out_size(conv_out_size(input_shape[1]))
        w = conv_out_size(conv_out_size(input_shape[2]))
        self.flat_size = 64 * h * w
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class A2CAgent:
    """A2C Agent for Minigrid"""
    def __init__(self, env, learning_rate=1e-4, gamma=0.99):
        self.env = env
        self.n_actions = env.action_space.n
        
        obs, _ = env.reset()
        self.obs_shape = obs['image'].shape if isinstance(obs, dict) else obs.shape
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(self.obs_shape, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def act(self, state, training=True):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
        
        if training:
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            return action.item()
        else:
            return torch.argmax(policy, dim=1).item()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Update actor and critic"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current values and policies
        policies, values = self.network(states)
        _, next_values = self.network(next_states)
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = returns - values.squeeze()
        
        # Calculate losses
        action_probs = policies.gather(1, actions.unsqueeze(1)).squeeze()
        actor_loss = -(torch.log(action_probs + 1e-8) * advantages.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns.detach())
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")

# ============ Training Functions ============
def train_dqn(env_name="MiniGrid-Empty-8x8-v0", episodes=500):
    """Train DQN agent"""
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    agent = DQNAgent(env)
    
    scores = deque(maxlen=100)
    best_score = -float('inf')
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = state['image'] if isinstance(state, dict) else state
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state['image'] if isinstance(next_state, dict) else next_state
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
            
            if done:
                break
        
        scores.append(total_reward)
        avg_score = np.mean(scores)
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Save best model
        if avg_score > best_score and episode > 100:
            best_score = avg_score
            agent.save(f"models/dqn_{env_name}_best.pt")
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Avg: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return agent

def train_a2c(env_name="MiniGrid-Empty-8x8-v0", episodes=500):
    """Train A2C agent"""
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    agent = A2CAgent(env)
    
    scores = deque(maxlen=100)
    best_score = -float('inf')
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = state['image'] if isinstance(state, dict) else state
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state['image'] if isinstance(next_state, dict) else next_state
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Train on episode
        if len(states) > 0:
            loss = agent.train_step(states, actions, rewards, next_states, dones)
        
        scores.append(total_reward)
        avg_score = np.mean(scores)
        
        # Save best model
        if avg_score > best_score and episode > 100:
            best_score = avg_score
            agent.save(f"models/a2c_{env_name}_best.pt")
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Avg: {avg_score:.2f}")
    
    env.close()
    return agent

def test_agent(agent, env_name="MiniGrid-Empty-8x8-v0", episodes=5, render=True):
    """Test trained agent"""
    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    env = ImgObsWrapper(env)
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = state['image'] if isinstance(state, dict) else state
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state['image'] if isinstance(next_state, dict) else next_state
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}, Steps = {steps}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "a2c"], default="dqn", help="Algorithm to use")
    parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--test", action="store_true", help="Test a trained model")
    parser.add_argument("--model", type=str, help="Path to saved model for testing")
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    if args.test:
        # Test existing model
        if not args.model:
            args.model = f"models/{args.algo}_{args.env}_best.pt"
        
        env = gym.make(args.env, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        
        if args.algo == "dqn":
            agent = DQNAgent(env, epsilon=0.01)  # Low epsilon for testing
        else:
            agent = A2CAgent(env)
        
        agent.load(args.model)
        test_agent(agent, args.env, episodes=5, render=True)
    else:
        # Train new model
        print(f"Training {args.algo.upper()} on {args.env} for {args.episodes} episodes...")
        
        if args.algo == "dqn":
            agent = train_dqn(args.env, args.episodes)
        else:
            agent = train_a2c(args.env, args.episodes)
        
        print("\nTesting trained agent...")
        test_agent(agent, args.env, episodes=3, render=False)