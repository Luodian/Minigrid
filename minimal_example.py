#!/usr/bin/env python3
"""Minimal script to run Minigrid environment"""

import gymnasium as gym
import minigrid  # Register Minigrid environments

# Create environment
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")

# Reset environment
obs, info = env.reset(seed=42)

# Run a few random steps
for i in range(100):
    # Take random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Reset if episode ends
    if terminated or truncated:
        obs, info = env.reset()

env.close()