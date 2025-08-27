#!/usr/bin/env python3
"""Record video of Minigrid environment gameplay"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import minigrid
import numpy as np

def record_random_play():
    """Record random agent playing"""
    # Create environment with RecordVideo wrapper
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_folder="./recordings",
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix="minigrid-random"
    )
    
    # Run episodes
    for episode in range(3):  # Record 3 episodes
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
        print(f"Episode {episode + 1} completed in {step_count} steps")
    
    env.close()
    print("\nVideos saved in ./recordings/")

def record_manual_play():
    """Record manual keyboard-controlled play"""
    import pygame
    from minigrid.manual_control import ManualControl
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    
    # Create environment
    env = gym.make(
        "MiniGrid-DoorKey-8x8-v0",
        tile_size=32,
        render_mode="rgb_array"
    )
    
    # Wrap with video recorder
    env = RecordVideo(
        env,
        video_folder="./recordings", 
        episode_trigger=lambda x: True,
        name_prefix="minigrid-manual"
    )
    
    # Manual control
    print("Controls:")
    print("  Arrow keys: move")
    print("  Space: toggle/open doors")
    print("  Tab: pickup")
    print("  Shift: drop")
    print("  Backspace: reset")
    print("  Escape: quit")
    print("\nRecording will save automatically when you reset or quit.\n")
    
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    
    env.close()
    print("\nVideo saved in ./recordings/")

def record_with_imageio():
    """Record using imageio for more control"""
    import imageio
    
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    
    # Create video writer
    writer = imageio.get_writer('./recordings/minigrid_imageio.mp4', fps=10)
    
    obs, info = env.reset(seed=42)
    
    for step in range(300):
        # Capture frame
        frame = env.render()
        writer.append_data(frame)
        
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    writer.close()
    env.close()
    print("Video saved as ./recordings/minigrid_imageio.mp4")

def save_frames_as_images():
    """Save individual frames as images"""
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs("./frames", exist_ok=True)
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    for i in range(50):
        # Get frame
        frame = env.render()
        
        # Save frame
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f'./frames/frame_{i:04d}.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    print(f"Frames saved in ./frames/")
    print("Convert to video with: ffmpeg -r 10 -i ./frames/frame_%04d.png -c:v libx264 output.mp4")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Record Minigrid gameplay")
    parser.add_argument(
        "--mode", 
        choices=["random", "manual", "imageio", "frames"],
        default="random",
        help="Recording mode: random play, manual control, imageio, or save frames"
    )
    
    args = parser.parse_args()
    
    # Create recordings directory
    os.makedirs("./recordings", exist_ok=True)
    
    if args.mode == "random":
        record_random_play()
    elif args.mode == "manual":
        record_manual_play()
    elif args.mode == "imageio":
        record_with_imageio()
    elif args.mode == "frames":
        save_frames_as_images()