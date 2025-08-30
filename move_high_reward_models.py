#!/usr/bin/env python3
"""
Script to move models corresponding to high reward environments to a target directory.
"""

import os
import shutil
from pathlib import Path

def get_high_reward_envs():
    """Return list of environments with reward >= 0.8 from our previous analysis."""
    return [
        "BabyAI-GoToLocalS6N3-v0",
        "BabyAI-GoToLocalS6N4-v0", 
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToObjS4-v0",
        "BabyAI-GoToObjS6-v1",
        "BabyAI-GoToRedBall-v0",
        "BabyAI-GoToRedBallNoDists-v0",
        "BabyAI-KeyCorridorS3R1-v0",
        "BabyAI-OneRoomS16-v0",
        "BabyAI-OneRoomS8-v0",
        "BabyAI-OpenDoor-v0",
        "BabyAI-OpenDoorColor-v0",
        "BabyAI-OpenRedDoor-v0",
        "BabyAI-PutNextS4N1-v0",
        "BabyAI-UnlockLocal-v0",
        "MiniGrid-DistShift1-v0",
        "MiniGrid-DistShift2-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        "MiniGrid-Empty-16x16-v0",
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-Random-5x5-v0",
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-KeyCorridorS3R2-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N3-v0"
    ]

def find_model_files(models_dir, env_names):
    """Find model files corresponding to the given environment names."""
    model_files = []
    
    for env_name in env_names:
        # Look for PPO model files with the pattern: ppo_{env_name}_final.zip
        model_pattern = f"ppo_{env_name}_final.zip"
        model_path = os.path.join(models_dir, model_pattern)
        
        if os.path.exists(model_path):
            model_files.append((env_name, model_path))
        else:
            print(f"Warning: Model not found for {env_name} (expected: {model_pattern})")
    
    return model_files

def move_models(model_files, target_dir):
    """Move model files to target directory."""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    moved_count = 0
    total_size = 0
    
    print(f"Moving {len(model_files)} models to {target_dir}...")
    print("-" * 60)
    
    for env_name, source_path in model_files:
        filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)
        
        try:
            # Get file size for reporting
            file_size = os.path.getsize(source_path)
            total_size += file_size
            
            # Move the file
            shutil.move(source_path, target_path)
            moved_count += 1
            
            print(f"✓ Moved {filename} ({file_size/1024/1024:.1f} MB)")
            
        except Exception as e:
            print(f"✗ Failed to move {filename}: {e}")
    
    print("-" * 60)
    print(f"Successfully moved {moved_count} models")
    print(f"Total size: {total_size/1024/1024:.1f} MB")
    
    return moved_count

def main():
    """Main function to move high reward models."""
    models_dir = "./models"
    target_dir = "/mnt/bn/seed-aws-va/brianli/prod/Minigrid/models"
    
    print("High Reward Models Mover")
    print("=" * 60)
    print(f"Source directory: {models_dir}")
    print(f"Target directory: {target_dir}")
    print("=" * 60)
    
    if not os.path.exists(models_dir):
        print(f"Error: Source directory {models_dir} not found!")
        return
    
    # Get high reward environment names
    high_reward_envs = get_high_reward_envs()
    print(f"Looking for models for {len(high_reward_envs)} high reward environments...")
    
    # Find corresponding model files
    model_files = find_model_files(models_dir, high_reward_envs)
    print(f"Found {len(model_files)} model files to move")
    
    if not model_files:
        print("No model files found to move!")
        return
    
    # Move the models
    moved_count = move_models(model_files, target_dir)
    
    print(f"\nOperation completed: {moved_count} models moved to {target_dir}")

if __name__ == "__main__":
    main()