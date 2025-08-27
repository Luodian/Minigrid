#!/usr/bin/env python3
"""Record trajectory data (states, actions, rewards) from Minigrid"""

import gymnasium as gym
import minigrid
import numpy as np
import json
import pickle
from datetime import datetime

class TrajectoryRecorder:
    """Record and save gameplay trajectories"""
    
    def __init__(self, env_name="MiniGrid-DoorKey-8x8-v0"):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.trajectories = []
        self.current_trajectory = None
        
    def start_episode(self, seed=None):
        """Start recording a new episode"""
        obs, info = self.env.reset(seed=seed)
        self.current_trajectory = {
            "env_name": self.env.spec.id,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "infos": [info],
            "frames": [self.env.render()],
            "terminated": False,
            "truncated": False
        }
        return obs, info
    
    def step(self, action):
        """Record a step in the environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_trajectory:
            self.current_trajectory["observations"].append(obs)
            self.current_trajectory["actions"].append(action)
            self.current_trajectory["rewards"].append(reward)
            self.current_trajectory["infos"].append(info)
            self.current_trajectory["frames"].append(self.env.render())
            
            if terminated or truncated:
                self.current_trajectory["terminated"] = terminated
                self.current_trajectory["truncated"] = truncated
                self.end_episode()
        
        return obs, reward, terminated, truncated, info
    
    def end_episode(self):
        """Finish recording current episode"""
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            print(f"Episode recorded: {len(self.current_trajectory['actions'])} steps, "
                  f"Total reward: {sum(self.current_trajectory['rewards']):.2f}")
            self.current_trajectory = None
    
    def save_trajectories(self, filename="trajectories"):
        """Save recorded trajectories to file"""
        # Save as pickle (preserves numpy arrays)
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self.trajectories, f)
        print(f"Trajectories saved to {filename}.pkl")
        
        # Save summary as JSON (human-readable)
        summary = []
        for i, traj in enumerate(self.trajectories):
            summary.append({
                "episode": i,
                "env_name": traj["env_name"],
                "seed": traj["seed"],
                "timestamp": traj["timestamp"],
                "num_steps": len(traj["actions"]),
                "total_reward": sum(traj["rewards"]),
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "actions": [int(a) for a in traj["actions"]]  # Convert to list of ints
            })
        
        with open(f"{filename}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {filename}_summary.json")
        
        return summary
    
    def save_video(self, episode_idx=0, filename="trajectory_video.mp4"):
        """Save specific episode as video"""
        import imageio
        
        if episode_idx < len(self.trajectories):
            traj = self.trajectories[episode_idx]
            writer = imageio.get_writer(filename, fps=10)
            
            for frame in traj["frames"]:
                writer.append_data(frame)
            
            writer.close()
            print(f"Video saved to {filename}")
    
    def replay_trajectory(self, trajectory_idx=0, delay_ms=100):
        """Replay a recorded trajectory visually"""
        import matplotlib.pyplot as plt
        from IPython import display
        import time
        
        if trajectory_idx >= len(self.trajectories):
            print(f"No trajectory at index {trajectory_idx}")
            return
        
        traj = self.trajectories[trajectory_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, frame in enumerate(traj["frames"]):
            ax1.clear()
            ax1.imshow(frame)
            ax1.set_title(f"Step {i}/{len(traj['frames'])-1}")
            ax1.axis('off')
            
            ax2.clear()
            rewards_so_far = traj["rewards"][:i] if i > 0 else []
            ax2.plot(rewards_so_far)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.set_title(f'Cumulative Reward: {sum(rewards_so_far):.2f}')
            
            plt.tight_layout()
            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(delay_ms / 1000.0)
        
        plt.close()

def record_human_play():
    """Record human playing with keyboard"""
    import pygame
    from minigrid.manual_control import ManualControl
    
    recorder = TrajectoryRecorder("MiniGrid-DoorKey-8x8-v0")
    
    class RecordingManualControl(ManualControl):
        def __init__(self, env, recorder, seed=None):
            super().__init__(env, seed)
            self.recorder = recorder
            self.recording = False
            
        def reset(self, seed=None):
            if self.recording:
                self.recorder.end_episode()
            obs, info = self.recorder.start_episode(seed)
            self.recording = True
            self.env.render()
            
        def step(self, action):
            obs, reward, terminated, truncated, info = self.recorder.step(action)
            print(f"step={self.env.unwrapped.step_count}, reward={reward:.2f}")
            
            if terminated:
                print("terminated!")
                self.reset(self.seed)
            elif truncated:
                print("truncated!")
                self.reset(self.seed)
            else:
                self.env.render()
    
    print("\nRecording human gameplay...")
    print("Controls: Arrow keys to move, Space to toggle, Escape to quit")
    print("Each episode will be recorded automatically\n")
    
    control = RecordingManualControl(recorder.env, recorder, seed=42)
    control.start()
    
    # Save when done
    if recorder.trajectories:
        recorder.save_trajectories("human_trajectories")
        recorder.save_video(0, "human_play.mp4")

def record_random_agent():
    """Record random agent playing"""
    recorder = TrajectoryRecorder("MiniGrid-FourRooms-v0")
    
    for episode in range(5):
        recorder.start_episode(seed=42 + episode)
        
        done = False
        steps = 0
        while not done and steps < 200:
            action = recorder.env.action_space.sample()
            obs, reward, terminated, truncated, info = recorder.step(action)
            done = terminated or truncated
            steps += 1
    
    recorder.save_trajectories("random_trajectories")
    recorder.save_video(0, "random_play.mp4")
    
    return recorder

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["human", "random"],
        default="random",
        help="Recording mode: human control or random agent"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to record (for random mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "human":
        record_human_play()
    else:
        print(f"Recording {args.episodes} episodes with random agent...")
        recorder = record_random_agent()
        print(f"\nRecorded {len(recorder.trajectories)} trajectories")
        print("Files saved: random_trajectories.pkl, random_trajectories_summary.json, random_play.mp4")