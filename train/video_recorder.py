#!/usr/bin/env python3
"""Unified video recording module for trained agents"""

import os
import json
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


class UnifiedVideoRecorder:
    """Universal video recorder for different environment types"""
    
    def __init__(
        self,
        env_id: str,
        model_path: str,
        env_config: Optional[Dict[str, Any]] = None,
        video_dir: str = "./videos"
    ):
        """
        Initialize video recorder
        
        Args:
            env_id: Environment ID
            model_path: Path to trained model
            env_config: Optional environment configuration
            video_dir: Directory to save videos
        """
        self.env_id = env_id
        self.model_path = model_path
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Import env_configs if not provided
        if env_config is None:
            from env_configs import get_environment_config
            self.env_config = get_environment_config(env_id)
        else:
            self.env_config = env_config
        
        self.env_type = self.env_config['env_type']
        self.wrapper_fn = self.env_config.get('wrapper_fn')
        
        # Load model
        self.model = PPO.load(model_path)
    
    def _create_env(self, render_mode: str = "rgb_array") -> gym.Env:
        """Create environment with appropriate wrappers"""
        env = gym.make(self.env_id, render_mode=render_mode)
        
        if self.wrapper_fn:
            env = self.wrapper_fn(env)
        
        return env
    
    def _get_action_name(self, env: gym.Env, action: Any) -> str:
        """Get human-readable action name if available"""
        action_id = int(action) if hasattr(action, "__int__") else int(np.array(action).item())
        
        # Try to get action meanings
        if hasattr(env, 'get_action_meanings'):
            meanings = env.get_action_meanings()
            if action_id < len(meanings):
                return meanings[action_id]
        elif hasattr(env.unwrapped, 'get_action_meanings'):
            meanings = env.unwrapped.get_action_meanings()
            if action_id < len(meanings):
                return meanings[action_id]
        
        # MiniGrid-specific actions
        if self.env_type == 'minigrid':
            try:
                from minigrid.core.actions import Actions
                return Actions(action_id).name
            except:
                pass
        
        return str(action_id)
    
    def _add_overlay(
        self,
        frame: np.ndarray,
        info: Dict[str, Any],
        position: str = "top-left"
    ) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        
        # Determine text position
        if position == "top-left":
            x, y = 10, 30
            y_step = 30
        elif position == "top-right":
            x, y = frame.shape[1] - 200, 30
            y_step = 30
        else:
            x, y = 10, frame.shape[0] - 100
            y_step = -30
        
        # Add semi-transparent background for better visibility
        bg_height = len(info) * abs(y_step) + 20
        bg_y1 = y - 20 if position.startswith("top") else y - bg_height
        bg_y2 = bg_y1 + bg_height
        overlay[bg_y1:bg_y2, x-5:x+200] = cv2.addWeighted(
            overlay[bg_y1:bg_y2, x-5:x+200], 0.3,
            np.zeros_like(overlay[bg_y1:bg_y2, x-5:x+200]), 0.7, 0
        )
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(overlay, text, (x, y), font, 0.6, (255, 255, 255), 2)
            y += y_step
        
        return overlay
    
    def record_episode(
        self,
        episode_num: int = 1,
        max_steps: int = None,
        add_overlay: bool = True,
        save_actions: bool = False,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Record a single episode
        
        Returns:
            Dictionary with episode statistics and file paths
        """
        # Create environment
        env = self._create_env()
        
        # Set max steps based on environment type
        if max_steps is None:
            max_steps = 10000 if self.env_type == 'atari' else 1000
        
        # Reset environment
        if seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        
        # Video setup
        video_path = self.video_dir / f"{self.env_id}_ep{episode_num}.mp4"
        fps = 30 if self.env_type == 'atari' else 10
        frames = []
        
        # Action logging setup
        actions_data = []
        
        # Episode loop
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get frame
            if hasattr(env, 'render'):
                frame = env.render()
            else:
                frame = env.unwrapped.render()
            
            # Add overlay if requested
            if add_overlay and frame is not None:
                overlay_info = {
                    "Step": step_count,
                    "Action": self._get_action_name(env, action),
                    "Reward": f"{reward:.2f}",
                    "Total": f"{total_reward:.2f}"
                }
                frame = self._add_overlay(frame, overlay_info)
            
            if frame is not None:
                frames.append(frame)
            
            # Log action if requested
            if save_actions:
                actions_data.append({
                    "step": step_count,
                    "action": self._get_action_name(env, action),
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated)
                })
            
            total_reward += reward
            step_count += 1
        
        # Save video
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Episode {episode_num}: Saved video to {video_path}")
        
        # Save actions if requested
        actions_path = None
        if save_actions and actions_data:
            actions_path = self.video_dir / f"{self.env_id}_actions_ep{episode_num}.json"
            with open(actions_path, 'w') as f:
                json.dump(actions_data, f, indent=2)
            print(f"Episode {episode_num}: Saved actions to {actions_path}")
        
        env.close()
        
        return {
            "episode": episode_num,
            "total_reward": total_reward,
            "steps": step_count,
            "video_path": str(video_path) if frames else None,
            "actions_path": str(actions_path) if actions_path else None
        }
    
    def record_episodes(
        self,
        num_episodes: int = 3,
        max_steps: int = None,
        add_overlay: bool = True,
        save_actions: bool = False,
        seed_start: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Record multiple episodes
        
        Returns:
            List of episode statistics
        """
        print(f"Recording {num_episodes} episodes for {self.env_id}")
        results = []
        
        for episode in range(num_episodes):
            result = self.record_episode(
                episode_num=episode + 1,
                max_steps=max_steps,
                add_overlay=add_overlay,
                save_actions=save_actions,
                seed=seed_start + episode if seed_start else None
            )
            results.append(result)
            print(f"Episode {episode + 1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
        
        # Print summary
        rewards = [r['total_reward'] for r in results]
        steps = [r['steps'] for r in results]
        print(f"\nSummary:")
        print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
        print(f"Videos saved in {self.video_dir}")
        
        return results
    
    def record_dual_view(
        self,
        episode_num: int = 1,
        max_steps: int = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Record dual view for MiniGrid environments (global + agent view)
        Only works for MiniGrid environments
        """
        if self.env_type != 'minigrid':
            print(f"Dual view recording only supported for MiniGrid environments")
            return self.record_episode(episode_num, max_steps, seed=seed)
        
        from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        
        # Create two environments
        global_env = gym.make(self.env_id, render_mode="rgb_array")
        agent_env = gym.make(self.env_id, render_mode="rgb_array", agent_view_size=7)
        agent_env = RGBImgPartialObsWrapper(agent_env, tile_size=32)
        agent_env = ImgObsWrapper(agent_env)
        
        # Create prediction environment
        predict_env = self._create_env()
        
        # Reset all with same seed
        if seed is not None:
            global_obs, _ = global_env.reset(seed=seed)
            agent_obs, _ = agent_env.reset(seed=seed)
            predict_obs, _ = predict_env.reset(seed=seed)
        else:
            global_obs, _ = global_env.reset()
            agent_obs, _ = agent_env.reset()
            predict_obs, _ = predict_env.reset()
        
        # Video setup for both views
        global_path = self.video_dir / f"{self.env_id}_global_ep{episode_num}.mp4"
        agent_path = self.video_dir / f"{self.env_id}_agent_ep{episode_num}.mp4"
        fps = 10
        global_frames = []
        agent_frames = []
        
        # Episode loop
        done = False
        total_reward = 0
        step_count = 0
        max_steps = max_steps or 1000
        
        while not done and step_count < max_steps:
            # Get action
            action, _ = self.model.predict(predict_obs, deterministic=True)
            
            # Step all environments
            global_obs, reward, g_term, g_trunc, _ = global_env.step(action)
            agent_obs, _, _, _, _ = agent_env.step(action)
            predict_obs, _, _, _, _ = predict_env.step(action)
            
            done = g_term or g_trunc
            
            # Get frames
            global_frame = global_env.render()
            agent_frame = agent_obs  # Agent view from wrapper
            
            # Resize agent view to match global if needed
            if agent_frame.shape[:2] != global_frame.shape[:2]:
                agent_frame = cv2.resize(
                    agent_frame,
                    (global_frame.shape[1], global_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            global_frames.append(global_frame)
            agent_frames.append(agent_frame)
            
            total_reward += reward
            step_count += 1
        
        # Save both videos
        for frames, path, view_name in [
            (global_frames, global_path, "global"),
            (agent_frames, agent_path, "agent")
        ]:
            if frames:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
                
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"Episode {episode_num}: Saved {view_name} view to {path}")
        
        global_env.close()
        agent_env.close()
        predict_env.close()
        
        return {
            "episode": episode_num,
            "total_reward": total_reward,
            "steps": step_count,
            "global_video": str(global_path),
            "agent_video": str(agent_path)
        }