#!/usr/bin/env python3
"""Train PPO agent using Stable Baselines3 with early stopping"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, FlatObsWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import json
from typing import Optional
import argparse

class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback based on improvement threshold and patience"""
    def __init__(self, patience=10, min_improvement=0.01, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.eval_rewards = []
    
    def _on_step(self) -> bool:
        # This is called after every env step
        # Early stopping logic handled in eval callback
        return True
    
    def check_improvement(self, mean_reward):
        """Check if there's sufficient improvement"""
        if mean_reward > self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.2f}")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"No improvement for {self.no_improvement_count} evaluations")
        
        # Only trigger early stopping if reward is above 0.5 and patience exceeded
        if self.no_improvement_count >= self.patience and mean_reward > 0.9:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.no_improvement_count} evaluations without improvement")
            return False  # Stop training
        return True  # Continue training

def make_env(env_id, rank, seed=0):
    """Create a wrapped Minigrid environment"""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)  # Use image observations
        env = Monitor(env)  # Monitor wrapper for logging
        env.reset(seed=seed + rank)
        return env
    return _init

def train_ppo(env_id="MiniGrid-DoorKey-8x8-v0", total_timesteps=100000, n_envs=4,
              reward_threshold=None, early_stopping_patience=10, min_improvement=0.01):
    """Train PPO agent with early stopping mechanism"""
    print(f"Training PPO on {env_id} with early stopping")
    print(f"Early stopping patience: {early_stopping_patience}, Min improvement: {min_improvement}")
    
    # Create vectorized environment
    env = make_vec_env(
        lambda: gym.wrappers.FlattenObservation(
            ImgObsWrapper(gym.make(env_id, render_mode="rgb_array"))
        ),
        n_envs=n_envs,
        seed=42
    )
    
    # Create eval environment
    eval_env = gym.wrappers.FlattenObservation(
        ImgObsWrapper(gym.make(env_id, render_mode="rgb_array"))
    )
    
    # Define model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/"
    )
    
    # Create callbacks list
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        patience=early_stopping_patience,
        min_improvement=min_improvement,
        verbose=1
    )
    
    # Modified eval callback that works with early stopping
    class EvalWithEarlyStopping(EvalCallback):
        def __init__(self, early_stop_callback, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.early_stop = early_stop_callback
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            
            # Check if evaluation was performed
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # Get last evaluation result
                if len(self.evaluations_results) > 0:
                    last_mean_reward = self.evaluations_results[-1][0]
                    # Check for improvement
                    should_continue = self.early_stop.check_improvement(last_mean_reward)
                    if not should_continue:
                        return False  # Stop training
            
            return result
    
    eval_callback = EvalWithEarlyStopping(
        early_stopping,
        eval_env,
        best_model_save_path="./models/ppo/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Reward threshold stopping (optional)
    if reward_threshold is not None:
        reward_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=1
        )
        callbacks.append(EvalCallback(
            eval_env,
            callback_on_new_best=reward_callback,
            eval_freq=5000,
            verbose=0
        ))
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Combine callbacks
    callback_list = CallbackList(callbacks)
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            tb_log_name=env_id
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    model.save(f"models/ppo_{env_id}_final")
    print(f"Model saved to models/ppo_{env_id}_final")
    
    return model


def test_agent(model_path, env_id, episodes=5, render=True):
    """Test a trained PPO agent"""
    print(f"\nTesting PPO agent on {env_id}")
    
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(ImgObsWrapper(env))
    
    # Load model
    model = PPO.load(model_path)
    
    # Test episodes
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    # Print statistics
    print(f"\nStatistics over {episodes} episodes:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    
    env.close()

def evaluate_model(model_path, env_id, n_eval_episodes=10):
    """Evaluate a trained PPO model"""
    print(f"\nEvaluating PPO model on {env_id}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create eval environment
    eval_env = gym.wrappers.FlattenObservation(
        ImgObsWrapper(gym.make(env_id, render_mode="rgb_array"))
    )
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

def record_video_with_agent(
    model_path: str,
    env_id: str,
    num_episodes: int = 3,
    record_actions: bool = False,
    actions_format: str = "csv",
    actions_dir: Optional[str] = None,
):
    """Record video of a trained PPO agent.

    Saves two videos per episode (global and front views). Optionally logs
    per-step actions to CSV or JSONL alongside the videos.
    """
    import cv2
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    try:
        from minigrid.core.actions import Actions  # for action names
    except Exception:
        Actions = None
    
    print(f"Recording dual-stream videos for {env_id}")
    
    # Create two environments - one for global view, one for front view
    global_env = gym.make(env_id, render_mode="rgb_array")
    front_env = gym.make(env_id, render_mode="rgb_array", agent_view_size=7)
    # Use larger tile_size to get higher resolution (7x7 grid * 32 pixels = 224x224, then we'll resize to 256x256)
    front_env = RGBImgPartialObsWrapper(front_env, tile_size=32)
    front_env = ImgObsWrapper(front_env)  # Wrap to get image only
    
    # Create flattened env for model prediction
    predict_env = gym.make(env_id, render_mode="rgb_array")
    predict_env = ImgObsWrapper(predict_env)
    predict_env = gym.wrappers.FlattenObservation(predict_env)
    
    # Load model
    model = PPO.load(model_path)
    
    # Record episodes
    for episode in range(num_episodes):
        # Reset all environments with same seed
        seed = 42 + episode
        global_obs, _ = global_env.reset(seed=seed)
        front_obs, _ = front_env.reset(seed=seed)
        predict_obs, _ = predict_env.reset(seed=seed)

        # Video writer setup for two separate streams
        global_video_name = f"./videos/ppo_global_view_ep{episode+1}.mp4"
        front_video_name = f"./videos/ppo_front_view_ep{episode+1}.mp4"
        fps = 10
        global_frames = []
        front_frames = []

        # Optional action logging
        actions_out_base = None
        actions_fp = None
        # Optional per-step frame export directories (tie frames to actions)
        frames_global_dir = None
        frames_front_dir = None
        if record_actions:
            out_dir = actions_dir or os.path.dirname(global_video_name) or "."
            os.makedirs(out_dir, exist_ok=True)
            actions_out_base = os.path.join(out_dir, f"ppo_actions_ep{episode+1}")
            # Create per-episode frame folders next to the action log
            frames_base_dir = os.path.join(out_dir, f"ppo_frames_ep{episode+1}")
            frames_global_dir = os.path.join(frames_base_dir, "global")
            frames_front_dir = os.path.join(frames_base_dir, "front")
            os.makedirs(frames_global_dir, exist_ok=True)
            os.makedirs(frames_front_dir, exist_ok=True)
            if actions_format.lower() == "jsonl":
                actions_fp = open(actions_out_base + ".jsonl", "w", encoding="utf-8")
            else:
                # default to CSV
                actions_format = "csv"
                actions_fp = open(actions_out_base + ".csv", "w", encoding="utf-8")
                # Include frame file paths to directly link images and actions
                actions_fp.write("t,action_id,action_name,reward,terminated,truncated,global_frame,front_frame\n")

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:
            # Get action from model using flattened observation
            action, _ = model.predict(predict_obs, deterministic=True)
            
            # Step all environments with same action
            global_obs, reward, g_terminated, g_truncated, _ = global_env.step(action)
            front_obs, _, f_terminated, f_truncated, _ = front_env.step(action)
            predict_obs, _, p_terminated, p_truncated, _ = predict_env.step(action)
            
            done = g_terminated or g_truncated

            # Get frames for both views
            global_frame = global_env.render()
            front_frame = front_obs  # Front view observation is the RGB image from the wrapper
            
            # Resize front view to match global view resolution if needed
            if front_frame.shape[:2] != global_frame.shape[:2]:
                front_frame = cv2.resize(front_frame, (global_frame.shape[1], global_frame.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Store frames separately without any text overlay
            global_frames.append(global_frame)
            front_frames.append(front_frame)
            
            total_reward += reward
            steps += 1

            # Log action and save per-step frames if enabled
            if record_actions and actions_fp is not None:
                a_id = int(action) if hasattr(action, "__int__") else int(np.array(action).item())
                a_name = None
                if Actions is not None:
                    try:
                        a_name = Actions(a_id).name
                    except Exception:
                        a_name = str(a_id)
                # Sanitize action name for filenames
                safe_name = "" if a_name is None else "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in a_name)
                # Build filenames (zero-padded step index for ordering)
                step_idx = steps - 1
                g_fname = f"g_{step_idx:04d}_a{a_id}{('_' + safe_name) if safe_name else ''}.png"
                f_fname = f"f_{step_idx:04d}_a{a_id}{('_' + safe_name) if safe_name else ''}.png"
                g_path = os.path.join(frames_global_dir, g_fname) if frames_global_dir else None
                f_path = os.path.join(frames_front_dir, f_fname) if frames_front_dir else None
                # Save image files mapped to this action/timestamp
                if g_path is not None:
                    # Convert RGB->BGR for OpenCV
                    from cv2 import cvtColor, COLOR_RGB2BGR, imwrite
                    imwrite(g_path, cvtColor(global_frame, COLOR_RGB2BGR))
                if f_path is not None:
                    from cv2 import cvtColor, COLOR_RGB2BGR, imwrite
                    imwrite(f_path, cvtColor(front_frame, COLOR_RGB2BGR))

                # Ensure JSON-serializable reward
                rew = float(reward) if isinstance(reward, (np.floating, float, int)) else float(np.array(reward).item())
                if actions_format == "jsonl":
                    rec = {
                        "t": step_idx,
                        "action_id": a_id,
                        "action_name": a_name,
                        "reward": rew,
                        "terminated": bool(g_terminated),
                        "truncated": bool(g_truncated),
                        "global_frame": g_path,
                        "front_frame": f_path,
                    }
                    actions_fp.write(json.dumps(rec) + "\n")
                else:
                    # CSV (include frame file paths)
                    actions_fp.write(
                        f"{step_idx},{a_id},{a_name if a_name is not None else ''},{rew},{int(g_terminated)},{int(g_truncated)},{g_path if g_path else ''},{f_path if f_path else ''}\n"
                    )
        
        # Write global view video
        if global_frames:
            height, width = global_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_global = cv2.VideoWriter(global_video_name, fourcc, fps, (width, height))
            
            for frame in global_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out_global.write(frame_bgr)
            
            out_global.release()
            print(f"Episode {episode + 1}: Saved {global_video_name}")
        
        # Write front view video
        if front_frames:
            height, width = front_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_front = cv2.VideoWriter(front_video_name, fourcc, fps, (width, height))
            
            for frame in front_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out_front.write(frame_bgr)
            
            out_front.release()
            print(f"Episode {episode + 1}: Saved {front_video_name}")
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {steps}")

        if actions_fp is not None:
            actions_fp.close()
            suffix = ".jsonl" if actions_format == "jsonl" else ".csv"
            print(f"Episode {episode + 1}: Saved actions to {actions_out_base + suffix}")
        
    global_env.close()
    front_env.close()
    predict_env.close()
    print(f"Videos saved in ./videos/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent with early stopping")
    parser.add_argument(
        "--env",
        default="MiniGrid-Empty-8x8-v0",
        help="Environment ID"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (number of evaluations without improvement)"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Minimum improvement threshold for early stopping"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.8,
        help="Stop training when this reward threshold is reached"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test a trained model"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to saved model (for testing/evaluation)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video of agent playing"
    )
    parser.add_argument(
        "--record-actions",
        action="store_true",
        help="Also save per-step actions while recording"
    )
    parser.add_argument(
        "--actions-format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Format for saved actions when recording"
    )
    parser.add_argument(
        "--actions-dir",
        type=str,
        default=None,
        help="Directory to save action logs (defaults to videos dir)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test/record episodes"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    if args.test:
        # Test existing model
        if not args.model:
            args.model = f"models/ppo_{args.env}_final"
        test_agent(args.model, args.env, episodes=args.episodes)
    
    elif args.evaluate:
        # Evaluate model
        if not args.model:
            args.model = f"models/ppo_{args.env}_final"
        evaluate_model(args.model, args.env, n_eval_episodes=args.episodes)
    
    elif args.record:
        # Record video
        if not args.model:
            args.model = f"models/ppo_{args.env}_final"
        record_video_with_agent(
            args.model,
            args.env,
            num_episodes=args.episodes,
            record_actions=args.record_actions,
            actions_format=args.actions_format,
            actions_dir=args.actions_dir,
        )
    
    else:
        # Train PPO with early stopping
        print(f"Training PPO on {args.env}")
        print(f"Settings: timesteps={args.timesteps}, n_envs={args.n_envs}")
        print(f"Early stopping: patience={args.patience}, min_improvement={args.min_improvement}")
        if args.reward_threshold:
            print(f"Reward threshold: {args.reward_threshold}")
        
        model = train_ppo(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            reward_threshold=args.reward_threshold,
            early_stopping_patience=args.patience,
            min_improvement=args.min_improvement
        )
        
        print("\nTraining complete! Testing agent...")
        test_agent(f"models/ppo_{args.env}_final", args.env, episodes=3, render=False)
