#!/usr/bin/env python3
"""Unified base trainer for PPO agents across different environment types"""

import os
import json
import gymnasium as gym
import numpy as np
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, 
    StopTrainingOnRewardThreshold, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any, Callable


class EarlyStoppingCallback(BaseCallback):
    """Universal early stopping callback"""
    def __init__(self, patience=10, min_improvement=0.01, min_reward_threshold=0.5, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_reward_threshold = min_reward_threshold
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
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
        
        # Only trigger early stopping if above threshold and patience exceeded
        if (self.no_improvement_count >= self.patience and 
            mean_reward > self.min_reward_threshold):
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.no_improvement_count} evaluations")
            return False
        return True


class EvalWithEarlyStopping(EvalCallback):
    """Evaluation callback with integrated early stopping"""
    def __init__(self, early_stop_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop = early_stop_callback
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if len(self.evaluations_results) > 0:
                last_mean_reward = self.evaluations_results[-1][0]
                should_continue = self.early_stop.check_improvement(last_mean_reward)
                if not should_continue:
                    return False
        
        return result


class UnifiedPPOTrainer:
    """Unified trainer for PPO agents across different environment types"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        Initialize trainer with environment configuration
        
        Args:
            env_config: Dictionary containing:
                - env_type: 'minigrid', 'atari', or 'gymnasium'
                - env_id: Environment ID
                - wrapper_fn: Optional function to wrap environment
                - hyperparams: Optional PPO hyperparameters
        """
        self.env_config = env_config
        self.env_type = env_config['env_type']
        self.env_id = env_config['env_id']
        self.wrapper_fn = env_config.get('wrapper_fn')
        
        # Set default hyperparameters based on environment type
        self.hyperparams = self._get_default_hyperparams()
        if 'hyperparams' in env_config:
            self.hyperparams.update(env_config['hyperparams'])
    
    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters based on environment type"""
        if self.env_type == 'atari':
            return {
                'learning_rate': 2.5e-4,
                'n_steps': 128,
                'batch_size': 256,
                'n_epochs': 4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.1,
                'ent_coef': 0.01,
                'policy': 'CnnPolicy',
                'use_frame_stack': True,
                'n_stack': 4
            }
        elif self.env_type == 'minigrid':
            return {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'policy': 'MlpPolicy',
                'use_frame_stack': False
            }
        else:  # gymnasium
            return {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'policy': 'MlpPolicy',
                'use_frame_stack': False
            }
    
    def create_env(self, n_envs: int = 4, seed: int = 42) -> Any:
        """Create vectorized environment with appropriate wrappers"""
        if self.wrapper_fn:
            env = make_vec_env(
                lambda: self.wrapper_fn(gym.make(self.env_id, render_mode="rgb_array")),
                n_envs=n_envs,
                seed=seed
            )
        else:
            env = make_vec_env(
                self.env_id,
                n_envs=n_envs,
                seed=seed
            )
        
        # Apply frame stacking if needed
        if self.hyperparams.get('use_frame_stack', False):
            env = VecFrameStack(env, n_stack=self.hyperparams.get('n_stack', 4))
        
        return env
    
    def create_eval_env(self) -> Any:
        """Create evaluation environment"""
        env = gym.make(self.env_id, render_mode="rgb_array")
        if self.wrapper_fn:
            env = self.wrapper_fn(env)
        return env
    
    def train(
        self,
        total_timesteps: int = 100000,
        n_envs: int = 4,
        eval_freq: int = 5000,
        save_freq: int = 10000,
        patience: int = 10,
        min_improvement: float = 0.01,
        reward_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        verbose: int = 1
    ) -> PPO:
        """
        Train PPO agent with unified interface
        
        Returns:
            Trained PPO model
        """
        print(f"Training PPO on {self.env_id} ({self.env_type})")
        print(f"Hyperparameters: {json.dumps(self.hyperparams, indent=2)}")
        
        # Create environments
        env = self.create_env(n_envs=n_envs)
        eval_env = self.create_eval_env()
        
        # Determine policy type
        policy = self.hyperparams.pop('policy', 'MlpPolicy')
        
        # Create model
        model = PPO(
            policy,
            env,
            verbose=verbose,
            tensorboard_log="./tensorboard_logs/",
            **{k: v for k, v in self.hyperparams.items() 
               if k not in ['use_frame_stack', 'n_stack']}
        )
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            patience=patience,
            min_improvement=min_improvement,
            verbose=verbose
        )
        
        # Evaluation with early stopping
        if not model_name:
            model_name = f"ppo_{self.env_type}_{self.env_id}"
        
        eval_callback = EvalWithEarlyStopping(
            early_stopping,
            eval_env,
            best_model_save_path=f"./models/{model_name}/",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=verbose
        )
        callbacks.append(eval_callback)
        
        # Reward threshold stopping
        if reward_threshold is not None:
            reward_callback = StopTrainingOnRewardThreshold(
                reward_threshold=reward_threshold,
                verbose=verbose
            )
            callbacks.append(EvalCallback(
                eval_env,
                callback_on_new_best=reward_callback,
                eval_freq=eval_freq,
                verbose=0
            ))
        
        # Checkpointing
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./models/checkpoints/",
            name_prefix=model_name
        )
        callbacks.append(checkpoint_callback)
        
        # Train
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                progress_bar=True,
                tb_log_name=model_name
            )
            print("\nTraining completed successfully!")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        # Save final model
        final_path = f"models/{model_name}_final"
        model.save(final_path)
        print(f"Model saved to {final_path}")
        
        return model
    
    def test(
        self,
        model_path: str,
        episodes: int = 5,
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Test trained model
        
        Returns:
            Dictionary with test statistics
        """
        print(f"\nTesting PPO agent on {self.env_id}")
        
        # Load model
        model = PPO.load(model_path)
        
        # Create test environment
        render_mode = "human" if render else "rgb_array"
        env = gym.make(self.env_id, render_mode=render_mode)
        if self.wrapper_fn:
            env = self.wrapper_fn(env)
        
        episode_rewards = []
        episode_steps = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 10000 if self.env_type == 'atari' else 1000
            
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            if verbose:
                print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        env.close()
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'std_steps': np.std(episode_steps)
        }
        
        if verbose:
            print(f"\nStatistics over {episodes} episodes:")
            print(f"Average Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Average Steps: {results['mean_steps']:.2f} ± {results['std_steps']:.2f}")
        
        return results