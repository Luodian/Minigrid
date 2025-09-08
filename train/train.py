#!/usr/bin/env python3
"""Unified training script for PPO agents across all environment types"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from base_trainer import UnifiedPPOTrainer
from env_configs import (
    get_environment_config,
    detect_environment_type,
    get_preset,
    list_presets,
    ENVIRONMENT_PRESETS
)
from video_recorder import UnifiedVideoRecorder


def main():
    parser = argparse.ArgumentParser(
        description="Unified PPO training for MiniGrid, Atari, and Gymnasium environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a single environment
  python train.py --env MiniGrid-DoorKey-8x8-v0 --timesteps 100000
  
  # Train with a preset
  python train.py --preset minigrid-easy
  
  # Test a trained model
  python train.py --test --env MiniGrid-DoorKey-8x8-v0 --model models/ppo_minigrid_MiniGrid-DoorKey-8x8-v0_final
  
  # Record video
  python train.py --record --env MiniGrid-DoorKey-8x8-v0 --episodes 3
  
  # Record dual view (MiniGrid only)
  python train.py --record --dual-view --env MiniGrid-Empty-8x8-v0
  
  # List available presets
  python train.py --list-presets
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", default=True, help="Train a new model (default)")
    mode_group.add_argument("--test", action="store_true", help="Test a trained model")
    mode_group.add_argument("--record", action="store_true", help="Record video of agent")
    mode_group.add_argument("--list-presets", action="store_true", help="List available environment presets")
    
    # Environment selection
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env", type=str, help="Environment ID to train/test")
    env_group.add_argument("--preset", type=str, help="Use a preset group of environments")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint save frequency")
    
    # Early stopping parameters
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Minimum improvement for early stopping")
    parser.add_argument("--reward-threshold", type=float, help="Stop training when reward threshold is reached")
    parser.add_argument("--reward-checkpoint-threshold", type=float, help="Save checkpoint when reward threshold is reached")
    
    # Model parameters
    parser.add_argument("--model", type=str, help="Path to saved model (for testing/recording)")
    parser.add_argument("--model-name", type=str, help="Custom name for saved model")
    
    # Test/Record parameters
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for testing/recording")
    parser.add_argument("--render", action="store_true", help="Render environment during testing")
    parser.add_argument("--save-actions", action="store_true", help="Save action logs when recording")
    parser.add_argument("--dual-view", action="store_true", help="Record dual view for MiniGrid (global + agent)")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay in recorded videos")
    
    # Hyperparameter overrides
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--n-steps", type=int, help="Override n_steps")
    parser.add_argument("--gamma", type=float, help="Override gamma")
    
    # Other options
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=silent, 1=normal, 2=debug)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Handle list-presets
    if args.list_presets:
        list_presets()
        return
    
    # Validate environment selection
    if not args.env and not args.preset:
        parser.error("Either --env or --preset must be specified")
    
    # Handle presets
    if args.preset:
        preset = get_preset(args.preset)
        environments = preset['environments']
        preset_params = preset['hyperparams']
        
        print(f"Training preset '{args.preset}' with {len(environments)} environments")
        print(f"Environments: {', '.join(environments)}")
        print(f"Parameters: {json.dumps(preset_params, indent=2)}")
        print("=" * 50)
        
        # Train each environment in the preset
        for env_id in environments:
            print(f"\nTraining {env_id}...")
            
            # Get environment config
            custom_config = {}
            if args.learning_rate:
                custom_config.setdefault('hyperparams', {})['learning_rate'] = args.learning_rate
            if args.batch_size:
                custom_config.setdefault('hyperparams', {})['batch_size'] = args.batch_size
            if args.n_steps:
                custom_config.setdefault('hyperparams', {})['n_steps'] = args.n_steps
            if args.gamma:
                custom_config.setdefault('hyperparams', {})['gamma'] = args.gamma
            
            env_config = get_environment_config(env_id, custom_config)
            
            # Create trainer
            trainer = UnifiedPPOTrainer(env_config)
            
            # Train
            trainer.train(
                total_timesteps=preset_params.get('total_timesteps', args.timesteps),
                n_envs=preset_params.get('n_envs', args.n_envs),
                eval_freq=args.eval_freq,
                save_freq=args.save_freq,
                patience=args.patience,
                min_improvement=args.min_improvement,
                reward_threshold=args.reward_threshold,
                reward_checkpoint_threshold=args.reward_checkpoint_threshold,
                model_name=f"{args.preset}_{env_id}" if not args.model_name else args.model_name,
                verbose=args.verbose
            )
        
        print(f"\nCompleted training all environments in preset '{args.preset}'")
        return
    
    # Single environment mode
    env_id = args.env
    
    # Get environment configuration
    custom_config = {}
    if args.learning_rate:
        custom_config.setdefault('hyperparams', {})['learning_rate'] = args.learning_rate
    if args.batch_size:
        custom_config.setdefault('hyperparams', {})['batch_size'] = args.batch_size
    if args.n_steps:
        custom_config.setdefault('hyperparams', {})['n_steps'] = args.n_steps
    if args.gamma:
        custom_config.setdefault('hyperparams', {})['gamma'] = args.gamma
    
    env_config = get_environment_config(env_id, custom_config)
    env_type = env_config['env_type']
    
    print(f"Environment: {env_id}")
    print(f"Type: {env_type}")
    print("=" * 50)
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Auto-generate model path
        model_name = args.model_name or f"ppo_{env_type}_{env_id}"
        model_path = f"models/{model_name}_final"
    
    # Handle different modes
    if args.test:
        # Test mode
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first or specify a valid model path with --model")
            return
        
        trainer = UnifiedPPOTrainer(env_config)
        results = trainer.test(
            model_path=model_path,
            episodes=args.episodes,
            render=args.render,
            verbose=True
        )
        
    elif args.record:
        # Record mode
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first or specify a valid model path with --model")
            return
        
        recorder = UnifiedVideoRecorder(
            env_id=env_id,
            model_path=model_path,
            env_config=env_config
        )
        
        if args.dual_view and env_type == 'minigrid':
            # Record dual view for MiniGrid
            print("Recording dual view (global + agent)...")
            for episode in range(args.episodes):
                recorder.record_dual_view(
                    episode_num=episode + 1,
                    seed=args.seed + episode if args.seed else None
                )
        else:
            # Standard recording
            recorder.record_episodes(
                num_episodes=args.episodes,
                add_overlay=not args.no_overlay,
                save_actions=args.save_actions,
                seed_start=args.seed
            )
        
    else:
        # Train mode (default)
        trainer = UnifiedPPOTrainer(env_config)
        
        model = trainer.train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            patience=args.patience,
            min_improvement=args.min_improvement,
            reward_threshold=args.reward_threshold,
            reward_checkpoint_threshold=args.reward_checkpoint_threshold,
            model_name=args.model_name,
            verbose=args.verbose
        )
        
        # Test the trained model
        print("\nTesting trained model...")
        trainer.test(
            model_path=model_path,
            episodes=3,
            render=False,
            verbose=True
        )


if __name__ == "__main__":
    main()