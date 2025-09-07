#!/usr/bin/env python3
"""Environment-specific configurations and wrappers"""

import gymnasium as gym
from typing import Dict, Any, Callable, Optional


def get_minigrid_wrapper():
    """Get wrapper function for MiniGrid environments"""
    from minigrid.wrappers import ImgObsWrapper
    
    def wrapper(env):
        env = ImgObsWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
        return env
    
    return wrapper


def get_atari_wrapper():
    """Get wrapper function for Atari environments"""
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    
    def wrapper(env):
        return AtariWrapper(env)
    
    return wrapper


def get_gymnasium_wrapper():
    """Get wrapper function for standard Gymnasium environments"""
    # Most gymnasium environments don't need special wrappers
    return None


def detect_environment_type(env_id: str) -> str:
    """
    Detect environment type from environment ID
    
    Args:
        env_id: Environment ID string
    
    Returns:
        Environment type: 'minigrid', 'atari', or 'gymnasium'
    """
    # MiniGrid environments
    if env_id.startswith('MiniGrid-') or env_id.startswith('BabyAI-'):
        return 'minigrid'
    
    # Atari environments
    if env_id.startswith('ALE/') or any(name in env_id for name in [
        'Breakout', 'Pong', 'SpaceInvaders', 'Asteroids', 'Qbert',
        'MsPacman', 'Seaquest', 'BeamRider', 'Enduro', 'Freeway'
    ]):
        return 'atari'
    
    # Default to gymnasium for everything else
    return 'gymnasium'


def get_environment_config(env_id: str, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get complete environment configuration
    
    Args:
        env_id: Environment ID
        custom_config: Optional custom configuration to override defaults
    
    Returns:
        Complete environment configuration dictionary
    """
    env_type = detect_environment_type(env_id)
    
    # Base configuration
    config = {
        'env_id': env_id,
        'env_type': env_type
    }
    
    # Add appropriate wrapper
    if env_type == 'minigrid':
        config['wrapper_fn'] = get_minigrid_wrapper()
        # MiniGrid-specific hyperparameters
        config['hyperparams'] = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'policy': 'MlpPolicy'
        }
    elif env_type == 'atari':
        config['wrapper_fn'] = get_atari_wrapper()
        # Atari-specific hyperparameters
        config['hyperparams'] = {
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
    else:  # gymnasium
        config['wrapper_fn'] = get_gymnasium_wrapper()
        # Try to detect if we need CNN or MLP policy
        try:
            test_env = gym.make(env_id)
            obs_shape = test_env.observation_space.shape
            test_env.close()
            
            # Use CNN for image observations
            if len(obs_shape) == 3:
                policy = 'CnnPolicy'
            else:
                policy = 'MlpPolicy'
        except:
            policy = 'MlpPolicy'
        
        config['hyperparams'] = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'policy': policy
        }
    
    # Override with custom config if provided
    if custom_config:
        if 'hyperparams' in custom_config:
            config['hyperparams'].update(custom_config['hyperparams'])
        for key, value in custom_config.items():
            if key != 'hyperparams':
                config[key] = value
    
    return config


# Presets for common environments
ENVIRONMENT_PRESETS = {
    # MiniGrid presets
    'minigrid-easy': {
        'environments': [
            'MiniGrid-Empty-5x5-v0',
            'MiniGrid-Empty-6x6-v0',
            'MiniGrid-Empty-8x8-v0',
            'MiniGrid-Empty-Random-5x5-v0',
            'MiniGrid-Empty-Random-6x6-v0'
        ],
        'hyperparams': {
            'total_timesteps': 50000,
            'n_envs': 4
        }
    },
    'minigrid-doorkey': {
        'environments': [
            'MiniGrid-DoorKey-5x5-v0',
            'MiniGrid-DoorKey-6x6-v0',
            'MiniGrid-DoorKey-8x8-v0',
            'MiniGrid-DoorKey-16x16-v0'
        ],
        'hyperparams': {
            'total_timesteps': 100000,
            'n_envs': 4
        }
    },
    'minigrid-hard': {
        'environments': [
            'MiniGrid-MultiRoom-N6-v0',
            'MiniGrid-KeyCorridorS6R3-v0',
            'MiniGrid-ObstructedMaze-Full-v0',
            'MiniGrid-LavaCrossingS11N5-v0'
        ],
        'hyperparams': {
            'total_timesteps': 500000,
            'n_envs': 8
        }
    },
    
    # Atari presets
    'atari-classic': {
        'environments': [
            'Breakout-v5',
            'Pong-v5',
            'SpaceInvaders-v5',
            'Qbert-v5'
        ],
        'hyperparams': {
            'total_timesteps': 1000000,
            'n_envs': 8
        }
    },
    
    # Gymnasium presets
    'gym-control': {
        'environments': [
            'CartPole-v1',
            'MountainCar-v0',
            'Acrobot-v1',
            'Pendulum-v1'
        ],
        'hyperparams': {
            'total_timesteps': 100000,
            'n_envs': 4
        }
    },
    'gym-mujoco': {
        'environments': [
            'HalfCheetah-v4',
            'Hopper-v4',
            'Walker2d-v4',
            'Ant-v4'
        ],
        'hyperparams': {
            'total_timesteps': 1000000,
            'n_envs': 4
        }
    }
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """Get preset configuration by name"""
    if preset_name not in ENVIRONMENT_PRESETS:
        available = ', '.join(ENVIRONMENT_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return ENVIRONMENT_PRESETS[preset_name]


def list_presets():
    """List all available presets"""
    print("Available environment presets:")
    print("=" * 50)
    for name, config in ENVIRONMENT_PRESETS.items():
        envs = config['environments']
        print(f"\n{name}:")
        print(f"  Environments: {', '.join(envs[:3])}...")
        print(f"  Total: {len(envs)} environments")
        print(f"  Timesteps: {config['hyperparams']['total_timesteps']:,}")