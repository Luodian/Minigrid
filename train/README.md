# Unified PPO Training Framework

A simplified, unified training framework for PPO agents across MiniGrid, Atari, and Gymnasium environments.

## Overview

This framework consolidates multiple redundant training scripts into a single, modular system with:
- Automatic environment type detection
- Environment-specific configurations
- Unified video recording
- Preset environment groups for batch training

## Structure

```
train/
├── train.py              # Main CLI entry point
├── base_trainer.py       # Core training logic
├── env_configs.py        # Environment configurations
├── video_recorder.py     # Video recording utilities
└── README.md            # This file
```

## Quick Start

### Train a single environment
```bash
python train.py --env MiniGrid-DoorKey-8x8-v0 --timesteps 100000
```

### Train with presets
```bash
# Train easy MiniGrid environments
python train.py --preset minigrid-easy

# Train classic Atari games
python train.py --preset atari-classic

# List all presets
python train.py --list-presets
```

### Test a trained model
```bash
python train.py --test --env MiniGrid-DoorKey-8x8-v0 --episodes 5
```

### Record videos
```bash
# Basic recording
python train.py --record --env MiniGrid-DoorKey-8x8-v0 --episodes 3

# Dual view for MiniGrid (global + agent view)
python train.py --record --dual-view --env MiniGrid-Empty-8x8-v0

# Record with action logging
python train.py --record --env CartPole-v1 --save-actions
```

## Environment Types

The framework automatically detects and configures three environment types:

### 1. MiniGrid Environments
- Prefix: `MiniGrid-*` or `BabyAI-*`
- Policy: MlpPolicy (flattened observations)
- Special features: Dual-view recording

### 2. Atari Environments
- Prefix: `ALE/*` or classic game names
- Policy: CnnPolicy
- Special features: Frame stacking, Atari wrappers

### 3. Gymnasium Environments
- Everything else (CartPole, MuJoCo, etc.)
- Policy: Auto-detected (CNN for images, MLP for vectors)
- Special features: Automatic policy selection

## Available Presets

| Preset | Environments | Timesteps |
|--------|--------------|-----------|
| `minigrid-easy` | Empty environments (5x5, 6x6, 8x8) | 50,000 |
| `minigrid-doorkey` | DoorKey environments | 100,000 |
| `minigrid-hard` | Complex mazes and multi-room | 500,000 |
| `atari-classic` | Breakout, Pong, SpaceInvaders, Qbert | 1,000,000 |
| `gym-control` | CartPole, MountainCar, Acrobot | 100,000 |
| `gym-mujoco` | HalfCheetah, Hopper, Walker2d, Ant | 1,000,000 |

## Command Line Options

### Training Options
- `--timesteps`: Total training timesteps (default: 100,000)
- `--n-envs`: Number of parallel environments (default: 4)
- `--eval-freq`: Evaluation frequency (default: 5,000)
- `--save-freq`: Checkpoint save frequency (default: 10,000)

### Early Stopping
- `--patience`: Early stopping patience (default: 10)
- `--min-improvement`: Minimum improvement threshold (default: 0.01)
- `--reward-threshold`: Stop when reward threshold reached

### Hyperparameter Overrides
- `--learning-rate`: Override learning rate
- `--batch-size`: Override batch size
- `--n-steps`: Override steps per update
- `--gamma`: Override discount factor

### Recording Options
- `--episodes`: Number of episodes to record (default: 5)
- `--save-actions`: Save action logs with videos
- `--dual-view`: Record dual view for MiniGrid
- `--no-overlay`: Disable text overlay in videos

## Migration from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `stable_baselines_agent.py --env MiniGrid-Empty-8x8-v0` | `python train.py --env MiniGrid-Empty-8x8-v0` |
| `atari_ppo_agent.py --env Breakout-v5` | `python train.py --env Breakout-v5` |
| `gymnasium_ppo_agent.py --env CartPole-v1` | `python train.py --env CartPole-v1` |
| `train_all_envs.sh -a` | `python train.py --preset minigrid-easy` (or other presets) |

## Benefits of the New Structure

1. **Reduced Code Duplication**: ~80% less code by sharing common components
2. **Automatic Configuration**: Environment type detection and optimal hyperparameters
3. **Unified Interface**: Single command for all environment types
4. **Better Organization**: Modular design with clear separation of concerns
5. **Preset Support**: Easy batch training of related environments
6. **Consistent Features**: All environments get video recording, early stopping, etc.

## Examples

### Complete Training Pipeline
```bash
# Train a model
python train.py --env MiniGrid-DoorKey-8x8-v0 --timesteps 200000

# Test the trained model
python train.py --test --env MiniGrid-DoorKey-8x8-v0 --episodes 10

# Record videos of the trained agent
python train.py --record --env MiniGrid-DoorKey-8x8-v0 --episodes 5 --save-actions
```

### Custom Hyperparameters
```bash
python train.py --env CartPole-v1 \
    --learning-rate 0.001 \
    --batch-size 128 \
    --gamma 0.95 \
    --timesteps 50000
```

### Batch Training with Early Stopping
```bash
python train.py --preset minigrid-doorkey \
    --patience 15 \
    --min-improvement 0.05 \
    --reward-threshold 0.95
```

## Old Files (Can be Removed)

The following files are now redundant and can be safely removed:
- `atari_ppo_agent.py` - Replaced by unified trainer
- `gymnasium_ppo_agent.py` - Replaced by unified trainer  
- `stable_baselines_agent.py` - Replaced by unified trainer
- `train_all_envs.sh` - Replaced by preset functionality
- `train_rl_agent.py` - Basic implementation, superseded by Stable Baselines3

Keep them for reference or remove them to complete the cleanup.