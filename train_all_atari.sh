#!/bin/bash

# Script to train all Atari environments using the existing train.py
# Automatically saves checkpoints when reward reaches 0.8

# List of all Atari environments (v5 versions)
ATARI_ENVS=(
    "ALE/Adventure-v5"
    "ALE/AirRaid-v5"
    "ALE/Alien-v5"
    "ALE/Amidar-v5"
    "ALE/Assault-v5"
    "ALE/Asterix-v5"
    "ALE/Asteroids-v5"
    "ALE/Atlantis-v5"
    "ALE/BankHeist-v5"
    "ALE/BattleZone-v5"
    "ALE/BeamRider-v5"
    "ALE/Berzerk-v5"
    "ALE/Bowling-v5"
    "ALE/Boxing-v5"
    "ALE/Breakout-v5"
    "ALE/Carnival-v5"
    "ALE/Centipede-v5"
    "ALE/ChopperCommand-v5"
    "ALE/CrazyClimber-v5"
    "ALE/Defender-v5"
    "ALE/DemonAttack-v5"
    "ALE/DoubleDunk-v5"
    "ALE/ElevatorAction-v5"
    "ALE/Enduro-v5"
    "ALE/FishingDerby-v5"
    "ALE/Freeway-v5"
    "ALE/Frostbite-v5"
    "ALE/Gopher-v5"
    "ALE/Gravitar-v5"
    "ALE/Hero-v5"
    "ALE/IceHockey-v5"
    "ALE/Jamesbond-v5"
    "ALE/JourneyEscape-v5"
    "ALE/Kangaroo-v5"
    "ALE/Krull-v5"
    "ALE/KungFuMaster-v5"
    "ALE/MontezumaRevenge-v5"
    "ALE/MsPacman-v5"
    "ALE/NameThisGame-v5"
    "ALE/Phoenix-v5"
    "ALE/Pitfall-v5"
    "ALE/Pong-v5"
    "ALE/Pooyan-v5"
    "ALE/PrivateEye-v5"
    "ALE/Qbert-v5"
    "ALE/Riverraid-v5"
    "ALE/RoadRunner-v5"
    "ALE/Robotank-v5"
    "ALE/Seaquest-v5"
    "ALE/Skiing-v5"
    "ALE/Solaris-v5"
    "ALE/SpaceInvaders-v5"
    "ALE/StarGunner-v5"
    "ALE/Tennis-v5"
    "ALE/TimePilot-v5"
    "ALE/Tutankham-v5"
    "ALE/UpNDown-v5"
    "ALE/Venture-v5"
    "ALE/VideoPinball-v5"
    "ALE/WizardOfWor-v5"
    "ALE/YarsRevenge-v5"
    "ALE/Zaxxon-v5"
)

# Default parameters (can be overridden with command line args)
TIMESTEPS=${1:-1000000}  # Default 1M timesteps
N_ENVS=${2:-8}           # Default 8 parallel environments
EVAL_FREQ=${3:-10000}    # Default eval every 10k steps
SAVE_FREQ=${4:-50000}    # Default save every 50k steps

# Create log directory for this training run
LOG_DIR="logs/atari_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="$LOG_DIR/training_summary.txt"

echo "=====================================" | tee "$SUMMARY_FILE"
echo "Starting Atari Training Run" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "Timesteps: $TIMESTEPS" | tee -a "$SUMMARY_FILE"
echo "Parallel envs: $N_ENVS" | tee -a "$SUMMARY_FILE"
echo "Eval frequency: $EVAL_FREQ" | tee -a "$SUMMARY_FILE"
echo "Save frequency: $SAVE_FREQ" | tee -a "$SUMMARY_FILE"
echo "Total environments: ${#ATARI_ENVS[@]}" | tee -a "$SUMMARY_FILE"
echo "Log directory: $LOG_DIR" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Track statistics
COMPLETED=0
FAILED=0
FAILED_ENVS=()

# Start time
START_TIME=$(date +%s)

# Train each environment
for i in "${!ATARI_ENVS[@]}"; do
    ENV="${ATARI_ENVS[$i]}"
    ENV_NUM=$((i + 1))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "[$ENV_NUM/${#ATARI_ENVS[@]}] Training: $ENV" | tee -a "$SUMMARY_FILE"
    echo "Progress: $COMPLETED completed, $FAILED failed" | tee -a "$SUMMARY_FILE"
    echo "-------------------------------------" | tee -a "$SUMMARY_FILE"
    
    # Create clean model name (replace / with _)
    MODEL_NAME="atari_${ENV//\//_}"
    
    # Check if model already exists (for resuming)
    if [ -f "models/${MODEL_NAME}_final.zip" ]; then
        echo "Model already exists, skipping..." | tee -a "$SUMMARY_FILE"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi
    
    # Run training
    LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"
    
    python train/train.py \
        --env "$ENV" \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --eval-freq "$EVAL_FREQ" \
        --save-freq "$SAVE_FREQ" \
        --reward-checkpoint-threshold 0.8 \
        --model-name "$MODEL_NAME" \
        --verbose 1 \
        2>&1 | tee "$LOG_FILE"
    
    # Check if training succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Successfully trained $ENV" | tee -a "$SUMMARY_FILE"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "✗ Failed to train $ENV" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        FAILED_ENVS+=("$ENV")
    fi
done

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Final summary
echo "" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "Training Complete!" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$SUMMARY_FILE"
echo "Completed: $COMPLETED / ${#ATARI_ENVS[@]}" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"

if [ ${#FAILED_ENVS[@]} -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Failed environments:" | tee -a "$SUMMARY_FILE"
    for ENV in "${FAILED_ENVS[@]}"; do
        echo "  - $ENV" | tee -a "$SUMMARY_FILE"
    done
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "Summary saved to: $SUMMARY_FILE"
echo "Logs saved to: $LOG_DIR"