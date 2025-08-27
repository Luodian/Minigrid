#!/bin/bash

# Training script for all MiniGrid and BabyAI environments
# Generated automatically - Total: 177 environments

# Activate virtual environment
source .venv/bin/activate

# Set default training parameters
TIMESTEPS=${TIMESTEPS:-1000000}
SCRIPT="stable_baselines_agent.py"

# Create results directory if it doesn't exist
mkdir -p training_results

# List of all available environments
ENVS=(
    "BabyAI-ActionObjDoor-v0"
    "BabyAI-BlockedUnlockPickup-v0"
    "BabyAI-BossLevel-v0"
    "BabyAI-BossLevelNoUnlock-v0"
    "BabyAI-FindObjS5-v0"
    "BabyAI-FindObjS6-v0"
    "BabyAI-FindObjS7-v0"
    "BabyAI-GoTo-v0"
    "BabyAI-GoToDoor-v0"
    "BabyAI-GoToImpUnlock-v0"
    "BabyAI-GoToLocal-v0"
    "BabyAI-GoToLocalS5N2-v0"
    "BabyAI-GoToLocalS6N2-v0"
    "BabyAI-GoToLocalS6N3-v0"
    "BabyAI-GoToLocalS6N4-v0"
    "BabyAI-GoToLocalS7N4-v0"
    "BabyAI-GoToLocalS7N5-v0"
    "BabyAI-GoToLocalS8N2-v0"
    "BabyAI-GoToLocalS8N3-v0"
    "BabyAI-GoToLocalS8N4-v0"
    "BabyAI-GoToLocalS8N5-v0"
    "BabyAI-GoToLocalS8N6-v0"
    "BabyAI-GoToLocalS8N7-v0"
    "BabyAI-GoToObj-v0"
    "BabyAI-GoToObjDoor-v0"
    "BabyAI-GoToObjMaze-v0"
    "BabyAI-GoToObjMazeOpen-v0"
    "BabyAI-GoToObjMazeS4-v0"
    "BabyAI-GoToObjMazeS4R2-v0"
    "BabyAI-GoToObjMazeS5-v0"
    "BabyAI-GoToObjMazeS6-v0"
    "BabyAI-GoToObjMazeS7-v0"
    "BabyAI-GoToObjS4-v0"
    "BabyAI-GoToObjS6-v1"
    "BabyAI-GoToOpen-v0"
    "BabyAI-GoToRedBall-v0"
    "BabyAI-GoToRedBallGrey-v0"
    "BabyAI-GoToRedBallNoDists-v0"
    "BabyAI-GoToRedBlueBall-v0"
    "BabyAI-GoToSeq-v0"
    "BabyAI-GoToSeqS5R2-v0"
    "BabyAI-KeyCorridor-v0"
    "BabyAI-KeyCorridorS3R1-v0"
    "BabyAI-KeyCorridorS3R2-v0"
    "BabyAI-KeyCorridorS3R3-v0"
    "BabyAI-KeyCorridorS4R3-v0"
    "BabyAI-KeyCorridorS5R3-v0"
    "BabyAI-KeyCorridorS6R3-v0"
    "BabyAI-KeyInBox-v0"
    "BabyAI-MiniBossLevel-v0"
    "BabyAI-MoveTwoAcrossS5N2-v0"
    "BabyAI-MoveTwoAcrossS8N9-v0"
    "BabyAI-OneRoomS12-v0"
    "BabyAI-OneRoomS16-v0"
    "BabyAI-OneRoomS20-v0"
    "BabyAI-OneRoomS8-v0"
    "BabyAI-Open-v0"
    "BabyAI-OpenDoor-v0"
    "BabyAI-OpenDoorColor-v0"
    "BabyAI-OpenDoorDebug-v0"
    "BabyAI-OpenDoorLoc-v0"
    "BabyAI-OpenDoorsOrderN2-v0"
    "BabyAI-OpenDoorsOrderN2Debug-v0"
    "BabyAI-OpenDoorsOrderN4-v0"
    "BabyAI-OpenDoorsOrderN4Debug-v0"
    "BabyAI-OpenRedBlueDoors-v0"
    "BabyAI-OpenRedBlueDoorsDebug-v0"
    "BabyAI-OpenRedDoor-v0"
    "BabyAI-OpenTwoDoors-v0"
    "BabyAI-Pickup-v0"
    "BabyAI-PickupAbove-v0"
    "BabyAI-PickupDist-v0"
    "BabyAI-PickupDistDebug-v0"
    "BabyAI-PickupLoc-v0"
    "BabyAI-PutNextLocal-v0"
    "BabyAI-PutNextLocalS5N3-v0"
    "BabyAI-PutNextLocalS6N4-v0"
    "BabyAI-PutNextS4N1-v0"
    "BabyAI-PutNextS5N1-v0"
    "BabyAI-PutNextS5N2-v0"
    "BabyAI-PutNextS5N2Carrying-v0"
    "BabyAI-PutNextS6N3-v0"
    "BabyAI-PutNextS6N3Carrying-v0"
    "BabyAI-PutNextS7N4-v0"
    "BabyAI-PutNextS7N4Carrying-v0"
    "BabyAI-Synth-v0"
    "BabyAI-SynthLoc-v0"
    "BabyAI-SynthS5R2-v0"
    "BabyAI-SynthSeq-v0"
    "BabyAI-UnblockPickup-v0"
    "BabyAI-Unlock-v0"
    "BabyAI-UnlockLocal-v0"
    "BabyAI-UnlockLocalDist-v0"
    "BabyAI-UnlockPickup-v0"
    "BabyAI-UnlockPickupDist-v0"
    "BabyAI-UnlockToUnlock-v0"
    "MiniGrid-BlockedUnlockPickup-v0"
    "MiniGrid-DistShift1-v0"
    "MiniGrid-DistShift2-v0"
    "MiniGrid-DoorKey-16x16-v0"
    "MiniGrid-DoorKey-5x5-v0"
    "MiniGrid-DoorKey-6x6-v0"
    "MiniGrid-DoorKey-8x8-v0"
    "MiniGrid-Dynamic-Obstacles-16x16-v0"
    "MiniGrid-Dynamic-Obstacles-5x5-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-8x8-v0"
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
    "MiniGrid-Empty-16x16-v0"
    "MiniGrid-Empty-5x5-v0"
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-5x5-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Fetch-5x5-N2-v0"
    "MiniGrid-Fetch-6x6-N2-v0"
    "MiniGrid-Fetch-8x8-N3-v0"
    "MiniGrid-FourRooms-v0"
    "MiniGrid-GoToDoor-5x5-v0"
    "MiniGrid-GoToDoor-6x6-v0"
    "MiniGrid-GoToDoor-8x8-v0"
    "MiniGrid-GoToObject-6x6-N2-v0"
    "MiniGrid-GoToObject-8x8-N2-v0"
    "MiniGrid-KeyCorridorS3R1-v0"
    "MiniGrid-KeyCorridorS3R2-v0"
    "MiniGrid-KeyCorridorS3R3-v0"
    "MiniGrid-KeyCorridorS4R3-v0"
    "MiniGrid-KeyCorridorS5R3-v0"
    "MiniGrid-KeyCorridorS6R3-v0"
    "MiniGrid-LavaCrossingS11N5-v0"
    "MiniGrid-LavaCrossingS9N1-v0"
    "MiniGrid-LavaCrossingS9N2-v0"
    "MiniGrid-LavaCrossingS9N3-v0"
    "MiniGrid-LavaGapS5-v0"
    "MiniGrid-LavaGapS6-v0"
    "MiniGrid-LavaGapS7-v0"
    "MiniGrid-LockedRoom-v0"
    "MiniGrid-MemoryS11-v0"
    "MiniGrid-MemoryS13-v0"
    "MiniGrid-MemoryS13Random-v0"
    "MiniGrid-MemoryS17Random-v0"
    "MiniGrid-MemoryS7-v0"
    "MiniGrid-MemoryS9-v0"
    "MiniGrid-MultiRoom-N2-S4-v0"
    "MiniGrid-MultiRoom-N4-S5-v0"
    "MiniGrid-MultiRoom-N6-v0"
    "MiniGrid-ObstructedMaze-1Dl-v0"
    "MiniGrid-ObstructedMaze-1Dlh-v0"
    "MiniGrid-ObstructedMaze-1Dlhb-v0"
    "MiniGrid-ObstructedMaze-1Q-v0"
    "MiniGrid-ObstructedMaze-1Q-v1"
    "MiniGrid-ObstructedMaze-2Dl-v0"
    "MiniGrid-ObstructedMaze-2Dlh-v0"
    "MiniGrid-ObstructedMaze-2Dlhb-v0"
    "MiniGrid-ObstructedMaze-2Dlhb-v1"
    "MiniGrid-ObstructedMaze-2Q-v0"
    "MiniGrid-ObstructedMaze-2Q-v1"
    "MiniGrid-ObstructedMaze-Full-v0"
    "MiniGrid-ObstructedMaze-Full-v1"
    "MiniGrid-Playground-v0"
    "MiniGrid-PutNear-6x6-N2-v0"
    "MiniGrid-PutNear-8x8-N3-v0"
    "MiniGrid-RedBlueDoors-6x6-v0"
    "MiniGrid-RedBlueDoors-8x8-v0"
    "MiniGrid-SimpleCrossingS11N5-v0"
    "MiniGrid-SimpleCrossingS9N1-v0"
    "MiniGrid-SimpleCrossingS9N2-v0"
    "MiniGrid-SimpleCrossingS9N3-v0"
    "MiniGrid-Unlock-v0"
    "MiniGrid-UnlockPickup-v0"
    "MiniGrid-WFC-DungeonMazeScaled-v0"
    "MiniGrid-WFC-MazeSimple-v0"
    "MiniGrid-WFC-ObstaclesAngular-v0"
    "MiniGrid-WFC-ObstaclesBlackdots-v0"
    "MiniGrid-WFC-ObstaclesHogs3-v0"
    "MiniGrid-WFC-RoomsFabric-v0"
)

# Function to train a single environment
train_env() {
    local env=$1
    echo "Training on environment: $env"
    echo "Command: python $SCRIPT --env $env --timesteps $TIMESTEPS"
    
    # Uncomment the line below to actually run the training
    # python $SCRIPT --env "$env" --timesteps $TIMESTEPS
    
    # For batch training with logging:
    # python $SCRIPT --env "$env" --timesteps $TIMESTEPS > "training_results/${env}.log" 2>&1
}

# Display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -l, --list           List all available environments"
    echo "  -e, --env ENV        Train a specific environment"
    echo "  -t, --timesteps N    Set number of timesteps (default: 1000000)"
    echo "  -a, --all            Train all environments (WARNING: This will take a long time!)"
    echo "  -b, --batch FILE     Train environments listed in FILE (one per line)"
    echo "  --dry-run            Print commands without executing them"
    echo ""
    echo "Examples:"
    echo "  $0 -e MiniGrid-DoorKey-8x8-v0           # Train single environment"
    echo "  $0 -e MiniGrid-Empty-5x5-v0 -t 500000   # Train with custom timesteps"
    echo "  $0 -l                                    # List all environments"
    echo "  $0 --dry-run -a                          # Show all training commands"
    echo ""
    echo "Total available environments: ${#ENVS[@]}"
}

# Parse command line arguments
DRY_RUN=false
TRAIN_ALL=false
SPECIFIC_ENV=""
BATCH_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -l|--list)
            echo "Available environments (${#ENVS[@]} total):"
            echo "======================================="
            for env in "${ENVS[@]}"; do
                echo "$env"
            done
            exit 0
            ;;
        -e|--env)
            SPECIFIC_ENV="$2"
            shift 2
            ;;
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -a|--all)
            TRAIN_ALL=true
            shift
            ;;
        -b|--batch)
            BATCH_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution logic
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands will be printed but not executed"
    echo "========================================================="
fi

if [ -n "$SPECIFIC_ENV" ]; then
    # Train specific environment
    if [[ " ${ENVS[@]} " =~ " ${SPECIFIC_ENV} " ]]; then
        if [ "$DRY_RUN" = true ]; then
            echo "python $SCRIPT --env $SPECIFIC_ENV --timesteps $TIMESTEPS"
        else
            train_env "$SPECIFIC_ENV"
        fi
    else
        echo "Error: Environment '$SPECIFIC_ENV' not found"
        echo "Use -l to list available environments"
        exit 1
    fi
elif [ -n "$BATCH_FILE" ]; then
    # Train environments from file
    if [ ! -f "$BATCH_FILE" ]; then
        echo "Error: File '$BATCH_FILE' not found"
        exit 1
    fi
    
    while IFS= read -r env; do
        if [[ " ${ENVS[@]} " =~ " ${env} " ]]; then
            if [ "$DRY_RUN" = true ]; then
                echo "python $SCRIPT --env $env --timesteps $TIMESTEPS"
            else
                train_env "$env"
            fi
        else
            echo "Warning: Skipping unknown environment '$env'"
        fi
    done < "$BATCH_FILE"
elif [ "$TRAIN_ALL" = true ]; then
    # Train all environments
    echo "WARNING: Training all ${#ENVS[@]} environments will take a VERY long time!"
    if [ "$DRY_RUN" = false ]; then
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Aborted."
            exit 0
        fi
    fi
    
    for env in "${ENVS[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            echo "python $SCRIPT --env $env --timesteps $TIMESTEPS"
        else
            train_env "$env"
        fi
    done
else
    # No specific action requested
    usage
fi

echo ""
echo "Script completed!"