#!/usr/bin/env python3
"""
Script to filter environments with high reward agents from training results.
Searches for environments where the average reward is at least 0.8.
"""

import os
import re
import glob
from typing import List, Tuple, Optional


def extract_avg_reward(log_file: str) -> Optional[float]:
    """
    Extract the average reward from a training log file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Average reward as float, or None if not found
    """
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Pattern to match "Average Reward: X.XX ± Y.YY"
        pattern = r'Average Reward:\s+([\d.]+)\s*±'
        matches = re.findall(pattern, content)
        
        if matches:
            # Return the last (most recent) average reward found
            return float(matches[-1])
        return None
        
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def extract_env_name(log_file: str) -> str:
    """
    Extract environment name from log file path.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Environment name without .log extension
    """
    return os.path.basename(log_file).replace('.log', '')


def find_high_reward_environments(results_dir: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find all environments with average reward >= threshold.
    
    Args:
        results_dir: Directory containing training result log files
        threshold: Minimum average reward threshold (default: 0.8)
        
    Returns:
        List of tuples (environment_name, average_reward) for high reward environments
    """
    high_reward_envs = []
    
    # Find all .log files in the results directory
    log_files = glob.glob(os.path.join(results_dir, "*.log"))
    
    print(f"Analyzing {len(log_files)} log files...")
    print(f"Filtering environments with average reward >= {threshold}")
    print("-" * 60)
    
    for log_file in sorted(log_files):
        env_name = extract_env_name(log_file)
        avg_reward = extract_avg_reward(log_file)
        
        if avg_reward is not None:
            if avg_reward >= threshold:
                high_reward_envs.append((env_name, avg_reward))
                print(f"✓ {env_name}: {avg_reward:.3f}")
            else:
                print(f"  {env_name}: {avg_reward:.3f}")
        else:
            print(f"? {env_name}: No statistics found")
    
    return high_reward_envs


def main():
    """Main function to run the analysis."""
    results_dir = "./training_results"
    threshold = 0.8
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found!")
        return
    
    print("High Reward Environment Finder")
    print("=" * 60)
    
    high_reward_envs = find_high_reward_environments(results_dir, threshold)
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: Found {len(high_reward_envs)} environments with average reward >= {threshold}")
    print("=" * 60)
    
    if high_reward_envs:
        print("\nHigh reward environments:")
        for i, (env_name, avg_reward) in enumerate(high_reward_envs, 1):
            print(f"{i:2d}. {env_name} (avg reward: {avg_reward:.3f})")
        
        # Save to file
        output_file = "high_reward_environments.txt"
        with open(output_file, 'w') as f:
            f.write(f"Environments with average reward >= {threshold}:\n")
            f.write("=" * 50 + "\n\n")
            for i, (env_name, avg_reward) in enumerate(high_reward_envs, 1):
                f.write(f"{i:2d}. {env_name} (avg reward: {avg_reward:.3f})\n")
        
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"\nNo environments found with average reward >= {threshold}")


if __name__ == "__main__":
    main()