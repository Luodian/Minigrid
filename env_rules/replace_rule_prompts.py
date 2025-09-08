#!/usr/bin/env python3
"""
Script to replace the first item in parquet files with corresponding rule prompts.

This script:
1. Reads the environment mapping from env_mapping.yaml
2. Loads rule prompts from the env_rules directory
3. Processes all parquet files in minigirid_v0_with_rules directory
4. Replaces the first item in the 'inputs' column with the corresponding rule prompt
5. Saves the updated parquet files
"""

import os
import json
import yaml
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional


def load_environment_mapping(mapping_file: str) -> Dict[str, str]:
    """Load the environment mapping from YAML file."""
    with open(mapping_file, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('reverse_mapping', {})


def load_rule_prompt(rule_file: str) -> str:
    """Load the rule prompt content from a rule file."""
    with open(rule_file, 'r') as f:
        return f.read().strip()


def extract_env_name_from_filename(filename: str) -> str:
    """Extract environment name from parquet filename."""
    # Example: BabyAI-GoToLocalS6N3-v0_rand0.2.w0.size1000_1756566136307_3701674_03e944be.parquet
    # Extract: BabyAI-GoToLocalS6N3-v0
    parts = filename.split('_')
    if len(parts) > 0:
        return parts[0]
    return ""


def create_rule_prompt_item(rule_content: str) -> Dict:
    """Create a rule prompt item in the required format."""
    return {
        "type": "text",
        "has_loss": 0,
        "text": rule_content
    }


def process_parquet_file(file_path: str, rule_content: str, output_path: str) -> bool:
    """Process a single parquet file and replace the first item."""
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        if 'inputs' not in df.columns:
            print(f"Warning: No 'inputs' column found in {file_path}")
            return False
        
        # Process each row
        for idx in range(len(df)):
            # Parse the existing inputs (which is stored as a JSON string)
            inputs = df.loc[idx, 'inputs']
            if isinstance(inputs, str):
                inputs_list = json.loads(inputs)
            else:
                inputs_list = inputs
            
            # Replace the first item if it exists
            if len(inputs_list) > 0:
                # Create new rule prompt item
                new_first_item = create_rule_prompt_item(rule_content)
                inputs_list[0] = new_first_item
                
                # Convert back to JSON string if it was originally a string
                if isinstance(df.loc[idx, 'inputs'], str):
                    df.loc[idx, 'inputs'] = json.dumps(inputs_list)
                else:
                    df.loc[idx, 'inputs'] = inputs_list
        
        # Save the updated dataframe
        df.to_parquet(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all parquet files."""
    # Define paths
    base_dir = Path("/home/tiger/Minigrid")
    env_rules_dir = base_dir / "env_rules"
    parquet_dir = "/home/tiger/minigirid_v0_with_rules"
    mapping_file = env_rules_dir / "env_mapping.yaml"
    
    # Create output directory
    output_dir = "/home/tiger/minigirid_v0_shorter_rules"
    output_dir.mkdir(exist_ok=True)
    
    print("Starting parquet file processing...")
    print(f"Input directory: {parquet_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Environment rules directory: {env_rules_dir}")
    
    # Load environment mapping
    try:
        env_mapping = load_environment_mapping(mapping_file)
        print(f"Loaded mapping for {len(env_mapping)} environments")
    except Exception as e:
        print(f"Error loading environment mapping: {e}")
        return
    
    # Load all rule prompts
    rule_prompts = {}
    for rule_file in env_rules_dir.glob("*_rule.txt"):
        try:
            rule_content = load_rule_prompt(rule_file)
            rule_prompts[rule_file.name] = rule_content
            print(f"Loaded rule prompt: {rule_file.name}")
        except Exception as e:
            print(f"Error loading rule file {rule_file}: {e}")
    
    print(f"Loaded {len(rule_prompts)} rule prompts")
    
    # Process all parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process")
    
    successful = 0
    failed = 0
    
    for parquet_file in parquet_files:
        # Extract environment name from filename
        env_name = extract_env_name_from_filename(parquet_file.name)
        
        if not env_name:
            print(f"Warning: Could not extract environment name from {parquet_file.name}")
            failed += 1
            continue
        
        # Find corresponding rule file
        if env_name not in env_mapping:
            print(f"Warning: No mapping found for environment {env_name}")
            failed += 1
            continue
        
        rule_file_name = env_mapping[env_name]
        if rule_file_name not in rule_prompts:
            print(f"Warning: Rule file {rule_file_name} not loaded")
            failed += 1
            continue
        
        # Process the file
        rule_content = rule_prompts[rule_file_name]
        output_path = output_dir / parquet_file.name
        
        print(f"Processing: {parquet_file.name} -> {env_name} -> {rule_file_name}")
        
        if process_parquet_file(str(parquet_file), rule_content, str(output_path)):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing completed:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
