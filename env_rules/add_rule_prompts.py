#!/usr/bin/env python3
"""
Script to add rule prompts to parquet files by inserting them as the first item.

This script:
1. Reads the environment mapping from env_mapping.yaml
2. Loads rule prompts from the env_rules directory
3. Processes all parquet files in the input directory
4. Inserts the corresponding rule prompt as the first item in the 'inputs' column
5. Shifts all existing items down by one position
6. Saves the updated parquet files to the output directory
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
        content = f.read().strip()
        # Remove the title line if it exists (e.g., "MiniGrid-DistShift Environment Rule Description")
        lines = content.split('\n')
        if lines and 'Environment Rule Description' in lines[0]:
            # Keep only the actual rule content (skip title and empty line)
            content = '\n'.join(lines[2:]) if len(lines) > 2 else ''
        return content.strip()


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
        "text": f"Environment Rule:\n{rule_content}"
    }


def process_parquet_file(file_path: str, rule_content: str, output_path: str) -> bool:
    """Process a single parquet file and add the rule prompt as the first item."""
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
            
            # Create new rule prompt item
            new_rule_item = create_rule_prompt_item(rule_content)
            
            # Insert the rule prompt at the beginning, shifting everything else
            new_inputs_list = [new_rule_item] + inputs_list
            
            # Convert back to JSON string if it was originally a string
            if isinstance(df.loc[idx, 'inputs'], str):
                df.loc[idx, 'inputs'] = json.dumps(new_inputs_list)
            else:
                df.loc[idx, 'inputs'] = new_inputs_list
        
        # Save the updated dataframe
        df.to_parquet(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all parquet files."""
    # Define paths - update these as needed for your environment
    base_dir = Path("/opt/tiger")  # Update path as needed
    env_rules_dir = base_dir / "Minigrid/one_paragraph_rules"
    
    # Input and output directories - update these as needed
    parquet_dir = Path("/opt/tiger/minigirid_v0")  # UPDATE THIS PATH
    output_dir = Path("/opt/tiger/minigirid_v0_one_paragraph_rules")  # UPDATE THIS PATH
    
    mapping_file = base_dir / "Minigrid/env_rules/env_mapping.yaml"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            print(f"Loaded rule prompt: {rule_file.name} (length: {len(rule_content)} chars)")
        except Exception as e:
            print(f"Error loading rule file {rule_file}: {e}")
    
    print(f"Loaded {len(rule_prompts)} rule prompts")
    
    # Process all parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process")
    
    if len(parquet_files) == 0:
        print("No parquet files found. Please check the input directory path.")
        return
    
    successful = 0
    failed = 0
    skipped = 0
    
    for parquet_file in parquet_files:
        # Extract environment name from filename
        env_name = extract_env_name_from_filename(parquet_file.name)
        
        if not env_name:
            print(f"Warning: Could not extract environment name from {parquet_file.name}")
            skipped += 1
            continue
        
        # Find corresponding rule file
        if env_name not in env_mapping:
            print(f"Warning: No mapping found for environment {env_name}")
            skipped += 1
            continue
        
        rule_file_name = env_mapping[env_name]
        if rule_file_name not in rule_prompts:
            print(f"Warning: Rule file {rule_file_name} not loaded")
            skipped += 1
            continue
        
        # Process the file
        rule_content = rule_prompts[rule_file_name]
        output_path = output_dir / parquet_file.name
        
        print(f"Processing: {parquet_file.name}")
        # print(f"  Environment: {env_name}")
        # print(f"  Rule file: {rule_file_name}")
        
        if process_parquet_file(str(parquet_file), rule_content, str(output_path)):
            successful += 1
            print(f"  ✓ Successfully processed")
        else:
            failed += 1
            print(f"  ✗ Failed to process")
    
    print(f"\nProcessing completed:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Skipped: {skipped} files")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()