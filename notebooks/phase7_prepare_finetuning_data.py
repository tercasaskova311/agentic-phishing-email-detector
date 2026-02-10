#!/usr/bin/env python3
"""
Phase 7: Prepare Data for Fine-tuning
Convert datasets to instruction format for LLM fine-tuning
"""

import pandas as pd
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def create_instruction_prompt(email_text: str, label: int) -> dict:
    """Create instruction-following format for fine-tuning"""
    
    # Truncate email
    email = email_text[:600]
    
    # Create instruction
    instruction = "Classify this email as either PHISHING or LEGITIMATE."
    
    # Create input
    input_text = f"Email: {email}"
    
    # Create output
    output = "PHISHING" if label == 1 else "LEGITIMATE"
    
    # Format for Unsloth (Alpaca format)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

def prepare_dataset(dataset_name: str, dataset_path: Path, train_size: int = 2000, test_size: int = 500):
    """Prepare train/test splits for fine-tuning"""
    print(f"\n{'='*60}")
    print(f"Preparing: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")
    print(f"  Phishing: {(df['label'] == 1).sum()}")
    print(f"  Legitimate: {(df['label'] == 0).sum()}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split train/test
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:train_size+test_size]
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"  Phishing: {(train_df['label'] == 1).sum()}")
    print(f"  Legitimate: {(train_df['label'] == 0).sum()}")
    
    print(f"\nTest set: {len(test_df)} samples")
    print(f"  Phishing: {(test_df['label'] == 1).sum()}")
    print(f"  Legitimate: {(test_df['label'] == 0).sum()}")
    
    # Convert to instruction format
    print("\nConverting to instruction format...")
    train_data = [create_instruction_prompt(row['text'], row['label']) 
                  for _, row in train_df.iterrows()]
    test_data = [create_instruction_prompt(row['text'], row['label']) 
                 for _, row in test_df.iterrows()]
    
    return train_data, test_data

def main():
    print("="*60)
    print("PHASE 7: PREPARE FINE-TUNING DATA")
    print("="*60)
    
    # Prepare Enron dataset (larger, more diverse)
    enron_train, enron_test = prepare_dataset(
        "Enron Dataset",
        RESULTS_DIR / "enron_preprocessed_3k.csv",
        train_size=2400,  # 80% for training
        test_size=600     # 20% for testing
    )
    
    # Save in JSON format for Unsloth
    print(f"\n{'='*60}")
    print("Saving datasets...")
    print(f"{'='*60}")
    
    train_file = MODELS_DIR / "train_data.json"
    test_file = MODELS_DIR / "test_data.json"
    
    with open(train_file, 'w') as f:
        json.dump(enron_train, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(enron_test, f, indent=2)
    
    print(f"\n✓ Training data saved: {train_file}")
    print(f"  {len(enron_train)} samples")
    
    print(f"\n✓ Test data saved: {test_file}")
    print(f"  {len(enron_test)} samples")
    
    # Show example
    print(f"\n{'='*60}")
    print("Example Training Sample:")
    print(f"{'='*60}")
    example = enron_train[0]
    print(f"\nInstruction: {example['instruction']}")
    print(f"\nInput: {example['input'][:200]}...")
    print(f"\nOutput: {example['output']}")
    
    print(f"\n{'='*60}")
    print("✓ DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print("\nNext step: Run fine-tuning script with Unsloth")

if __name__ == "__main__":
    main()
