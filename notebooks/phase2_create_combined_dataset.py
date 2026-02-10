#!/usr/bin/env python3
"""
Phase 2: Create Combined Balanced Dataset
Combine legit.csv and phishing.csv to create a balanced 2k dataset (1k each)
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def create_combined_dataset():
    """Create combined balanced dataset from legit and phishing"""
    print("="*60)
    print("PHASE 2: CREATING COMBINED BALANCED DATASET")
    print("="*60)
    
    # Load preprocessed datasets
    print("\n1. Loading preprocessed datasets...")
    legit_df = pd.read_csv(RESULTS_DIR / "legit_preprocessed_1.5k.csv")
    phishing_df = pd.read_csv(RESULTS_DIR / "phishing_preprocessed_1.5k.csv")
    
    print(f"   Legitimate emails: {len(legit_df)}")
    print(f"   Phishing emails: {len(phishing_df)}")
    
    # Take equal samples
    print("\n2. Creating balanced sample...")
    n_samples = min(len(legit_df), len(phishing_df))
    print(f"   Using {n_samples} samples from each class")
    
    legit_sample = legit_df.sample(n=n_samples, random_state=42)
    phishing_sample = phishing_df.sample(n=n_samples, random_state=42)
    
    # Combine
    combined_df = pd.concat([legit_sample, phishing_sample])
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ✓ Created balanced dataset: {len(combined_df)} emails")
    print(f"   Phishing: {(combined_df['label'] == 1).sum()}")
    print(f"   Legitimate: {(combined_df['label'] == 0).sum()}")
    
    # Save
    output_file = RESULTS_DIR / "combined_preprocessed_2k.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\n3. Saved to: {output_file}")
    
    # Statistics
    print("\n4. Final Statistics:")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Phishing: {(combined_df['label'] == 1).sum()} ({(combined_df['label'] == 1).sum()/len(combined_df)*100:.1f}%)")
    print(f"   Legitimate: {(combined_df['label'] == 0).sum()} ({(combined_df['label'] == 0).sum()/len(combined_df)*100:.1f}%)")
    print(f"   Avg text length: {combined_df['text'].str.len().mean():.0f} chars")
    
    print("\n" + "="*60)
    print("COMBINED DATASET CREATION COMPLETE!")
    print("="*60)
    
    print("\n📊 SUMMARY OF PREPROCESSED DATASETS:")
    print("-" * 60)
    print(f"1. Enron Dataset:    3,000 emails (1,500 phishing + 1,500 legit)")
    print(f"2. Combined Dataset: 2,000 emails (1,000 phishing + 1,000 legit)")
    print("-" * 60)
    print("\nWe'll use these two datasets for all subsequent phases.")

if __name__ == "__main__":
    create_combined_dataset()
