#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "first_datasets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def clean_text(text):
    """Clean email text"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def preprocess_legit():
    """Preprocess Legit dataset"""
    print("="*60)
    print("PHASE 2: PREPROCESSING LEGIT DATASET")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(DATA_DIR / "legit.csv")
    print(f"   Total emails: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check label distribution
    print("\n2. Analyzing labels...")
    print(f"   Label column: 'label'")
    print(f"   Value counts:")
    print(df['label'].value_counts())
    print(f"   Note: This dataset should contain only legitimate emails (label=0)")
    
    # Verify all are legitimate
    if (df['label'] == 0).all():
        print(f"   ✓ Confirmed: All {len(df)} emails are legitimate")
    else:
        print(f"   ⚠ Warning: Found {(df['label'] != 0).sum()} non-legitimate emails")
        df = df[df['label'] == 0]
        print(f"   Filtered to {len(df)} legitimate emails")
    
    # Clean text fields
    print("\n3. Cleaning text...")
    df['subject_clean'] = df['subject'].apply(clean_text)
    df['body_clean'] = df['body'].apply(clean_text)
    df['sender_clean'] = df['sender'].apply(clean_text)
    
    # Combine fields for full text
    df['full_text'] = (df['sender_clean'] + " " + 
                       df['subject_clean'] + " " + 
                       df['body_clean'])
    
    # Remove empty emails
    df = df[df['full_text'].str.len() > 10]
    print(f"   Emails after removing empty: {len(df)}")
    
    # Sample 1.5k legitimate emails
    print("\n4. Sampling 1.5k legitimate emails...")
    n_samples = 1500
    
    if len(df) >= n_samples:
        sample_df = df.sample(n=n_samples, random_state=42)
        print(f"   ✓ Sampled {len(sample_df)} legitimate emails")
    else:
        print(f"   ⚠ Only {len(df)} emails available (need {n_samples})")
        sample_df = df
    
    # Create final dataset
    print("\n5. Creating final dataset...")
    final_df = sample_df[['full_text', 'subject_clean', 'body_clean', 'sender_clean', 'date', 'label']].copy()
    final_df.columns = ['text', 'subject', 'message', 'sender', 'date', 'label']
    
    # Save
    output_file = RESULTS_DIR / "legit_preprocessed_1.5k.csv"
    final_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved to: {output_file}")
    
    # Statistics
    print("\n6. Final Statistics:")
    print(f"   Total samples: {len(final_df)}")
    print(f"   All legitimate: {(final_df['label'] == 0).all()}")
    print(f"   Avg text length: {final_df['text'].str.len().mean():.0f} chars")
    print(f"   Min text length: {final_df['text'].str.len().min()} chars")
    print(f"   Max text length: {final_df['text'].str.len().max()} chars")
    
    print("\n" + "="*60)
    print("LEGIT PREPROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    preprocess_legit()
