#!/usr/bin/env python3
"""
Phase 2: Preprocessing Enron Dataset
Format: Message ID, Subject, Message, Spam/Ham, Date
Goal: Create balanced 3k sample (1.5k phishing + 1.5k legitimate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
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

def preprocess_enron():
    """Preprocess Enron dataset"""
    print("="*60)
    print("PHASE 2: PREPROCESSING ENRON DATASET")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(DATA_DIR / "enron.csv")
    print(f"   Total emails: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check label distribution
    print("\n2. Analyzing labels...")
    print(f"   Label column: 'Spam/Ham'")
    print(f"   Value counts:")
    print(df['Spam/Ham'].value_counts())
    
    # Convert labels to binary
    print("\n3. Converting labels...")
    df['label'] = df['Spam/Ham'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
    print(f"   Phishing (spam): {(df['label'] == 1).sum()}")
    print(f"   Legitimate (ham): {(df['label'] == 0).sum()}")
    
    # Clean text fields
    print("\n4. Cleaning text...")
    df['subject_clean'] = df['Subject'].apply(clean_text)
    df['message_clean'] = df['Message'].apply(clean_text)
    
    # Combine subject and message
    df['full_text'] = df['subject_clean'] + " " + df['message_clean']
    
    # Remove empty emails
    df = df[df['full_text'].str.len() > 10]
    print(f"   Emails after removing empty: {len(df)}")
    
    # Create balanced sample
    print("\n5. Creating balanced 3k sample...")
    phishing_df = df[df['label'] == 1]
    legit_df = df[df['label'] == 0]
    
    print(f"   Available phishing: {len(phishing_df)}")
    print(f"   Available legitimate: {len(legit_df)}")
    
    # Sample 1.5k from each
    n_samples = 1500
    
    if len(phishing_df) >= n_samples and len(legit_df) >= n_samples:
        phishing_sample = phishing_df.sample(n=n_samples, random_state=42)
        legit_sample = legit_df.sample(n=n_samples, random_state=42)
        
        balanced_df = pd.concat([phishing_sample, legit_sample])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✓ Created balanced sample: {len(balanced_df)} emails")
        print(f"   Phishing: {(balanced_df['label'] == 1).sum()}")
        print(f"   Legitimate: {(balanced_df['label'] == 0).sum()}")
    else:
        print(f"   ✗ Not enough samples!")
        print(f"   Need {n_samples} of each, have {len(phishing_df)} phishing and {len(legit_df)} legitimate")
        return
    
    # Create final dataset
    print("\n6. Creating final dataset...")
    final_df = balanced_df[['full_text', 'subject_clean', 'message_clean', 'Date', 'label']].copy()
    final_df.columns = ['text', 'subject', 'message', 'date', 'label']
    
    # Save
    output_file = RESULTS_DIR / "enron_preprocessed_3k.csv"
    final_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved to: {output_file}")
    
    # Statistics
    print("\n7. Final Statistics:")
    print(f"   Total samples: {len(final_df)}")
    print(f"   Phishing: {(final_df['label'] == 1).sum()} ({(final_df['label'] == 1).sum()/len(final_df)*100:.1f}%)")
    print(f"   Legitimate: {(final_df['label'] == 0).sum()} ({(final_df['label'] == 0).sum()/len(final_df)*100:.1f}%)")
    print(f"   Avg text length: {final_df['text'].str.len().mean():.0f} chars")
    print(f"   Min text length: {final_df['text'].str.len().min()} chars")
    print(f"   Max text length: {final_df['text'].str.len().max()} chars")
    
    print("\n" + "="*60)
    print("ENRON PREPROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    preprocess_enron()
