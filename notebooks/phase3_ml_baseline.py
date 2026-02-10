#!/usr/bin/env python3
"""
Phase 3: Traditional ML Baseline
Test Logistic Regression, Naive Bayes, and Random Forest on both datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import json

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def calculate_metrics(y_true, y_pred, processing_time, n_samples):
    """Calculate all metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    emails_per_second = n_samples / processing_time if processing_time > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(emails_per_second, 2),
        "total_time": round(processing_time, 2)
    }

def test_ml_models(dataset_name, dataset_path):
    """Test all ML models on a dataset"""
    print(f"\n{'='*60}")
    print(f"TESTING: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Phishing: {(df['label'] == 1).sum()}")
    print(f"   Legitimate: {(df['label'] == 0).sum()}")
    
    # Prepare data
    X = df['text'].fillna("")
    y = df['label']
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Vectorize text
    print("\n3. Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   Feature dimensions: {X_train_vec.shape[1]}")
    
    # Models to test
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # Test each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        # Train
        print("Training...")
        train_start = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - train_start
        print(f"✓ Training completed in {train_time:.2f}s")
        
        # Predict
        print("Predicting...")
        pred_start = time.time()
        y_pred = model.predict(X_test_vec)
        pred_time = time.time() - pred_start
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, pred_time, len(X_test))
        
        # Display results
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.1f}%)")
        print(f"  Speed:     {metrics['emails_per_second']:.2f} emails/second")
        print(f"  Test Time: {metrics['total_time']:.2f}s")
        
        results[model_name] = {
            **metrics,
            "train_time": round(train_time, 2),
            "test_samples": len(X_test)
        }
    
    return results

def main():
    """Main function to test all models on all datasets"""
    print("="*60)
    print("PHASE 3: TRADITIONAL ML BASELINE")
    print("="*60)
    
    # Datasets to test
    datasets = {
        "Enron (3k)": RESULTS_DIR / "enron_preprocessed_3k.csv",
        "Combined (2k)": RESULTS_DIR / "combined_preprocessed_2k.csv"
    }
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, dataset_path in datasets.items():
        results = test_ml_models(dataset_name, dataset_path)
        all_results[dataset_name] = results
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: ALL RESULTS")
    print(f"{'='*60}\n")
    
    for dataset_name, models_results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 60)
        print(f"{'Model':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed':<12}")
        print("-" * 60)
        
        for model_name, metrics in models_results.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<8.4f} "
                  f"{metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics['emails_per_second']:<12.2f}")
    
    # Save results
    output_file = RESULTS_DIR / "phase3_ml_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ PHASE 3 COMPLETE!")
    print("Traditional ML baselines established for both datasets.")

if __name__ == "__main__":
    main()
