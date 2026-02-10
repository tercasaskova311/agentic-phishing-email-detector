#!/usr/bin/env python3
"""
Phase 4: Single LLM Evaluation (Quick Test)
Test each LLM on small samples (20 emails each) to get initial results
"""

import pandas as pd
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ollama

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# LLM Models
MODELS = {
    "qwen2.5:3b-instruct": "Qwen2.5-3B",
    "llama3.2:latest": "Llama-3.2-3B",
    "gemma:2b": "Gemma-2B"
}

def classify_email(model: str, email_text: str) -> dict:
    """Classify email using LLM"""
    try:
        start = time.time()
        response = ollama.generate(
            model=model,
            prompt=f"Is this email PHISHING or LEGITIMATE? Answer with one word only.\n\nEmail: {email_text[:300]}",
            options={'temperature': 0.1, 'num_predict': 5}
        )
        elapsed = time.time() - start
        
        result = response['response'].upper()
        classification = "PHISHING" if "PHISHING" in result else "LEGITIMATE"
        
        return {"classification": classification, "time": elapsed, "success": True}
    except Exception as e:
        return {"classification": "LEGITIMATE", "time": 0, "success": False}

def test_model(model_key, model_name, df, n_samples=20):
    """Test model on dataset"""
    print(f"\n{'='*50}")
    print(f"{model_name} on {len(df)} emails")
    print(f"{'='*50}")
    
    # Sample
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    predictions = []
    times = []
    
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"  {i}/{len(sample)}...", end='', flush=True)
        result = classify_email(model_key, row['text'])
        predictions.append(result["classification"])
        times.append(result["time"])
        print(f" {result['classification']} ({result['time']:.1f}s)")
    
    # Metrics
    y_true = [1 if label == 1 else 0 for label in sample['label']]
    y_pred = [1 if p == "PHISHING" else 0 for p in predictions]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    speed = 1.0 / (sum(times) / len(times)) if times else 0
    
    print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Speed:     {speed:.3f} emails/s")
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(speed, 3),
        "avg_time": round(sum(times)/len(times), 2)
    }

def main():
    print("="*50)
    print("PHASE 4: SINGLE LLM EVALUATION (QUICK)")
    print("="*50)
    
    # Load datasets
    enron = pd.read_csv(RESULTS_DIR / "enron_preprocessed_3k.csv")
    combined = pd.read_csv(RESULTS_DIR / "combined_preprocessed_2k.csv")
    
    datasets = {
        "Enron": enron,
        "Combined": combined
    }
    
    all_results = {}
    
    # Test each model
    for model_key, model_name in MODELS.items():
        print(f"\n\n{'#'*50}")
        print(f"TESTING: {model_name}")
        print(f"{'#'*50}")
        
        all_results[model_name] = {}
        
        for ds_name, df in datasets.items():
            results = test_model(model_key, f"{model_name} - {ds_name}", df, n_samples=20)
            all_results[model_name][ds_name] = results
    
    # Summary
    print(f"\n\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}\n")
    
    for ds_name in datasets.keys():
        print(f"\n{ds_name} Dataset:")
        print("-" * 50)
        print(f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed'}")
        print("-" * 50)
        for model_name, results in all_results.items():
            m = results[ds_name]
            print(f"{model_name:<15} {m['accuracy']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1_score']:<8.4f} {m['emails_per_second']:.3f}")
    
    # Save
    with open(RESULTS_DIR / "phase4_llm_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("✓ PHASE 4 COMPLETE!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
