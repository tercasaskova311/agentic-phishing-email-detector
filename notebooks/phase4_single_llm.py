#!/usr/bin/env python3
"""
Phase 4: Single LLM Evaluation
Test each LLM individually on both datasets using Ollama
Models: Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ollama

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# LLM Models to test
MODELS = {
    "qwen2.5:3b-instruct": "Qwen2.5-3B",
    "llama3.2:latest": "Llama-3.2-3B",
    "gemma:2b": "Gemma-2B"
}

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        ollama.list()
        return True
    except:
        return False

def call_ollama(model: str, prompt: str, max_retries: int = 3) -> dict:
    """Call Ollama API with retry logic"""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 10  # Short response
                }
            )
            
            inference_time = time.time() - start_time
            
            return {
                "response": response['response'].strip(),
                "inference_time": inference_time,
                "success": True
            }
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    return {
        "response": "",
        "inference_time": 0,
        "success": False,
        "error": "Max retries exceeded"
    }

def classify_email(model: str, email_text: str) -> dict:
    """Classify a single email using LLM"""
    # Truncate email to avoid token limits and speed up inference
    email_snippet = email_text[:400]  # Reduced from 800
    
    prompt = f"""Analyze this email and classify it as either PHISHING or LEGITIMATE.

EMAIL:
{email_snippet}

Respond with ONLY one word: either "PHISHING" or "LEGITIMATE"."""

    result = call_ollama(model, prompt)
    
    if not result["success"]:
        return {
            "classification": "LEGITIMATE",  # Conservative default
            "confidence": "error",
            "inference_time": result["inference_time"],
            "success": False
        }
    
    # Parse response
    response = result["response"].upper()
    
    if "PHISHING" in response and "LEGITIMATE" not in response:
        classification = "PHISHING"
    elif "LEGITIMATE" in response and "PHISHING" not in response:
        classification = "LEGITIMATE"
    elif "PHISHING" in response:
        classification = "PHISHING"  # If both, prefer phishing
    else:
        classification = "LEGITIMATE"  # Conservative default
    
    return {
        "classification": classification,
        "inference_time": result["inference_time"],
        "success": True
    }

def calculate_metrics(y_true, y_pred, times):
    """Calculate all metrics"""
    y_true_bin = [1 if label == 1 else 0 for label in y_true]
    y_pred_bin = [1 if label == "PHISHING" else 0 for label in y_pred]
    
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    avg_time = sum(times) / len(times) if times else 0
    emails_per_second = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(emails_per_second, 3),
        "avg_time_per_email": round(avg_time, 2),
        "total_time": round(sum(times), 2)
    }

def test_llm_on_dataset(model_key: str, model_name: str, dataset_name: str, dataset_path: Path, sample_size: int = 10):
    """Test a single LLM on a dataset"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Sample for testing (use smaller sample for LLMs due to speed)
    print(f"\n2. Sampling {sample_size} emails for testing...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Ensure balanced sample
    phishing_count = (df_sample['label'] == 1).sum()
    legit_count = (df_sample['label'] == 0).sum()
    print(f"   Phishing: {phishing_count}")
    print(f"   Legitimate: {legit_count}")
    
    # Classify emails
    print(f"\n3. Classifying {len(df_sample)} emails...")
    predictions = []
    times = []
    success_count = 0
    
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        print(f"   Email {i+1}/{len(df_sample)}...", end='', flush=True)
        
        result = classify_email(model_key, row['text'])
        predictions.append(result["classification"])
        times.append(result["inference_time"])
        
        if result["success"]:
            success_count += 1
            print(f" ✓ {result['classification']} ({result['inference_time']:.1f}s)")
        else:
            print(f" ✗ Error")
    
    print(f"   Completed: {success_count}/{len(df_sample)} successful")
    
    # Calculate metrics
    true_labels = df_sample['label'].tolist()
    metrics = calculate_metrics(true_labels, predictions, times)
    
    # Display results
    print(f"\n4. Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.1f}%)")
    print(f"   Speed:     {metrics['emails_per_second']:.3f} emails/second")
    print(f"   Avg Time:  {metrics['avg_time_per_email']:.2f}s per email")
    print(f"   Total:     {metrics['total_time']/60:.1f} minutes")
    
    return {
        **metrics,
        "sample_size": len(df_sample),
        "success_rate": round(success_count / len(df_sample), 4)
    }

def main():
    """Main function to test all LLMs on all datasets"""
    print("="*60)
    print("PHASE 4: SINGLE LLM EVALUATION")
    print("="*60)
    
    # Check Ollama
    print("\nChecking Ollama installation...")
    if not check_ollama():
        print("✗ Ollama is not installed or not running!")
        print("Please install Ollama from: https://ollama.ai")
        print("Then run: ollama pull qwen2.5:3b")
        print("          ollama pull llama3.2:3b")
        print("          ollama pull gemma2:2b")
        return
    print("✓ Ollama is ready")
    
    # Datasets to test
    datasets = {
        "Enron (3k)": RESULTS_DIR / "enron_preprocessed_3k.csv",
        "Combined (2k)": RESULTS_DIR / "combined_preprocessed_2k.csv"
    }
    
    all_results = {}
    
    # Test each model on each dataset
    for model_key, model_name in MODELS.items():
        all_results[model_name] = {}
        
        for dataset_name, dataset_path in datasets.items():
            results = test_llm_on_dataset(
                model_key, 
                model_name, 
                dataset_name, 
                dataset_path,
                sample_size=10  # Test on 10 emails per dataset for speed
            )
            all_results[model_name][dataset_name] = results
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: ALL RESULTS")
    print(f"{'='*60}\n")
    
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name}:")
        print("-" * 60)
        print(f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed':<10}")
        print("-" * 60)
        
        for model_name, datasets_results in all_results.items():
            metrics = datasets_results[dataset_name]
            print(f"{model_name:<15} "
                  f"{metrics['accuracy']:<8.4f} "
                  f"{metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics['emails_per_second']:<10.3f}")
    
    # Save results
    output_file = RESULTS_DIR / "phase4_llm_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ PHASE 4 COMPLETE!")
    print("Single LLM evaluation completed for all models and datasets.")

if __name__ == "__main__":
    main()
