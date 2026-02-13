#!/usr/bin/env python3
"""
Phase 4: Single LLM Evaluation using Groq API
Test each LLM individually on both datasets using fast Groq inference
Models: Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B (via Groq equivalents)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

# LLM Models to test (Groq available models)
MODELS = {
    "llama-3.1-8b-instant": "Llama-3.1-8B-Instant",
    "llama-3.3-70b-versatile": "Llama-3.3-70B",
    "mixtral-8x7b-32768": "Mixtral-8x7B"
}

def classify_email(client: Groq, model: str, email_text: str) -> dict:
    """Classify a single email using Groq LLM"""
    try:
        start_time = time.time()
        
        # Truncate email
        email_snippet = email_text[:600]
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an email security classifier. Respond with ONLY one word: PHISHING or LEGITIMATE."
                },
                {
                    "role": "user",
                    "content": f"Classify this email:\n\n{email_snippet}"
                }
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        inference_time = time.time() - start_time
        result = response.choices[0].message.content.upper()
        
        # Parse classification
        if "PHISHING" in result and "LEGITIMATE" not in result:
            classification = "PHISHING"
        elif "LEGITIMATE" in result and "PHISHING" not in result:
            classification = "LEGITIMATE"
        elif "PHISHING" in result:
            classification = "PHISHING"
        else:
            classification = "LEGITIMATE"
        
        return {
            "classification": classification,
            "inference_time": inference_time,
            "success": True
        }
        
    except Exception as e:
        return {
            "classification": "LEGITIMATE",
            "inference_time": 0,
            "success": False,
            "error": str(e)
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

def test_llm_on_dataset(client: Groq, model_key: str, model_name: str, dataset_name: str, dataset_path: Path, sample_size: int = 100):
    """Test a single LLM on a dataset"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Sample for testing
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
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(df_sample)} emails...")
        
        result = classify_email(client, model_key, row['text'])
        predictions.append(result["classification"])
        times.append(result["inference_time"])
        
        if result["success"]:
            success_count += 1
    
    print(f"   ✓ Completed: {success_count}/{len(df_sample)} successful")
    
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
    print(f"   Total:     {metrics['total_time']:.1f} seconds ({metrics['total_time']/60:.1f} minutes)")
    
    return {
        **metrics,
        "sample_size": len(df_sample),
        "success_rate": round(success_count / len(df_sample), 4)
    }

def main():
    """Main function to test all LLMs on all datasets"""
    print("="*60)
    print("PHASE 4: SINGLE LLM EVALUATION (GROQ API)")
    print("="*60)
    
    # Initialize Groq client
    print("\nInitializing Groq client...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("✓ Groq client ready")
    except Exception as e:
        print(f"✗ Failed to initialize Groq: {e}")
        return
    
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
            try:
                results = test_llm_on_dataset(
                    client,
                    model_key, 
                    model_name, 
                    dataset_name, 
                    dataset_path,
                    sample_size=100  # Test on 100 emails per dataset
                )
                all_results[model_name][dataset_name] = results
            except Exception as e:
                print(f"\n✗ Error testing {model_name} on {dataset_name}: {e}")
                all_results[model_name][dataset_name] = {
                    "error": str(e),
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "emails_per_second": 0
                }
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: ALL RESULTS")
    print(f"{'='*60}\n")
    
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name}:")
        print("-" * 60)
        print(f"{'Model':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed':<10}")
        print("-" * 60)
        
        for model_name, datasets_results in all_results.items():
            if dataset_name in datasets_results:
                metrics = datasets_results[dataset_name]
                print(f"{model_name:<20} "
                      f"{metrics['accuracy']:<8.4f} "
                      f"{metrics['precision']:<8.4f} "
                      f"{metrics['recall']:<8.4f} "
                      f"{metrics['f1_score']:<8.4f} "
                      f"{metrics['emails_per_second']:<10.3f}")
    
    # Save results
    output_file = RESULTS_DIR / "phase4_llm_groq_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ PHASE 4 COMPLETE!")
    print("Single LLM evaluation completed using Groq API.")
    print("Much faster than local inference!")

if __name__ == "__main__":
    main()
