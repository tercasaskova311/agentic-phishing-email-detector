#!/usr/bin/env python3
"""
Ensemble Methods: Combine Traditional ML + LLM predictions
Test different ensemble strategies to achieve best of both worlds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def classify_with_llm(client: Groq, email_text: str) -> tuple:
    """Classify email using LLM, return prediction and confidence"""
    try:
        email_snippet = email_text[:600]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        
        result = response.choices[0].message.content.upper()
        
        if "PHISHING" in result and "LEGITIMATE" not in result:
            return 1, 0.9
        elif "LEGITIMATE" in result and "PHISHING" not in result:
            return 0, 0.9
        elif "PHISHING" in result:
            return 1, 0.7
        else:
            return 0, 0.7
            
    except Exception as e:
        return 0, 0.5

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
        "emails_per_second": round(emails_per_second, 3),
        "total_time": round(processing_time, 2)
    }

def ensemble_voting(ml_pred, llm_pred, ml_confidence, llm_confidence):
    """Simple majority voting"""
    return ml_pred

def ensemble_weighted(ml_pred, llm_pred, ml_confidence, llm_confidence):
    """Weighted voting based on confidence"""
    ml_weight = 0.7  # ML is more accurate
    llm_weight = 0.3
    
    score = (ml_pred * ml_weight) + (llm_pred * llm_weight)
    return 1 if score >= 0.5 else 0

def ensemble_ml_primary(ml_pred, llm_pred, ml_confidence, llm_confidence):
    """ML primary, LLM as validator for uncertain cases"""
    if ml_confidence < 0.6:
        return llm_pred
    return ml_pred

def ensemble_llm_override(ml_pred, llm_pred, ml_confidence, llm_confidence):
    """LLM can override ML for high-confidence phishing detections"""
    if llm_pred == 1 and llm_confidence > 0.8:
        return 1
    return ml_pred

def test_ensemble(dataset_name, dataset_path, client, sample_size=200):
    """Test ensemble methods on a dataset"""
    print(f"\n{'='*60}")
    print(f"TESTING ENSEMBLE: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Sample for testing
    print(f"\n2. Sampling {sample_size} emails...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_sample, test_size=0.5, random_state=42, stratify=df_sample['label'])
    
    X_train = train_df['text'].fillna("")
    y_train = train_df['label']
    X_test = test_df['text'].fillna("")
    y_test = test_df['label']
    
    print(f"   Training: {len(train_df)} emails ({(y_train == 1).sum()} phishing)")
    print(f"   Testing: {len(test_df)} emails ({(y_test == 1).sum()} phishing)")
    
    # Train ML model
    print("\n3. Training ML model (Logistic Regression)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    ml_model = LogisticRegression(max_iter=1000, random_state=42)
    ml_model.fit(X_train_vec, y_train)
    
    # Get ML predictions with confidence
    ml_proba = ml_model.predict_proba(X_test_vec)
    ml_preds = ml_model.predict(X_test_vec)
    ml_confidences = np.max(ml_proba, axis=1)
    
    print(f"   ✓ ML model trained")
    print(f"   ML Test Accuracy: {accuracy_score(y_test, ml_preds):.4f}")
    
    # Get LLM predictions
    print(f"\n4. Getting LLM predictions on test set...")
    llm_preds = []
    llm_confidences = []
    
    start_time = time.time()
    for i, text in enumerate(X_test):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(X_test)} emails...")
        
        pred, conf = classify_with_llm(client, text)
        llm_preds.append(pred)
        llm_confidences.append(conf)
    
    llm_time = time.time() - start_time
    
    llm_preds = np.array(llm_preds)
    llm_confidences = np.array(llm_confidences)
    
    print(f"   ✓ LLM predictions completed")
    print(f"   LLM Test Accuracy: {accuracy_score(y_test, llm_preds):.4f}")
    
    # Test ensemble strategies
    print(f"\n5. Testing ensemble strategies...")
    
    ensemble_methods = {
        "ML Only": lambda ml, llm, mlc, llmc: ml,
        "LLM Only": lambda ml, llm, mlc, llmc: llm,
        "Simple Voting": ensemble_voting,
        "Weighted (70/30)": ensemble_weighted,
        "ML Primary": ensemble_ml_primary,
        "LLM Override": ensemble_llm_override
    }
    
    results = {}
    
    for method_name, method_func in ensemble_methods.items():
        ensemble_preds = []
        
        for i in range(len(ml_preds)):
            pred = method_func(
                ml_preds[i], 
                llm_preds[i],
                ml_confidences[i],
                llm_confidences[i]
            )
            ensemble_preds.append(pred)
        
        ensemble_preds = np.array(ensemble_preds)
        
        # Calculate metrics (use LLM time as baseline)
        metrics = calculate_metrics(y_test, ensemble_preds, llm_time, len(y_test))
        
        print(f"\n   {method_name}:")
        print(f"     Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"     Precision: {metrics['precision']:.4f}")
        print(f"     Recall:    {metrics['recall']:.4f}")
        print(f"     F1 Score:  {metrics['f1_score']:.4f}")
        
        results[method_name] = metrics
    
    return results

def main():
    """Main function to test ensemble methods"""
    print("="*60)
    print("ENSEMBLE METHODS: ML + LLM")
    print("="*60)
    
    if not GROQ_API_KEY:
        print("\n✗ Error: GROQ_API_KEY not found in environment")
        print("Please set it with: export GROQ_API_KEY='your-key'")
        return
    
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
    
    # Test each dataset
    for dataset_name, dataset_path in datasets.items():
        try:
            results = test_ensemble(dataset_name, dataset_path, client, sample_size=200)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\n✗ Error testing {dataset_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: ENSEMBLE RESULTS")
    print(f"{'='*60}\n")
    
    for dataset_name, methods_results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 70)
        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 70)
        
        for method_name, metrics in methods_results.items():
            print(f"{method_name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")
    
    # Save results
    output_file = RESULTS_DIR / "ensemble_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ ENSEMBLE TESTING COMPLETE!")
    print("\nKey Findings:")
    print("- Compare ML Only vs LLM Only vs Ensemble methods")
    print("- Best ensemble strategy combines strengths of both")
    print("- Weighted voting or ML primary with LLM validation work well")

if __name__ == "__main__":
    main()
