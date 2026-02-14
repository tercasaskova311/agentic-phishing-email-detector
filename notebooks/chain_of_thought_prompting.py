
import pandas as pd
from groq import Groq
import os
import time
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-api-key-here"))

CHAIN_OF_THOUGHT_PROMPT = """You are a phishing detection expert. Analyze this email step-by-step:

Email:
{email_text}

Think through this systematically:
1. Sender Analysis: Is the sender legitimate or suspicious?
2. Content Analysis: Are there urgency tactics, threats, or requests for sensitive info?
3. Link Analysis: Are there suspicious URLs or requests to click links?
4. Language Analysis: Is the grammar and tone professional or suspicious?
5. Context Analysis: Does this match expected communication patterns?

Based on your analysis, classify as 'phishing' or 'legitimate'.

Format your response as:
Analysis: [your step-by-step reasoning]
Classification: [phishing or legitimate]
"""

def classify_email_cot(email_text, model="llama-3.3-70b-versatile"):
    """Classify email using chain-of-thought prompting"""
    prompt = CHAIN_OF_THOUGHT_PROMPT.format(email_text=email_text)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract classification from response
        if "classification:" in result:
            classification_line = result.split("classification:")[-1].strip()
            if "phishing" in classification_line:
                return 1
            elif "legitimate" in classification_line:
                return 0
        
        # Fallback: check entire response
        if "phishing" in result and "legitimate" not in result:
            return 1
        elif "legitimate" in result and "phishing" not in result:
            return 0
        
        return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def evaluate_cot(dataset_path, dataset_name, sample_size=100):
    """Evaluate chain-of-thought prompting on a dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating Chain-of-Thought on {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Sample for testing
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Testing on {len(df)} emails...")
    
    predictions = []
    true_labels = []
    failed = 0
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        pred = classify_email_cot(row['text'])
        
        if pred is not None:
            predictions.append(pred)
            true_labels.append(row['label'])
        else:
            failed += 1
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(df)} emails...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    if len(predictions) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        speed = len(df) / total_time
        success_rate = (len(predictions) / len(df)) * 100
        
        print(f"\n{dataset_name} Results:")
        print("="*60)
        print(f"Accuracy:      {accuracy*100:.2f}%")
        print(f"Precision:     {precision*100:.2f}%")
        print(f"Recall:        {recall*100:.2f}%")
        print(f"F1 Score:      {f1*100:.2f}%")
        print(f"Speed:         {speed:.3f} emails/second")
        print(f"Success Rate:  {success_rate:.2f}% ({len(predictions)}/{len(df)})")
        print(f"Failed:        {failed}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'speed': speed,
            'success_rate': success_rate
        }
    else:
        print("All classifications failed!")
        return None

if __name__ == "__main__":
    # Test on both datasets
    print("\nPhase 4C: Chain-of-Thought Prompting Evaluation")
    print("="*60)
    print("Hypothesis: Step-by-step reasoning improves accuracy")
    print("Expected: 92-98% accuracy (up from 91-97% zero-shot)")
    print("="*60)
    
    # Enron dataset
    enron_results = evaluate_cot(
        "../results/enron_preprocessed_3k.csv",
        "Enron Dataset"
    )
    
    # Combined dataset
    combined_results = evaluate_cot(
        "../results/combined_preprocessed_2k.csv",
        "Combined Dataset"
    )
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Zero-Shot vs Chain-of-Thought")
    print("="*80)
    print(f"\n{'Dataset':<20} {'Zero-Shot':<15} {'CoT':<15} {'Improvement':<15}")
    print("-"*80)
    
    if enron_results:
        print(f"{'Enron':<20} {'91.00%':<15} {f'{enron_results[\"accuracy\"]*100:.2f}%':<15} {f'+{(enron_results[\"accuracy\"]*100 - 91.00):.2f}%':<15}")
    
    if combined_results:
        print(f"{'Combined':<20} {'97.00%':<15} {f'{combined_results[\"accuracy\"]*100:.2f}%':<15} {f'+{(combined_results[\"accuracy\"]*100 - 97.00):.2f}%':<15}")
    
    print("="*80)
