
import pandas as pd
from groq import Groq
import os
import time
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-api-key-here"))

# Models to ensemble
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768"
]

def classify_email_single(email_text, model):
    """Classify email with a single model"""
    prompt = f"""You are a phishing email detection expert. Analyze this email and determine if it's phishing or legitimate.

Email:
{email_text}

Respond with ONLY one word: "phishing" or "legitimate"
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if "phishing" in result:
            return 1
        elif "legitimate" in result:
            return 0
        else:
            return None
            
    except Exception as e:
        print(f"Error with {model}: {e}")
        return None

def classify_email_ensemble(email_text, models=MODELS):
    """Classify email using ensemble of models (majority vote)"""
    votes = []
    
    for model in models:
        pred = classify_email_single(email_text, model)
        if pred is not None:
            votes.append(pred)
    
    if len(votes) == 0:
        return None
    
    # Majority vote
    vote_counts = Counter(votes)
    majority_vote = vote_counts.most_common(1)[0][0]
    
    return majority_vote

def evaluate_ensemble(dataset_path, dataset_name, sample_size=100):
    """Evaluate ensemble approach on a dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating LLM Ensemble on {dataset_name}")
    print(f"Models: {', '.join(MODELS)}")
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
        pred = classify_email_ensemble(row['text'])
        
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
    print("\nPhase 4D: LLM Ensemble Evaluation")
    print("="*60)
    print("Hypothesis: Multiple models voting improves accuracy")
    print("Expected: 93-98% accuracy (up from 91-97% single model)")
    print("="*60)
    
    # Enron dataset
    enron_results = evaluate_ensemble(
        "../results/enron_preprocessed_3k.csv",
        "Enron Dataset"
    )
    
    # Combined dataset
    combined_results = evaluate_ensemble(
        "../results/combined_preprocessed_2k.csv",
        "Combined Dataset"
    )
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Single Model vs Ensemble")
    print("="*80)
    print(f"\n{'Dataset':<20} {'Single Model':<15} {'Ensemble':<15} {'Improvement':<15}")
    print("-"*80)
    
    if enron_results:
        print(f"{'Enron':<20} {'91.00%':<15} {f'{enron_results[\"accuracy\"]*100:.2f}%':<15} {f'+{(enron_results[\"accuracy\"]*100 - 91.00):.2f}%':<15}")
    
    if combined_results:
        print(f"{'Combined':<20} {'97.00%':<15} {f'{combined_results[\"accuracy\"]*100:.2f}%':<15} {f'+{(combined_results[\"accuracy\"]*100 - 97.00):.2f}%':<15}")
    
    print("="*80)
    print("\nNote: Ensemble is 3x slower (3 API calls per email)")
