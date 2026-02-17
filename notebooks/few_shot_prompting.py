
import pandas as pd
from groq import Groq
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-groq-api-key-here"))

FEW_SHOT_EXAMPLES = """
Example 1:
Email: Dear customer, your PayPal account has been suspended. Click here immediately to verify your identity or your account will be permanently closed.
Classification: phishing
Reason: Urgency, suspicious link, impersonation

Example 2:
Email: Hi team, please review the Q4 budget proposal attached. Let me know if you have any questions before Friday's meeting.
Classification: legitimate
Reason: Internal communication, specific context, no suspicious elements

Example 3:
Email: URGENT: Your bank account shows unusual activity. Confirm your details now to prevent account closure: http://secure-bank-verify.com
Classification: phishing
Reason: Fake urgency, suspicious URL, impersonation, threatening language

Example 4:
Email: Thanks for your order #12345. Your package will arrive on Tuesday. Track your shipment at amazon.com/orders
Classification: legitimate
Reason: Order confirmation, specific details, legitimate domain

Now classify this email:
"""

def classify_email_few_shot(email_text, model="llama-3.3-70b-versatile"):
    prompt = f"""{FEW_SHOT_EXAMPLES}
Email: {email_text}
Classification:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if "phishing" in result:
            return 1
        elif "legitimate" in result:
            return 0
        else:
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def evaluate_few_shot(dataset_path, dataset_name, sample_size=100):
    print(f"\n{'='*60}")
    print(f"Evaluating Few-Shot Prompting on {dataset_name}")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(dataset_path)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Testing on {len(df)} emails...")
    
    predictions = []
    true_labels = []
    failed = 0
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        pred = classify_email_few_shot(row['text'])
        
        if pred is not None:
            predictions.append(pred)
            true_labels.append(row['label'])
        else:
            failed += 1
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(df)} emails...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
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
    print("\nFew-Shot Prompting Evaluation")
    print("="*60)
    print("Testing with 4 example classifications in prompt")
    print("="*60)
    
    enron_results = evaluate_few_shot(
        "phishing-detection-project/results/enron_preprocessed_3k.csv",
        "Enron Dataset"
    )
    
    combined_results = evaluate_few_shot(
        "phishing-detection-project/results/combined_preprocessed_2k.csv",
        "Combined Dataset"
    )
    
    print("\n" + "="*80)
    print("COMPARISON: Zero-Shot vs Few-Shot")
    print("="*80)
    print(f"\n{'Dataset':<20} {'Zero-Shot':<15} {'Few-Shot':<15} {'Improvement':<15}")
    print("-"*80)
    
    if enron_results:
        enron_acc = enron_results["accuracy"]*100
        enron_improvement = enron_acc - 91.00
        print(f"{'Enron':<20} {'91.00%':<15} {enron_acc:.2f}%{'':<9} +{enron_improvement:.2f}%")
    
    if combined_results:
        combined_acc = combined_results["accuracy"]*100
        combined_improvement = combined_acc - 97.00
        print(f"{'Combined':<20} {'97.00%':<15} {combined_acc:.2f}%{'':<9} +{combined_improvement:.2f}%")
    
    print("="*80)
