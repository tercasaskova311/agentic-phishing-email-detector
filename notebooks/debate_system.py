#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import time
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

class ImprovedDebateSystem:
    """Improved debate system with better prompts"""
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def call_agent(self, model: str, prompt: str, temperature: float = 0.1) -> dict:
        """Call agent with error handling"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200
            )
            
            return {
                "response": response.choices[0].message.content,
                "time": time.time() - start_time,
                "success": True
            }
        except Exception as e:
            return {
                "response": "",
                "time": 0,
                "success": False,
                "error": str(e)
            }
    
    def debate_email(self, email_text: str) -> dict:
        """Run improved debate"""
        start = time.time()
        
        # Truncate email
        email = email_text[:500]
        
        # Step 1: Attacker finds threats
        attacker_prompt = f"""You are a cybersecurity expert. Analyze this email for phishing indicators.

EMAIL: {email}

List 3-5 specific red flags if this is phishing, or state "NO MAJOR THREATS" if it seems legitimate.
Be specific and concise."""

        attacker = self.call_agent("llama-3.1-8b-instant", attacker_prompt, temperature=0.7)
        
        if not attacker["success"]:
            return {"classification": "LEGITIMATE", "time": time.time() - start, "success": False}
        
        # Step 2: Defender counters
        defender_prompt = f"""You are a business email analyst. Review this email and the security concerns.

EMAIL: {email}

SECURITY CONCERNS: {attacker['response']}

Provide 3-5 reasons why this email could be LEGITIMATE, or confirm it's suspicious.
Be specific and concise."""

        defender = self.call_agent("llama-3.1-8b-instant", defender_prompt, temperature=0.3)
        
        if not defender["success"]:
            # If defender fails, trust attacker
            classification = "PHISHING" if "NO MAJOR THREATS" not in attacker["response"].upper() else "LEGITIMATE"
            return {"classification": classification, "time": time.time() - start, "success": True}
        
        # Step 3: Judge decides
        judge_prompt = f"""You are a security judge. Review this email classification debate.

EMAIL: {email}

THREAT ANALYSIS: {attacker['response']}

LEGITIMACY ANALYSIS: {defender['response']}

Make your final decision. Respond with EXACTLY one of these two words:
PHISHING
LEGITIMATE

Your answer:"""

        judge = self.call_agent("llama-3.3-70b-versatile", judge_prompt, temperature=0.1)
        
        if not judge["success"]:
            return {"classification": "LEGITIMATE", "time": time.time() - start, "success": False}
        
        # Parse decision
        response = judge["response"].strip().upper()
        
        # Simple parsing - look for the keywords
        if "PHISHING" in response.split()[0:3]:  # Check first 3 words
            classification = "PHISHING"
        elif "LEGITIMATE" in response.split()[0:3]:
            classification = "LEGITIMATE"
        elif "PHISHING" in response and "LEGITIMATE" not in response:
            classification = "PHISHING"
        elif "LEGITIMATE" in response:
            classification = "LEGITIMATE"
        else:
            # Default based on attacker's assessment
            classification = "PHISHING" if "NO MAJOR THREATS" not in attacker["response"].upper() else "LEGITIMATE"
        
        return {
            "classification": classification,
            "time": time.time() - start,
            "success": True,
            "debate": {
                "attacker": attacker["response"][:200],
                "defender": defender["response"][:200],
                "judge": judge["response"][:100]
            }
        }

def calculate_metrics(y_true, y_pred, times):
    """Calculate metrics"""
    y_true_bin = [1 if label == 1 else 0 for label in y_true]
    y_pred_bin = [1 if label == "PHISHING" else 0 for label in y_pred]
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    avg_time = sum(times) / len(times) if times else 0
    speed = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(speed, 3),
        "avg_time": round(avg_time, 2),
        "total_time": round(sum(times), 2)
    }

def test_debate(dataset_name: str, dataset_path: Path, n_samples: int = 100):
    """Test improved debate system"""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*60}")
    
    debate = ImprovedDebateSystem()
    
    # Load and sample
    df = pd.read_csv(dataset_path)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    print(f"\nSample: {len(sample)} emails")
    print(f"  Phishing: {(sample['label'] == 1).sum()}")
    print(f"  Legitimate: {(sample['label'] == 0).sum()}")
    
    # Run debates
    print(f"\nRunning debates...")
    predictions = []
    times = []
    success = 0
    
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        if i % 20 == 0:
            print(f"  {i}/{len(sample)}...")
        
        result = debate.debate_email(row['text'])
        predictions.append(result["classification"])
        times.append(result["time"])
        
        if result["success"]:
            success += 1
    
    print(f"  ✓ Completed: {success}/{len(sample)} successful ({success/len(sample)*100:.1f}%)")
    
    # Metrics
    metrics = calculate_metrics(sample['label'].tolist(), predictions, times)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.1f}%)")
    print(f"  Speed:     {metrics['emails_per_second']:.3f} emails/s")
    print(f"  Avg Time:  {metrics['avg_time']:.2f}s per email")
    print(f"  Total:     {metrics['total_time']/60:.1f} minutes")
    
    return metrics

def main():
    print("="*60)
    print("PHASE 5: IMPROVED MULTI-AGENT DEBATE SYSTEM")
    print("="*60)
    print("\nImprovements:")
    print("  - Clearer, more concise prompts")
    print("  - Better output parsing")
    print("  - Improved error handling")
    print("  - Shorter responses (200 tokens max)")
    
    datasets = {
        "Enron (3k)": RESULTS_DIR / "enron_preprocessed_3k.csv",
        "Combined (2k)": RESULTS_DIR / "combined_preprocessed_2k.csv"
    }
    
    all_results = {}
    
    for name, path in datasets.items():
        results = test_debate(name, path, n_samples=100)
        all_results[name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Dataset':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed'}")
    print("-" * 60)
    
    for name, m in all_results.items():
        print(f"{name:<15} {m['accuracy']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1_score']:<8.4f} {m['emails_per_second']:.3f}")
    
    # Save
    with open(RESULTS_DIR / "phase5_debate_improved_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ IMPROVED DEBATE SYSTEM COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
