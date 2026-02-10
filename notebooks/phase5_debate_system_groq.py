#!/usr/bin/env python3
"""
Phase 5: Multi-Agent Debate System using Groq API
Three agents debate to classify emails:
- Attacker Agent: Identifies phishing indicators
- Defender Agent: Argues for legitimacy
- Judge Agent: Makes final decision based on debate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

class DebateSystem:
    """Multi-agent debate system for phishing detection"""
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # Use same model with different temperatures for different perspectives
        self.agents = {
            "attacker": {
                "model": "llama-3.1-8b-instant",
                "temperature": 0.7,  # More creative in finding threats
                "role": "Security Threat Hunter",
                "personality": "You are an aggressive cybersecurity analyst who identifies phishing threats. Look for suspicious patterns, social engineering tactics, and malicious indicators."
            },
            "defender": {
                "model": "llama-3.1-8b-instant",
                "temperature": 0.3,  # More conservative
                "role": "Legitimate Email Advocate",
                "personality": "You are a business analyst who defends legitimate emails. Look for signs of authenticity, proper business communication, and reasons why an email might be legitimate."
            },
            "judge": {
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1,  # Very analytical
                "role": "Impartial Security Judge",
                "personality": "You are an impartial judge who weighs evidence from both sides. Make a balanced decision based on the strongest arguments."
            }
        }
    
    def call_agent(self, agent_key: str, prompt: str) -> dict:
        """Call a specific agent"""
        try:
            agent = self.agents[agent_key]
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=agent["model"],
                messages=[
                    {"role": "system", "content": f"{agent['personality']} You are the {agent['role']}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=agent["temperature"],
                max_tokens=300
            )
            
            inference_time = time.time() - start_time
            
            return {
                "response": response.choices[0].message.content,
                "inference_time": inference_time,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def debate_email(self, email_text: str) -> dict:
        """Run debate on email"""
        start_time = time.time()
        
        # Truncate email
        email_snippet = email_text[:600]
        
        # Round 1: Attacker identifies threats
        attacker_prompt = f"""Analyze this email and identify ALL potential phishing indicators and threats.

EMAIL:
{email_snippet}

List specific red flags, suspicious patterns, and reasons why this could be PHISHING.
Be thorough and aggressive in identifying threats."""

        attacker_result = self.call_agent("attacker", attacker_prompt)
        
        if not attacker_result["success"]:
            return {
                "classification": "LEGITIMATE",
                "confidence": "error",
                "debate_log": "Error in attacker analysis",
                "total_time": time.time() - start_time,
                "success": False
            }
        
        # Round 2: Defender argues for legitimacy
        defender_prompt = f"""Analyze this email and argue why it could be LEGITIMATE.

EMAIL:
{email_snippet}

ATTACKER'S CONCERNS:
{attacker_result['response']}

Counter the attacker's arguments. List reasons why this email appears legitimate, signs of authentic communication, and explanations for any suspicious elements."""

        defender_result = self.call_agent("defender", defender_prompt)
        
        if not defender_result["success"]:
            return {
                "classification": "PHISHING",  # If defender fails, trust attacker
                "confidence": "error",
                "debate_log": "Error in defender analysis",
                "total_time": time.time() - start_time,
                "success": False
            }
        
        # Round 3: Judge makes final decision
        judge_prompt = f"""Review this email security debate and make the FINAL DECISION.

EMAIL:
{email_snippet}

ATTACKER ARGUMENT (Phishing Indicators):
{attacker_result['response']}

DEFENDER ARGUMENT (Legitimacy Indicators):
{defender_result['response']}

Based on BOTH arguments, make your final decision. Respond with:
CLASSIFICATION: [PHISHING or LEGITIMATE]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASONING: [Your balanced analysis]"""

        judge_result = self.call_agent("judge", judge_prompt)
        
        if not judge_result["success"]:
            return {
                "classification": "LEGITIMATE",
                "confidence": "error",
                "debate_log": "Error in judge decision",
                "total_time": time.time() - start_time,
                "success": False
            }
        
        # Parse judge's decision
        judge_response = judge_result["response"].upper()
        
        if "CLASSIFICATION: PHISHING" in judge_response or ("PHISHING" in judge_response and "LEGITIMATE" not in judge_response):
            classification = "PHISHING"
        elif "CLASSIFICATION: LEGITIMATE" in judge_response or "LEGITIMATE" in judge_response:
            classification = "LEGITIMATE"
        else:
            classification = "LEGITIMATE"  # Conservative default
        
        # Parse confidence
        confidence = "medium"
        if "CONFIDENCE: HIGH" in judge_response:
            confidence = "high"
        elif "CONFIDENCE: LOW" in judge_response:
            confidence = "low"
        
        # Create debate log
        debate_log = f"""
=== ATTACKER (Threat Hunter) ===
{attacker_result['response']}

=== DEFENDER (Legitimacy Advocate) ===
{defender_result['response']}

=== JUDGE (Final Decision) ===
{judge_result['response']}
"""
        
        total_time = time.time() - start_time
        
        return {
            "classification": classification,
            "confidence": confidence,
            "debate_log": debate_log,
            "total_time": total_time,
            "agent_times": {
                "attacker": attacker_result["inference_time"],
                "defender": defender_result["inference_time"],
                "judge": judge_result["inference_time"]
            },
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

def test_debate_system(dataset_name: str, dataset_path: Path, sample_size: int = 100):
    """Test debate system on dataset"""
    print(f"\n{'='*60}")
    print(f"Testing Debate System on {dataset_name}")
    print(f"{'='*60}")
    
    # Initialize debate system
    debate_system = DebateSystem()
    
    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Sample
    print(f"\n2. Sampling {sample_size} emails for testing...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    phishing_count = (df_sample['label'] == 1).sum()
    legit_count = (df_sample['label'] == 0).sum()
    print(f"   Phishing: {phishing_count}")
    print(f"   Legitimate: {legit_count}")
    
    # Run debates
    print(f"\n3. Running debates on {len(df_sample)} emails...")
    predictions = []
    times = []
    success_count = 0
    
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(df_sample)} emails...")
        
        result = debate_system.debate_email(row['text'])
        predictions.append(result["classification"])
        times.append(result["total_time"])
        
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
    """Main function"""
    print("="*60)
    print("PHASE 5: MULTI-AGENT DEBATE SYSTEM")
    print("="*60)
    print("\nDebate Structure:")
    print("  1. ATTACKER: Identifies phishing threats (Llama-3.1-8B, temp=0.7)")
    print("  2. DEFENDER: Argues for legitimacy (Llama-3.1-8B, temp=0.3)")
    print("  3. JUDGE: Makes final decision (Llama-3.3-70B, temp=0.1)")
    
    # Datasets
    datasets = {
        "Enron (3k)": RESULTS_DIR / "enron_preprocessed_3k.csv",
        "Combined (2k)": RESULTS_DIR / "combined_preprocessed_2k.csv"
    }
    
    all_results = {}
    
    # Test on each dataset
    for dataset_name, dataset_path in datasets.items():
        results = test_debate_system(dataset_name, dataset_path, sample_size=100)
        all_results[dataset_name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: DEBATE SYSTEM RESULTS")
    print(f"{'='*60}\n")
    
    print(f"{'Dataset':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Speed':<10}")
    print("-" * 60)
    
    for dataset_name, metrics in all_results.items():
        print(f"{dataset_name:<15} "
              f"{metrics['accuracy']:<8.4f} "
              f"{metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} "
              f"{metrics['f1_score']:<8.4f} "
              f"{metrics['emails_per_second']:<10.3f}")
    
    # Save results
    output_file = RESULTS_DIR / "phase5_debate_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ PHASE 5 COMPLETE!")
    print("Multi-agent debate system evaluation completed!")

if __name__ == "__main__":
    main()
