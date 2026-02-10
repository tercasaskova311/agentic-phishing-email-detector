#!/usr/bin/env python3
"""
Phase 6: Graph-Based Agent System using LangGraph
Structured workflow with better error handling and state management
"""

import pandas as pd
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

# Define state
class EmailState(TypedDict):
    """State for email classification workflow"""
    email: str
    threat_analysis: str
    legitimacy_analysis: str
    final_decision: str
    classification: str
    confidence: str
    error: str

class LangGraphPhishingDetector:
    """LangGraph-based phishing detection system"""
    
    def __init__(self):
        # Initialize LLMs
        self.attacker_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.7
        )
        
        self.defender_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        
        self.judge_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        # Build graph
        self.graph = self.build_graph()
    
    def analyze_threats(self, state: EmailState) -> EmailState:
        """Attacker node: Identify threats"""
        try:
            prompt = f"""Analyze this email for phishing indicators. List 3-5 specific red flags or state "NO MAJOR THREATS".

EMAIL: {state['email'][:500]}

Be concise."""
            
            response = self.attacker_llm.invoke(prompt)
            state["threat_analysis"] = response.content
            
        except Exception as e:
            state["error"] = f"Threat analysis failed: {str(e)}"
            state["threat_analysis"] = "Analysis failed"
        
        return state
    
    def analyze_legitimacy(self, state: EmailState) -> EmailState:
        """Defender node: Argue for legitimacy"""
        try:
            prompt = f"""Review this email and the security concerns. Provide 3-5 reasons why it could be legitimate.

EMAIL: {state['email'][:500]}

CONCERNS: {state['threat_analysis']}

Be concise."""
            
            response = self.defender_llm.invoke(prompt)
            state["legitimacy_analysis"] = response.content
            
        except Exception as e:
            state["error"] = f"Legitimacy analysis failed: {str(e)}"
            state["legitimacy_analysis"] = "Analysis failed"
        
        return state
    
    def make_decision(self, state: EmailState) -> EmailState:
        """Judge node: Final decision"""
        try:
            prompt = f"""Review this debate and decide: PHISHING or LEGITIMATE?

EMAIL: {state['email'][:500]}

THREATS: {state['threat_analysis']}

LEGITIMACY: {state['legitimacy_analysis']}

Respond with ONE WORD: PHISHING or LEGITIMATE"""
            
            response = self.judge_llm.invoke(prompt)
            decision = response.content.strip().upper()
            
            # Parse decision
            if "PHISHING" in decision.split()[0:3]:
                state["classification"] = "PHISHING"
            elif "LEGITIMATE" in decision.split()[0:3]:
                state["classification"] = "LEGITIMATE"
            elif "PHISHING" in decision:
                state["classification"] = "PHISHING"
            else:
                state["classification"] = "LEGITIMATE"
            
            state["final_decision"] = decision
            state["confidence"] = "medium"
            
        except Exception as e:
            state["error"] = f"Decision failed: {str(e)}"
            state["classification"] = "LEGITIMATE"
            state["final_decision"] = "Error"
        
        return state
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(EmailState)
        
        # Add nodes
        workflow.add_node("analyze_threats", self.analyze_threats)
        workflow.add_node("analyze_legitimacy", self.analyze_legitimacy)
        workflow.add_node("make_decision", self.make_decision)
        
        # Define edges
        workflow.set_entry_point("analyze_threats")
        workflow.add_edge("analyze_threats", "analyze_legitimacy")
        workflow.add_edge("analyze_legitimacy", "make_decision")
        workflow.add_edge("make_decision", END)
        
        return workflow.compile()
    
    def classify_email(self, email_text: str) -> dict:
        """Classify email using graph"""
        start = time.time()
        
        try:
            # Initialize state
            initial_state = {
                "email": email_text,
                "threat_analysis": "",
                "legitimacy_analysis": "",
                "final_decision": "",
                "classification": "",
                "confidence": "",
                "error": ""
            }
            
            # Run graph
            result = self.graph.invoke(initial_state)
            
            return {
                "classification": result.get("classification", "LEGITIMATE"),
                "confidence": result.get("confidence", "low"),
                "time": time.time() - start,
                "success": result.get("error", "") == "",
                "error": result.get("error", "")
            }
            
        except Exception as e:
            return {
                "classification": "LEGITIMATE",
                "confidence": "error",
                "time": time.time() - start,
                "success": False,
                "error": str(e)
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

def test_langgraph(dataset_name: str, dataset_path: Path, n_samples: int = 100):
    """Test LangGraph system"""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*60}")
    
    detector = LangGraphPhishingDetector()
    
    # Load and sample
    df = pd.read_csv(dataset_path)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    print(f"\nSample: {len(sample)} emails")
    print(f"  Phishing: {(sample['label'] == 1).sum()}")
    print(f"  Legitimate: {(sample['label'] == 0).sum()}")
    
    # Classify
    print(f"\nClassifying with LangGraph...")
    predictions = []
    times = []
    success = 0
    
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        if i % 20 == 0:
            print(f"  {i}/{len(sample)}...")
        
        result = detector.classify_email(row['text'])
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
    print("PHASE 6: LANGGRAPH-BASED AGENT SYSTEM")
    print("="*60)
    print("\nGraph Structure:")
    print("  1. Analyze Threats (Attacker)")
    print("  2. Analyze Legitimacy (Defender)")
    print("  3. Make Decision (Judge)")
    print("  → Structured workflow with state management")
    
    datasets = {
        "Enron (3k)": RESULTS_DIR / "enron_preprocessed_3k.csv",
        "Combined (2k)": RESULTS_DIR / "combined_preprocessed_2k.csv"
    }
    
    all_results = {}
    
    for name, path in datasets.items():
        results = test_langgraph(name, path, n_samples=100)
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
    with open(RESULTS_DIR / "phase6_langgraph_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ LANGGRAPH SYSTEM COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
