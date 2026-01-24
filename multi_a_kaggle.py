"""
CrewAI multi-agent phishing detector for Kaggle notebooks.
Optimized batch processing with comprehensive metrics.
"""
import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
from urllib.parse import urlparse

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'models': {
        'email_agent': 'Qwen/Qwen2.5-3B-Instruct',
        'url_agent': 'microsoft/Phi-3-mini-4k-instruct',
        'pattern_agent': 'meta-llama/Llama-3.2-3B-Instruct',
        'judge_agent': 'Qwen/Qwen2.5-7B-Instruct',
    },
    
    'datasets': {
        'aigen': '/kaggle/input/phishing-emails/aigen.csv',
        'enron': '/kaggle/input/phishing-emails/enron.csv',
        'trec': '/kaggle/input/phishing-emails/trec.csv'
    },
    
    'sample_sizes': {
        'aigen': None,
        'enron': 3000,
        'trec': 3000
    },
    
    'balanced_sampling': True,
    'save_errors': True,
    'checkpoint_every': 100,
    'batch_size': 10,
    'experiment_name': 'multi_agent_crewai',
}

#helper fucntion =============================
def email_analysis_tool(email_text: str) -> dict:
        """
    Analyze email content for phishing cues.
        
        Returns a dictionary of indicators:
        - suspicious_words: list of keywords like 'urgent', 'verify', etc.
        - excessive_punctuation: count of '!!' or more
        - all_caps: count of words in all caps
        """
    indicators = {}

        # Suspicious keywords
    suspicious_words_list = [
        "urgent", "verify", "password", "login", "account suspended", "click here", "confirm"
    ]
    indicators['suspicious_words'] = [
        w for w in suspicious_words_list if w in email_text.lower()
    ]

        # Excessive punctuation
    indicators['excessive_punctuation'] = len(re.findall(r"[!]{2,}", email_text))

        # All caps words
    indicators['all_caps'] = len(re.findall(r"\b[A-Z]{4,}\b", email_text))

    return indicators


def url_extraction_tool(email_text: str) -> dict:
        """
        Extract URLs from email and check for suspicious patterns.
        
        Returns a dictionary with:
        - urls: list of dictionaries for each URL
        - url: the URL itself
        - is_ip: True if URL uses IP instead of domain
        - suspicious_length: True if URL length > 75 characters
        - odd_subdomains: True if URL has >3 subdomain levels
        - total_urls: total number of URLs found
        """
    urls = re.findall(r'(https?://[^\s]+)', email_text)
    url_indicators = []

    for url in urls:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        indicator = {
            "url": url,
            "is_ip": all(c.isdigit() or c == '.' for c in hostname),
            "suspicious_length": len(url) > 75,
            "odd_subdomains": hostname.count(".") > 3
        }
        url_indicators.append(indicator)

    return {
        "urls": url_indicators,
        "total_urls": len(urls)
    }

def get_llm_from_hf(model_name: str):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    return HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        temperature=0.2,
        max_new_tokens=512,
        top_p=0.9,
        repetition_penalty=1.1,
        timeout=120,
    )
# ============================================================================
# AGENTS SETUP
# ============================================================================

def create_agents():
    """Create all agents with specified models."""
    email_agent = Agent(
        role="Email Content Analyst",
        goal="Detect phishing language in emails - context, tricky and spam language",
        backstory="Cybersecurity analyst specializing in linguistic phishing cues.",
        llm=get_llm_from_hf(CONFIG['models']['email_agent']),
        tools=[Tools.email_analysis_tool],
        allow_delegation=False,
        verbose=False,
    )
    
    url_agent = Agent(
        role="URL Inspector",
        goal="Analyze URLs for phishing indicators",
        backstory="Expert in malicious URLs and domain obfuscation.",
        llm=get_llm_from_hf(CONFIG['models']['url_agent']),
        tools=[Tools.url_extraction_tool],
        allow_delegation=False,
        verbose=False,
    )
    
    pattern_agent = Agent(
        role="Email Pattern Analyst",
        goal="Detect known phishing templates and metadata inconsistencies",
        backstory="Specialist in email headers, subject, sender spoofing, and scam patterns.",
        llm=get_llm_from_hf(CONFIG['models']['pattern_agent']),
        allow_delegation=False,
        verbose=False,
    )
    
    judge_agent = Agent(
        role="Final Judge",
        goal="Aggregate all analyses and decide if the email is phishing",
        backstory="Senior security expert making final classification decisions.",
        llm=get_llm_from_hf(CONFIG['models']['judge_agent']),
        allow_delegation=False,
        verbose=False,
    )
    
    return email_agent, url_agent, pattern_agent, judge_agent

# ============================================================================
# TASK BUILDER
# ============================================================================

def build_tasks(email_text: str, agents: Tuple):
    """Build task graph for a single email."""
    email_agent, url_agent, pattern_agent, judge_agent = agents
    
    t1 = Task(
        description=f"""
Analyze the email text for phishing language.
Email:
{email_text[:800]}

Provide: Suspicious language indicators
""",
        agent=email_agent,
        expected_output="Suspicious language indicators",
    )
    
    t2 = Task(
        description=f"""
Extract and analyze URLs in the email.
Email:
{email_text[:800]}

Provide: URL-based phishing indicators
""",
        agent=url_agent,
        expected_output="URL-based phishing indicators",
    )
    
    t3 = Task(
        description=f"""
Analyze sender, email subject formatting, and known phishing patterns.
Email:
{email_text[:800]}

Provide: Pattern-based phishing indicators
""",
        agent=pattern_agent,
        expected_output="Pattern-based phishing indicators",
    )
    
    t4 = Task(
        description="""
You are given three analyses:
1) Email language analysis
2) URL analysis
3) Pattern analysis

Combine them and respond EXACTLY in this format:
DECISION: PHISHING or SAFE
CONFIDENCE: integer 0-100
REASON: one sentence

Your response:
""",
        agent=judge_agent,
        context=[t1, t2, t3],
        expected_output="Final phishing verdict",
    )
    
    return [t1, t2, t3, t4]

# ============================================================================
# RESPONSE PARSER
# ============================================================================

def parse_response(text: str) -> Tuple[str, float, str]:
    """Parse agent response into structured format."""
    decision_match = re.search(r"DECISION:\s*(PHISHING|SAFE)", text, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text)
    
    decision = decision_match.group(1).upper() if decision_match else "SAFE"
    confidence = int(conf_match.group(1)) / 100.0 if conf_match else 0.5
    
    # Map to expected labels
    prediction = "phishing_email" if decision == "PHISHING" else "safe_email"
    
    return prediction, confidence, text.strip()[:200]

# ============================================================================
# MULTI-AGENT DETECTOR
# ============================================================================

class MultiAgentDetector:
    """Multi-agent phishing detector with CrewAI."""
    
    def __init__(self):
        print("Initializing multi-agent system...")
        self.agents = create_agents()
        print("✓ All agents created")
    
    def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """Analyze a single email with multi-agent crew."""
        try:
            tasks = build_tasks(email_text, self.agents)
            crew = Crew(
                agents=list(self.agents),
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )
            
            result = crew.kickoff()
            prediction, confidence, reasoning = parse_response(str(result))
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            return {
                'prediction': 'error',
                'confidence': 0.5,
                'reasoning': f'Error: {str(e)[:100]}'
            }
    
    def analyze_batch(self, emails: List[str], show_progress: bool = True) -> List[Dict]:
        """Process batch of emails."""
        results = []
        iterator = tqdm(emails, desc="Processing") if show_progress else emails
        
        for email in iterator:
            result = self.analyze_email(email)
            results.append(result)
        
        return results

# ============================================================================
# DATA LOADING (same as single-agent)
# ============================================================================

def load_dataset(
    filepath: str,
    sample_size: int = None,
    balanced: bool = True
) -> pd.DataFrame:
    """Load and optionally sample dataset."""
    print(f"\nLoading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original size: {len(df):,}")
    
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if sample_size and sample_size < len(df):
        if balanced:
            n_per_class = sample_size // len(class_counts)
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_per_class), random_state=42)
            ).reset_index(drop=True)
        else:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        
        print(f"  Sampled size: {len(df):,}")
        sampled_counts = df['label'].value_counts()
        print(f"  Sampled distribution:")
        for label, count in sampled_counts.items():
            print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df

# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    """Evaluate multi-agent detector on datasets."""
    
    def __init__(self, detector: MultiAgentDetector, checkpoint_dir: Path = Path('.')):
        self.detector = detector
        self.checkpoint_dir = checkpoint_dir
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        dataset_name: str,
        checkpoint_every: int = 100,
        batch_size: int = 10
    ) -> Dict:
        """Evaluate detector on dataset with checkpointing."""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Total emails: {len(df):,}")
        print(f"Batch size: {batch_size}")
        
        results = []
        errors = []
        start_time = time.time()
        
        # Process in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            batch_emails = batch_df['message'].tolist()
            
            # Analyze batch
            batch_results = self.detector.analyze_batch(batch_emails, show_progress=False)
            
            # Store results
            for idx, result in enumerate(batch_results):
                actual_idx = batch_start + idx
                row = batch_df.iloc[idx]
                
                full_result = {
                    'email_id': actual_idx,
                    'true_label': row['label'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'correct': result['prediction'] == row['label']
                }
                results.append(full_result)
                
                if result['prediction'] == 'error':
                    errors.append(full_result)
            
            # Progress update
            processed = batch_end
            if processed % 10 == 0 or processed == len(df):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(df) - processed) / rate if rate > 0 else 0
                
                print(f"  Progress: {processed:,}/{len(df):,} "
                      f"({processed/len(df)*100:.1f}%) | "
                      f"Rate: {rate:.1f} emails/s | "
                      f"ETA: {remaining/60:.1f}m | "
                      f"Errors: {len(errors)}", 
                      end='\r')
            
            # Checkpoint
            if checkpoint_every and processed % checkpoint_every < batch_size:
                self._save_checkpoint(results, dataset_name, processed)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        metrics = self._calculate_metrics(results, elapsed)
        self._print_summary(dataset_name, metrics, len(errors))
        
        return {
            'results': results,
            'metrics': metrics,
            'errors': errors if CONFIG['save_errors'] else []
        }
    
    def _calculate_metrics(self, results: List[Dict], elapsed: float) -> Dict:
        """Calculate comprehensive metrics."""
        valid_results = [r for r in results if r['prediction'] != 'error']
        
        if not valid_results:
            return {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'error_rate': 1.0,
                'time_seconds': elapsed
            }
        
        correct = sum(1 for r in valid_results if r['correct'])
        total = len(valid_results)
        error_count = len(results) - total
        
        tp = sum(1 for r in valid_results if r['true_label'] == 'phishing_email' and r['prediction'] == 'phishing_email')
        fp = sum(1 for r in valid_results if r['true_label'] == 'safe_email' and r['prediction'] == 'phishing_email')
        tn = sum(1 for r in valid_results if r['true_label'] == 'safe_email' and r['prediction'] == 'safe_email')
        fn = sum(1 for r in valid_results if r['true_label'] == 'phishing_email' and r['prediction'] == 'safe_email')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': correct / total,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'correct': correct,
            'total': total,
            'errors': error_count,
            'error_rate': error_count / len(results),
            'time_seconds': elapsed,
            'emails_per_second': len(results) / elapsed if elapsed > 0 else 0,
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        }
    
    def _print_summary(self, dataset_name: str, metrics: Dict, error_count: int):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print(f"RESULTS: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Accuracy:   {metrics['accuracy']:.2%} ({metrics['correct']:,}/{metrics['total']:,})")
        print(f"Precision:  {metrics['precision']:.2%}")
        print(f"Recall:     {metrics['recall']:.2%}")
        print(f"F1-Score:   {metrics['f1_score']:.2%}")
        print(f"Errors:     {error_count:,} ({metrics['error_rate']:.1%})")
        print(f"Time:       {metrics['time_seconds']/60:.1f} minutes")
        print(f"Speed:      {metrics['emails_per_second']:.1f} emails/second")
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TP: {cm['true_positives']:,}  |  FP: {cm['false_positives']:,}")
        print(f"  FN: {cm['false_negatives']:,}  |  TN: {cm['true_negatives']:,}")
        print(f"{'='*70}")
    
    def _save_checkpoint(self, results: List[Dict], dataset_name: str, count: int):
        """Save checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{dataset_name}_{count}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f)
        print(f"\n  ✓ Checkpoint saved: {checkpoint_file.name}")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_metrics(all_results: Dict):
    """Plot comparison across datasets."""
    datasets = []
    accuracies = []
    f1_scores = []
    
    for dataset_name, data in all_results['datasets'].items():
        datasets.append(dataset_name)
        accuracies.append(data['metrics']['accuracy'])
        f1_scores.append(data['metrics']['f1_score'])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Agent Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_agent_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    print("\n" + "="*70)
    print("MULTI-AGENT PHISHING DETECTION - CrewAI")
    print("="*70)
    print(f"Email Agent:   {CONFIG['models']['email_agent']}")
    print(f"URL Agent:     {CONFIG['models']['url_agent']}")
    print(f"Pattern Agent: {CONFIG['models']['pattern_agent']}")
    print(f"Judge Agent:   {CONFIG['models']['judge_agent']}")
    print(f"Batch size:    {CONFIG['batch_size']}")
    print("="*70)
    
    # Create detector
    detector = MultiAgentDetector()
    evaluator = Evaluator(detector)
    
    # Store results
    all_results = {
        'models': CONFIG['models'],
        'config': CONFIG,
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {}
    }
    
    # Process each dataset
    for dataset_name, filepath in CONFIG['datasets'].items():
        try:
            df = load_dataset(
                filepath,
                sample_size=CONFIG['sample_sizes'][dataset_name],
                balanced=CONFIG['balanced_sampling']
            )
            
            dataset_results = evaluator.evaluate(
                df,
                dataset_name,
                checkpoint_every=CONFIG['checkpoint_every'],
                batch_size=CONFIG['batch_size']
            )
            
            all_results['datasets'][dataset_name] = dataset_results
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user!")
            print("Saving partial results...")
            break
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_file = f"results_multiagent_{CONFIG['experiment_name']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("Multi-Agent System Performance")
    print("-"*70)
    
    for dataset_name, data in all_results['datasets'].items():
        m = data['metrics']
        print(f"{dataset_name:10s} | "
              f"Acc: {m['accuracy']:6.1%} | "
              f"F1: {m['f1_score']:6.1%} | "
              f"Time: {m['time_seconds']/60:5.1f}m | "
              f"Speed: {m['emails_per_second']:4.1f}/s")
    
    print(f"\n✓ Results saved: {output_file}")
    print("="*70)
    
    # Plot results
    try:
        plot_metrics(all_results)
    except:
        print("⚠️  Could not generate plots")
    
    return all_results

if __name__ == '__main__':
    results = main()