# ============================================================================
# PHISHING DETECTOR - ZERO-SHOT BASELINE
# Models: Open-Source LLMs from HuggingFace
# GPU: Kaggle T4/P100

"""
class ModelLoader:
    Loads LLM with optimal settings for Kaggle GPUs
class PhishingDetector:
    Takes email → returns classification
class Evaluator:
    Takes detector + dataset → returns metrics
"""

# ============================================================================
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'model_name': 'meta-llama/Llama-3.2-3B-Instruct',
    
    """
        MODELS_TO_TEST = [
            'meta-llama/Llama-3.2-3B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'google/gemma-2-2b-it',
            'microsoft/Phi-3-mini-4k-instruct'
        ]
    """
    
    'datasets': {
        'aigen': '/kaggle/datasets/tercasaskova/agentic-detector/aigen.csv',
        'enron': '/kaggle/datasets/tercasaskova/agentic-detector/enron.csv',
        'trec': '/kaggle/datasets/tercasaskova/agentic-detector/trec.csv'
    },
    
    # Sample sizes (None = use all data)
    'sample_sizes': {
        'aigen': None,     # 4k - use all
        'enron': 10000,    # 33k - sample 10k
        'trec': 20000      # 75k - sample 20k
    },
    
    'generation': {
        'max_new_tokens': 50,
        'temperature': 0.1,      # Low temp for consistent answers - classification task...
        'do_sample': True,
        'top_p': 0.9
    },
    
    #important - especially for eron - not balanced....
    'balanced_sampling': True,   # Equal samples per class
    'save_errors': True,         # Save failed predictions
    'checkpoint_every': 1000     # Save progress every N emails
}


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class ModelLoader:
    """Handles model loading with optimal Kaggle GPU settings"""
    
    @staticmethod
    def load(model_name: str):
        
        print(f"LOADING MODEL: {model_name}")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Fix padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,      # FP16 for speed
            device_map="auto",               # Auto GPU placement
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        load_time = time.time() - start_time
        print(f"\n✓ Model loaded in {load_time:.1f}s")
        print(f"  Device: {generator.device}")
        
        # Test generation
        print("\nTesting generation...")
        test_output = generator(
            "Say 'READY' if you can classify emails:",
            max_new_tokens=10,
            **CONFIG['generation']
        )
        print(f"  Test output: {test_output[0]['generated_text'][:100]}")
        print("="*70)
        
        return generator, tokenizer

"""
You are a {ROLE}.              # Sets domain context
                               
Your task: {TASK}              # Clear objective

Constraints:                   # Forces structured output
- Respond with ONLY: X or Y
- Do not explain
"""

class PhishingDetector:
    def __init__(self, generator, tokenizer):
        self.generator = generator
        self.tokenizer = tokenizer
        self.prompt_template = self._get_prompt_template()
    
    def _get_prompt_template(self) -> str:
        
        # Clean, minimal prompt
        return """You are a cybersecurity expert analyzing emails for phishing attempts.

Email to analyze:
{email_text}

Instructions:
1. Determine if this email is a phishing attempt or safe
2. Respond with ONLY one word: "PHISHING" or "SAFE"

Your answer:"""

#now important part = why did the model decide to classify wrongly??
# 
    def analyze(self, email_text: str) -> Tuple[str, str]:
        # Format prompt
        prompt = self.prompt_template.format(
            email_text=email_text[:800]  # Limit to 800 chars
        )
        
        try:
            # Generate response
            output = self.generator(
                prompt,
                max_new_tokens=CONFIG['generation']['max_new_tokens'],
                temperature=CONFIG['generation']['temperature'],
                do_sample=CONFIG['generation']['do_sample'],
                top_p=CONFIG['generation']['top_p'],
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = output[0]['generated_text'].strip()
            response_upper = response.upper()
            
            valid_results = [r for r in results if r['prediction'] != 'error']
            error_count = len(results) - len(valid_results)

            #LLMs fail in 3 ways: generation errors, parsing errors, wrong predictions.
            # Parse response
            if 'PHISHING' in response_upper:
                return 'phishing_email', response
            elif 'SAFE' in response_upper:
                return 'safe_email', response
            else:
                return 'error', f'Unclear response: {response}'
                
        except Exception as e:
            return 'error', f'Generation error: {str(e)}'
            

def load_dataset(
    filepath: str,
    sample_size: Optional[int] = None,
    balanced: bool = True
) -> pd.DataFrame:
    """
    Load and optionally sample dataset
    
    Args:
        filepath: Path to CSV file
        sample_size: Number of samples (None = use all)
        balanced: Whether to balance classes in sampling
    """
    
    print(f"\nLoading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original size: {len(df):,}")
    
    # Show class distribution
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        if balanced:
            # Stratified sampling - equal from each class
            n_per_class = sample_size // len(class_counts)
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_per_class), random_state=42)
            ).reset_index(drop=True)
        else:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        
        print(f"  Sampled size: {len(df):,}")
        
        # Show sampled distribution
        sampled_counts = df['label'].value_counts()
        print(f"  Sampled distribution:")
        for label, count in sampled_counts.items():
            print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


class Evaluator:    
    def __init__(self, detector: PhishingDetector, checkpoint_dir: Path = Path('.')):
        self.detector = detector
        self.checkpoint_dir = checkpoint_dir
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        dataset_name: str,
        checkpoint_every: int = 1000
    ) -> Dict:
        print(f"EVALUATING: {dataset_name.upper()}")
        print(f"Total emails: {len(df):,}")
        
        results = []
        errors = []
        start_time = time.time()
        
        for idx, row in df.iterrows():
            # Generate prediction
            prediction, response = self.detector.analyze(row['message'])
            
            result = {
                'email_id': idx,
                'true_label': row['label'],
                'prediction': prediction,
                'response': response[:200],  # Truncate for storage
                'correct': prediction == row['label']
            }
            results.append(result)
            
            if prediction == 'error':
                errors.append(result)
            
        #Key Metrics to Track:
            #Rate (items/sec): Detect slowdowns
            #ETA: Plan your day
            #Error count: Catch broken runs early
            #Memory (for long runs)

            # Progress update
            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(df) - idx - 1) / rate if rate > 0 else 0
                
                print(f"  Progress: {idx+1:,}/{len(df):,} "
                      f"({(idx+1)/len(df)*100:.1f}%) | "
                      f"Rate: {rate:.1f} emails/s | "
                      f"ETA: {remaining/60:.1f}m | "
                      f"Errors: {len(errors)}", 
                      end='\r')
            
            # Checkpoint
            if checkpoint_every and (idx + 1) % checkpoint_every == 0:
                self._save_checkpoint(results, dataset_name, idx + 1)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, elapsed)
        
        # Print summary
        self._print_summary(dataset_name, metrics, len(errors))
        
        return {
            'results': results,
            'metrics': metrics,
            'errors': errors if CONFIG['save_errors'] else []
        }
    
    def _calculate_metrics(self, results: List[Dict], elapsed: float) -> Dict:
        """Calculate evaluation metrics"""
        
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
        
        # Confusion matrix
        tp = sum(1 for r in valid_results 
                if r['true_label'] == 'phishing_email' 
                and r['prediction'] == 'phishing_email')
        fp = sum(1 for r in valid_results 
                if r['true_label'] == 'safe_email' 
                and r['prediction'] == 'phishing_email')
        tn = sum(1 for r in valid_results 
                if r['true_label'] == 'safe_email' 
                and r['prediction'] == 'safe_email')
        fn = sum(1 for r in valid_results 
                if r['true_label'] == 'phishing_email' 
                and r['prediction'] == 'safe_email')
        
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
            'emails_per_second': len(results) / elapsed,
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        }
    
    def _print_summary(self, dataset_name: str, metrics: Dict, error_count: int):
        """Print evaluation summary"""
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Accuracy:   {metrics['accuracy']:.2%} "
              f"({metrics['correct']:,}/{metrics['total']:,})")
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
        """Save checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{dataset_name}_{count}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f)
        print(f"\n  ✓ Checkpoint saved: {checkpoint_file.name}")

# ============================================================================
# CELL 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("PHISHING DETECTION - ZERO-SHOT BASELINE")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print("="*70)
    
    # Load model
    generator, tokenizer = ModelLoader.load(CONFIG['model_name'])
    
    # Create detector
    detector = PhishingDetector(generator, tokenizer)
    evaluator = Evaluator(detector)
    
    # Store all results
    all_results = {
        'model': CONFIG['model_name'],
        'config': CONFIG,
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {}
    }
    
    # Process each dataset
    for dataset_name, filepath in CONFIG['datasets'].items():
        # Load data
        df = load_dataset(
            filepath,
            sample_size=CONFIG['sample_sizes'][dataset_name],
            balanced=CONFIG['balanced_sampling']
        )
        
        # Evaluate
        dataset_results = evaluator.evaluate(
            df, 
            dataset_name,
            checkpoint_every=CONFIG['checkpoint_every']
        )
        
        all_results['datasets'][dataset_name] = dataset_results
    
    # Save final results
    model_short_name = CONFIG['model_name'].split('/')[-1]
    output_file = f"results_{model_short_name}_zero_shot.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print("-"*70)
    
    for dataset_name, data in all_results['datasets'].items():
        m = data['metrics']
        print(f"{dataset_name:10s} | "
              f"Acc: {m['accuracy']:6.1%} | "
              f"F1: {m['f1_score']:6.1%} | "
              f"Time: {m['time_seconds']/60:5.1f}m | "
              f"Speed: {m['emails_per_second']:4.1f}/s")
    
    print(f"✓ Results saved: {output_file}")
    
    return all_results

# ============================================================================
# CELL 8: RUN
# ============================================================================

if __name__ == '__main__':
    results = main()