#SINGLE AGENT - A/B testing - 
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from configs.config import CONFIG
from configs zero_shot import ZERO_SHOT_A, ZERO_SHOT_B

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_kaggle_gpu():
    """Configure GPU and authenticate with HuggingFace"""
    print("="*70)
    print("KAGGLE GPU SETUP")
    print("="*70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"✓ Free GPU Memory: {free_mem / 1e9:.1f} GB")
    else:
        print("⚠️  WARNING: No GPU detected!")
        raise RuntimeError("GPU not available! Please enable GPU in Kaggle settings.")
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        
        from huggingface_hub import login
        login(token=hf_token)
        print("✓ HuggingFace authentication successful")
        
    except Exception as e:
        print(f"⚠️  HuggingFace token not found: {e}")
        print("   Continuing with open models only")
    
    torch.backends.cudnn.benchmark = True
    print("="*70)
    print()

# ============================================================================
# DATASET CLASS
# ============================================================================

class EmailDataset(Dataset):
    """
    PyTorch Dataset for efficient batching
    """
    def __init__(self, emails: List[str], labels: List[str], prompt_template: str):
        self.emails = emails
        self.labels = labels
        self.prompt_template = prompt_template
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email_text = self.emails[idx][:800]  # Truncate long emails
        prompt = self.prompt_template.format(email_text=email_text)
        return {
            'prompt': prompt,
            'label': self.labels[idx],
            'idx': idx
        }

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Handles model loading with Kaggle GPU optimizations"""
    
    @staticmethod
    def load(model_name: str, use_4bit: bool = False):
        print(f"\n{'='*70}")
        print(f"LOADING MODEL: {model_name}")
        print(f"{'='*70}")
        start_time = time.time()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
            print("✓ Using 4-bit quantization")
        else:
            model_kwargs['torch_dtype'] = torch.float16
            model_kwargs['device_map'] = 'auto'
            print("✓ Using FP16 (half precision)")
        
        print("Loading model to GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        print("Creating inference pipeline...")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map='auto',
            batch_size=CONFIG['batch_size']  # Set default batch size
        )
        
        load_time = time.time() - start_time
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n✓ Model loaded in {load_time:.1f}s")
            print(f"✓ GPU Memory Allocated: {allocated:.2f} GB")
            print(f"✓ GPU Memory Reserved: {reserved:.2f} GB")
        
        print(f"{'='*70}\n")
        
        return generator, tokenizer

# ============================================================================
# PHISHING DETECTOR 
# ============================================================================

class PhishingDetector:
    def __init__(self, generator, tokenizer, config: Dict = CONFIG):
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config
        self.strategy = config['strategy']
        self.prompt_template = config['prompt_template']

    def analyze_batch(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """
         batching for efficient GPU usage
        """
        try:
            # Single pipeline call with all prompts
            outputs = self.generator(
                prompts,
                max_new_tokens=self.config['generation']['max_new_tokens'],
                temperature=self.config['generation']['temperature'],
                do_sample=self.config['generation']['do_sample'],
                top_p=self.config['generation']['top_p'],
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Parse results
            results = []
            for output in outputs:
                response = output[0]['generated_text'].strip()
                result = self._parse_simple_response(response)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"\n⚠️  Batch processing error: {e}")
            # Fallback: process one at a time
            results = []
            for prompt in prompts:
                try:
                    output = self.generator(
                        prompt,
                        max_new_tokens=self.config['generation']['max_new_tokens'],
                        temperature=self.config['generation']['temperature'],
                        do_sample=self.config['generation']['do_sample'],
                        top_p=self.config['generation']['top_p'],
                        return_full_text=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    response = output[0]['generated_text'].strip()
                    result = self._parse_simple_response(response)
                    results.append(result)
                except Exception as e2:
                    results.append(('error', f'Error: {str(e2)[:100]}'))
            
            return results
    
    def _parse_simple_response(self, response: str) -> Tuple[str, str]:
        response_upper = response.upper()
        if 'PHISHING' in response_upper:
            return 'phishing_email', response[:100]
        elif 'SAFE' in response_upper:
            return 'safe_email', response[:100]
        else:
            return 'error', f'Unclear: {response[:100]}'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(
    filepath: str,
    sample_size: Optional[int] = None,
    balanced: bool = True
) -> pd.DataFrame:
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
# EVALUATOR - W. BATCHING
# ============================================================================

class Evaluator:
    def __init__(self, detector: PhishingDetector, checkpoint_dir: Path = Path('.')):
        self.detector = detector
        self.checkpoint_dir = checkpoint_dir
    
    def evaluate(self, df: pd.DataFrame, dataset_name: str, 
                 checkpoint_every: int = 500, batch_size: int = 16) -> Dict:
        """
        FIXED: Proper batching without nested loops
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Total emails: {len(df):,}")
        print(f"Batch size: {batch_size}")
        
        results = []
        errors = []
        start_time = time.time()
        
        # Prepare all prompts at once
        prompts = [
            self.detector.prompt_template.format(email_text=text[:800])
            for text in df['message']
        ]
        
        # Process in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_prompts = prompts[batch_start:batch_end]
            batch_df = df.iloc[batch_start:batch_end]
            
            # Get predictions for this batch
            batch_predictions = self.detector.analyze_batch(batch_prompts)
            
            # Store results
            for idx, (prediction, response) in enumerate(batch_predictions):
                actual_idx = batch_start + idx
                row = batch_df.iloc[idx]
                
                result = {
                    'email_id': actual_idx,
                    'true_label': row['label'],
                    'prediction': prediction,
                    'response': response[:200],
                    'correct': prediction == row['label']
                }
                results.append(result)
                
                if prediction == 'error':
                    errors.append(result)
            
            # Progress update
            processed = batch_end
            if processed % 50 == 0 or processed == len(df):
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
        
        print()  # New line after progress bar
        elapsed = time.time() - start_time
        metrics = self._calculate_metrics(results, elapsed)
        self._print_summary(dataset_name, metrics, len(errors))
        
        return {
            'results': results,
            'metrics': metrics,
            'errors': errors if CONFIG['save_errors'] else []
        }
    
    def _calculate_metrics(self, results: List[Dict], elapsed: float) -> Dict:
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
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{dataset_name}_{count}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f)
        print(f"\n  ✓ Checkpoint saved: {checkpoint_file.name}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Setup GPU
    setup_kaggle_gpu()
    
    print("\n" + "="*70)
    print("PHISHING DETECTION - ZERO-SHOT BASELINE (FIXED)")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Strategy: {CONFIG['strategy']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("="*70)
    
    # Load model
    generator, tokenizer = ModelLoader.load(
        CONFIG['model_name'],
        use_4bit=CONFIG['use_4bit']
    )
    
    # Create detector
    detector = PhishingDetector(generator, tokenizer)
    evaluator = Evaluator(detector)
    
    # Store results
    all_results = {
        'model': CONFIG['model_name'],
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
            print(f"\n Error processing {dataset_name}: {e}")
            continue
    
    # Save results
    model_short_name = CONFIG['model_name'].split('/')[-1]
    output_file = f"results_{model_short_name}_{CONFIG['strategy']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
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
    
    print(f"\n✓ Results saved: {output_file}")
    print("="*70)
    
    return all_results

if __name__ == '__main__':
    results = main()