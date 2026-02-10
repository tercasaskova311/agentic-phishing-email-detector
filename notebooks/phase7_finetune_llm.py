#!/usr/bin/env python3
"""
Phase 7: Fine-Tuning LLMs for Phishing Detection
Using Unsloth for efficient fine-tuning with GPU support
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ No GPU available - will use CPU (slower)")
        return False

def prepare_training_data(dataset_path: Path, max_samples: int = 2000):
    """Prepare data for fine-tuning"""
    print(f"\nPreparing training data from {dataset_path.name}...")
    
    df = pd.read_csv(dataset_path)
    
    # Create instruction-response pairs
    training_data = []
    
    for _, row in df.iterrows():
        email_text = row['text'][:500]  # Truncate for efficiency
        label = "PHISHING" if row['label'] == 1 else "LEGITIMATE"
        
        # Create training example
        instruction = f"Classify this email as PHISHING or LEGITIMATE:\n\n{email_text}"
        response = label
        
        training_data.append({
            "instruction": instruction,
            "response": response
        })
    
    # Limit samples
    if len(training_data) > max_samples:
        training_data = training_data[:max_samples]
    
    print(f"  Created {len(training_data)} training examples")
    
    # Split train/test
    train_data, test_data = train_test_split(
        training_data, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    return train_data, test_data

def finetune_with_unsloth(train_data, model_name="unsloth/Qwen2.5-3B-Instruct"):
    """Fine-tune model using Unsloth"""
    print(f"\n{'='*60}")
    print("FINE-TUNING WITH UNSLOTH")
    print(f"{'='*60}")
    
    try:
        from unsloth import FastLanguageModel
        
        print(f"\nLoading base model: {model_name}")
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization for efficiency
        )
        
        print("✓ Model loaded")
        
        # Add LoRA adapters
        print("\nAdding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        print("✓ LoRA adapters added")
        
        # Prepare dataset
        print("\nPreparing dataset...")
        
        def format_prompt(example):
            return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
        
        formatted_data = [
            {"text": format_prompt(ex)} for ex in train_data
        ]
        
        # Create dataset
        from datasets import Dataset
        dataset = Dataset.from_list(formatted_data)
        
        print(f"✓ Dataset prepared: {len(dataset)} examples")
        
        # Training arguments
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        print("\nStarting training...")
        print("  This may take 10-30 minutes depending on GPU...")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=100,  # Quick training for testing
                learning_rate=2e-4,
                fp16=not torch.cuda.is_available(),
                logging_steps=10,
                output_dir=str(MODELS_DIR / "finetuned_phishing"),
                optim="adamw_8bit",
                save_strategy="steps",
                save_steps=50,
            ),
        )
        
        # Train
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
        # Save model
        model_path = MODELS_DIR / "finetuned_phishing_final"
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        print(f"✓ Model saved to: {model_path}")
        
        return model, tokenizer, training_time
        
    except ImportError:
        print("\n✗ Unsloth not installed!")
        print("Install with: pip install unsloth")
        return None, None, 0
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {e}")
        return None, None, 0

def test_finetuned_model(model, tokenizer, test_data):
    """Test fine-tuned model"""
    print(f"\n{'='*60}")
    print("TESTING FINE-TUNED MODEL")
    print(f"{'='*60}")
    
    if model is None:
        print("No model to test")
        return {}
    
    print(f"\nTesting on {len(test_data)} examples...")
    
    predictions = []
    true_labels = []
    times = []
    
    for i, example in enumerate(test_data):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_data)}...")
        
        start = time.time()
        
        # Generate prediction
        inputs = tokenizer(
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n",
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse prediction
        if "PHISHING" in prediction.upper():
            pred_label = "PHISHING"
        else:
            pred_label = "LEGITIMATE"
        
        predictions.append(pred_label)
        true_labels.append(example['response'])
        times.append(time.time() - start)
    
    # Calculate metrics
    y_true = [1 if label == "PHISHING" else 0 for label in true_labels]
    y_pred = [1 if label == "PHISHING" else 0 for label in predictions]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    avg_time = sum(times) / len(times)
    speed = 1.0 / avg_time if avg_time > 0 else 0
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(speed, 3),
        "avg_time": round(avg_time, 2)
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Speed:     {metrics['emails_per_second']:.3f} emails/s")
    
    return metrics

def main():
    print("="*60)
    print("PHASE 7: FINE-TUNING LLMs FOR PHISHING DETECTION")
    print("="*60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    if not has_gpu:
        print("\n⚠ WARNING: No GPU detected!")
        print("Fine-tuning will be very slow on CPU.")
        print("Consider:")
        print("  1. Restarting to fix GPU")
        print("  2. Using Google Colab with GPU")
        print("  3. Using cloud GPU (AWS, Azure, etc.)")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Prepare data
    dataset_path = RESULTS_DIR / "enron_preprocessed_3k.csv"
    train_data, test_data = prepare_training_data(dataset_path, max_samples=1000)
    
    # Fine-tune
    model, tokenizer, training_time = finetune_with_unsloth(train_data)
    
    if model is None:
        print("\nFine-tuning failed. Exiting...")
        return
    
    # Test
    metrics = test_finetuned_model(model, tokenizer, test_data)
    
    # Save results
    results = {
        "training_time_minutes": round(training_time / 60, 2),
        "training_samples": len(train_data),
        "test_samples": len(test_data),
        "metrics": metrics
    }
    
    with open(RESULTS_DIR / "phase7_finetuning_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ PHASE 7 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    print(f"Final accuracy: {metrics['accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()
