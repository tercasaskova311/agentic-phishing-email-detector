#!/usr/bin/env python3
"""
Phase 7: Fine-tune LLM using Unsloth
Fine-tune Llama-3.2-3B on phishing detection task
"""

import json
from pathlib import Path
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model configuration
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True  # Use 4-bit quantization for memory efficiency

def load_data():
    """Load training and test data"""
    print("Loading datasets...")
    
    with open(MODELS_DIR / "train_data.json", 'r') as f:
        train_data = json.load(f)
    
    with open(MODELS_DIR / "test_data.json", 'r') as f:
        test_data = json.load(f)
    
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    
    return train_data, test_data

def format_prompt(sample):
    """Format sample for Alpaca-style instruction following"""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

def main():
    print("="*60)
    print("PHASE 7: FINE-TUNING WITH UNSLOTH")
    print("="*60)
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ Warning: No GPU detected. Fine-tuning will be very slow on CPU.")
        print("Consider using Google Colab with GPU for faster training.")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting. Please run on a machine with GPU.")
            return
    
    # Load data
    train_data, test_data = load_data()
    
    # Load model
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    print("✓ Model loaded successfully")
    
    # Configure LoRA
    print(f"\n{'='*60}")
    print("Configuring LoRA...")
    print(f"{'='*60}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("✓ LoRA configured")
    
    # Prepare datasets
    print(f"\n{'='*60}")
    print("Preparing datasets...")
    print(f"{'='*60}")
    
    # Format data
    train_formatted = [{"text": format_prompt(sample)} for sample in train_data]
    
    train_dataset = Dataset.from_list(train_formatted)
    
    print(f"✓ Training dataset prepared: {len(train_dataset)} samples")
    
    # Training arguments
    print(f"\n{'='*60}")
    print("Training configuration...")
    print(f"{'='*60}")
    
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "finetuned_llama3.2_3b"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=500,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
    )
    
    print("Training parameters:")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Max steps: {training_args.max_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  BF16: {training_args.bf16}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    print("This may take 30-60 minutes depending on your GPU...")
    
    trainer.train()
    
    print("\n✓ Training complete!")
    
    # Save model
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")
    
    model_save_path = MODELS_DIR / "finetuned_llama3.2_3b_final"
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    
    print(f"✓ Model saved to: {model_save_path}")
    
    # Save in GGUF format for Ollama (optional)
    print(f"\n{'='*60}")
    print("Saving in GGUF format for Ollama...")
    print(f"{'='*60}")
    
    try:
        model.save_pretrained_gguf(
            str(MODELS_DIR / "finetuned_llama3.2_3b_gguf"),
            tokenizer,
            quantization_method="q4_k_m"
        )
        print("✓ GGUF model saved")
    except Exception as e:
        print(f"⚠ GGUF export failed: {e}")
        print("You can still use the regular saved model")
    
    print(f"\n{'='*60}")
    print("✓ FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Test the fine-tuned model")
    print("2. Compare with baseline models")
    print("3. Evaluate on test set")

if __name__ == "__main__":
    main()
