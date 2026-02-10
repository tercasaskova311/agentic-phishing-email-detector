# Phase 7: Fine-Tuning Setup

## Overview
Phase 7 involves fine-tuning an LLM (Llama-3.2-3B) on the phishing detection task using Unsloth.

## Data Preparation ✓ Complete

**Training Data**: 2,400 samples (balanced)
- Phishing: 1,197 emails
- Legitimate: 1,203 emails

**Test Data**: 600 samples (balanced)
- Phishing: 303 emails
- Legitimate: 297 emails

**Format**: Alpaca-style instruction following
```json
{
  "instruction": "Classify this email as either PHISHING or LEGITIMATE.",
  "input": "Email: [email content]",
  "output": "PHISHING" or "LEGITIMATE"
}
```

## Fine-Tuning Configuration

**Model**: unsloth/Llama-3.2-3B-Instruct
**Method**: LoRA (Low-Rank Adaptation)
**Quantization**: 4-bit for memory efficiency

**LoRA Parameters**:
- Rank (r): 16
- Alpha: 16
- Dropout: 0
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Training Parameters**:
- Batch size: 2
- Gradient accumulation: 4 (effective batch size: 8)
- Max steps: 500
- Learning rate: 2e-4
- Optimizer: AdamW 8-bit
- Scheduler: Linear

## Requirements

### Hardware
- **GPU Required**: NVIDIA GPU with CUDA support
- **Minimum VRAM**: 8GB (16GB recommended)
- **Training Time**: 30-60 minutes on modern GPU

### Software
```bash
pip install unsloth
pip install torch transformers datasets trl
```

## Running Fine-Tuning

### Option 1: Local GPU
If you have a compatible GPU:
```bash
python phishing-detection-project/notebooks/phase7_finetune_unsloth.py
```

### Option 2: Google Colab (Recommended)
1. Upload the training data to Google Drive
2. Open a new Colab notebook with GPU runtime
3. Copy the fine-tuning script
4. Run training (free T4 GPU available)

### Option 3: Cloud GPU
- AWS SageMaker
- Azure ML
- Lambda Labs
- RunPod

## Expected Results

Based on similar fine-tuning tasks:

**Baseline (Pre-trained Llama-3.2-3B)**:
- Accuracy: ~70-80% (zero-shot)

**After Fine-tuning (Expected)**:
- Accuracy: 90-95%
- Precision: 90-95%
- Recall: 85-95%
- F1 Score: 88-95%

**Goal**: Match or exceed traditional ML (98-99% accuracy)

## Post Fine-Tuning Steps

1. **Evaluate on Test Set**:
   - Test on 600 held-out samples
   - Calculate all metrics
   - Compare with baselines

2. **Test on Both Datasets**:
   - Enron dataset
   - Combined dataset

3. **Compare All Approaches**:
   - Traditional ML
   - Single LLM (pre-trained)
   - Single LLM (fine-tuned)
   - Debate systems
   - Graph systems

## Current Status

✓ Data prepared (2,400 train + 600 test)
✓ Fine-tuning script created
⏳ Waiting for GPU to run training

## Alternative: Simulated Results

Since fine-tuning requires GPU and significant time, we can:

1. **Document the approach** (completed above)
2. **Estimate expected performance** based on literature
3. **Create evaluation framework** for when model is trained
4. **Generate comprehensive final report** with all phases

## Next Steps

**If GPU Available**:
- Run fine-tuning script
- Evaluate fine-tuned model
- Complete Phase 7

**If No GPU**:
- Document fine-tuning approach
- Create final comprehensive report
- Summarize all findings from Phases 1-6

Would you like to:
1. Proceed with fine-tuning (requires GPU)
2. Create final comprehensive report with all results
3. Both (document approach + create report)
