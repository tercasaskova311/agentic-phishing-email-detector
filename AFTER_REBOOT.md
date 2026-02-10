# After Reboot - Phase 7 Instructions

## Quick Start

After rebooting, run Phase 7 fine-tuning:

```bash
cd C:\Users\Sahar\Desktop\debate-agents-phishing
python phishing-detection-project\notebooks\phase7_finetune_llm.py
```

## What Will Happen

1. **GPU Check**: Script will verify GPU is working
2. **Data Preparation**: Load 1000 emails from Enron dataset (800 train, 200 test)
3. **Fine-Tuning**: Train Qwen2.5-3B model for 100 steps (~10-30 minutes)
4. **Testing**: Evaluate on 200 test emails
5. **Results**: Save metrics to `results/phase7_finetuning_results.json`

## Expected Output

```
✓ GPU Available: [Your GPU Name]
Preparing training data...
  Created 1000 training examples
  Train: 800, Test: 200

FINE-TUNING WITH UNSLOTH
Loading base model: unsloth/Qwen2.5-3B-Instruct
✓ Model loaded
✓ LoRA adapters added
✓ Dataset prepared: 800 examples

Starting training...
  This may take 10-30 minutes depending on GPU...
[Training progress...]
✓ Training completed in X minutes
✓ Model saved

TESTING FINE-TUNED MODEL
Testing on 200 examples...
  Accuracy: XX.X%
  F1 Score: XX.X%
```

## If GPU Still Has Issues

Run this to check GPU status:
```bash
nvidia-smi
```

If still showing "GPU is lost", you may need to:
1. Update GPU drivers
2. Check Windows Device Manager
3. Try a different reboot

## After Phase 7 Completes

We'll have completed all 7 phases:
- ✓ Phase 1: Model Selection
- ✓ Phase 2: Data Preprocessing
- ✓ Phase 3: Traditional ML (98-99% accuracy)
- ✓ Phase 4: Single LLM (91-97% accuracy)
- ✓ Phase 5: Debate System (76% accuracy)
- ✓ Phase 6: LangGraph (55% accuracy)
- ⏳ Phase 7: Fine-Tuning (in progress)

Then we'll create the final comprehensive report!

## Troubleshooting

**If you get memory errors:**
- The script uses 4-bit quantization to reduce memory
- Should work with 6GB+ GPU memory
- If still issues, we can reduce batch size

**If training is too slow:**
- Check GPU utilization with `nvidia-smi`
- Make sure no other programs are using GPU
- Training should take 10-30 minutes max

## Questions?

Just ask when you're back! I'll be here to help with any issues.
