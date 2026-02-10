# Phase 4: Single LLM Evaluation Results

## Overview
Tested individual LLMs on both datasets using Groq API for fast cloud inference.
Each model tested on 100 balanced emails per dataset.

## Models Tested
- **Llama-3.1-8B-Instant**: Fast, efficient Llama model
- **Llama-3.3-70B**: Larger, more capable Llama model
- **Mixtral-8x7B**: Failed (API issues)

## Results

### Enron Dataset (100 emails: 50 phishing + 50 legitimate)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Llama-3.1-8B-Instant | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 |
| **Llama-3.3-70B** | **91.00%** | **95.56%** | **86.00%** | **90.53%** | 0.625 |

**Best Model**: Llama-3.3-70B (91.0% accuracy, 90.5% F1)

### Combined Dataset (100 emails: 47 phishing + 53 legitimate)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Llama-3.1-8B-Instant | 91.00% | 85.19% | 97.87% | 91.09% | 0.372 |
| **Llama-3.3-70B** | **97.00%** | **100.00%** | **93.62%** | **96.70%** | 0.453 |

**Best Model**: Llama-3.3-70B (97.0% accuracy, 96.7% F1)

## Key Findings

1. **Llama-3.3-70B Excels**: The larger 70B model significantly outperforms the 8B model
   - Enron: 91% accuracy vs 72%
   - Combined: 97% accuracy vs 91%

2. **Combined Dataset Easier**: Both models performed better on Combined dataset
   - Llama-3.1-8B: 91% vs 72%
   - Llama-3.3-70B: 97% vs 91%

3. **Speed**: Groq API provides fast inference
   - 0.4-0.6 emails/second (~2 seconds per email)
   - Much faster than local Ollama (30-50 seconds per email)

4. **High Recall**: Both models tend to be aggressive in detecting phishing
   - Llama-3.1-8B: 98% recall (catches almost all phishing)
   - Llama-3.3-70B: 86-94% recall (more balanced)

## Comparison with Traditional ML (Phase 3)

### Enron Dataset
- **Traditional ML Best**: Logistic Regression (98.0% accuracy, 98.0% F1)
- **LLM Best**: Llama-3.3-70B (91.0% accuracy, 90.5% F1)
- **Winner**: Traditional ML by 7%

### Combined Dataset
- **Traditional ML Best**: Naive Bayes & Random Forest (99.5% accuracy, 99.5% F1)
- **LLM Best**: Llama-3.3-70B (97.0% accuracy, 96.7% F1)
- **Winner**: Traditional ML by 2.5%

## Analysis

**Traditional ML still outperforms single LLMs** on this task, likely because:
1. TF-IDF features capture spam/phishing patterns well
2. Training data is large and clean
3. Task is relatively straightforward pattern matching

**However, LLMs offer advantages**:
- No training required (zero-shot)
- Can understand context and semantics
- Potential for better generalization to novel attacks
- Can explain reasoning (not tested here)

## Next Phase

Phase 5 will test **multi-agent debate systems** where multiple LLMs collaborate, which may improve performance through:
- Different perspectives (attacker vs defender)
- Consensus building
- Error correction through debate
