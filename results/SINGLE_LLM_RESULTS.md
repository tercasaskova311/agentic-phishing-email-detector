# Single LLM Results

## Overview
Individual LLMs tested on phishing detection using zero-shot classification via Groq API.

## Datasets
- **Enron Dataset**: 100 emails (50 phishing + 50 legitimate)
- **Combined Dataset**: 100 emails (47 phishing + 53 legitimate)

## Methodology
- Zero-shot classification (no training)
- Groq API for fast cloud inference
- Temperature: 0.1 (low for consistency)
- Max tokens: 10 (just classification)
- Email truncated to 600 characters

## Models Tested

### 1. Llama-3.1-8B-Instant
- 8 billion parameters
- Fast inference
- Groq optimized

### 2. Llama-3.3-70B-Versatile
- 70 billion parameters
- More capable
- Better reasoning

### 3. Llama-3.3-70B (Few-Shot)
- Same model with 4 examples
- Improved prompting
- Better context

---

## Results

### Enron Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Notes |
|-------|----------|-----------|--------|----------|------------------|-------|
| Llama-3.1-8B | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 | High recall, low precision |
| **Llama-3.3-70B (Zero-Shot)** | **91.00%** | **95.56%** | **86.00%** | **90.53%** | 0.625 | Best balance |
| **Llama-3.3-70B (Few-Shot)** | **94.37%** | **N/A** | **N/A** | **N/A** | 0.580 | +3.37% improvement |

**Best Model**: Llama-3.3-70B with Few-Shot (94.37% accuracy)

### Combined Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Notes |
|-------|----------|-----------|--------|----------|------------------|-------|
| Llama-3.1-8B | 91.00% | 85.19% | 97.87% | 91.09% | 0.372 | Good performance |
| **Llama-3.3-70B (Zero-Shot)** | **97.00%** | **96.00%** | **93.62%** | **96.70%** | 0.453 | Excellent |

**Best Model**: Llama-3.3-70B Zero-Shot (97.0% accuracy)

---

## Key Findings

### Model Size Matters
- **70B vs 8B**: Larger model significantly better
  - Enron: 91% vs 72% (+19%)
  - Combined: 97% vs 91% (+6%)
- Better reasoning and context understanding

### Few-Shot Learning Helps
- Adding 4 examples improved Enron performance
- Zero-shot: 91.00% → Few-shot: 94.37% (+3.37%)
- Demonstrates LLMs can learn from examples

### Dataset Difficulty
- **Combined easier than Enron** for LLMs
  - Llama-3.3-70B: 97% vs 91%
  - More diverse patterns easier to recognize

### Speed
- Groq API: 0.4-0.6 emails/second (~2 seconds per email)
- Much faster than local inference (30-50 seconds)
- Still slower than traditional ML (100k+ emails/second)

### Precision vs Recall Trade-off
- **Llama-3.1-8B**: High recall (98%), low precision (64%)
  - Catches all phishing but many false positives
- **Llama-3.3-70B**: Balanced (95% precision, 86% recall)
  - Better overall performance

---

## Comparison with Traditional ML

### Enron Dataset
| Approach | Accuracy | F1 Score | Speed | Winner |
|----------|----------|----------|-------|--------|
| Traditional ML (Logistic Regression) | 98.00% | 98.03% | 601,765 emails/s | ✅ ML |
| Single LLM (Llama-3.3-70B Zero-Shot) | 91.00% | 90.53% | 0.625 emails/s | |
| Single LLM (Llama-3.3-70B Few-Shot) | 94.37% | N/A | 0.580 emails/s | |

**Gap**: ML leads by 3.63-7% on Enron

### Combined Dataset
| Approach | Accuracy | F1 Score | Speed | Winner |
|----------|----------|----------|-------|--------|
| Traditional ML (Naive Bayes) | 99.50% | 99.50% | 125,178 emails/s | ✅ ML |
| Single LLM (Llama-3.3-70B) | 97.00% | 96.70% | 0.453 emails/s | |

**Gap**: ML leads by 2.5% on Combined

---

## Analysis

### Why Traditional ML Still Wins

1. **Pattern Matching**: TF-IDF captures spam/phishing patterns effectively
2. **Training Data**: ML learns from 2,400 training samples
3. **Task Nature**: Phishing detection is largely pattern-based
4. **Speed**: ML is 100,000x faster

### Where LLMs Excel

1. **Zero-Shot**: No training required
2. **Context Understanding**: Can reason about email content
3. **Novel Attacks**: Better generalization potential
4. **Explainability**: Can explain reasoning (not tested)
5. **Few-Shot Learning**: Improves with examples

### LLM Advantages Not Fully Utilized

- This evaluation uses simple classification
- LLMs could provide explanations
- Could handle more complex reasoning tasks
- Could adapt to new attack patterns without retraining

---

## Prompting Strategies Tested

### Zero-Shot Prompt
```
You are an email security classifier. 
Respond with ONLY one word: PHISHING or LEGITIMATE.

Classify this email:
[email text]
```

### Few-Shot Prompt
```
You are an email security classifier.

Examples:
1. [phishing example] → PHISHING
2. [legitimate example] → LEGITIMATE
3. [phishing example] → PHISHING
4. [legitimate example] → LEGITIMATE

Now classify:
[email text]
```

**Result**: Few-shot improved accuracy by 3.37% on Enron

---

## Recommendations

### When to Use Single LLM

1. **No Training Data**: Zero-shot capability valuable
2. **Novel Attacks**: Better generalization than ML
3. **Explainability Needed**: Can provide reasoning
4. **Low Volume**: Speed not critical (<1000 emails/day)
5. **Diverse Patterns**: Works well on varied datasets

### When to Use Traditional ML

1. **High Accuracy Required**: ML achieves 98-99%
2. **High Volume**: Need to process millions of emails
3. **Speed Critical**: Real-time filtering required
4. **Training Data Available**: Can learn patterns
5. **Production Deployment**: Proven and reliable

### Best of Both Worlds

Consider ensemble approaches:
- ML for fast initial filtering
- LLM for uncertain cases
- Combine predictions for robustness

---

## Files
- Results JSON: `single_llm_results.json`
- Training scripts: `../notebooks/single_llm_groq.py`, `../notebooks/few_shot_prompting.py`
