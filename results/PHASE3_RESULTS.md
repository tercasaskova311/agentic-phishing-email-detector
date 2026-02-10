# Phase 3: Traditional ML Baseline Results

## Overview
Tested three classical machine learning models on both datasets using TF-IDF vectorization (5000 features).

## Datasets
- **Enron Dataset**: 3,000 emails (1,500 phishing + 1,500 legitimate)
- **Combined Dataset**: 2,000 emails (1,000 phishing + 1,000 legitimate)

## Train/Test Split
- 80% training, 20% testing
- Stratified split to maintain class balance

## Results

### Enron Dataset (3k emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Logistic Regression | 98.00% | 96.45% | 99.67% | 98.03% | 601,765 |
| Naive Bayes | 97.83% | 97.99% | 97.67% | 97.83% | 125,178 |
| Random Forest | 97.17% | 96.39% | 98.00% | 97.19% | 13,104 |

**Best Model**: Logistic Regression (98.0% accuracy, 98.0% F1)

### Combined Dataset (2k emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Logistic Regression | 99.25% | 100.00% | 98.50% | 99.24% | Very Fast |
| Naive Bayes | 99.50% | 99.50% | 99.50% | 99.50% | Very Fast |
| Random Forest | 99.50% | 100.00% | 99.00% | 99.50% | 12,318 |

**Best Models**: Naive Bayes & Random Forest (99.5% accuracy, 99.5% F1)

## Key Findings

1. **Excellent Performance**: All models achieved >97% accuracy on both datasets
2. **Combined Dataset Easier**: Models performed slightly better on the Combined dataset (99%+) vs Enron (97-98%)
3. **Speed**: Logistic Regression and Naive Bayes are extremely fast (100k+ emails/second)
4. **Balanced Metrics**: High precision and recall across all models, indicating good balance between false positives and false negatives

## Baseline Established

These results provide strong baselines for comparison with LLM-based approaches in subsequent phases. The challenge for LLMs will be to match or exceed these performance levels while potentially offering better interpretability or handling of novel phishing patterns.

## Next Phase

Phase 4 will evaluate single LLM agents (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) on the same datasets.
