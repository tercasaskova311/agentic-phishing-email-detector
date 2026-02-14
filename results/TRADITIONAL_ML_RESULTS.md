# Traditional ML Results

## Overview
Classical machine learning models tested on phishing detection using TF-IDF vectorization (5000 features).

## Datasets
- **Enron Dataset**: 3,000 emails (1,500 phishing + 1,500 legitimate)
- **Combined Dataset**: 2,000 emails (1,000 phishing + 1,000 legitimate)

## Methodology
- Train/Test Split: 80% training, 20% testing
- Stratified split to maintain class balance
- TF-IDF vectorization with 5000 features
- English stop words removed

## Models Tested

### 1. Logistic Regression
- Max iterations: 1000
- Random state: 42
- Fast training and inference

### 2. Naive Bayes (MultinomialNB)
- Default parameters
- Extremely fast
- Works well with text data

### 3. Random Forest
- 100 estimators
- Parallel processing enabled
- Random state: 42

---

## Results

### Enron Dataset (3,000 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Training Time |
|-------|----------|-----------|--------|----------|------------------|---------------|
| **Logistic Regression** | **98.00%** | **96.45%** | **99.67%** | **98.03%** | **601,765** | 0.04s |
| Naive Bayes | 97.83% | 97.99% | 97.67% | 97.83% | 125,178 | <0.01s |
| Random Forest | 97.17% | 96.39% | 98.00% | 97.19% | 13,104 | 0.55s |

**Best Model**: Logistic Regression
- Highest accuracy (98.0%)
- Excellent recall (99.67%) - catches almost all phishing
- Extremely fast inference (600k+ emails/second)

### Combined Dataset (2,000 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Training Time |
|-------|----------|-----------|--------|----------|------------------|---------------|
| **Naive Bayes** | **99.50%** | **99.50%** | **99.50%** | **99.50%** | **Very Fast** | <0.01s |
| **Random Forest** | **99.50%** | **99.00%** | **99.00%** | **99.50%** | **12,318** | 0.29s |
| Logistic Regression | 99.25% | 99.00% | 98.50% | 99.24% | Very Fast | 0.02s |

**Best Models**: Naive Bayes & Random Forest (tied at 99.5%)
- Near-perfect accuracy
- Balanced precision and recall
- Fast training and inference

---

## Key Findings

### Performance
1. **Excellent Accuracy**: All models achieved 97-99.5% accuracy
2. **Combined Dataset Easier**: Models performed slightly better on Combined (99%+) vs Enron (97-98%)
3. **Balanced Metrics**: High precision and recall across all models
4. **Consistent Results**: All three models performed similarly, indicating robust patterns

### Speed
1. **Logistic Regression**: Fastest (600k+ emails/second)
2. **Naive Bayes**: Very fast (125k+ emails/second)
3. **Random Forest**: Slower but still fast (12-13k emails/second)

### Training Time
1. **Naive Bayes**: Instant (<0.01s)
2. **Logistic Regression**: Very fast (0.02-0.04s)
3. **Random Forest**: Moderate (0.29-0.55s)

### Model Characteristics

**Logistic Regression**:
- ✅ Best overall performance on Enron
- ✅ Extremely fast
- ✅ Interpretable coefficients
- ✅ Good for production

**Naive Bayes**:
- ✅ Best on Combined dataset
- ✅ Fastest training
- ✅ Works well with text
- ✅ Simple and reliable

**Random Forest**:
- ✅ Tied best on Combined
- ✅ Robust to overfitting
- ✅ Good feature importance
- ⚠️ Slower than others

---

## Comparison with LLM Approaches

Traditional ML establishes strong baselines:
- **Enron**: 98.0% accuracy (Logistic Regression)
- **Combined**: 99.5% accuracy (Naive Bayes/Random Forest)

LLM approaches need to match or exceed these levels while offering:
- Better handling of novel phishing patterns
- Zero-shot capability
- Contextual understanding
- Explainability

---

## Recommendations

### For Production Use
1. **Enron-like datasets**: Use Logistic Regression (98% accuracy, extremely fast)
2. **Diverse datasets**: Use Naive Bayes or Random Forest (99.5% accuracy)
3. **Real-time filtering**: Logistic Regression or Naive Bayes (fastest)
4. **Batch processing**: Any model works well

### For Research
- These results provide strong baselines for LLM comparison
- Focus on cases where LLMs might excel (novel attacks, context-dependent)
- Consider ensemble approaches combining ML + LLM

---

## Files
- Results JSON: `traditional_ml_results.json`
- Training scripts: `../notebooks/traditional_ml_baseline.py`
