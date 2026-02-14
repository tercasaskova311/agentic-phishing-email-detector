# Results Directory

This directory contains all evaluation results from the phishing email detection project.

## Result Files by Approach

### 1. Traditional ML Results
- **File**: `TRADITIONAL_ML_RESULTS.md`
- **JSON**: `traditional_ml_results.json`
- **Models**: Logistic Regression, Naive Bayes, Random Forest
- **Best Performance**: 98-99.5% accuracy
- **Key Finding**: Excellent baseline, extremely fast

### 2. Single LLM Results
- **File**: `SINGLE_LLM_RESULTS.md`
- **JSON**: `single_llm_results.json`
- **Models**: Llama-3.1-8B, Llama-3.3-70B (zero-shot and few-shot)
- **Best Performance**: 91-97% accuracy
- **Key Finding**: Good zero-shot capability, slower than ML

### 3. Multi-Agent Debate Results
- **File**: `MULTI_AGENT_DEBATE_RESULTS.md`
- **JSON**: `multi_agent_debate_results.json`
- **Structure**: 3-agent debate (Attacker, Defender, Judge)
- **Best Performance**: 76% accuracy (Enron), 54% (Combined)
- **Key Finding**: Complexity hurts performance

### 4. Ensemble Methods Results
- **File**: `ENSEMBLE_AND_EXPLAINABILITY.md`
- **JSON**: `ensemble_results.json`
- **Strategies**: ML Only, LLM Only, Simple Voting, Weighted, ML Primary, LLM Override
- **Best Performance**: 97-99% accuracy
- **Key Finding**: ML Primary best for Enron, ensemble adds value

### 5. Explainability Analysis
- **File**: `ENSEMBLE_AND_EXPLAINABILITY.md`
- **JSON**: `explainability/explainability_results.json`
- **Methods**: SHAP, LIME, Feature Importance, Pattern Analysis
- **Key Finding**: Model decisions are interpretable and logical

---

## Datasets

### Preprocessed Data
- `enron_preprocessed_3k.csv` - 3,000 Enron emails (balanced)
- `combined_preprocessed_2k.csv` - 2,000 combined emails (balanced)
- `legit_preprocessed_1.5k.csv` - 1,500 legitimate emails
- `phishing_preprocessed_1.5k.csv` - 1,500 phishing emails

---

## Performance Summary

### Enron Dataset (3,000 emails)

| Approach | Accuracy | F1 Score | Speed | Rank |
|----------|----------|----------|-------|------|
| Traditional ML | 98.00% | 98.03% | 601,765 emails/s | 🥇 |
| Fine-Tuned LLM | 96.39% | 96.77% | 0.664 emails/s | 🥈 |
| Ensemble (ML Primary) | 97.00% | 97.20% | 0.133 emails/s | 🥈 |
| Single LLM (Few-Shot) | 94.37% | N/A | 0.580 emails/s | 🥉 |
| Single LLM (Zero-Shot) | 91.00% | 90.53% | 0.625 emails/s | 4th |
| Multi-Agent Debate | 76.00% | 72.09% | 0.133 emails/s | 5th |

### Combined Dataset (2,000 emails)

| Approach | Accuracy | F1 Score | Speed | Rank |
|----------|----------|----------|-------|------|
| Traditional ML | 99.50% | 99.50% | 125,178 emails/s | 🥇 |
| Ensemble (ML Only) | 99.00% | 99.00% | 0.120 emails/s | 🥈 |
| Single LLM | 97.00% | 96.70% | 0.453 emails/s | 🥉 |
| Fine-Tuned LLM | 85.14% | 87.91% | 0.659 emails/s | 4th |
| Multi-Agent Debate | 54.00% | 4.17% | 0.120 emails/s | Failed |

---

## Key Insights

### What Works Best
1. **Traditional ML**: 98-99.5% accuracy, extremely fast
2. **Fine-Tuned LLM**: 96.39% accuracy on Enron (close to ML)
3. **Ensemble Methods**: 97-99% accuracy, combines strengths
4. **Single LLM**: 91-97% accuracy, zero-shot capability

### What Doesn't Work
1. **Multi-Agent Debate**: 54-76% accuracy, too complex
2. **LangGraph**: Implementation issues, unreliable results

### Speed Comparison
- Traditional ML: 100,000x faster than LLMs
- Single LLM: 5x faster than debate systems
- Ensemble: Similar speed to debate (multiple calls)

### Accuracy vs Speed Trade-off
- **Need 98%+ accuracy + speed**: Traditional ML
- **Need 96%+ accuracy**: Fine-tuned LLM or Ensemble
- **Need zero-shot**: Single LLM (91-97%)
- **Avoid**: Multi-agent systems for this task

---

## Recommendations by Use Case

### Production Deployment
→ Use **Traditional ML** (Logistic Regression or Naive Bayes)
- 98-99% accuracy
- Extremely fast (600k+ emails/second)
- Proven and reliable

### Research / Novel Attacks
→ Use **Single LLM** or **Fine-Tuned LLM**
- Zero-shot capability
- Better generalization
- Can explain reasoning

### Maximum Accuracy
→ Use **Ensemble (ML Primary)**
- 97-99% accuracy
- Combines ML speed with LLM intelligence
- Robust to edge cases

### Low Volume / Explainability
→ Use **Single LLM with Explainability**
- Can provide reasoning
- Good for user education
- Acceptable speed for low volume

---

## Files Organization

```
results/
├── README.md (this file)
├── TRADITIONAL_ML_RESULTS.md
├── SINGLE_LLM_RESULTS.md
├── MULTI_AGENT_DEBATE_RESULTS.md
├── ENSEMBLE_AND_EXPLAINABILITY.md
├── traditional_ml_results.json
├── single_llm_results.json
├── multi_agent_debate_results.json
├── ensemble_results.json
├── explainability/
│   └── explainability_results.json
├── enron_preprocessed_3k.csv
├── combined_preprocessed_2k.csv
├── legit_preprocessed_1.5k.csv
└── phishing_preprocessed_1.5k.csv
```

---

## Next Steps

1. ✅ Traditional ML baseline established
2. ✅ Single LLM evaluated
3. ✅ Multi-agent systems tested (not recommended)
4. ✅ Fine-tuning completed (96.39% on Enron)
5. ✅ Ensemble methods tested (97-99% accuracy)
6. ✅ Explainability analysis completed

### Future Work
- Test fine-tuned LLM on larger datasets
- Explore ensemble with fine-tuned models
- Deploy best model as production API
- Add real-time monitoring and feedback loop
