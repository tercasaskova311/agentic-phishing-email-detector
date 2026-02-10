# Phase 5: Multi-Agent Debate System Results

## Overview
Implemented a three-agent debate system where agents collaborate to classify emails:
- **Attacker Agent**: Identifies phishing threats (Llama-3.1-8B, temp=0.7)
- **Defender Agent**: Argues for legitimacy (Llama-3.1-8B, temp=0.3)
- **Judge Agent**: Makes final decision (Llama-3.3-70B, temp=0.1)

Each model tested on 100 emails per dataset.

## Results

### Enron Dataset (100 emails: 50 phishing + 50 legitimate)

| Metric | Value |
|--------|-------|
| Accuracy | 69.00% |
| Precision | 75.68% |
| Recall | 56.00% |
| F1 Score | 64.37% |
| Speed | 0.096 emails/second (~10.4s per email) |
| Success Rate | 58/100 (58%) |

### Combined Dataset (100 emails: 47 phishing + 53 legitimate)

| Metric | Value |
|--------|-------|
| Accuracy | 55.00% |
| Precision | 100.00% |
| Recall | 4.26% |
| F1 Score | 8.16% |
| Speed | 0.091 emails/second (~11s per email) |
| Success Rate | 2/100 (2%) ⚠️ |

## Issues Encountered

1. **Low Success Rate on Combined Dataset**: Only 2% of debates completed successfully
   - Likely API errors or parsing issues
   - Judge agent may have failed to provide proper format

2. **Slower Than Single LLM**: ~10 seconds per email vs 2 seconds for single LLM
   - 3 sequential API calls per email
   - Could be parallelized for speed

3. **Lower Accuracy Than Single LLM**: 
   - Enron: 69% vs 91% (single Llama-3.3-70B)
   - Combined: 55% vs 97% (single Llama-3.3-70B)

## Comparison with Previous Phases

### Enron Dataset
| Approach | Accuracy | F1 Score |
|----------|----------|----------|
| Traditional ML (Logistic Regression) | 98.00% | 98.03% |
| Single LLM (Llama-3.3-70B) | 91.00% | 90.53% |
| **Debate System** | **69.00%** | **64.37%** |

### Combined Dataset
| Approach | Accuracy | F1 Score |
|----------|----------|----------|
| Traditional ML (Naive Bayes) | 99.50% | 99.50% |
| Single LLM (Llama-3.3-70B) | 97.00% | 96.70% |
| **Debate System** | **55.00%** | **8.16%** |

## Analysis

**The debate system underperformed expectations:**

1. **Complexity Hurts**: Multiple agents introduced more failure points
2. **Sequential Processing**: Slower than single LLM
3. **Parsing Issues**: Judge's responses may not follow expected format
4. **No Clear Benefit**: Debate didn't improve accuracy over single LLM

**Possible Improvements:**
- Better prompt engineering for consistent output format
- Parallel agent calls instead of sequential
- Simpler debate structure (2 agents instead of 3)
- Better error handling and retry logic
- Use structured output (JSON) instead of free text

## Conclusion

For this phishing detection task:
- **Traditional ML remains best** (98-99% accuracy)
- **Single LLM is competitive** (91-97% accuracy)
- **Debate system needs refinement** (55-69% accuracy)

The debate approach may work better with:
- More complex tasks requiring multiple perspectives
- Better prompt engineering
- Structured output formats
- Parallel processing

## Next Phase

Phase 6 will explore **graph-based agent systems** using frameworks like LangGraph, which may provide better coordination and error handling than the simple sequential debate approach.
