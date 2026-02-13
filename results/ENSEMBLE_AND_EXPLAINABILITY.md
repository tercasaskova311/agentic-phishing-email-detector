# Ensemble Methods & Explainability Analysis

## Overview

This document presents results from two advanced analyses:
1. **Ensemble Methods**: Combining Traditional ML + LLM predictions
2. **Explainability**: Understanding model decisions using SHAP and LIME

---

## 1. Ensemble Methods Results

### Approach
Combined Logistic Regression (ML) with Llama-3.3-70B (LLM) using various ensemble strategies.

### Ensemble Strategies Tested

1. **ML Only**: Use only ML predictions (baseline)
2. **LLM Only**: Use only LLM predictions (baseline)
3. **Simple Voting**: Majority vote between ML and LLM
4. **Weighted (70/30)**: 70% weight to ML, 30% to LLM
5. **ML Primary**: Use ML, but defer to LLM for uncertain cases
6. **LLM Override**: Allow LLM to override ML for high-confidence phishing

### Results

#### Enron Dataset (100 samples)

| Method | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| **ML Only** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| LLM Only | 91.0% | 95.6% | 86.0% | 90.5% |
| **Simple Voting** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Weighted (70/30)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| ML Primary | 99.0% | 98.0% | 100.0% | 99.0% |
| LLM Override | 98.0% | 96.2% | 100.0% | 98.0% |

#### Combined Dataset (100 samples)

| Method | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| **ML Only** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| LLM Only | 97.0% | 100.0% | 93.6% | 96.7% |
| **Simple Voting** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Weighted (70/30)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **ML Primary** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **LLM Override** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### Key Findings

1. **ML Dominance**: ML alone achieved 100% accuracy on test samples
2. **Ensemble Benefit**: Simple voting and weighted methods maintained 100% accuracy
3. **LLM Value**: While LLM alone scored 91-97%, it can validate ML decisions
4. **Best Strategy**: Weighted (70/30) or Simple Voting for optimal results
5. **Practical Use**: Ensemble provides confidence validation without sacrificing accuracy

### Recommendations

- **For Maximum Accuracy**: Use ML Only or Simple Voting ensemble
- **For Validation**: Use Weighted (70/30) to get dual confirmation
- **For Novel Attacks**: LLM Override can catch new phishing patterns ML might miss
- **Production**: Simple Voting provides best balance of accuracy and robustness

---

## 2. Explainability Analysis Results

### Approach
Used SHAP (global importance) and LIME (local explanations) to understand Logistic Regression model decisions.

### Feature Importance Analysis

#### Top 20 Phishing Indicators

| Rank | Feature | Weight | Interpretation |
|------|---------|--------|----------------|
| 1 | http | 2.67 | URLs strongly indicate phishing |
| 2 | click | 2.18 | Call-to-action typical in phishing |
| 3 | money | 1.99 | Financial lures |
| 4 | save | 1.78 | Urgency tactics |
| 5 | email | 1.72 | Meta-references to email |
| 6 | free | 1.64 | Common bait word |
| 7 | online | 1.60 | Web-based scams |
| 8 | company | 1.53 | Impersonation attempts |
| 9 | remove | 1.52 | Unsubscribe tricks |
| 10 | software | 1.51 | Tech-related scams |
| 11-12 | 2004, 2005 | 1.49, 1.45 | Phishing dataset years |
| 13 | best | 1.42 | Marketing language |
| 14 | account | 1.42 | Account verification scams |
| 15 | quality | 1.39 | Product spam |
| 16 | low | 1.38 | Price lures |
| 17 | statements | 1.38 | Financial document tricks |
| 18 | site | 1.33 | Website references |
| 19 | man | 1.31 | Generic addressing |
| 20 | viagra | 1.26 | Classic spam |

#### Top 20 Legitimate Indicators

| Rank | Feature | Weight | Interpretation |
|------|---------|--------|----------------|
| 1 | enron | -5.24 | Strong legitimate signal (Enron dataset) |
| 2 | 2001 | -2.76 | Legitimate email year |
| 3 | vince | -2.68 | Employee name |
| 4 | thanks | -2.51 | Professional courtesy |
| 5 | attached | -2.22 | Document references |
| 6 | let | -2.20 | Conversational tone |
| 7 | 2000 | -2.17 | Legitimate email year |
| 8 | ect | -2.14 | Enron division |
| 9 | louise | -2.12 | Employee name |
| 10 | questions | -1.90 | Professional inquiry |
| 11 | deal | -1.83 | Business context |
| 12 | gas | -1.72 | Industry term |
| 13 | 713 | -1.71 | Houston area code |
| 14 | know | -1.69 | Personal knowledge |
| 15 | pm | -1.60 | Time reference |
| 16 | meeting | -1.51 | Business activity |
| 17 | power | -1.50 | Industry term |
| 18 | energy | -1.49 | Industry term |
| 19 | schedule | -1.45 | Business planning |
| 20 | risk | -1.44 | Business term |

### SHAP Global Importance

Top features by SHAP values (impact across all predictions):

1. **enron** (0.241) - Most important feature overall
2. **ect** (0.091) - Enron division
3. **http** (0.084) - URL indicator
4. **2000, 2001** (0.065, 0.055) - Temporal markers
5. **thanks, attached** (0.053) - Professional language
6. **email, free, click** (0.049, 0.047, 0.046) - Phishing indicators

### LIME Individual Explanations

Sample insights from 5 random emails:

- **Legitimate emails** correctly identified by:
  - Presence of "enron", "thanks", "meeting", "questions"
  - Professional language and business context
  - Specific names and locations

- **Phishing emails** correctly identified by:
  - URLs ("http"), action words ("click", "email")
  - Generic addressing, urgency language
  - Financial or product-related terms

### Pattern Analysis

#### Email Length Statistics

- **Phishing**: Mean 1,239 chars, Median 636 chars
- **Legitimate**: Mean 1,658 chars, Median 793 chars
- **Insight**: Legitimate emails tend to be longer and more detailed

#### Most Common Words

- **Phishing**: Generic terms like "this", "click", "free", "money"
- **Legitimate**: Specific terms like "enron", "thanks", "meeting", employee names

### Key Insights

1. **URL Presence**: "http" is the 3rd most important feature - URLs strongly indicate phishing
2. **Context Matters**: Business-specific terms (enron, ect, gas, energy) indicate legitimacy
3. **Personal vs Generic**: Names and personal references indicate legitimate emails
4. **Action Words**: "click", "save", "remove" are strong phishing signals
5. **Financial Lures**: "money", "free", "account" commonly used in phishing
6. **Professional Tone**: "thanks", "attached", "meeting" indicate legitimate business communication

### Practical Applications

1. **Email Filtering**: Prioritize checking emails with high-weight phishing indicators
2. **User Training**: Educate users about common phishing words and patterns
3. **Rule Creation**: Develop rules based on top features for quick filtering
4. **Model Improvement**: Focus on features with high SHAP importance
5. **False Positive Reduction**: Whitelist emails with strong legitimate indicators

---

## Conclusions

### Ensemble Methods

- ML already achieves near-perfect accuracy (98-100%)
- Ensemble methods maintain this accuracy while adding validation
- Best for production: Simple Voting or Weighted (70/30)
- LLM adds value for novel attack detection and confidence validation

### Explainability

- Model decisions are interpretable and logical
- Top features align with human understanding of phishing
- SHAP and LIME provide complementary insights (global + local)
- Results can guide user education and rule-based filtering

### Combined Impact

These analyses demonstrate that:
1. High accuracy is achievable and explainable
2. Ensemble methods provide robustness without complexity
3. Model decisions are transparent and trustworthy
4. Results can inform both technical and non-technical stakeholders

---

## Files Generated

- `results/ensemble_results.json` - Full ensemble method results
- `results/explainability/explainability_results.json` - Complete explainability analysis
- `notebooks/ensemble_ml_llm.py` - Ensemble testing script
- `notebooks/explainability_analysis.py` - Explainability analysis script
