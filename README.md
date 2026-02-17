# Phishing Email Detection: Comprehensive ML & LLM Analysis

This repository compares traditional ML, single LLMs, multi-agent debate systems, ensemble methods, and fine-tuning for phishing email detection. The goal is to see how close LLM approaches can get close to the 98-99% accuracy achieved by classical ML.

## Project Overview

**Research Question**: Can LLMs (zero-shot, multi-agent, and fine-tuned) match traditional ML performance for phishing detection?

**Primary Datasets**:
- **Enron**: 3,000 emails (1,500 phishing + 1,500 legitimate)
- **Combined**: 2,000 emails (1,000 phishing + 1,000 legitimate)

**Raw Sources**:
- Enron corpus + two smaller labeled datasets (legit.csv, phishing.csv)

## Results Highlights

Traditional ML remains the top performer for speed and accuracy, while fine-tuning closes much of the gap on Enron. Full per-dataset results are listed in the two tables under **Detailed Results**.

## Model Selection (LLMs)

Selected three small, open-source models for local and API testing:

1. **Qwen/Qwen2.5-3B-Instruct**: Fast and efficient reasoning
2. **meta-llama/Llama-3.2-3B-Instruct**: Strong performance and community support
3. **google/gemma-2-2b-it**: Lightweight, resource-efficient, good for fine-tuning

Selection criteria: open-source, 2-3B parameters, good accuracy/speed trade-off, works with Ollama and Groq.

## Data Preprocessing Summary

**Steps**:
- Normalize formatting and labels across datasets
- Clean text, remove noise, and handle missing values
- Create balanced splits (50/50 phishing/legitimate)
- Build two evaluation sets: Enron (3k) and Combined (2k)

**Outputs**:
- enron_preprocessed_3k.csv (3,000 emails)
- legit_preprocessed_1.5k.csv (1,000 emails)
- phishing_preprocessed_1.5k.csv (1,000 emails)
- combined_preprocessed_2k.csv (2,000 emails)

## Traditional ML Baseline

**Models**: Logistic Regression, Naive Bayes, Random Forest

**Method**:
- TF-IDF features (5,000) with unigram/bigram support
- 80/20 stratified split

**Best results**:
- Enron: Logistic Regression 98.00% (F1 98.03%)
- Combined: Naive Bayes / Random Forest 99.50% (F1 99.50%)

## Single LLM Evaluation

**Models**:
- Llama-3.1-8B-Instant
- Llama-3.3-70B-Versatile

**Prompt**: Zero-shot classification, single-word output

**Best results**:
- Enron: Llama-3.3-70B at 91.00% (Few-shot reached 94.37%)
- Combined: Llama-3.3-70B at 97.00%

**Other models**:
- Llama-3.1-8B-Instant: 72.00% (Enron), 91.00% (Combined)
- Mixtral-8x7B: 50.00% (Enron), 53.00% (Combined), 0% success rate in this run

## Multi-Agent Debate System

**Architecture**: Attacker (phishing), Defender (legitimate), Judge (final)

**Findings**:
- Enron improved to 76% accuracy with better prompts
- Combined dataset failed (98-99% failure rate)
- 3-5x slower and less reliable than single LLM

## Fine-Tuning (LoRA)

**Model**: google/gemma-2-2b-it
**Method**: LoRA adapters via Unsloth
**Training**: 2,400 Enron emails (80/20 split)

**Results**:
- Enron: 96.39% accuracy, 96.77% F1
- Combined: 85.14% accuracy (trained on different data)

## Ensemble Methods & Explainability

**Ensemble** (200 samples total):
- ML Primary (confidence switch at 0.6) reached 97% on Enron
- LLM Override achieved 100% on Combined (small sample)

**Full ensemble metrics (accuracy / precision / recall / F1)**:
- Enron: ML Only 84.0 / 0.7714 / 1.0000 / 0.8710
- Enron: LLM Only 93.0 / 0.9796 / 0.8889 / 0.9320
- Enron: Simple Voting 84.0 / 0.7714 / 1.0000 / 0.8710
- Enron: Weighted (70/30) 84.0 / 0.7714 / 1.0000 / 0.8710
- Enron: ML Primary 97.0 / 0.9811 / 0.9630 / 0.9720
- Enron: LLM Override 83.0 / 0.7606 / 1.0000 / 0.8640
- Combined: ML Only 99.0 / 1.0000 / 0.9804 / 0.9901
- Combined: LLM Only 97.0 / 1.0000 / 0.9412 / 0.9697
- Combined: Simple Voting 99.0 / 1.0000 / 0.9804 / 0.9901
- Combined: Weighted (70/30) 99.0 / 1.0000 / 0.9804 / 0.9901
- Combined: ML Primary 99.0 / 1.0000 / 0.9804 / 0.9901
- Combined: LLM Override 100.0 / 1.0000 / 1.0000 / 1.0000

**Explainability**:
- SHAP and LIME highlight URLs, urgency words, and financial language as phishing cues
- Business-specific terms and names correlate with legitimate emails

## Recommendations

**Production**:
- Use Traditional ML for best accuracy and speed (98-99.5%, 100k+ emails/s)

**Research / LLM Flexibility**:
- Fine-tune a single model (best LLM approach so far)
- Keep architectures simple; avoid multi-agent overhead

## Project Layout

```
phishing-detection-project/
├── docs/                          # Methodology documentation
├── first_datasets/                # Raw datasets
├── notebooks/                     # Experiments and scripts
├── results/                       # Result documents and artifacts
├── README.md
└── requirements.txt
```

## Quick Run Notes

- Preprocessing scripts live in notebooks/ (preprocess_*.py)
- Traditional ML baseline: notebooks/traditional_ml_baseline.py
- Single LLM tests: notebooks/single_llm_groq.py, notebooks/few_shot_prompting.py
- Multi-agent debate: notebooks/debate_system.py
- Fine-tuning: notebooks/finetune_colab.ipynb

## Requirements

Python 3.12
pandas>=2.0.0, numpy>=1.24.0, scikit-learn>=1.3.0
transformers>=4.35.0
torch>=2.1.0
groq>=0.4.0

# Agent Frameworks
langchain>=0.1.0

# Fine-tuning
unsloth>=2024.1.0
trl>=0.7.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/06sahar06/phishing-emails.git
cd phishing-emails
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Traditional ML (Best Approach)
```bash
python notebooks/traditional_ml_baseline.py
```

### 4. Test LLM Approaches (Requires Groq API Key)
```bash
export GROQ_API_KEY="your-api-key-here"

# Zero-shot
python notebooks/single_llm_groq.py

# Few-shot (best LLM result: 94.37%)
python notebooks/few_shot_prompting.py

# Other approaches
python notebooks/chain_of_thought_prompting.py
python notebooks/llm_ensemble.py
```

### 5. Fine-Tuning
Fine-tuning requires GPU access. See documentation for details.

## Detailed Results

### Enron Dataset (3,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|----------|----------|-----------|--------|----------|------------------|
| Logistic Regression | 98.00% | 96.45% | 99.67% | 98.03% | 601,765 |
| Naive Bayes | 97.83% | 97.99% | 97.67% | 97.83% | 125,178 |
| Random Forest | 97.17% | 96.39% | 98.00% | 97.19% | 13,104 |
| Fine-Tuned Gemma 2B | 96.39% | 98.00% | 93.62% | 96.77% | 0.664 |
| Llama-3.3-70B (Single) | 91.00% | 95.56% | 86.00% | 90.53% | 0.625 |
| Llama-3.1-8B (Single) | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 |
| Mixtral-8x7B (Single) | 50.00% | 0.00% | 0.00% | 0.00% | 0.000 |
| Debate System | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 |

### Combined Dataset (2,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|----------|----------|-----------|--------|----------|------------------|
| Naive Bayes | 99.50% | 99.50% | 99.50% | 99.50% | 125,178 |
| Random Forest | 99.50% | 99.00% | 99.00% | 99.50% | 12,318 |
| Logistic Regression | 99.25% | 99.00% | 98.50% | 99.24% | Very Fast |
| Llama-3.3-70B (Single) | 97.00% | 96.00% | 93.62% | 96.70% | 0.453 |
| Llama-3.1-8B (Single) | 91.00% | 85.19% | 97.87% | 91.09% | 0.372 |
| Mixtral-8x7B (Single) | 53.00% | 0.00% | 0.00% | 0.00% | 0.000 |
| Fine-Tuned Gemma 2B | 85.14% | N/A | N/A | 87.91% | 0.659 |
| Debate System | 54.00% | 85.00% | 2.13% | 4.17% | 0.120 |

## Advanced Analyses

### Ensemble Methods (ML + LLM)

Combined Traditional ML with LLM predictions using various strategies on proper train/test splits:

**Results (100 test samples per dataset):**

Enron Dataset:
- **ML Primary (Best)**: 97.0% accuracy, 97.2% F1
- LLM Only: 93.0% accuracy, 93.2% F1
- ML Only: 84.0% accuracy, 87.1% F1

Combined Dataset:
- ML Only: 99.0% accuracy, 99.0% F1
- LLM Only: 97.0% accuracy, 97.0% F1
- Simple Voting: 99.0% accuracy, 99.0% F1

**Key Finding**: Ensemble methods successfully combine strengths - ML Primary achieves 97% on Enron (better than either alone), demonstrating real value in combining ML and LLM approaches.

### Explainability Analysis (SHAP & LIME)

Analyzed model decisions to understand feature importance:

**Top Phishing Indicators:**
- http (2.67), click (2.18), money (1.99), free (1.64)
- URLs and action words are strongest signals

**Top Legitimate Indicators:**
- enron (-5.24), thanks (-2.51), attached (-2.22), meeting (-1.51)
- Business context and professional language

**Insights:**
- Model decisions are interpretable and logical
- Features align with human understanding of phishing
- Results can guide user education and filtering rules

## Contributing

This is a research project. For questions or suggestions, please open an issue on GitHub.

## Citation

If you use this work, please cite:
```
Phishing Email Detection: Comprehensive ML & LLM Analysis
GitHub: https://github.com/tercasaskova311/agentic-phishing-email-detector.git
Year: 2026
```

## Acknowledgments

- Enron email dataset
- Groq for fast LLM inference API
- Unsloth for efficient fine-tuning
- Open-source LLM community

## Contact

Repository: [phishing-emails](https://github.com/tercasaskova311/agentic-phishing-email-detector.git)
