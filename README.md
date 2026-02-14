# Phishing Email Detection: Comprehensive ML & LLM Analysis

A systematic comparison of traditional ML models, single LLM agents, multi-agent debate systems, and graph-based approaches for phishing email detection.

## Project Overview

This project explores how far we can push Large Language Models (LLMs) for phishing email detection, with the goal of matching or exceeding traditional ML performance (98-99% accuracy).

**Research Question**: Can LLMs, through various techniques (zero-shot, multi-agent, fine-tuning), achieve performance comparable to traditional ML for phishing detection?

**Datasets**:
- **Enron Dataset**: 3,000 emails (1,500 phishing + 1,500 legitimate)
- **Combined Dataset**: 2,000 emails (1,000 phishing + 1,000 legitimate)



## Results Summary

### Enron Dataset (3,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Status |
|----------|----------|-----------|--------|----------|------------------|--------|
| **Traditional ML** | **98.00%** | **96.45%** | **99.67%** | **98.03%** | **601,765** | Best |
| **Fine-Tuned LLM** | **96.39%** | **98.00%** | **93.62%** | **96.77%** | **0.664** | Very Good |
| **Single LLM** | **91.00%** | **95.56%** | **86.00%** | **90.53%** | **0.625** | Good |
| Debate System | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 | Poor |

### Combined Dataset (2,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Status |
|----------|----------|-----------|--------|----------|------------------|--------|
| **Traditional ML** | **99.50%** | **99.50%** | **99.50%** | **99.50%** | **125,178** | Best |
| **Single LLM** | **97.00%** | **96.00%** | **93.62%** | **96.70%** | **0.453** | Very Good |
| **Fine-Tuned LLM** | **85.14%** | **N/A** | **N/A** | **87.91%** | **0.659** | Good |
| Debate System | 54.00% | 85.00% | 2.13% | 4.17% | 0.120 | Failed |



**Progress Toward Goal**: 
- **Baseline (ML)**: 98-99% accuracy - the target to match
- **Fine-Tuned LLM**: 96.39% (Enron), 85.14% (Combined) - significant improvement
- **Zero-Shot LLM**: 91-97% accuracy - good starting point
- **Multi-Agent Attempts**: 53-76% accuracy - complexity hurt performance

**Key Achievement**: Fine-tuning closed the gap on Enron from 7% to just 1.61% (96.39% vs 98%)

## Evaluation Approaches

### Model Selection
Selected 3 open-source LLMs for testing:
- **Qwen/Qwen2.5-3B-Instruct**: Fast, efficient, good reasoning
- **meta-llama/Llama-3.2-3B-Instruct**: Strong performance, widely adopted
- **google/gemma-2-2b-it**: Lightweight, resource-efficient

[Detailed Documentation](docs/MODEL_SELECTION.md)

### Data Preprocessing
- Cleaned and standardized three raw datasets
- Created balanced samples (50/50 phishing/legitimate)
- Generated two test datasets: Enron (3k) and Combined (2k)
- Separate preprocessing for each dataset due to different formats

[Detailed Documentation](docs/DATA_PREPROCESSING.md)

### Traditional ML Baseline
Tested classical ML models with TF-IDF features on both datasets:

**Enron Dataset**:
- **Logistic Regression**: 98.00% accuracy, 98.03% F1
- **Naive Bayes**: 97.83% accuracy, 97.83% F1
- **Random Forest**: 97.17% accuracy, 97.19% F1

**Combined Dataset**:
- **Naive Bayes**: 99.50% accuracy, 99.50% F1
- **Random Forest**: 99.50% accuracy, 99.50% F1
- **Logistic Regression**: 99.25% accuracy, 99.24% F1

**Result**: Excellent baseline (97-99% accuracy), extremely fast (100k+ emails/second)

[Detailed Documentation](docs/TRADITIONAL_ML.md)

### Single LLM Evaluation
Tested LLMs via Groq API (zero-shot classification) on both datasets:

**Enron Dataset**:
- **Llama-3.3-70B**: 91% accuracy, 90.53% F1
- **Llama-3.1-8B**: 72% accuracy, 77.78% F1

**Combined Dataset**:
- **Llama-3.3-70B**: 97% accuracy, 96.70% F1
- **Llama-3.1-8B**: 91% accuracy, 91.09% F1

**Result**: Competitive performance (91-97%), but slower and below traditional ML

[Detailed Documentation](docs/SINGLE_LLM.md)

### Multi-Agent Debate System
Three-agent system (Attacker, Defender, Judge) tested on both datasets:

**Enron Dataset**:
- **Improved Version**: 76% accuracy, 72.09% F1
- **Original Version**: 69% accuracy, 64.37% F1

**Combined Dataset**:
- **Both Versions Failed**: 54-55% accuracy, 2-8% F1, 98-99% failure rate

**Result**: Underperformed single LLM, high failure rates, 3-5x slower

[Detailed Documentation](docs/MULTI_AGENT_DEBATE.md)

### Fine-Tuning

Fine-tuned Gemma 2B model with LoRA on phishing detection task:
- **Model**: google/gemma-2-2b-it
- **Method**: LoRA fine-tuning
- **Training**: 2,400 Enron emails
- **Results**:
  - Enron: 96.39% accuracy, 96.77% F1 (only 1.61% below ML!)
  - Combined: 85.14% accuracy, 87.91% F1
  - Speed: 0.664-0.906 emails/second

**Achievement**: Closed the gap from 7% to 1.61% on Enron dataset through fine-tuning

[Detailed Documentation](docs/FINETUNING.md)

## Evaluation Metrics

All approaches evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Speed**: Emails processed per second

## Complete Documentation

- [Model Selection](docs/MODEL_SELECTION.md)
- [Data Preprocessing](docs/DATA_PREPROCESSING.md)
- [Traditional ML](docs/TRADITIONAL_ML.md)
- [Single LLM](docs/SINGLE_LLM.md)
- [Multi-Agent Debate](docs/MULTI_AGENT_DEBATE.md)
- [Fine-Tuning](docs/FINETUNING.md)

## Project Structure

```
phishing-detection-project/
├── first_datasets/                # Raw datasets
│   ├── enron.csv
│   ├── legit.csv
│   └── phishing.csv
├── docs/                          # Documentation
│   ├── MODEL_SELECTION.md
│   ├── DATA_PREPROCESSING.md
│   ├── TRADITIONAL_ML.md
│   ├── SINGLE_LLM.md
│   ├── MULTI_AGENT_DEBATE.md
│   └── FINETUNING.md
├── notebooks/                     # Evaluation scripts
│   ├── preprocess_*.py           # Data preprocessing
│   ├── traditional_ml_baseline.py # ML baseline
│   ├── single_llm_groq.py        # Zero-shot LLM
│   ├── few_shot_prompting.py     # Few-shot LLM (best)
│   ├── chain_of_thought_prompting.py
│   ├── llm_ensemble.py
│   ├── debate_system.py
│   └── finetune_colab.ipynb      # Fine-tuning notebook
├── results/                       # Evaluation results
│   ├── TRADITIONAL_ML_RESULTS.md
│   ├── SINGLE_LLM_RESULTS.md
│   ├── MULTI_AGENT_DEBATE_RESULTS.md
│   ├── ENSEMBLE_AND_EXPLAINABILITY.md
│   ├── traditional_ml_results.json
│   ├── single_llm_results.json
│   ├── multi_agent_debate_results.json
│   ├── ensemble_results.json
│   ├── enron_preprocessed_3k.csv
│   ├── combined_preprocessed_2k.csv
│   ├── legit_preprocessed_1.5k.csv
│   └── phishing_preprocessed_1.5k.csv
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

## Notebooks Guide

### Data Preprocessing Scripts
- `preprocess_enron.py` - Clean and prepare Enron dataset
- `preprocess_legit.py` - Clean and prepare legitimate emails dataset
- `preprocess_phishing.py` - Clean and prepare phishing emails dataset
- `create_combined_dataset.py` - Merge legit + phishing into combined dataset

### Running Preprocessing
```bash
cd notebooks
python preprocess_enron.py
python preprocess_legit.py
python preprocess_phishing.py
python create_combined_dataset.py
```

### Evaluation Scripts

**Traditional ML**:
- `traditional_ml_baseline.py` - Test Logistic Regression, Naive Bayes, Random Forest (98-99% accuracy)

**Single LLM Approaches**:
- `single_llm_groq.py` - Zero-shot classification with Groq API (91-97% accuracy)
- `few_shot_prompting.py` - Provide 4 examples in prompt (94.37% accuracy on Enron)
- `chain_of_thought_prompting.py` - Ask for step-by-step reasoning
- `llm_ensemble.py` - Multiple models voting

**Multi-Agent Systems**:
- `debate_system.py` - Three agents: Attacker, Defender, Judge (76% accuracy, underperformed single LLM)

**Fine-Tuning**:
- `finetune_colab.ipynb` - Google Colab notebook for fine-tuning (requires T4 GPU, 15-30 minutes)

### Running Evaluations
```bash
cd notebooks

# Traditional ML baseline
python traditional_ml_baseline.py

# Single LLM (requires Groq API key)
export GROQ_API_KEY="your-api-key-here"
python single_llm_groq.py

# Few-shot prompting (best LLM result)
python few_shot_prompting.py

# Other approaches
python chain_of_thought_prompting.py
python llm_ensemble.py
python debate_system.py
```

## Results Files

All evaluation results are stored in the `results/` directory:

### Result Documents
- **TRADITIONAL_ML_RESULTS.md** - Logistic Regression, Naive Bayes, Random Forest (98-99.5% accuracy)
- **SINGLE_LLM_RESULTS.md** - Llama-3.1-8B, Llama-3.3-70B zero-shot and few-shot (91-97% accuracy)
- **MULTI_AGENT_DEBATE_RESULTS.md** - 3-agent debate system (76% Enron, 54% Combined)
- **ENSEMBLE_AND_EXPLAINABILITY.md** - ML+LLM ensemble methods and SHAP/LIME analysis (97-99% accuracy)

### Result Data Files
- `traditional_ml_results.json` - ML model metrics
- `single_llm_results.json` - LLM evaluation metrics
- `multi_agent_debate_results.json` - Debate system metrics
- `ensemble_results.json` - Ensemble method metrics
- `explainability/explainability_results.json` - SHAP and LIME analysis

### Preprocessed Datasets
- `enron_preprocessed_3k.csv` - 3,000 Enron emails (balanced)
- `combined_preprocessed_2k.csv` - 2,000 combined emails (balanced)
- `legit_preprocessed_1.5k.csv` - 1,500 legitimate emails
- `phishing_preprocessed_1.5k.csv` - 1,500 phishing emails

### Performance Rankings

**Enron Dataset (3,000 emails)**:
1. Traditional ML: 98.00% accuracy, 601,765 emails/s
2. Fine-Tuned LLM: 96.39% accuracy, 0.664 emails/s
3. Ensemble (ML Primary): 97.00% accuracy, 0.133 emails/s
4. Single LLM (Few-Shot): 94.37% accuracy, 0.580 emails/s
5. Single LLM (Zero-Shot): 91.00% accuracy, 0.625 emails/s
6. Multi-Agent Debate: 76.00% accuracy, 0.133 emails/s

**Combined Dataset (2,000 emails)**:
1. Traditional ML: 99.50% accuracy, 125,178 emails/s
2. Ensemble (ML Only): 99.00% accuracy, 0.120 emails/s
3. Single LLM: 97.00% accuracy, 0.453 emails/s
4. Fine-Tuned LLM: 85.14% accuracy, 0.659 emails/s
5. Multi-Agent Debate: 54.00% accuracy (failed)

### Key Insights from Results
- **Traditional ML** is best for production: 98-99.5% accuracy, extremely fast
- **Fine-Tuned LLM** closes gap to 1.61% on Enron (96.39% vs 98%)
- **Ensemble methods** combine strengths: ML Primary achieves 97% on Enron
- **Multi-agent debate** underperforms: complexity hurts performance (54-76%)
- **Model decisions are interpretable**: SHAP/LIME show logical feature importance

## Key Findings

### LLM Performance Journey

**Goal**: Push LLMs to match traditional ML (98-99% accuracy)

**Attempts**:
1. **Zero-Shot LLM**: 91-97% accuracy
   - Good starting point, but 2-7% gap from ML
   - No training required
   - Shows LLMs understand phishing concepts

2. **Multi-Agent Debate**: 54-76% accuracy
   - Hypothesis: Multiple perspectives would improve accuracy
   - Reality: Complexity introduced more errors
   - Lesson: More agents does not equal better performance

3. **Graph-Based Systems**: 53-55% accuracy
   - Hypothesis: Structured workflows would help coordination
   - Reality: Added overhead without benefits
   - Lesson: Framework complexity doesn't solve fundamental issues

4. **Fine-Tuning**: 96.39% accuracy (Enron), 85.14% (Combined)
   - Hypothesis: Task-specific training will close the gap
   - Result: Successfully closed gap to just 1.61% on Enron!
   - Achievement: Proved LLMs can nearly match ML with proper training

### What We Learned

**What Helps LLMs**:
- Larger models (70B > 8B)
- Task-specific fine-tuning (expected)
- Simple, direct prompts
- Lower temperature for consistency

**What Hurts LLMs**:
- Multi-agent complexity
- Sequential API calls
- Complex orchestration
- Graph-based overhead

**The Path Forward**:
- Fine-tuning single LLM is most promising
- Keep it simple - avoid multi-agent complexity
- Focus on model quality over system architecture

## Recommendations

### Research Goal: Pushing LLMs to Match ML

**Current Status**:
- ML Baseline: 98-99% accuracy ← Target
- Fine-Tuned LLM: 96.39% accuracy (Enron) ← Only 1.61% gap!
- Best LLM (Zero-Shot): 91-97% accuracy ← 2-7% gap

**For Achieving ML-Level Performance with LLMs**:

**Recommended Approach**:
1. Fine-tune single LLM on task-specific data (achieved 96.39% on Enron)
2. Use larger models (70B parameters for zero-shot)
3. Keep architecture simple (avoid multi-agent)
4. Optimize prompts for consistency

**Avoid**:
- Multi-agent systems (76% accuracy, worse than single LLM)
- Graph-based orchestration (55% accuracy, added complexity)
- Complex workflows (more failure points)

### For Production Use (Current State)

**If you need 98-99% accuracy NOW**:
- Use Traditional ML (Logistic Regression or Naive Bayes)
- Extremely fast (600k emails/second)
- Proven and reliable

**If you want LLM flexibility**:
- Use Fine-Tuned LLM (Gemma 2B) at 96.39% accuracy (Enron)
- Or use Single LLM (Llama-3.3-70B) at 91-97% for zero-shot
- Fine-tuning successfully closed the gap to within 1.61% of ML

### Future Work to Further Improve

1. **Ensemble Approach**
   - Combine fine-tuned LLM + Traditional ML
   - Could achieve best of both worlds (98%+ accuracy)

2. **Fine-Tune Larger Models**
   - Apply same technique to 7B or 13B models
   - May close the remaining 1.61% gap

3. **Prompt Optimization**
   - Few-shot learning with examples
   - Chain-of-thought for complex cases

4. **Cross-Dataset Training**
   - Train on Combined dataset to improve generalization
   - Current fine-tuning only used Enron data

## Technologies Used

- **Languages**: Python 3.12
- **ML Libraries**: scikit-learn, pandas, numpy
- **LLM Frameworks**: Transformers, Unsloth, LangChain
- **APIs**: Groq (fast LLM inference), Ollama (local inference)
- **Fine-tuning**: Unsloth with LoRA
- **Platforms**: Local (Windows), Kaggle (GPU training)

## Requirements

```bash
# Traditional ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# LLM Inference
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
| Debate System | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 |

### Combined Dataset (2,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|----------|----------|-----------|--------|----------|------------------|
| Naive Bayes | 99.50% | 99.50% | 99.50% | 99.50% | 125,178 |
| Random Forest | 99.50% | 99.00% | 99.00% | 99.50% | 12,318 |
| Logistic Regression | 99.25% | 99.00% | 98.50% | 99.24% | Very Fast |
| Llama-3.3-70B (Single) | 97.00% | 96.00% | 93.62% | 96.70% | 0.453 |
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

[Full Results](results/ENSEMBLE_AND_EXPLAINABILITY.md)

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

[Full Analysis](results/ENSEMBLE_AND_EXPLAINABILITY.md)
