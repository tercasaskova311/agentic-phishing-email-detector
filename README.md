# Phishing Email Detection: Comprehensive ML & LLM Analysis

A systematic comparison of traditional ML models, single LLM agents, multi-agent debate systems, and graph-based approaches for phishing email detection.

## Project Overview

This project evaluates multiple approaches to phishing email detection across three different datasets:
- **Enron Dataset**: Historical email corpus with spam/ham labels
- **Legit Dataset**: Legitimate emails from various sources
- **Phishing Dataset**: Known phishing emails with metadata

## Evaluation Phases

### Phase 1: Model Selection ✓
Selected 3 open-source LLMs for testing:
- **Qwen/Qwen2.5-3B-Instruct**: Fast, efficient, good reasoning
- **meta-llama/Llama-3.2-3B-Instruct**: Strong performance, widely adopted
- **google/gemma-2-2b-it**: Lightweight, good for resource constraints

### Phase 2: Data Preprocessing (In Progress)
- Clean and standardize three datasets
- Create balanced 3k samples (1.5k phishing + 1.5k legitimate)
- Separate preprocessing scripts for each dataset due to different formats

### Phase 3: Traditional ML Baseline
Test classical ML models on each dataset:
- Logistic Regression
- Naive Bayes
- Random Forest

### Phase 4: Single LLM Evaluation
Test each of the 3 LLMs individually on each dataset

### Phase 5: Multi-Agent Debate System
Implement debate-based classification:
- Agent 1: Attacker (identifies phishing indicators)
- Agent 2: Defender (argues for legitimacy)
- Agent 3: Judge (makes final decision)

Frameworks to test:
- Ollama (local inference)
- CrewAI (agent orchestration)
- Direct API calls

### Phase 6: Graph-Based Systems
Implement graph-based agent systems:
- GraphRAG
- LangGraph
- CrewAI with graph workflows

### Phase 7: Fine-tuning & Optimization
Fine-tune LLMs using Unsloth and test:
- Single agent approach
- Debate system approach
- Graph-based approach

## Metrics

All approaches evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Speed**: Emails processed per second

## Project Structure

```
phishing-detection-project/
├── data/                          # Raw datasets
│   ├── enron.csv
│   ├── legit.csv
│   └── phishing.csv
├── notebooks/                     # Jupyter notebooks for each phase
│   ├── phase2_preprocess_enron.py
│   ├── phase2_preprocess_legit.py
│   ├── phase2_preprocess_phishing.py
│   ├── phase3_ml_enron.py
│   ├── phase3_ml_legit.py
│   ├── phase3_ml_phishing.py
│   └── ...
├── src/                          # Reusable utilities
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── models.py
├── results/                      # Evaluation results
│   ├── phase3_ml_results.csv
│   ├── phase4_llm_results.csv
│   └── ...
├── models/                       # Saved models
└── README.md
```

## Requirements

```bash
# Traditional ML
pandas
numpy
scikit-learn

# LLM Inference
ollama
transformers
torch

# Agent Frameworks
crewai
langchain
langgraph

# Fine-tuning
unsloth
```

## Results Summary

Results will be updated as each phase completes.

| Phase | Dataset | Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|---------|----------|----------|-----------|--------|----------|------------------|
| TBD   | TBD     | TBD      | TBD      | TBD       | TBD    | TBD      | TBD              |

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run preprocessing for each dataset:
```bash
python notebooks/phase2_preprocess_enron.py
python notebooks/phase2_preprocess_legit.py
python notebooks/phase2_preprocess_phishing.py
```

3. Continue with subsequent phases...

## Notes

- All tests use balanced 3k samples (1.5k phishing + 1.5k legitimate)
- Each dataset processed separately due to different formats
- All tools and models are open-source
- Results tracked systematically for comparison
