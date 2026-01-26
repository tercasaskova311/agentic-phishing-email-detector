# Agentic Phishing Email Detector

Detect phishing emails using **LLMs** with Single-Agent, Multi-Agent, and Debating architectures. Supports **traditional ML baselines** (Logistic Regression, Naive Bayes, Random Forest) for comparison.

---

## Features

* **LLM-based detection:** semantic understanding, context-aware.
* **Multi-Agent / Debating systems:** specialized agents + consensus voting.
* **Prompt engineering:** Zero-Shot / Few-Shot modes.
* **Datasets:** ENRON, AIGEN, TREC.

---

## Tech Stack

* Python 3.10+
* PyTorch, Hugging Face Transformers
* Ollama for local LLM inference
* Scikit-learn ML baselines
* Kaggle notebooks for reproducible experiments with GPU

---

## Quick Setup (Direct from `models` folder)

```bash
# 1. Clone repo
git clone https://github.com/tercasaskova311/agentic-phishing-email-detector.git
cd agentic-phishing-email-detector/models

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets crewai ollama scikit-learn

# 3. Download Kaggle dataset
mkdir -p datasets
kaggle datasets download -d tercasaskova/phishing-emails -p datasets/
unzip datasets/phishing-emails.zip -d datasets/

# 4. Hugging Face token - get thought hf acount + grand access to models: llama + qwen
export HF_TOKEN="YOUR_HF_TOKEN"  # or set as Kaggle secret
```

---

## Running the Notebooks

### Multi-Agent / Graph (Kaggle-ready)

* Open `models/agents/*.ipynb` in Kaggle.com
* Ensure **GPU runtime** and **internet access**
* Update dataset paths if necessary: `datasets/phishing-emails`
* Set `HF_TOKEN` as Kaggle secret
* Run cells to reproduce experiments

### ML Baselines 

* `models/ML` contains scripts for **Logistic Regression, Naive Bayes, Random Forest**

---

## Architecture 

* **Single-Agent:** Zero-Shot / Few-Shot prompts.
* **Multi-Agent:** Email Agent → URL Agent → Pattern Agent → Judge Agent.
* **Debating Agents:** Multiple LLMs vote in parallel, output weighted consensus.

---

## References

* [ENRON Dataset](https://github.com/MWiechmann/enron_spam_data)
* [TREC Spam Corpus](https://github.com/imdeepmind/Preprocessed-TREC-2007-Public-CA)
* [AIGEN Kaggle Dataset](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails/data)
* CrewAI, Hugging Face Transformers, Ollama


