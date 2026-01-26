# Phishing-only evaluation (trained on combined data, tested on phishing.csv)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import time

sys.path.append('src/models/baseline_ML')
from src.models.baseline_ML.logistic_regression import LogisticRegressionModel
from src.models.baseline_ML.naive_bayes import NaiveBayesModel
from src.models.baseline_ML.random_forest import RandomForestModel

def load_all_datasets():
    enron = pd.read_csv('datasets/processed/enron_clean.csv')
    legit = pd.read_csv('datasets/processed/legit_clean.csv')
    phishing = pd.read_csv('datasets/processed/phishing_clean.csv')
    enron['text'] = enron['message']
    legit['text'] = legit['body']
    phishing['text'] = phishing['body']
    combined = pd.concat([
        enron[['text','label']],
        legit[['text','label']],
        phishing[['text','label']]
    ], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined

def main():
    print("\n=== LOADING COMBINED DATASET AND TRAINING MODELS ===")
    combined = load_all_datasets()
    X = combined['text'].fillna('')
    y = combined['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    models = [
        LogisticRegressionModel(),
        NaiveBayesModel(),
        RandomForestModel()
    ]
    for model in models:
        print(f"Training {model.name}...")
        model.train(X_train_vec, y_train)

    print("\n=== TESTING ON PHISHING DATASET ===")
    phishing_df = pd.read_csv('datasets/processed/phishing_clean.csv')
    X_phishing = phishing_df['body'].fillna('')
    y_phishing = phishing_df['label']
    X_phishing_vec = vectorizer.transform(X_phishing)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    results = []
    for model in models:
        print(f"\n{model.name}:")
        start = time.time()
        y_pred = model.predict(X_phishing_vec)
        dt = time.time() - start
        accuracy = accuracy_score(y_phishing, y_pred)
        precision = precision_score(y_phishing, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_phishing, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_phishing, y_pred, average='binary', zero_division=0)
        speed = len(X_phishing) / dt if dt > 0 else 0
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Speed:     {speed:.2f} emails/second")
        results.append({
            'dataset': 'phishing',
            'model': model.name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'speed_emails_per_second': speed,
            'num_test_samples': len(X_phishing)
        })

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/ml_phishing_test_results.csv', index=False)
    print("\nSaved results to results/ml_phishing_test_results.csv")

if __name__ == "__main__":
    main()
