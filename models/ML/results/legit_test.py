# Legit-only testing (models trained on sampled Enron, tested on legit.csv)
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

def prepare_dataset(dataset_name, sample_size=None):
    df = pd.read_csv(f'datasets/processed/{dataset_name}_clean.csv')
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    if 'body' in df.columns:
        text_col = 'body'
    elif 'message' in df.columns:
        text_col = 'message'
    else:
        raise ValueError(f"No text column found in {dataset_name}")
    X = df[text_col].fillna('')
    y = df['label']
    return X, y

def main():
    print("\n=== TRAINING ON ENRON (sampled 5000 for speed) ===")
    X_enron, y_enron = prepare_dataset('enron', sample_size=5000)
    X_train, _, y_train, _ = train_test_split(
        X_enron, y_enron, test_size=0.2, random_state=42, stratify=y_enron
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

    print("\n=== TESTING ON LEGIT DATASET ===")
    X_legit, y_legit = prepare_dataset('legit')
    X_legit_vec = vectorizer.transform(X_legit)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    results = []
    for model in models:
        print(f"\n{model.name}:")
        start = time.time()
        y_pred = model.predict(X_legit_vec)
        dt = time.time() - start
        accuracy = accuracy_score(y_legit, y_pred)
        precision = precision_score(y_legit, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_legit, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_legit, y_pred, average='binary', zero_division=0)
        speed = len(X_legit) / dt if dt > 0 else 0
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Speed:     {speed:.2f} emails/second")
        results.append({
            'dataset': 'legit',
            'model': model.name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'speed_emails_per_second': speed,
            'num_test_samples': len(X_legit)
        })

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/ml_legit_results.csv', index=False)
    print("\nSaved results to results/ml_legit_results.csv")

if __name__ == "__main__":
    main()
