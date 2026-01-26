# Enron-only evaluation (train/test on enron_clean)
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

def prepare_enron(sample_size=None):
    df = pd.read_csv('datasets/processed/enron_clean.csv')
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    X = df['message'].fillna('')
    y = df['label']
    return X, y

def evaluate_model(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model.train(X_train_vec, y_train)
    start = time.time()
    y_pred = model.predict(X_test_vec)
    dt = time.time() - start
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'speed_emails_per_second': len(X_test) / dt if dt > 0 else 0,
        'num_test_samples': len(X_test)
    }

def main():
    print("\n=== ENRON TRAIN/TEST ===")
    X, y = prepare_enron()
    print(f"Samples: {len(X)} | Label distribution: {y.value_counts().to_dict()}")

    models = [
        LogisticRegressionModel(),
        NaiveBayesModel(),
        RandomForestModel()
    ]
    results = []
    for model in models:
        print(f"\nEvaluating {model.name}...")
        metrics = evaluate_model(model, X, y)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Speed:     {metrics['speed_emails_per_second']:.2f} emails/second")
        results.append({
            'dataset': 'enron',
            'model': model.name,
            **metrics
        })

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/enron_only_results.csv', index=False)
    print("\nSaved results to results/enron_only_results.csv")

if __name__ == "__main__":
    main()
