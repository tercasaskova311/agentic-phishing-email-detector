import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

sys.path.append('src/models/baseline_ML')
from logistic_regression import LogisticRegressionModel
from naive_bayes import NaiveBayesModel
from random_forest import RandomForestModel

def prepare_dataset(dataset_name):
    df = pd.read_csv(f'datasets/processed/{dataset_name}_clean.csv')
    
    if 'body' in df.columns:
        text_col = 'body'
    elif 'message' in df.columns:
        text_col = 'message'
    else:
        raise ValueError(f"No text column found in {dataset_name}")
    
    X = df[text_col].fillna('')
    y = df['label']
    
    return X, y

def evaluate_model_on_dataset(model, X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model.train(X_train_vec, y_train)
    metrics = model.evaluate(X_test_vec, y_test)
    
    return metrics

def prepare_combined_dataset():
    phishing = pd.read_csv('datasets/processed/phishing_clean.csv')
    legit = pd.read_csv('datasets/processed/legit_clean.csv')
    
    combined = pd.concat([phishing, legit], ignore_index=True)
    
    if 'body' in combined.columns:
        X = combined['body'].fillna('')
    else:
        X = combined['message'].fillna('')
    y = combined['label']
    
    return X, y

def main():
    datasets_config = [
        ('combined', lambda: prepare_combined_dataset()),
        ('enron', lambda: prepare_dataset('enron'))
    ]
    
    models = [
        LogisticRegressionModel(),
        NaiveBayesModel(),
        RandomForestModel()
    ]
    
    results = []
    
    for dataset_name, loader in datasets_config:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print('='*80)
        
        X, y = loader()
        print(f"Samples: {len(X)}, Labels: {y.value_counts().to_dict()}")
        
        for model in models:
            print(f"\nTraining {model.name}...")
            metrics = evaluate_model_on_dataset(model, X, y, dataset_name)
            
            results.append({
                'dataset': dataset_name,
                'model': model.name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    results_df = pd.DataFrame(results)
    
    pivot = results_df.pivot_table(
        index='model',
        columns='dataset',
        values='f1',
        aggfunc='first'
    )
    
    print("\nF1 Scores:")
    print(pivot.to_string())
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ml_evaluation.csv', index=False)
    print("\nFull results saved to results/ml_evaluation.csv")

if __name__ == "__main__":
    main()
