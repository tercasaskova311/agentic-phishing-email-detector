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
    
    return X, y, df

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

def main():
    datasets = ['trec', 'enron', 'aigen']
    models = [
        LogisticRegressionModel(),
        NaiveBayesModel(),
        RandomForestModel()
    ]
    
    results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print('='*80)
        
        X, y, df = prepare_dataset(dataset_name)
        unique_labels = y.nunique()
        print(f"Samples: {len(X)}, Labels: {y.value_counts().to_dict()}")
        
        if unique_labels < 2:
            print(f"Skipping {dataset_name} - only has {unique_labels} class(es)")
            continue
        
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
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print('='*80)
    
    results_df = pd.DataFrame(results)
    pivot = results_df.pivot_table(
        index='model',
        columns='dataset',
        values='f1',
        aggfunc='first'
    )
    
    print("\nF1 Scores by Model and Dataset:")
    print(pivot.to_string())
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ml_evaluation_all.csv', index=False)
    print("\n✓ Full results saved to results/ml_evaluation_all.csv")

if __name__ == "__main__":
    main()