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
    """Load preprocessed dataset - all have 'message' column."""
    df = pd.read_csv(f'datasets/{dataset_name}.csv')
    
    X = df['message'].fillna('')
    y = df['label']
    
    return X, y

def evaluate_model_on_dataset(model, X, y, dataset_name):
    """Train and evaluate a model on the dataset."""
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
    # All three datasets with 'message' column
    datasets = ['enron', 'trec', 'aigen']
    
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
        
        X, y = prepare_dataset(dataset_name)
        
        print(f"Total samples: {len(X):,}")
        print(f"Label distribution:")
        for label, count in y.value_counts().items():
            print(f"  {label}: {count:,} ({count/len(y)*100:.1f}%)")
        
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
    
    # Final Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print('='*80)
    
    results_df = pd.DataFrame(results)
    
    # Pivot table for easy comparison
    pivot = results_df.pivot_table(
        index='model',
        columns='dataset',
        values='f1',
        aggfunc='first'
    )
    
    print("\nF1 Scores by Model and Dataset:")
    print(pivot.to_string())
    
    # Show best model per dataset
    print("\n" + "-"*80)
    print("Best Model per Dataset (by F1 Score):")
    print("-"*80)
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        best = dataset_results.loc[dataset_results['f1'].idxmax()]
        print(f"{dataset.upper():10s}: {best['model']:20s} (F1: {best['f1']:.4f})")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ml_evaluation_all.csv', index=False)
    print(f"\n✓ Full results saved to results/ml_evaluation_all.csv")

if __name__ == "__main__":
    main()