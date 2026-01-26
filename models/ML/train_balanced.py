import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os


def evaluate_with_method(model_class, model_name, X_train, X_test, y_train, y_test, method_name, **model_kwargs):
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'method': method_name,
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0, pos_label='phishing_email'),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0, pos_label='phishing_email'),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0, pos_label='phishing_email'),
        'precision_safe': precision_score(y_test, y_pred, pos_label='safe_email', zero_division=0),
        'recall_safe': recall_score(y_test, y_pred, pos_label='safe_email', zero_division=0),
        'precision_phishing': precision_score(y_test, y_pred, pos_label='phishing_email', zero_division=0),
        'recall_phishing': recall_score(y_test, y_pred, pos_label='phishing_email', zero_division=0)
    }

def main():
    print("="*80)
    print("CLASS IMBALANCE HANDLING EXPERIMENT")
    print("="*80)
    
    # Load aigen dataset (already concatenated)
    df = pd.read_csv('datasets/aigen.csv')
    X = df['message'].fillna('')
    y = df['label']
    
    print(f"\nDataset: AIGen (Combined Phishing + Safe)")
    print(f"Total samples: {len(X):,}")
    print(f"Class distribution:")
    for label, count in y.value_counts().items():
        print(f"  {label}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Check if balanced or imbalanced
    counts = y.value_counts()
    if len(counts) == 2:
        ratio = max(counts) / min(counts)
        print(f"Imbalance ratio: {ratio:.2f}:1")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train class distribution:")
    for label, count in pd.Series(y_train).value_counts().items():
        print(f"  {label}: {count:,}")
    
    results = []
    
    print("\n" + "="*80)
    print("METHOD 1: BASELINE (No Balancing)")
    print("="*80)
    
    for model_class, name, kwargs in [
        (LogisticRegression, "Logistic Regression", {'max_iter': 1000, 'random_state': 42}),
        (MultinomialNB, "Naive Bayes", {'alpha': 1.0}),
        (RandomForestClassifier, "Random Forest", {'n_estimators': 100, 'random_state': 42})
    ]:
        print(f"\n{name}:")
        metrics = evaluate_with_method(model_class, name, X_train_vec, X_test_vec, 
                                      y_train, y_test, "Baseline", **kwargs)
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Safe email - P: {metrics['precision_safe']:.4f} | R: {metrics['recall_safe']:.4f}")
        print(f"  Phishing email - P: {metrics['precision_phishing']:.4f} | R: {metrics['recall_phishing']:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 2: CLASS WEIGHTS")
    print("="*80)
    
    for model_class, name, kwargs in [
        (LogisticRegression, "Logistic Regression", {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}),
        (RandomForestClassifier, "Random Forest", {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'})
    ]:
        print(f"\n{name}:")
        metrics = evaluate_with_method(model_class, name, X_train_vec, X_test_vec, 
                                      y_train, y_test, "Class Weights", **kwargs)
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Safe email - P: {metrics['precision_safe']:.4f} | R: {metrics['recall_safe']:.4f}")
        print(f"  Phishing email - P: {metrics['precision_phishing']:.4f} | R: {metrics['recall_phishing']:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 3: SMOTE (Synthetic Minority Oversampling)")
    print("="*80)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)
    print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
    
    for model_class, name, kwargs in [
        (LogisticRegression, "Logistic Regression", {'max_iter': 1000, 'random_state': 42}),
        (MultinomialNB, "Naive Bayes", {'alpha': 1.0}),
        (RandomForestClassifier, "Random Forest", {'n_estimators': 100, 'random_state': 42})
    ]:
        print(f"\n{name}:")
        metrics = evaluate_with_method(model_class, name, X_train_smote, X_test_vec, 
                                      y_train_smote, y_test, "SMOTE", **kwargs)
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Safe email - P: {metrics['precision_safe']:.4f} | R: {metrics['recall_safe']:.4f}")
        print(f"  Phishing email - P: {metrics['precision_phishing']:.4f} | R: {metrics['recall_phishing']:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 4: RANDOM OVERSAMPLING")
    print("="*80)
    
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_vec, y_train)
    print(f"After Random Oversampling: {pd.Series(y_train_ros).value_counts().to_dict()}")
    
    for model_class, name, kwargs in [
        (LogisticRegression, "Logistic Regression", {'max_iter': 1000, 'random_state': 42}),
        (MultinomialNB, "Naive Bayes", {'alpha': 1.0}),
        (RandomForestClassifier, "Random Forest", {'n_estimators': 100, 'random_state': 42})
    ]:
        print(f"\n{name}:")
        metrics = evaluate_with_method(model_class, name, X_train_ros, X_test_vec, 
                                      y_train_ros, y_test, "Random Oversample", **kwargs)
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Safe email - P: {metrics['precision_safe']:.4f} | R: {metrics['recall_safe']:.4f}")
        print(f"  Phishing email - P: {metrics['precision_phishing']:.4f} | R: {metrics['recall_phishing']:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 5: RANDOM UNDERSAMPLING")
    print("="*80)
    
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_vec, y_train)
    print(f"After Random Undersampling: {pd.Series(y_train_rus).value_counts().to_dict()}")
    
    for model_class, name, kwargs in [
        (LogisticRegression, "Logistic Regression", {'max_iter': 1000, 'random_state': 42}),
        (MultinomialNB, "Naive Bayes", {'alpha': 1.0}),
        (RandomForestClassifier, "Random Forest", {'n_estimators': 100, 'random_state': 42})
    ]:
        print(f"\n{name}:")
        metrics = evaluate_with_method(model_class, name, X_train_rus, X_test_vec, 
                                      y_train_rus, y_test, "Random Undersample", **kwargs)
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Safe email - P: {metrics['precision_safe']:.4f} | R: {metrics['recall_safe']:.4f}")
        print(f"  Phishing email - P: {metrics['precision_phishing']:.4f} | R: {metrics['recall_phishing']:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY & COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    print("\nF1 Scores by Method and Model:")
    pivot = results_df.pivot_table(index='model', columns='method', values='f1', aggfunc='first')
    print(pivot.to_string())
    
    print("\n\nRecall for Phishing Emails by Method and Model:")
    pivot_recall = results_df.pivot_table(index='model', columns='method', values='recall_phishing', aggfunc='first')
    print(pivot_recall.to_string())
    
    print("\n\nBest F1 per Model:")
    best_f1 = results_df.loc[results_df.groupby('model')['f1'].idxmax()][['model', 'method', 'f1', 'recall_phishing']]
    print(best_f1.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/balanced_comparison.csv', index=False)
    print("\n✓ Full results saved to results/balanced_comparison.csv")

if __name__ == "__main__":
    main()