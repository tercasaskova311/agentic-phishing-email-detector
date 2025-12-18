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

def prepare_combined_dataset():
    phishing = pd.read_csv('datasets/processed/phishing_clean.csv')
    legit = pd.read_csv('datasets/processed/legit_clean.csv')
    
    combined = pd.concat([phishing, legit], ignore_index=True)
    X = combined['body'].fillna('')
    y = combined['label']
    
    return X, y

def evaluate_with_method(model_class, model_name, X_train, X_test, y_train, y_test, method_name, **model_kwargs):
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'method': method_name,
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'precision_class0': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        'recall_class0': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'precision_class1': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    }

def main():
    print("="*80)
    print("CLASS IMBALANCE HANDLING EXPERIMENT")
    print("="*80)
    
    X, y = prepare_combined_dataset()
    print(f"\nDataset: Combined (Phishing + Legit)")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1 (ham:spam)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
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
        print(f"  Class 0 (ham) - P: {metrics['precision_class0']:.4f} | R: {metrics['recall_class0']:.4f}")
        print(f"  Class 1 (spam) - P: {metrics['precision_class1']:.4f} | R: {metrics['recall_class1']:.4f}")
    
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
        print(f"  Class 0 (ham) - P: {metrics['precision_class0']:.4f} | R: {metrics['recall_class0']:.4f}")
        print(f"  Class 1 (spam) - P: {metrics['precision_class1']:.4f} | R: {metrics['recall_class1']:.4f}")
    
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
        print(f"  Class 0 (ham) - P: {metrics['precision_class0']:.4f} | R: {metrics['recall_class0']:.4f}")
        print(f"  Class 1 (spam) - P: {metrics['precision_class1']:.4f} | R: {metrics['recall_class1']:.4f}")
    
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
        print(f"  Class 0 (ham) - P: {metrics['precision_class0']:.4f} | R: {metrics['recall_class0']:.4f}")
        print(f"  Class 1 (spam) - P: {metrics['precision_class1']:.4f} | R: {metrics['recall_class1']:.4f}")
    
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
        print(f"  Class 0 (ham) - P: {metrics['precision_class0']:.4f} | R: {metrics['recall_class0']:.4f}")
        print(f"  Class 1 (spam) - P: {metrics['precision_class1']:.4f} | R: {metrics['recall_class1']:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY & COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    print("\nF1 Scores by Method and Model:")
    pivot = results_df.pivot_table(index='model', columns='method', values='f1', aggfunc='first')
    print(pivot.to_string())
    
    print("\n\nRecall for Class 1 (Spam - Minority) by Method and Model:")
    pivot_recall = results_df.pivot_table(index='model', columns='method', values='recall_class1', aggfunc='first')
    print(pivot_recall.to_string())
    
    print("\n\nBest F1 per Model:")
    best_f1 = results_df.loc[results_df.groupby('model')['f1'].idxmax()][['model', 'method', 'f1', 'recall_class1']]
    print(best_f1.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/balanced_comparison.csv', index=False)
    print("\n\nFull results saved to results/balanced_comparison.csv")

if __name__ == "__main__":
    main()
