import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import engineer_features, get_combined_features

def create_enhanced_dataset():
    enron = pd.read_csv('datasets/processed/enron_clean.csv')
    phishing = pd.read_csv('datasets/processed/phishing_clean.csv')
    legit = pd.read_csv('datasets/processed/legit_clean.csv')
    
    enron_spam = enron[enron['label'] == 1].copy()
    enron_ham = enron[enron['label'] == 0].copy()
    
    enron_spam['body'] = enron_spam['message']
    enron_ham['body'] = enron_ham['message']
    enron_spam['sender'] = 'unknown'
    enron_ham['sender'] = 'unknown'
    
    enhanced_spam = pd.concat([phishing, enron_spam[['body', 'sender', 'label']].head(1000)], ignore_index=True)
    enhanced_ham = pd.concat([legit, enron_ham[['body', 'sender', 'label']].head(1000)], ignore_index=True)
    
    enhanced = pd.concat([enhanced_spam, enhanced_ham], ignore_index=True)
    
    return enhanced

def prepare_data_with_features(df, text_col='body'):
    X_text = df[text_col].fillna('')
    y = df['label']
    X_features = engineer_features(df, text_col=text_col)
    
    return X_text, X_features, y

def grid_search_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_params = {
        'C': [0.1, 1, 10],
        'max_iter': [1000],
        'class_weight': ['balanced']
    }
    
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    
    print("Grid Search - Logistic Regression")
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, 
                           cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    lr_grid.fit(X_train, y_train)
    print(f"  Best params: {lr_grid.best_params_}")
    print(f"  Best CV F1: {lr_grid.best_score_:.4f}")
    
    print("\nGrid Search - Random Forest")
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params,
                           cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    print(f"  Best params: {rf_grid.best_params_}")
    print(f"  Best CV F1: {rf_grid.best_score_:.4f}")
    
    print("\nNaive Bayes skipped (incompatible with scaled features)")
    
    return lr_grid.best_estimator_, rf_grid.best_estimator_

def create_ensemble_models(lr, rf):
    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf)],
        voting='soft',
        weights=[1, 2]
    )
    
    stacking = StackingClassifier(
        estimators=[('lr', lr)],
        final_estimator=rf,
        cv=5
    )
    
    return voting, stacking

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, cv=5):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    print(f"\n{model_name}:")
    print(f"  Test F1: {f1:.4f} | Test Recall: {recall:.4f}")
    print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return {'model': model_name, 'f1': f1, 'recall': recall, 'cv_f1_mean': cv_scores.mean(), 'cv_f1_std': cv_scores.std()}

def main():
    print("="*80)
    print("ADVANCED ML PIPELINE: FEATURE ENGINEERING + GRID SEARCH + CV + ENSEMBLES")
    print("="*80)
    
    results = []
    
    datasets_config = [
        ('Enhanced Combined', create_enhanced_dataset(), 'body'),
        ('Enron', pd.read_csv('datasets/processed/enron_clean.csv'), 'message')
    ]
    
    for dataset_name, df, text_col in datasets_config:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Samples: {len(df)}, Distribution: {df['label'].value_counts().to_dict()}")
        
        X_text, X_features, y = prepare_data_with_features(df, text_col)
        
        from sklearn.model_selection import train_test_split
        X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
            X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        X_train_combined, _, scaler = get_combined_features(
            X_text_train, X_features_train, vectorizer, fit=True
        )
        X_test_combined, _, _ = get_combined_features(
            X_text_test, X_features_test, vectorizer, fit=False
        )
        
        if scaler:
            numeric_cols = ['url_count', 'ip_url_count', 'suspicious_tld_count', 'suspicious_sender',
                           'text_length', 'word_count', 'uppercase_ratio', 'special_char_count', 'exclamation_count']
            X_features_test[numeric_cols] = scaler.transform(X_features_test[numeric_cols])
            X_test_combined, _, _ = get_combined_features(X_text_test, X_features_test, vectorizer, fit=False)
        
        print(f"\nFeature dimensions: {X_train_combined.shape}")
        
        print("\n" + "-"*80)
        print("GRID SEARCH WITH 5-FOLD CV")
        print("-"*80)
        lr_best, rf_best = grid_search_models(X_train_combined, y_train)
        
        print("\n" + "-"*80)
        print("APPLYING SMOTE FOR BALANCE")
        print("-"*80)
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_combined, y_train)
        print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
        
        print("\n" + "-"*80)
        print("CREATING ENSEMBLE MODELS")
        print("-"*80)
        voting, stacking = create_ensemble_models(lr_best, rf_best)
        
        print("\n" + "-"*80)
        print("EVALUATING ALL MODELS")
        print("-"*80)
        
        for model, name in [
            (lr_best, 'Logistic Regression (Tuned)'),
            (rf_best, 'Random Forest (Tuned)'),
            (voting, 'Voting Ensemble'),
            (stacking, 'Stacking Ensemble')
        ]:
            metrics = evaluate_model(model, X_train_smote, X_test_combined, 
                                    y_train_smote, y_test, name)
            metrics['dataset'] = dataset_name
            results.append(metrics)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    print("\nF1 Scores:")
    pivot_f1 = results_df.pivot_table(index='model', columns='dataset', values='f1', aggfunc='first')
    print(pivot_f1.to_string())
    
    print("\n\nRecall Scores:")
    pivot_recall = results_df.pivot_table(index='model', columns='dataset', values='recall', aggfunc='first')
    print(pivot_recall.to_string())
    
    print("\n\nBest Results Per Dataset:")
    for ds in results_df['dataset'].unique():
        ds_results = results_df[results_df['dataset'] == ds]
        best = ds_results.loc[ds_results['f1'].idxmax()]
        print(f"\n{ds}:")
        print(f"  Best Model: {best['model']}")
        print(f"  F1: {best['f1']:.4f} | Recall: {best['recall']:.4f}")
    
    import os
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/advanced_results.csv', index=False)
    print("\n\nResults saved to results/advanced_results.csv")

if __name__ == "__main__":
    main()
