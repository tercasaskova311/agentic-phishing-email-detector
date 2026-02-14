#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"
EXPLAINABILITY_DIR.mkdir(exist_ok=True)

def analyze_feature_importance(model, vectorizer, top_n=20):
    """Analyze feature importance from model coefficients"""
    print("\n1. Feature Importance Analysis")
    print("-" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        # For Logistic Regression
        coefficients = model.coef_[0]
        
        # Top phishing indicators (positive coefficients)
        top_phishing_idx = np.argsort(coefficients)[-top_n:][::-1]
        top_phishing_features = [(feature_names[i], coefficients[i]) for i in top_phishing_idx]
        
        # Top legitimate indicators (negative coefficients)
        top_legit_idx = np.argsort(coefficients)[:top_n]
        top_legit_features = [(feature_names[i], coefficients[i]) for i in top_legit_idx]
        
        print(f"\nTop {top_n} Phishing Indicators:")
        for i, (feature, coef) in enumerate(top_phishing_features, 1):
            print(f"  {i:2d}. {feature:<20} (weight: {coef:>7.4f})")
        
        print(f"\nTop {top_n} Legitimate Indicators:")
        for i, (feature, coef) in enumerate(top_legit_features, 1):
            print(f"  {i:2d}. {feature:<20} (weight: {coef:>7.4f})")
        
        return {
            "phishing_indicators": top_phishing_features,
            "legitimate_indicators": top_legit_features
        }
    
    elif hasattr(model, 'feature_log_prob_'):
        # For Naive Bayes
        log_prob_phishing = model.feature_log_prob_[1]
        log_prob_legit = model.feature_log_prob_[0]
        
        # Calculate log probability ratio
        log_ratio = log_prob_phishing - log_prob_legit
        
        top_phishing_idx = np.argsort(log_ratio)[-top_n:][::-1]
        top_phishing_features = [(feature_names[i], log_ratio[i]) for i in top_phishing_idx]
        
        top_legit_idx = np.argsort(log_ratio)[:top_n]
        top_legit_features = [(feature_names[i], log_ratio[i]) for i in top_legit_idx]
        
        print(f"\nTop {top_n} Phishing Indicators:")
        for i, (feature, ratio) in enumerate(top_phishing_features, 1):
            print(f"  {i:2d}. {feature:<20} (log ratio: {ratio:>7.4f})")
        
        print(f"\nTop {top_n} Legitimate Indicators:")
        for i, (feature, ratio) in enumerate(top_legit_features, 1):
            print(f"  {i:2d}. {feature:<20} (log ratio: {ratio:>7.4f})")
        
        return {
            "phishing_indicators": top_phishing_features,
            "legitimate_indicators": top_legit_features
        }
    
    return None

def lime_explain_samples(model, vectorizer, X_test, y_test, n_samples=5):
    """Use LIME to explain individual predictions"""
    print("\n2. LIME Explanations (Individual Samples)")
    print("-" * 60)
    
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
    
    # Prediction function for LIME
    def predict_proba_text(texts):
        X_vec = vectorizer.transform(texts)
        return model.predict_proba(X_vec)
    
    explanations = []
    
    # Explain a few samples
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        text = X_test.iloc[idx]
        true_label = y_test.iloc[idx]
        
        # Get prediction
        X_vec = vectorizer.transform([text])
        pred = model.predict(X_vec)[0]
        pred_proba = model.predict_proba(X_vec)[0]
        
        print(f"\nSample {i+1}:")
        print(f"  True Label: {'Phishing' if true_label == 1 else 'Legitimate'}")
        print(f"  Predicted:  {'Phishing' if pred == 1 else 'Legitimate'}")
        print(f"  Confidence: {max(pred_proba):.2%}")
        
        # Generate explanation
        try:
            exp = explainer.explain_instance(
                text, 
                predict_proba_text,
                num_features=10,
                top_labels=1
            )
            
            # Get top features
            label = 1 if pred == 1 else 0
            top_features = exp.as_list(label=label)
            
            print(f"  Top Contributing Features:")
            for feature, weight in top_features[:5]:
                direction = "→ Phishing" if weight > 0 else "→ Legitimate"
                print(f"    '{feature}' ({weight:+.3f}) {direction}")
            
            explanations.append({
                "text_snippet": text[:100] + "...",
                "true_label": int(true_label),
                "predicted": int(pred),
                "confidence": float(max(pred_proba)),
                "top_features": top_features
            })
            
        except Exception as e:
            print(f"  Error generating explanation: {e}")
    
    return explanations

def shap_analysis(model, vectorizer, X_train, X_test, n_samples=100):
    """Use SHAP for global feature importance"""
    print("\n3. SHAP Analysis (Global Feature Importance)")
    print("-" * 60)
    
    try:
        # Transform data
        X_train_vec = vectorizer.transform(X_train[:1000])  # Use subset for speed
        X_test_vec = vectorizer.transform(X_test[:n_samples])
        
        # Create SHAP explainer
        print("  Creating SHAP explainer (this may take a moment)...")
        
        if hasattr(model, 'coef_'):
            # For linear models, use LinearExplainer
            explainer = shap.LinearExplainer(model, X_train_vec)
            shap_values = explainer.shap_values(X_test_vec)
        else:
            # For other models, use KernelExplainer (slower)
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_train_vec, 100)
            )
            shap_values = explainer.shap_values(X_test_vec)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[-20:][::-1]
        
        print("\n  Top 20 Most Important Features (by SHAP):")
        for i, idx in enumerate(top_features_idx, 1):
            if idx < len(feature_names):
                print(f"    {i:2d}. {feature_names[idx]:<20} (importance: {mean_shap[idx]:.4f})")
        
        return {
            "top_features": [(feature_names[idx], float(mean_shap[idx])) 
                           for idx in top_features_idx if idx < len(feature_names)]
        }
        
    except Exception as e:
        print(f"  Error in SHAP analysis: {e}")
        print("  SHAP analysis requires additional memory and computation")
        return None

def analyze_common_patterns(df, label_col='label'):
    """Analyze common patterns in phishing vs legitimate emails"""
    print("\n4. Common Pattern Analysis")
    print("-" * 60)
    
    phishing = df[df[label_col] == 1]['text']
    legitimate = df[df[label_col] == 0]['text']
    
    # Common words/patterns
    phishing_words = ' '.join(phishing.fillna('')).lower().split()
    legit_words = ' '.join(legitimate.fillna('')).lower().split()
    
    from collections import Counter
    
    phishing_counter = Counter(phishing_words)
    legit_counter = Counter(legit_words)
    
    # Find distinctive words
    print("\n  Most Common in Phishing Emails:")
    for word, count in phishing_counter.most_common(15):
        if len(word) > 3 and word.isalpha():
            print(f"    {word:<15} ({count} times)")
    
    print("\n  Most Common in Legitimate Emails:")
    for word, count in legit_counter.most_common(15):
        if len(word) > 3 and word.isalpha():
            print(f"    {word:<15} ({count} times)")
    
    # Length analysis
    phishing_lengths = phishing.str.len()
    legit_lengths = legitimate.str.len()
    
    print("\n  Email Length Statistics:")
    print(f"    Phishing - Mean: {phishing_lengths.mean():.0f}, Median: {phishing_lengths.median():.0f}")
    print(f"    Legitimate - Mean: {legit_lengths.mean():.0f}, Median: {legit_lengths.median():.0f}")
    
    return {
        "phishing_common": phishing_counter.most_common(20),
        "legitimate_common": legit_counter.most_common(20),
        "length_stats": {
            "phishing_mean": float(phishing_lengths.mean()),
            "phishing_median": float(phishing_lengths.median()),
            "legitimate_mean": float(legit_lengths.mean()),
            "legitimate_median": float(legit_lengths.median())
        }
    }

def main():
    """Main function for explainability analysis"""
    print("="*60)
    print("EXPLAINABILITY ANALYSIS: SHAP & LIME")
    print("="*60)
    
    # Load dataset
    dataset_path = RESULTS_DIR / "enron_preprocessed_3k.csv"
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Phishing: {(df['label'] == 1).sum()}")
    print(f"Legitimate: {(df['label'] == 0).sum()}")
    
    # Prepare data
    X = df['text'].fillna("")
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    accuracy = model.score(X_test_vec, y_test)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Run analyses
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSES")
    print(f"{'='*60}")
    
    all_results = {}
    
    # 1. Feature importance
    feature_importance = analyze_feature_importance(model, vectorizer, top_n=20)
    if feature_importance:
        all_results['feature_importance'] = feature_importance
    
    # 2. LIME explanations
    lime_results = lime_explain_samples(model, vectorizer, X_test, y_test, n_samples=5)
    all_results['lime_examples'] = lime_results
    
    # 3. SHAP analysis
    shap_results = shap_analysis(model, vectorizer, X_train, X_test, n_samples=100)
    if shap_results:
        all_results['shap_analysis'] = shap_results
    
    # 4. Pattern analysis
    pattern_results = analyze_common_patterns(df)
    all_results['pattern_analysis'] = pattern_results
    
    # Save results
    output_file = EXPLAINABILITY_DIR / "explainability_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    print("\n✓ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("\nKey Insights:")
    print("- Feature importance shows which words drive predictions")
    print("- LIME explains individual email classifications")
    print("- SHAP provides global feature importance across all samples")
    print("- Pattern analysis reveals common characteristics")

if __name__ == "__main__":
    main()
