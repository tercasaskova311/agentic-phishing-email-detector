import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

def extract_url_features(text):
    if pd.isna(text):
        return 0, 0, 0
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(text))
    url_count = len(urls)
    
    ip_urls = len([u for u in urls if re.search(r'\d+\.\d+\.\d+\.\d+', u)])
    suspicious_tlds = len([u for u in urls if any(tld in u.lower() for tld in ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz'])])
    
    return url_count, ip_urls, suspicious_tlds

def extract_sender_features(email):
    if pd.isna(email) or email == 'unknown':
        return 'unknown', 0
    
    email = str(email).lower()
    domain = email.split('@')[-1] if '@' in email else 'unknown'
    
    suspicious = int(any(char in email for char in ['!', '#', '$', '%']))
    
    return domain, suspicious

def extract_text_features(text):
    if pd.isna(text):
        return 0, 0, 0, 0, 0
    
    text = str(text)
    length = len(text)
    word_count = len(text.split())
    
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    special_chars = len(re.findall(r'[!?$%&*]', text))
    exclamation_count = text.count('!')
    
    return length, word_count, uppercase_ratio, special_chars, exclamation_count

def engineer_features(df, text_col='body', sender_col='sender'):
    features = pd.DataFrame()
    
    text_data = df[text_col] if text_col in df.columns else df.get('message', '')
    sender_data = df[sender_col] if sender_col in df.columns else pd.Series(['unknown'] * len(df))
    
    url_features = text_data.apply(extract_url_features)
    features['url_count'] = url_features.apply(lambda x: x[0])
    features['ip_url_count'] = url_features.apply(lambda x: x[1])
    features['suspicious_tld_count'] = url_features.apply(lambda x: x[2])
    
    sender_features = sender_data.apply(extract_sender_features)
    features['sender_domain'] = sender_features.apply(lambda x: x[0])
    features['suspicious_sender'] = sender_features.apply(lambda x: x[1])
    
    text_features = text_data.apply(extract_text_features)
    features['text_length'] = text_features.apply(lambda x: x[0])
    features['word_count'] = text_features.apply(lambda x: x[1])
    features['uppercase_ratio'] = text_features.apply(lambda x: x[2])
    features['special_char_count'] = text_features.apply(lambda x: x[3])
    features['exclamation_count'] = text_features.apply(lambda x: x[4])
    
    return features

def get_combined_features(X_text, X_features, vectorizer=None, fit=False):
    from scipy.sparse import hstack
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    
    if fit:
        X_tfidf = vectorizer.fit_transform(X_text)
    else:
        X_tfidf = vectorizer.transform(X_text)
    
    numeric_cols = ['url_count', 'ip_url_count', 'suspicious_tld_count', 'suspicious_sender',
                    'text_length', 'word_count', 'uppercase_ratio', 'special_char_count', 'exclamation_count']
    
    X_numeric = X_features[numeric_cols].values
    
    if fit:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
    else:
        scaler = None
        X_numeric_scaled = X_numeric
    
    X_combined = hstack([X_tfidf, X_numeric_scaled])
    
    return X_combined, vectorizer, scaler
