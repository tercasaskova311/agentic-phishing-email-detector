"""
Simple data loader for CSV files.
"""

import pandas as pd


def load_csv(filepath: str, sample_size: int = None) -> list:
    """
    Load emails from CSV.
    
    Args:
        filepath: Path to CSV file
        sample_size: Number of samples (None = all)
    
    Returns:
        List of dicts: [{"message": "...", "label": "phishing_email"}, ...]
    """
    df = pd.read_csv(filepath)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    emails = []
    for _, row in df.iterrows():
        emails.append({
            "message": str(row.get('message', '')),
            "label": row.get('label'),
            "subject": row.get('subject'),
        })
    
    return emails


# Test
if __name__ == "__main__":
    emails = load_csv("data/enron.csv", sample_size=5)
    print(f"Loaded {len(emails)} emails")
    print(emails[0])