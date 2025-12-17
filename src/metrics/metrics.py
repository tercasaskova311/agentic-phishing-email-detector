"""
Simple metrics calculator.
"""


def calculate_metrics(results: list) -> dict:
    """
    Calculate accuracy, precision, recall, F1.
    
    Args:
        results: [{"prediction": "phishing_email", "true_label": "phishing_email"}, ...]
    
    Returns:
        {"accuracy": 0.85, "precision": 0.88, ...}
    """
    # Filter valid results
    valid = [r for r in results if r.get('prediction') and r.get('true_label')]
    
    if not valid:
        return {"error": "No valid results"}
    
    # Count
    tp = sum(1 for r in valid if r['prediction'] == 'phishing_email' and r['true_label'] == 'phishing_email')
    fp = sum(1 for r in valid if r['prediction'] == 'phishing_email' and r['true_label'] == 'safe_email')
    tn = sum(1 for r in valid if r['prediction'] == 'safe_email' and r['true_label'] == 'safe_email')
    fn = sum(1 for r in valid if r['prediction'] == 'safe_email' and r['true_label'] == 'phishing_email')
    
    total = len(valid)
    
    # Calculate
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "total": total,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# Test
if __name__ == "__main__":
    test_results = [
        {"prediction": "phishing_email", "true_label": "phishing_email"},
        {"prediction": "phishing_email", "true_label": "safe_email"},
        {"prediction": "safe_email", "true_label": "safe_email"},
    ]
    
    metrics = calculate_metrics(test_results)
    print(metrics)