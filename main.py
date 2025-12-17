from src.models.agents.single_agent_simple import SimpleAgent
from llm_wrapper import OllamaLLM
from data import load_csv
from src.metrics.metrics import calculate_metrics
import json


def main():
    print("\n[1/4] Creating LLM...")
    llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
    print("✓ Using Ollama with llama3.1")
    
    # 2. Create Agent
    print("\n[2/4] Creating agent...")
    agent = SimpleAgent(llm=llm)
    print("✓ Agent ready")
    
    # 3. Load data
    print("\n[3/4] Loading data...")
    emails = load_csv("data/processed/enron.csv", sample_size=10)
    print(f"✓ Loaded {len(emails)} emails")
    
    # 4. Analyze
    print("\n[4/4] Analyzing...")
    results = []
    
    for i, email in enumerate(emails):
        print(f"  {i+1}/{len(emails)}...", end=" ")
        
        result = agent.analyze(email['message'])
        result['true_label'] = email['label']
        results.append(result)
        
        print(f"{result['prediction']}")
    
    # 5. Metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    metrics = calculate_metrics(results)
    print(f"\nAccuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")
    
    # 6. Save
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()