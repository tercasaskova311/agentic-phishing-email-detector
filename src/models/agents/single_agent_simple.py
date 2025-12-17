import json
import os
import pandas as pd


class SimpleDetector:
    def __init__(self, llm_function):
        self.llm = llm_function
    
    def analyze(self, email_text: str):
        prompt = f"""
        You are a cybersecurity expert. Analyze if this email is phishing email or safe email?
        Phishing emails often contain urgent requests, suspicious links, or requests for sensitive information.

Email: {email_text[:1000]}

Respond with ONLY: "PHISHING" or "SAFE" """
        
        try:
            response = self.llm(prompt)
            
            if 'PHISHING' in response.upper():
                return {'prediction': 'phishing_email', 'reasoning': response[:100]}
            elif 'SAFE' in response.upper():
                return {'prediction': 'safe_email', 'reasoning': response[:100]}
            else:
                return {'prediction': 'error', 'reasoning': f'Unclear: {response[:50]}'}
        except Exception as e:
            return {'prediction': 'error', 'reasoning': str(e)}


def create_simple_hf_llm(model_name: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        print("Step 1/3: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Step 2/3: Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        )
        
        print("Step 3/3: Setting up generator...")
        
        # Detect device
        if torch.cuda.is_available():
            device = 0
            print("✓ Using CUDA (GPU)")
        else:
            device = -1
            print("✓ Using CPU (slower but works)")
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        print("✓ Model ready!")
        
        def llm(prompt):
            result = generator(
                prompt,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            return result[0]['generated_text']
        
        return llm
        
    except Exception as e:
        raise Exception(f"Failed to load model: {e}\n\nTry: pip install --upgrade transformers torch")


def load_data(filepath: str, n: int = 10):
    df = pd.read_csv(filepath)
    sample = df.sample(min(n, len(df)))
    return [{'message': row['message'], 'label': row['label']} for _, row in sample.iterrows()]


def calculate_accuracy(results: list):
    correct = sum(1 for r in results if r['prediction'] == r['true_label'])
    total = len([r for r in results if r['prediction'] != 'error'])
    return {'accuracy': correct / total if total > 0 else 0, 'correct': correct, 'total': total}


RECOMMENDED_MODELS = {
    '1': ('meta-llama/Llama-3.2-1B-Instruct', 'Llama 3.2 1B (fastest, 2GB RAM)'),
    '2': ('Qwen/Qwen2.5-3B-Instruct', 'Qwen 2.5 3B (best balance, 6GB RAM)'),
    '3': ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3 Mini (efficient, 8GB RAM)'),
    '4': ('google/gemma-2-2b-it', 'Gemma 2 2B (good quality, 5GB RAM)'),
}


def main():
    print("SIMPLE HUGGINGFACE PHISHING DETECTOR")

    print("\nRecommended models:")
    for key, (model_id, desc) in RECOMMENDED_MODELS.items():
        print(f"  {key}. {desc}")
    
    print("\nOr enter any HuggingFace model name directly")
    
    choice = input("\nChoose model (1-4) or enter model name [1]: ").strip() or '1'
    
    if choice in RECOMMENDED_MODELS:
        model_name = RECOMMENDED_MODELS[choice][0]
    else:
        model_name = choice
    
    print(f"\nModel: {model_name}")
    
    try:
        llm = create_simple_hf_llm(model_name)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have transformers installed:")
        print("     pip install transformers torch")
        print("  2. Try updating:")
        print("     pip install --upgrade transformers torch")
        print("  3. Check you have enough RAM for the model")
        return
    
    detector = SimpleDetector(llm)
    
    dataset = input("\nDataset (enron/aigen/trec) [enron]: ").strip() or 'enron'
    n = int(input("How many emails?: ").strip() or '10')
    
    try:
        emails = load_data(f'data/processed/{dataset}.csv', n)
        print(f"\n✓ Loaded {len(emails)} emails")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print("\nAnalyzing (this will take a while on CPU)...")
    results = []
    
    for i, email in enumerate(emails):
        print(f"  {i+1}/{len(emails)}...", end=' ', flush=True)
        
        result = detector.analyze(email['message'])
        result['true_label'] = email['label']
        results.append(result)
        
        if result['prediction'] == 'error':
            print(f"ERROR: {result['reasoning'][:30]}")
        else:
            print(result['prediction'])
    
    metrics = calculate_accuracy(results)
    print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    
    output_file = f'results_hf_{model_name.split("/")[-1]}_{dataset}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset,
            'metrics': metrics,
            'results': results
        }, f, indent=2)
    print(f"\n✓ Saved to {output_file}")


if __name__ == '__main__':
    main()

