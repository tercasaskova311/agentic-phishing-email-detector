CONFIG = {
    'model_name': 'Qwen/Qwen2.5-3B-Instruct',
    # 'meta-llama/Llama-3.2-3B-Instruct',      
    # 'google/gemma-2-2b-it',                  
    # 'microsoft/Phi-3-mini-4k-instruct'       
    
    'datasets': {
        'aigen': '/kaggle/input/phishing-emails/aigen.csv',
        'enron': '/kaggle/input/phishing-emails/enron.csv',
        'trec': '/kaggle/input/phishing-emails/trec.csv'
    },
    
    'sample_sizes': {
        'aigen': None,
        'enron': 1000,
        'trec': 1000
    },
    
    'generation': {
        'max_new_tokens': 50,
        'temperature': 0.1,
        'do_sample': True,
        'top_p': 0.9
    },
    
    'balanced_sampling': True,
    'save_errors': True,
    'checkpoint_every': 500,  # More frequent checkpoints
    'experiment_name': 'zero_shot_baseline',
    'strategy': 'zero-shot',
    'prompt_template': ZERO_SHOT_PROMPT,
    'use_4bit': False,
    'batch_size': 16,  # Increased for better GPU utilization
}