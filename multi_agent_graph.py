"""
CrewAI phishing detector with FREE/OPEN-SOURCE model support.
Supports: Ollama (free, local), Groq (free tier), OpenAI, Anthropic
"""
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from tools import email_analysis_tool, url_extraction_tool
from data import load_csv
from src.metrics.metrics import calculate_metrics
import json
import argparse
import re

# Load environment variables
load_dotenv()


def create_llm(provider: str, model: str = None):
    """Create LLM instance based on provider."""
    
    if provider == 'ollama':
        # OLLAMA - Free, Local
        from langchain_community.llms import Ollama
        model = model or 'llama3.1'
        print(f"Using Ollama {model} (local)")
        return Ollama(model=model, base_url='http://localhost:11434')
    
    elif provider == 'groq':
        # GROQ - Free tier available
        from langchain_groq import ChatGroq
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Get one free at https://console.groq.com")
        model = model or 'llama-3.1-8b-instant'
        print(f"Using Groq {model} (free tier)")
        return ChatGroq(model=model, api_key=api_key, temperature=0.1)
    
    elif provider == 'openai':
        # OPENAI - Paid
        from langchain_openai import ChatOpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        model = model or 'gpt-4o-mini'
        print(f"Using OpenAI {model}")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.1)
    
    elif provider == 'anthropic':
        # ANTHROPIC - Paid
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        model = model or 'claude-3-5-sonnet-20241022'
        print(f"Using Anthropic {model}")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0.1)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_phishing_detector_agent(llm):
    """Create the phishing detection agent."""
    return Agent(
        role='Email Security Expert',
        goal='Accurately identify phishing emails and protect users',
        backstory="""Expert cybersecurity analyst with 15 years of experience 
        analyzing phishing emails. You can spot subtle indicators and patterns.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[email_analysis_tool, url_extraction_tool]
    )


def analyze_email(email_content: str, email_label: str, llm, agent) -> dict:
    """Analyze a single email."""
    
    task = Task(
        description=f"""Analyze this email for phishing:

{email_content[:1500]}

Use your tools to check for suspicious patterns and URLs.

Respond in this format:
DECISION: [PHISHING or SAFE]
CONFIDENCE: [0-100]
REASON: [brief explanation]""",
        agent=agent,
        expected_output="Analysis with decision and confidence"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    
    try:
        result = crew.kickoff()
        parsed = parse_response(str(result))
        parsed['true_label'] = email_label
        return parsed
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'reasoning': str(e),
            'true_label': email_label
        }


def parse_response(response: str) -> dict:
    """Parse agent response."""
    decision_match = re.search(r'DECISION:\s*(PHISHING|SAFE)', response, re.IGNORECASE)
    
    if decision_match:
        decision = decision_match.group(1).upper()
        prediction = 'phishing_email' if decision == 'PHISHING' else 'safe_email'
    else:
        response_lower = response.lower()
        if 'phishing' in response_lower:
            prediction = 'phishing_email'
        elif 'safe' in response_lower:
            prediction = 'safe_email'
        else:
            prediction = 'error'
    
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
    confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'reasoning': response[:200]
    }


def main():
    parser = argparse.ArgumentParser(description='CrewAI Phishing Detector (Free Models)')
    parser.add_argument('--provider', type=str, default='ollama',
                       choices=['ollama', 'groq', 'openai', 'anthropic'],
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (optional)')
    parser.add_argument('--dataset', type=str, default='enron',
                       choices=['enron', 'aigen', 'trec'])
    parser.add_argument('--sample-size', type=int, default=10)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CREWAI PHISHING DETECTOR - FREE/OPEN-SOURCE MODELS")
    print("="*60)
    
    # 1. Create LLM
    print(f"\n[1/4] Creating LLM ({args.provider})...")
    try:
        llm = create_llm(args.provider, args.model)
    except Exception as e:
        print(f"✗ Error: {e}")
        if args.provider == 'ollama':
            print("\nTroubleshooting:")
            print("  1. Install: https://ollama.com")
            print("  2. Run: ollama serve")
            print("  3. Pull model: ollama pull llama3.1")
        elif args.provider == 'groq':
            print("\nGet free API key: https://console.groq.com")
            print("Set it: export GROQ_API_KEY='your-key'")
        return
    
    # 2. Create Agent
    print("\n[2/4] Creating agent...")
    agent = create_phishing_detector_agent(llm)
    print("✓ Agent ready")
    
    # 3. Load Data
    print(f"\n[3/4] Loading data ({args.dataset})...")
    try:
        emails = load_csv(f"data/processed/{args.dataset}.csv", args.sample_size)
        print(f"✓ Loaded {len(emails)} emails")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 4. Analyze
    print(f"\n[4/4] Analyzing {len(emails)} emails...")
    results = []
    errors = 0
    
    for i, email in enumerate(emails):
        print(f"  [{i+1}/{len(emails)}] ", end="", flush=True)
        
        result = analyze_email(email['message'], email['label'], llm, agent)
        results.append(result)
        
        if result['prediction'] == 'error':
            errors += 1
            print("ERROR")
        else:
            print(f"{result['prediction']}")
    
    # 5. Metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if errors > 0:
        print(f"\n⚠ {errors}/{len(emails)} emails had errors")
    
    metrics = calculate_metrics(results)
    
    if 'error' not in metrics:
        print(f"\nAccuracy:  {metrics['accuracy']:.1%}")
        print(f"Precision: {metrics['precision']:.1%}")
        print(f"Recall:    {metrics['recall']:.1%}")
        print(f"F1 Score:  {metrics['f1']:.1%}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")
    
    # 6. Save
    output_file = f'results_crewai_{args.provider}_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'provider': args.provider,
                'model': args.model or 'default',
                'dataset': args.dataset,
                'sample_size': len(emails)
            },
            'metrics': metrics,
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()