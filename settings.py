import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

class LLMConfig:    
    # Current provider (set via environment variable)
    PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to free/open source
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # Optional for proxies
    
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    
    HF_API_KEY = os.getenv("HF_API_KEY")
    HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    HF_BASE_URL = os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co/models")
    
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
    
    # Groq Configuration (Fast & Free Tier)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    
    # General LLM settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))



class EvalConfig: # Sample sizes for each dataset
    ENRON_SAMPLE_SIZE = int(os.getenv("ENRON_SAMPLE_SIZE", "100"))
    TREC_SAMPLE_SIZE = int(os.getenv("TREC_SAMPLE_SIZE", "100"))
    AIGEN_SAMPLE_SIZE = int(os.getenv("AIGEN_SAMPLE_SIZE", "100"))
    
    VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
    SAVE_RESULTS = os.getenv("SAVE_RESULTS", "true").lower() == "true"
    
    # Batch processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    USE_PARALLEL = os.getenv("USE_PARALLEL", "false").lower() == "true"
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

class DataConfig:
    
    # Dataset file paths
    ENRON_PATH = DATA_DIR / os.getenv("ENRON_FILE", "enron.csv")
    TREC_PATH = DATA_DIR / os.getenv("TREC_FILE", "trec.csv")
    AIGEN_PATH = DATA_DIR / os.getenv("AIGEN_FILE", "aigen.csv")
    
    # Data processing
    EXTRACT_URLS = os.getenv("EXTRACT_URLS", "true").lower() == "true"
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

class AgentConfig:
    # Prompt template to use
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "default")
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Enable/disable features
    USE_RISK_INDICATORS = os.getenv("USE_RISK_INDICATORS", "true").lower() == "true"
    USE_RECOMMENDATIONS = os.getenv("USE_RECOMMENDATIONS", "true").lower() == "true"

def get_llm_config() -> dict:
    provider = LLMConfig.PROVIDER.lower()
    
    configs = {
        "openai": {
            "provider": "openai",
            "api_key": LLMConfig.OPENAI_API_KEY,
            "model": LLMConfig.OPENAI_MODEL,
            "base_url": LLMConfig.OPENAI_BASE_URL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
        "anthropic": {
            "provider": "anthropic",
            "api_key": LLMConfig.ANTHROPIC_API_KEY,
            "model": LLMConfig.ANTHROPIC_MODEL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
        "ollama": {
            "provider": "ollama",
            "base_url": LLMConfig.OLLAMA_BASE_URL,
            "model": LLMConfig.OLLAMA_MODEL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
        "huggingface": {
            "provider": "huggingface",
            "api_key": LLMConfig.HF_API_KEY,
            "model": LLMConfig.HF_MODEL,
            "base_url": LLMConfig.HF_BASE_URL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
        "together": {
            "provider": "together",
            "api_key": LLMConfig.TOGETHER_API_KEY,
            "model": LLMConfig.TOGETHER_MODEL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
        "groq": {
            "provider": "groq",
            "api_key": LLMConfig.GROQ_API_KEY,
            "model": LLMConfig.GROQ_MODEL,
            "max_tokens": LLMConfig.MAX_TOKENS,
            "temperature": LLMConfig.TEMPERATURE,
        },
    }
    
    if provider not in configs:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Available: {', '.join(configs.keys())}"
        )
    
    return configs[provider]


def validate_config():
    """
    Validate that required configuration is present.
    
    Raises:
        ValueError: If required configuration is missing
    """
    provider = LLMConfig.PROVIDER.lower()
    
    # Check provider-specific requirements
    if provider == "openai" and not LLMConfig.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
    
    if provider == "anthropic" and not LLMConfig.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
    
    if provider == "huggingface" and not LLMConfig.HF_API_KEY:
        raise ValueError("HF_API_KEY is required for HuggingFace provider")
    
    if provider == "groq" and not LLMConfig.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for Groq provider")


def print_config():
    """Print current configuration (without sensitive data)."""
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"\nLLM Provider: {LLMConfig.PROVIDER}")
    print(f"Model: {get_llm_config()['model']}")
    print(f"Temperature: {LLMConfig.TEMPERATURE}")
    print(f"Max Tokens: {LLMConfig.MAX_TOKENS}")
    print(f"\nDatasets:")
    print(f"  ENRON: {EvalConfig.ENRON_SAMPLE_SIZE} samples")
    print(f"  TREC: {EvalConfig.TREC_SAMPLE_SIZE} samples")
    print(f"  AIGEN: {EvalConfig.AIGEN_SAMPLE_SIZE} samples")
    print(f"\nVerbose: {EvalConfig.VERBOSE}")
    print(f"Batch Size: {EvalConfig.BATCH_SIZE}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")