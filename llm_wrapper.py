"""
Unified LLM interface supporting multiple providers.

Supports: OpenAI, Anthropic, Ollama, HuggingFace, LM Studio, Together AI, Groq
"""

import json
import requests
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for all LLM implementations."""
    
    @abstractmethod
    def get_response(self, messages: List[Dict]) -> Dict:
        """Get response from LLM."""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI GPT models."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.1):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return {"content": response.choices[0].message.content}


class AnthropicLLM(BaseLLM):
    """Anthropic Claude models."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.1):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        # Extract system message
        system = next(
            (m['content'] for m in messages if m['role'] == 'system'),
            None
        )
        user_messages = [m for m in messages if m['role'] != 'system']
        
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return {"content": response.content[0].text}


class OllamaLLM(BaseLLM):
    """Local Ollama models."""
    
    def __init__(self, base_url: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.1):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        # Combine messages for Ollama
        prompt = self._format_messages(messages)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        return {"content": response.json()['response']}
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for Ollama."""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)


class HuggingFaceLLM(BaseLLM):
    """HuggingFace Inference API."""
    
    def __init__(self, api_key: str, model: str, base_url: str,
                 max_tokens: int = 1024, temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        # Format messages into a single prompt
        prompt = self._format_messages(messages)
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/{self.model}",
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "return_full_text": False
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list):
            result = result[0]
        
        return {"content": result.get('generated_text', str(result))}
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for HuggingFace."""
        formatted = []
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)


class LMStudioLLM(BaseLLM):
    """LM Studio local server."""
    
    def __init__(self, base_url: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.1):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            },
            timeout=120
        )
        response.raise_for_status()
        
        return {"content": response.json()['choices'][0]['message']['content']}


class TogetherLLM(BaseLLM):
    """Together AI API."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://api.together.xyz/v1"
    
    def get_response(self, messages: List[Dict]) -> Dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            },
            timeout=120
        )
        response.raise_for_status()
        
        return {"content": response.json()['choices'][0]['message']['content']}


class GroqLLM(BaseLLM):
    """Groq API."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.1):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, messages: List[Dict]) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return {"content": response.choices[0].message.content}


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(config: Dict) -> BaseLLM:
        """
        Create an LLM instance based on configuration.
        
        Args:
            config: Configuration dictionary from settings.get_llm_config()
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider = config['provider'].lower()
        
        if provider == 'openai':
            return OpenAILLM(
                api_key=config['api_key'],
                model=config['model'],
                base_url=config.get('base_url'),
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'anthropic':
            return AnthropicLLM(
                api_key=config['api_key'],
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'ollama':
            return OllamaLLM(
                base_url=config['base_url'],
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'huggingface':
            return HuggingFaceLLM(
                api_key=config['api_key'],
                model=config['model'],
                base_url=config['base_url'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'lmstudio':
            return LMStudioLLM(
                base_url=config['base_url'],
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'together':
            return TogetherLLM(
                api_key=config['api_key'],
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        elif provider == 'groq':
            return GroqLLM(
                api_key=config['api_key'],
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Convenience function
def create_llm_from_config(config: Dict) -> BaseLLM:
    """
    Create LLM instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM instance
    """
    return LLMFactory.create_llm(config)


if __name__ == "__main__":
    # Test LLM creation
    print("Testing LLM Factory...")
    
    # Test Ollama (no API key needed)
    test_config = {
        'provider': 'ollama',
        'base_url': 'http://localhost:11434',
        'model': 'llama3.1',
        'max_tokens': 100,
        'temperature': 0.1
    }
    
    try:
        llm = create_llm_from_config(test_config)
        print(f"✓ Created {test_config['provider']} LLM")
        
        # Test basic call
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ]
        
        response = llm.get_response(messages)
        print(f"Response: {response['content'][:50]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")