#!/usr/bin/env python3
"""
Phase 5: Multi-Agent Debate System
Three agents debate to classify emails:
- Agent 1 (Attacker): Argues the email is phishing
- Agent 2 (Defender): Argues the email is legitimate
- Agent 3 (Judge): Makes final decision based on debate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

class DebateSystem:
    """Multi-agent debate system for phishing detection"""
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.1-8b-instant"  # Fast model for all agents
    
    def call_agent(self, role: str, prompt: str) -> dict:
        """Call an agent with specific role"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tok