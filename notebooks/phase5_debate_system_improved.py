#!/usr/bin/env python3
"""
Phase 5: Multi-Agent Debate System (IMPROVED)
Better prompts, structured JSON output, improved error handling
"""

import pandas as pd
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from groq import Groq
import re

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq API Key
GROQ_API_KEY = "gsk_XkgbAtl7TpAlByllFO6CWGdyb3FYKSp4w1w