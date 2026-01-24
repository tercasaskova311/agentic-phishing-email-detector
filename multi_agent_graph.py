"""
CrewAI multi-agent phishing detector using Hugging Face open-source models.
Graph structure:
Email -> (Content Agent, URL Agent, Pattern Agent) -> Judge Agent
"""

import os
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint

from tools import Tools 
from config import get_llm_from_hf

load_dotenv()

#======= Agents =======================================

#content + context
email_agent = Agent(
    role="Email Content Analyst",
    goal="Detect phishing language in emails - context, tricky and spam language",
    backstory="Cybersecurity analyst specializing in linguistic phishing cues.",
    llm=get_llm_from_hf("Qwen/Qwen2.5-3B-Instruct"),
    tools=[Tools.email_analysis_tool],
    allow_delegation=False,
    verbose=True,
)

#check if url exists
url_agent = Agent(
    role="URL Inspector",
    goal="Analyze URLs for phishing indicators",
    backstory="Expert in malicious URLs and domain obfuscation.",
    llm=get_llm_from_hf("microsoft/Phi-3-mini-4k-instruct"),
    tools=[Tools.url_extraction_tool],
    allow_delegation=False,
    verbose=True,
)

#header + sender check 
pattern_agent = Agent(
    role="Email Pattern Analyst",
    goal="Detect known phishing templates and metadata inconsistencies",
    backstory="Specialist in email headers, subject, sender spoofing, and scam patterns.",
    llm=get_llm_from_hf("meta-llama/Llama-3.2-3B-Instruct"),
    allow_delegation=False,
    verbose=True,
)

#final judge => agg everythign together
judge_agent = Agent(
    role="Final Judge",
    goal="Aggregate all analyses and decide if the email is phishing",
    backstory="Senior security expert making final classification decisions.",
    llm=get_llm_from_hf("Qwen/Qwen2.5-7B-Instruct"),
    allow_delegation=False,
    verbose=True,
)


#==== GRAPH EDGES =========================================
def build_tasks(email_text: str):
    t1 = Task(
        description=f"""
Analyze the email text for phishing language.
Email:
{email_text}
""",
        agent=email_agent,
        expected_output="Suspicious language indicators",
    )

    t2 = Task(
        description=f"""
Extract and analyze URLs in the email.
Email:
{email_text}
""",
        agent=url_agent,
        expected_output="URL-based phishing indicators",
    )

    t3 = Task(
        description=f"""
Analyze sender,email subject formatting, and known phishing patterns.
Email:
{email_text}
""",
        agent=pattern_agent,
        expected_output="Pattern-based phishing indicators",
    )

    t4 = Task(
        description="""
You are given three analyses:
1) Email language analysis
2) URL analysis
3) Pattern analysis

Combine them and respond EXACTLY in this format:
DECISION: PHISHING or SAFE
CONFIDENCE: integer 0-100
REASON: one sentence
""",
        agent=judge_agent,
        context=[t1, t2, t3],  # graph dependency
        expected_output="Final phishing verdict",
    )

    return [t1, t2, t3, t4]


# Parsing
# ---------------------------
def parse_response(text: str):
    decision_match = re.search(r"DECISION:\s*(PHISHING|SAFE)", text, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text)

    decision = decision_match.group(1).upper() if decision_match else "ERROR"
    confidence = int(conf_match.group(1)) / 100.0 if conf_match else 0.5

    return {
        "prediction": "phishing_email" if decision == "PHISHING" else "safe_email",
        "confidence": confidence,
        "reasoning": text.strip()[:300],
    }


# Run single email
# ---------------------------
def analyze_email(email_text: str):
    tasks = build_tasks(email_text)

    crew = Crew(
        agents=[email_agent, url_agent, pattern_agent, judge_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return parse_response(str(result))




