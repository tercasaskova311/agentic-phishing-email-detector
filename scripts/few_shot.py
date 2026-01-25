#Few-shot prompts for phishing detection

FEW_SHOT_PROMPT = """You are a cybersecurity expert analyzing emails for phishing attempts (spam email - scams).

Here are examples to guide your analysis:

Example 1 (PHISHING):
"Dear User, your account has been compromised. Click here immediately to verify: http://suspicious-link.com"
→ PHISHING because: urgent threat + suspicious link + impersonal greeting

Example 2 (SAFE):
"Hi Team, please find attached the Q4 report. Let me know if you have questions. - Sarah"
→ SAFE because: professional tone + no suspicious requests + legitimate context

Example 3 (PHISHING):
"URGENT: Your package is waiting. Confirm delivery details now or it will be returned: bit.ly/pkg123"
→ PHISHING because: urgency + shortened link + vague sender

Example 4 (SAFE):
"Meeting reminder: Project sync tomorrow at 2pm in Conference Room B. See you there!"
→ SAFE because: simple reminder + no links + no requests

Now analyze this email:
{email_text}

Think about:
- Is there urgency or threats?
- Are there suspicious links?
- Does it request sensitive information?
- Is the sender legitimate?

Respond in this format:
REASONING: [Your brief analysis in 1-2 sentences]
VERDICT: [PHISHING or SAFE]

Your response:"""

