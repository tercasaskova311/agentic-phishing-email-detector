#Zero-shot prompts for phishing detection
#A/B testing

#A version ===========================
ZERO_SHOT_A = """You are a cybersecurity expert analyzing emails for phishing attempts.

Email to analyze:
{email_text}

Determine if this email is PHISHING or SAFE.

Respond with ONLY one word: PHISHING or SAFE

Your answer:"""


#B version ===========================
ZERO_SHOT_B = """You are a cybersecurity analyst. Classify this email, decide if its a phishing email or normal email. Phishig email means its a spam.

Email:
{email_text}

Is this phishing? Reply: PHISHING or SAFE only.

Classification:"""