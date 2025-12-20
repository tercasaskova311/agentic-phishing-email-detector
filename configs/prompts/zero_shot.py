#Zero-shot prompts for phishing detection

ZERO_SHOT_PROMPT = """You are a cybersecurity expert analyzing emails for phishing attempts.

Email to analyze:
{email_text}

Determine if this email is PHISHING or SAFE.

Respond with ONLY one word: PHISHING or SAFE

Your answer:"""


# Alternative versions for A/B testing
ZERO_SHOT_PROMPT_V2 = """Analyze this email for phishing:

{email_text}

Classification: PHISHING or SAFE

Answer:"""


ZERO_SHOT_PROMPT_STRICT = """You are a security analyst. Classify this email.

Email:
{email_text}

Is this phishing? Reply: PHISHING or SAFE only.

Classification:"""