import json
from typing import List, Dict


class SimpleAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a phishing email detector.

Analyze emails and respond with JSON:
{
    "prediction": "phishing_email" or "safe_email",
    "confidence": 0.0 to 1.0,
    "reasoning": "why you made this decision"
}

Look for: urgent language, suspicious links, credential requests, spoofed senders."""
    
    def analyze(self, email_text: str) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Analyze this email:\n\n{email_text}"}
        ]
        
        try:
            response = self.llm.get_response(messages)
            content = response.get('content', str(response))
            
            # Extract JSON
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                result = json.loads(content[start:end])
            else:
                result = {"prediction": "safe_email", "confidence": 0.5, "reasoning": content}
            
            return result
            
        except Exception as e:
            return {"prediction": "error", "confidence": 0.0, "reasoning": str(e)}
    
    def analyze_batch(self, emails: List[str]) -> List[Dict]:
        """Analyze multiple emails."""
        return [self.analyze(email) for email in emails]

if __name__ == "__main__":
    class MockLLM:
        def get_response(self, messages):
            return {"content": '{"prediction": "phishing_email", "confidence": 0.9, "reasoning": "urgent language"}'}
    
    agent = SimpleAgent(llm=MockLLM())
    email = "URGENT! Your account will be closed. Click here now!"
    result = agent.analyze(email)
    
    print(json.dumps(result, indent=2))