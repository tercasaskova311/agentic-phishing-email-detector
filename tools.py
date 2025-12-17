"""
Custom tools for phishing email detection.
"""
from crewai_tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import re
import pandas as pd


class EmailAnalysisInput(BaseModel):
    """Input schema for email analysis."""
    email_content: str = Field(..., description="The email content to analyze")


class URLExtractionInput(BaseModel):
    """Input schema for URL extraction."""
    text: str = Field(..., description="Text to extract URLs from")


class DatasetLoaderInput(BaseModel):
    """Input schema for loading dataset."""
    dataset_name: str = Field(..., description="Dataset name: enron, aigen, or trec")
    sample_size: int = Field(default=10, description="Number of emails to sample")


class EmailAnalysisTool(BaseTool):
    """Tool for analyzing email content for phishing indicators."""
    
    name: str = "email_analyzer"
    description: str = (
        "Analyzes email content to identify phishing indicators. "
        "Returns a list of suspicious patterns found in the email."
    )
    args_schema: Type[BaseModel] = EmailAnalysisInput
    
    def _run(self, email_content: str) -> str:
        """Analyze email for phishing indicators."""
        indicators = []
        
        # Check for urgent language
        urgent_patterns = [
            r'urgent', r'immediate', r'act now', r'within \d+ hours',
            r'expire', r'suspended', r'verify now', r'confirm immediately'
        ]
        for pattern in urgent_patterns:
            if re.search(pattern, email_content, re.IGNORECASE):
                indicators.append(f"Urgent language detected: '{pattern}'")
                break
        
        # Check for sensitive info requests
        sensitive_patterns = [
            r'password', r'ssn', r'social security', r'credit card',
            r'bank account', r'pin', r'account number', r'routing number'
        ]
        for pattern in sensitive_patterns:
            if re.search(pattern, email_content, re.IGNORECASE):
                indicators.append(f"Requests sensitive information: '{pattern}'")
                break
        
        # Check for suspicious URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, email_content)
        if urls:
            indicators.append(f"Contains {len(urls)} URLs")
            # Check for suspicious domains
            suspicious_domains = ['bit.ly', 'tinyurl', 'shortened']
            for url in urls:
                for domain in suspicious_domains:
                    if domain in url:
                        indicators.append(f"Contains shortened URL: {url}")
                        break
        
        # Check for money/payment requests
        money_patterns = [
            r'\$\d+', r'payment', r'transfer', r'gift card', r'wire',
            r'bitcoin', r'cryptocurrency', r'paypal', r'venmo'
        ]
        for pattern in money_patterns:
            if re.search(pattern, email_content, re.IGNORECASE):
                indicators.append(f"Money/payment related: '{pattern}'")
                break
        
        # Check for impersonation
        impersonation_patterns = [
            r'dear (customer|user|member)', r'from: (admin|support|security)',
            r'(bank|paypal|amazon|microsoft|apple|google) (team|support)'
        ]
        for pattern in impersonation_patterns:
            if re.search(pattern, email_content, re.IGNORECASE):
                indicators.append(f"Possible impersonation: '{pattern}'")
                break
        
        if not indicators:
            return "No obvious phishing indicators detected."
        
        return "Phishing indicators found:\n" + "\n".join(f"- {ind}" for ind in indicators)


class URLExtractionTool(BaseTool):
    """Tool for extracting URLs from text."""
    
    name: str = "url_extractor"
    description: str = (
        "Extracts all URLs from the given text. "
        "Useful for analyzing links in suspicious emails."
    )
    args_schema: Type[BaseModel] = URLExtractionInput
    
    def _run(self, text: str) -> str:
        """Extract URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        if not urls:
            return "No URLs found in the text."
        
        return f"Found {len(urls)} URLs:\n" + "\n".join(f"- {url}" for url in urls)


class DatasetLoaderTool(BaseTool):
    """Tool for loading and sampling email datasets."""
    
    name: str = "dataset_loader"
    description: str = (
        "Loads email datasets (enron, aigen, or trec) and returns sampled emails. "
        "Use this to get test emails for analysis."
    )
    args_schema: Type[BaseModel] = DatasetLoaderInput
    
    def _run(self, dataset_name: str, sample_size: int = 10) -> str:
        """Load and sample dataset."""
        try:
            # Map dataset names to file paths
            dataset_paths = {
                'enron': 'data/processed/enron.csv',
                'aigen': 'data/processed/aigen.csv',
                'trec': 'data/processed/trec.csv'
            }
            
            if dataset_name not in dataset_paths:
                return f"Unknown dataset: {dataset_name}. Available: enron, aigen, trec"
            
            # Load dataset
            df = pd.read_csv(dataset_paths[dataset_name])
            
            # Sample
            sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            # Format output
            result = f"Loaded {len(sample)} emails from {dataset_name} dataset:\n\n"
            for idx, row in sample.iterrows():
                result += f"Email {idx}:\n"
                result += f"Subject: {row.get('subject', 'N/A')}\n"
                result += f"Label: {row.get('label', 'N/A')}\n"
                result += f"Message: {str(row.get('message', ''))[:200]}...\n\n"
            
            return result
            
        except Exception as e:
            return f"Error loading dataset: {str(e)}"


# Initialize tools
email_analysis_tool = EmailAnalysisTool()
url_extraction_tool = URLExtractionTool()
dataset_loader_tool = DatasetLoaderTool()

# Export for easy import
__all__ = ['email_analysis_tool', 'url_extraction_tool', 'dataset_loader_tool']