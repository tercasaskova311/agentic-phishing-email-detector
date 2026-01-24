import re
from urllib.parse import urlparse

class Tools:
    @staticmethod
    def email_analysis_tool(email_text: str) -> dict:
        """
        Analyze email content for phishing cues.
        
        Returns a dictionary of indicators:
        - suspicious_words: list of keywords like 'urgent', 'verify', etc.
        - excessive_punctuation: count of '!!' or more
        - all_caps: count of words in all caps
        """
        indicators = {}

        # Suspicious keywords
        suspicious_words_list = [
            "urgent", "verify", "password", "login", "account suspended", "click here", "confirm"
        ]
        indicators['suspicious_words'] = [
            w for w in suspicious_words_list if w in email_text.lower()
        ]

        # Excessive punctuation
        indicators['excessive_punctuation'] = len(re.findall(r"[!]{2,}", email_text))

        # All caps words
        indicators['all_caps'] = len(re.findall(r"\b[A-Z]{4,}\b", email_text))

        return indicators

    @staticmethod
    def url_extraction_tool(email_text: str) -> dict:
        """
        Extract URLs from email and check for suspicious patterns.
        
        Returns a dictionary with:
        - urls: list of dictionaries for each URL
        - url: the URL itself
        - is_ip: True if URL uses IP instead of domain
        - suspicious_length: True if URL length > 75 characters
        - odd_subdomains: True if URL has >3 subdomain levels
        - total_urls: total number of URLs found
        """
        urls = re.findall(r'(https?://[^\s]+)', email_text)
        url_indicators = []

        for url in urls:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            indicator = {
                "url": url,
                "is_ip": all(c.isdigit() or c == '.' for c in hostname),
                "suspicious_length": len(url) > 75,
                "odd_subdomains": hostname.count(".") > 3
            }
            url_indicators.append(indicator)

        return {
            "urls": url_indicators,
            "total_urls": len(urls)
        }
