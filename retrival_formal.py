# Standard format
{
    'message': str,
    'label': 'phishing_email' | 'safe_email',
    'subject': str | None,
    'sender': str | None,
    'urls': list[str] | None,
    'dataset_source': 'enron' | 'aigen' | 'trec'
}


#For missing fields, extract what you can (e.g., URLs from message text) and fill with `None` otherwise. The agents can handle missing data.




