import re
from labelling.whitelist_check import whitelist_check

# Patterns
SPAM_P = [r'unsubscribe', r'click here', r'buy now', r'won', r'lottery', r'casino', r'viagra']
URGENT_P = ['urgent', 'asap', 'immediately', 'deadline', 'critical', 'action required', 'must respond']
IMPORTANT_P = ['please review', 'can you', 'i need', 'follow up', 'thoughts?', 'please advise']

def label_email(row: dict) -> tuple:
    sub = str(row.get('subject', '')).lower()
    body = str(row.get('body', '')).lower()
    
    # 1. Whitelist pass
    _, _, _, is_whitelisted = whitelist_check(row)

    # 2. Spam Pass (Pass 1 of logic)
    if any(re.search(p, sub + body) for p in SPAM_P):
        return ('spam', 0.95, 'Spam keywords detected', 'rule')
    
    # 3. Importance Pass (Pass 2 of logic)
    if any(kw in sub for kw in URGENT_P):
        return ('urgent', 0.88, 'Urgent keyword in subject', 'rule')
    
    if '?' in body and any(kw in body for kw in IMPORTANT_P):
        return ('important', 0.82, 'Question + Action keyword', 'rule')
    
    if len(body) > 150 and not '?' in body:
        return ('normal', 0.70, 'Informational text', 'rule')
    
    if is_whitelisted:
        return ('normal', 0.60, 'Whitelisted domain', 'rule')

    return (None, 0.0, 'Ambiguous', 'rule')