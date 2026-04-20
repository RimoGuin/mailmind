def whitelist_check(row: dict) -> tuple:
    """
    Returns: (label, confidence, reason, is_whitelisted)
    """
    trusted_domains = ['enron.com', 'google.com', 'p360.com']
    from_addr = str(row.get('from', '')).lower()
    
    for domain in trusted_domains:
        if from_addr.endswith(domain):
            # We don't auto-label as 'normal' yet, just flag that it's from a trusted source
            return (None, 0.0, "Trusted domain", True)
            
    return (None, 0.0, "Not whitelisted", False)