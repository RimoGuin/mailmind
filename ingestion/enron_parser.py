import email as email_lib
import pandas as pd
import json
import hashlib
import re
from tqdm import tqdm
from labelling.rule_filter import label_email

def clean_body(text: str) -> str:
    """The Claude regex cleaning logic."""
    text = re.split(r'-{3,}\s*(original message|forwarded by)\s*-{3,}', text, flags=re.IGNORECASE)[0]
    text = '\n'.join([l for l in text.split('\n') if not l.strip().startswith('>')])
    text = re.split(r'\n\s*(regards|thanks|best regards|sincerely|cheers)\s*,?\s*\n', text, flags=re.IGNORECASE)[0]
    return re.sub(r'\n{3,}', '\n\n', text).strip()

if __name__ == "__main__":
    IN = 'data\\raw\\datasets\\emails.csv'
    OUT = 'data/labelled/enron_processed.jsonl'
    
    print(f" Processing Enron: {IN}")
    reader = pd.read_csv(IN, chunksize=5000)
    
    with open(OUT, 'w', encoding='utf-8') as f:
        for chunk in tqdm(reader, desc="Enron"):
            for _, row in chunk.iterrows():
                try:
                    msg = email_lib.message_from_string(str(row['message']))
                    body = clean_body(msg.get_payload() if not msg.is_multipart() else "")
                    
                    parsed = {
                        'id': msg.get('Message-ID', 'syn_' + hashlib.md5(body[:50].encode()).hexdigest()[:8]),
                        'from': msg.get('From', ''),
                        'to': msg.get('To', ''),
                        'cc': msg.get('Cc', ''),
                        'thread_id': msg.get('In-Reply-To', 'root'),
                        'subject': msg.get('Subject', ''),
                        'body': body,
                        'timestamp': str(pd.to_datetime(msg.get('Date', ''))),
                        'is_reply': 're:' in msg.get('Subject', '').lower(),
                        'cc_count': len(msg.get('Cc', '').split(',')) if msg.get('Cc') else 0,
                        'reply_chain_depth': body.count('>'),
                        'email_length': len(body),
                        'source_dataset': 'enron',
                    }
                    
                    lbl, conf, res, src = label_email(parsed)
                    parsed.update({'label': lbl, 'confidence': conf, 'label_source': src, 'reason': res})
                    f.write(json.dumps(parsed) + '\n')
                except: continue