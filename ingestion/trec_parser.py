import pandas as pd
import json
import hashlib
from tqdm import tqdm
from ingestion.enron_parser import clean_body
from labelling.rule_filter import label_email

if __name__ == "__main__":
    IN = 'data\\raw\\datasets\\processed_data.csv'
    OUT = 'data/labelled/trec_processed.jsonl'
    
    print(f"🚀 Processing TREC: {IN}")
    reader = pd.read_csv(IN, chunksize=5000)
    
    with open(OUT, 'w', encoding='utf-8') as f:
        for chunk in tqdm(reader, desc="TREC"):
            for _, row in chunk.iterrows():
                body = clean_body(str(row.get('message', '')))
                
                parsed = {
                    'id': f"trec_{hashlib.md5(body[:50].encode()).hexdigest()[:8]}",
                    'from': str(row.get('email_from', '')),
                    'to': str(row.get('email_to', '')),
                    'cc': '',
                    'thread_id': 'root',
                    'subject': str(row.get('subject', '')),
                    'body': body,
                    'timestamp': None,
                    'is_reply': False,
                    'cc_count': 0,
                    'reply_chain_depth': 0,
                    'email_length': len(body),
                    'source_dataset': 'trec',
                }
                
                # TREC Mapping: 1=Spam, 0=Ham
                if str(row.get('label')) == '1':
                    parsed.update({'label': 'spam', 'confidence': 0.99, 'label_source': 'trec_index', 'reason': 'TREC Spam'})
                else:
                    lbl, conf, res, src = label_email(parsed)
                    parsed.update({'label': lbl, 'confidence': conf, 'label_source': src, 'reason': res})
                
                f.write(json.dumps(parsed) + '\n')