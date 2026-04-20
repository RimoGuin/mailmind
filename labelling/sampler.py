import json
import random
import os
from collections import defaultdict
from tqdm import tqdm

def get_ambiguous_sample(input_path, output_path, n=20):
    candidates = []
    
    print(f"🧐 Hunting for ambiguous emails in {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning Master"):
            email = json.loads(line)
            if email.get('confidence') == 0.0 and email.get('reason') == 'Ambiguous':
                candidates.append(email)
    
    if not candidates:
        print("⚠️ No ambiguous emails found! Check your rule_filter logic.")
        return

    print(f"📊 Found {len(candidates)} total ambiguous emails.")
    sample_size = min(len(candidates), n)
    sampled_emails = random.sample(candidates, sample_size)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for email in sampled_emails:
            f_out.write(json.dumps(email) + '\n')
            
    print(f"✅ Successfully sampled {sample_size} random ambiguous emails to {output_path}")


# Hard caps not percentages — percentages on 575K emails = 40K+ API calls (~23 hrs).
# These caps catch systematic rule errors with ~1,200 calls total (~40 mins on Groq free tier).
VALIDATION_CAPS = {
    'urgent':    200,
    'important': 300,
    'normal':    300,
    'low':       200,
    'spam':      200,
}

def get_rule_sample(input_path, output_path):
    """
    Stratified sample of rule-labeled emails for LLM validation.
    Only picks label_source='rule' and confidence >= 0.60.
    For future phases: add source_dataset filter to avoid re-validating old data.
    """
    buckets = defaultdict(list)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning for rule labels"):
            email = json.loads(line)
            lbl  = email.get('label')
            src  = email.get('label_source', '')
            conf = email.get('confidence', 0.0)
            if lbl and src == 'rule' and conf >= 0.60:
                buckets[lbl].append(email)

    sampled = []
    for label, emails in buckets.items():
        cap   = VALIDATION_CAPS.get(label, 200)
        picked = random.sample(emails, min(cap, len(emails)))
        print(f"  {label:10s}: {len(emails):>6} available → {len(picked):>5} sampled")
        sampled.extend(picked)

    random.shuffle(sampled)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for email in sampled:
            f_out.write(json.dumps(email) + '\n')

    print(f"✅ Rule sample: {len(sampled)} emails → {output_path}")


if __name__ == "__main__":
    MASTER_IN  = 'data/labelled/final_master.jsonl'
    SAMPLE_OUT = 'data/labelled/test_sample_ambiguous.jsonl'
    get_ambiguous_sample(MASTER_IN, SAMPLE_OUT, n=20)