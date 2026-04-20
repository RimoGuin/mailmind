# labelling/llm_validator.py
import json
import random
import time
import os
from tqdm import tqdm
from collections import defaultdict
from labelling.llm_worker import call_llm

# What % of each rule-labeled class to validate
SAMPLE_RATES = {
    'urgent':    0.20,   # High stakes — validate more
    'important': 0.15,
    'normal':    0.08,
    'low':       0.08,
    'spam':      0.05,   # Rule is already very confident here
}

VALIDATION_PROMPT = """You are auditing a rule-based email classifier.
The rule system labeled this email as: [{rule_label}]

Subject: {subject}
Body: {body}

Do you agree? Classify independently.
Categories: urgent, important, normal, low, spam.
Return JSON: {{"label": "...", "confidence": "high | medium | low", 
               "agrees_with_rule": true/false, "reason": "..."}}"""

def validate_rule_labels(
    master_path: str,
    output_path: str,
    provider: str = 'groq',
    min_rule_confidence: float = 0.60,
):
    """
    Samples rule-labeled emails per class and asks LLM to independently
    classify them. Disagreements are flagged and overridden.
    """

    # Resume support — load already-validated IDs
    validated_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                validated_ids.add(json.loads(line)['id'])
    print(f"↩️  Resuming — {len(validated_ids)} already validated.")

    # ── Pass 1: Bucket rule-labeled emails by class ────────────
    buckets = defaultdict(list)
    with open(master_path, 'r') as f:
        for line in tqdm(f, desc="Scanning master"):
            email = json.loads(line)
            lbl = email.get('label')
            src = email.get('label_source', '')
            conf = email.get('confidence', 0.0)

            # Only validate rule-sourced labels, skip LLM-labeled & already done
            if (lbl and 'rule' in src
                    and conf >= min_rule_confidence
                    and email['id'] not in validated_ids):
                buckets[lbl].append(email)

    # ── Pass 2: Stratified sample per class ───────────────────
    to_validate = []
    for label, emails in buckets.items():
        rate = SAMPLE_RATES.get(label, 0.10)
        n = max(1, int(len(emails) * rate))
        sampled = random.sample(emails, min(n, len(emails)))
        print(f"  {label:10s}: {len(emails):>6} rule-labeled → validating {len(sampled)}")
        to_validate.extend(sampled)

    random.shuffle(to_validate)
    print(f"\n🔍 Total to validate: {len(to_validate)} emails\n")

    # ── Pass 3: LLM validation ─────────────────────────────────
    stats = defaultdict(int)

    with open(output_path, 'a') as f_out:
        for email in tqdm(to_validate, desc="Validating"):
            prompt_body = VALIDATION_PROMPT.format(
                rule_label=email['label'],
                subject=email['subject'][:200],
                body=email['body'][:800],
            )

            # Reuse existing call_llm but pass custom prompt
            result = _call_with_custom_prompt(prompt_body, provider)

            if not result:
                stats['api_failures'] += 1
                continue

            agrees = result.get('agrees_with_rule', True)
            llm_label = result.get('label', email['label'])
            stats['total'] += 1

            if not agrees and llm_label != email['label']:
                # LLM disagrees — override
                email.update({
                    'label':            llm_label,
                    'confidence':       0.88 if result.get('confidence') == 'high' else 0.75,
                    'label_source':     f'llm_corrected_{provider}',
                    'reason':           result.get('reason', ''),
                    'rule_label':       email['label'],   # preserve original
                })
                stats['overridden'] += 1
            else:
                # LLM agrees — keep rule label, just mark as validated
                email.update({
                    'label_source': f'rule_validated_{provider}',
                    'llm_validation': 'agreed',
                })
                stats['agreed'] += 1

            f_out.write(json.dumps(email) + '\n')
            f_out.flush()

            time.sleep(2 if provider == 'groq' else 4)

    # ── Report ─────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"✅ Validation complete")
    print(f"   Validated :  {stats['total']}")
    print(f"   Agreed    :  {stats['agreed']}  ({_pct(stats['agreed'], stats['total'])}%)")
    print(f"   Overridden:  {stats['overridden']}  ({_pct(stats['overridden'], stats['total'])}%)")
    print(f"   API fails :  {stats['api_failures']}")
    print(f"{'─'*40}")
    print(f"💾 Results → {output_path}")

    _print_disagreement_report(output_path)


def _call_with_custom_prompt(prompt: str, provider: str):
    """Thin wrapper — same infra as llm_worker but with a custom prompt."""
    try:
        if provider == 'groq':
            from groq import Groq
            import json as _json
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": 
                     "You are an expert email auditor. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return _json.loads(completion.choices[0].message.content)

        elif provider == 'gemini':
            import re, json as _json
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            clean = re.sub(r'```json|```', '', response.text).strip()
            return _json.loads(clean)

    except Exception as e:
        return None


def _print_disagreement_report(output_path: str):
    """Shows which rule→LLM label flips happened most."""
    from collections import Counter
    flips = Counter()
    with open(output_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            if 'rule_label' in r:
                flips[f"{r['rule_label']} → {r['label']}"] += 1

    if flips:
        print("\n📊 Top disagreement patterns (rule → LLM):")
        for pattern, count in flips.most_common(10):
            print(f"   {pattern:30s}  ×{count}")


def _pct(a, b):
    return round(100 * a / b, 1) if b else 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    validate_rule_labels(
        master_path='data/labelled/final_master.jsonl',
        output_path='data/labelled/llm_validated.jsonl',
        provider=os.getenv('LLM_PROVIDER', 'groq'),
    )