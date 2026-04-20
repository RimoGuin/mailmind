import re
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('GROQ_API_KEY') and not os.getenv('GEMINI_API_KEY'):
    print("### ERROR: No API keys found in .env file!")
    exit()

SYSTEM_PROMPT = """You are an expert business email classifier. 
If an email is a general newsletter, industry update, or mailing list blast without a direct question to the recipient, prioritize 'low' over 'normal'.
Return ONLY valid JSON. No preamble. No markdown blocks."""

USER_PROMPT="""Classify this email:
Subject: {subject}
Body: {body}

Categories: urgent, important, normal, low, spam.
Return JSON: {{"label": "...", "confidence": "high | medium | low", "reason": "..."}}"""

def call_llm(subject, body, provider='groq'):
    """Handles the API call with basic retry logic."""
    prompt = USER_PROMPT.format(subject=subject[:200], body=body[:800])
    
    try:
        if provider == 'groq':
            from groq import Groq
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.1, response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        
        elif provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(SYSTEM_PROMPT + "\n\n" + prompt)
            # Clean potential markdown fences from Gemini
            clean_json = re.sub(r'```json|```', '', response.text).strip()
            return json.loads(clean_json)
    except Exception as e:
        return None

if __name__ == "__main__":
    MASTER_IN = 'data/labelled/final_master.jsonl'
    LLM_OUT = 'data/labelled/llm_results.jsonl'
    PROVIDER = os.getenv('LLM_PROVIDER', 'groq')

    # Load IDs we've already labeled to allow resume
    processed_ids = set()
    if os.path.exists(LLM_OUT):
        with open(LLM_OUT, 'r') as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])

    print(f"🧠 Starting LLM Labelling ({PROVIDER})...")
    
    with open(MASTER_IN, 'r') as f_in, open(LLM_OUT, 'a') as f_out:
        for line in tqdm(f_in, desc="Scanning for ambiguous emails"):
            email = json.loads(line)
            
            # Target: Ambiguous or Low Confidence emails only
            if (email['label'] is None or email['confidence'] < 0.70) and email['id'] not in processed_ids:
                
                result = call_llm(email['subject'], email['body'], provider=PROVIDER)
                
                if result:
                    email.update({
                        'label': result.get('label'),
                        'confidence': 0.90 if result.get('confidence') == 'high' else 0.75,
                        'reason': result.get('reason'),
                        'label_source': f'llm_{PROVIDER}'
                    })
                    f_out.write(json.dumps(email) + '\n')
                    f_out.flush() # Save immediately
                    
                    # Rate limiting for free tiers
                    time.sleep(2 if PROVIDER == 'groq' else 4)