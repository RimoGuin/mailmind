import os
import json
import time
from dotenv import load_dotenv
from labelling.llm_worker import call_llm

load_dotenv()

def run_smoke_test():
    TEST_IN = 'data/labelled/test_sample_ambiguous.jsonl'
    TEST_OUT = 'data/processed/test_results.jsonl'
    PROVIDER = os.getenv('LLM_PROVIDER', 'groq')

    if not os.path.exists(TEST_IN):
        print(f"❌ Error: {TEST_IN} not found. Run the sampler first!")
        return

    print(f"🧪 Running Smoke Test on {PROVIDER}...")
    
    with open(TEST_IN, 'r') as f_in, open(TEST_OUT, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            email = json.loads(line)
            
            print(f"👉 Processing: {email['subject'][:50]}...")
            
            # API Call
            result = call_llm(email['subject'], email['body'], provider=PROVIDER)
            
            if result:
                email.update({
                        'label': result.get('label'),
                        'confidence': 0.90 if result.get('confidence') == 'high' else 0.75,
                        'reason': result.get('reason'),
                        'label_source': f'llm_{PROVIDER}'
                    })
                f_out.write(json.dumps(email) + '\n')
                
                # Print a live snippet for you to watch
                print(f"   Result: [{email['label']}] - {email['reason']}")
                
                # Rate limit respect
                time.sleep(2 if PROVIDER == 'groq' else 4)

    print(f"\n✅ Test complete. Results at {TEST_OUT}")

if __name__ == "__main__":
    run_smoke_test()