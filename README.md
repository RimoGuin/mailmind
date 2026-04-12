# Email Intelligence — Data Pipeline v2

Medallion architecture: **raw → cleaned → results**, all in a local SQLite file.

## Structure

```
email_pipeline_v2/
├── run_pipeline.py          ← orchestrator (full + per-stage CLI)
├── requirements.txt
├── db/
│   ├── schema.py            ← all table definitions (3 metadata layers)
│   └── connection.py        ← SQLite engine + pragma setup
└── pipeline/
    ├── adapters.py          ← per-dataset column mappers (Enron, SpamHam, generic)
    ├── stage1_load.py       ← CSV → raw.*
    ├── stage2_transform.py  ← raw.* → cleaned.*
    └── stage3_enrich.py     ← cleaned.* → results.* (Claude API + aggregates)
```

## Schema

```
RAW LAYER
  raw_emails          id, dataset_name, dataset_row, sender_raw, recipients_raw,
                      subject_raw, body_raw, timestamp_raw, thread_id_raw, extra_fields
  raw_senders         id, sender_str, dataset_name
  raw_recipients      id, raw_email_id → raw_emails, recipient_str, recipient_type

CLEANED LAYER
  cleaned_senders     id, email_address, display_name, domain, email_count
  cleaned_emails      id, raw_id, dataset_name, sender_id → cleaned_senders,
                      subject, body, body_length, timestamp, year, month,
                      day_of_week, hour, thread_id, content_hash
  cleaned_recipients  id, email_address, display_name, domain
  cleaned_email_recipients  email_id → cleaned_emails, recipient_id → cleaned_recipients

RESULTS LAYER
  results_classifications   email_id, priority, sender_type, intent, sentiment, summary
  results_sender_profiles   sender_id, dominant_type, total_emails, urgent_count,
                            avg_sentiment_score, sentiment_trend
  results_vendor_metrics    vendor_domain, avg_gap_hours, sla_breach, urgent_ratio_pct
  results_client_trails     sender_id, email_id, sequence_num, priority, intent, timestamp
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Full pipeline (load + transform + enrich)
python run_pipeline.py --input data/enron.csv --dataset enron

# Multiple datasets
python run_pipeline.py --input data/enron.csv   --dataset enron
python run_pipeline.py --input data/spamham.csv --dataset spamham
python run_pipeline.py --stage transform         # transform all pending
python run_pipeline.py --stage enrich            # enrich all pending

# Individual stages
python run_pipeline.py --stage load      --input data/emails.csv
python run_pipeline.py --stage transform
python run_pipeline.py --stage enrich --limit 100   # test with 100 emails first

# Rebuild aggregates only (no API calls)
python run_pipeline.py --stage enrich --skip-classification
```

## Adding a New Dataset

1. Add a new class in `pipeline/adapters.py` extending `BaseAdapter`
2. Map its columns in `iter_rows()`
3. Add detection logic to `detect_adapter()` (or pass explicitly)

No other files need to change.
