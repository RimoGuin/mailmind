"""
db/schema.py
SQLAlchemy Core schema definitions for all 3 layers:
  raw.*      — ingested as-is from any Kaggle dataset
  cleaned.*  — normalized, deduped, parsed
  results.*  — Claude labels + analytics aggregates
"""

from sqlalchemy import (
    MetaData, Table, Column,
    Integer, String, Text, Float, Boolean,
    DateTime, JSON,
    ForeignKey, UniqueConstraint, Index,
    text
)

# One metadata per schema layer — keeps them logically separated
raw_meta     = MetaData()
cleaned_meta = MetaData()
results_meta = MetaData()


# ════════════════════════════════════════════════════════════════
#  RAW LAYER  — exact dump, no transforms, source-of-truth
# ════════════════════════════════════════════════════════════════

raw_emails = Table("raw_emails", raw_meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("dataset_name", String(100), nullable=False),        # e.g. "enron", "spamham"
    Column("dataset_row",  Integer, nullable=False),            # original row index
    Column("sender_raw",   Text),
    Column("recipients_raw", Text),                             # raw to/cc string
    Column("subject_raw",  Text),
    Column("body_raw",     Text),
    Column("timestamp_raw",String(200)),                        # unparsed string
    Column("thread_id_raw",String(200)),
    Column("extra_fields", JSON),                               # any extra dataset columns
    Column("ingested_at",  DateTime, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("dataset_name", "dataset_row", name="uq_raw_source"),
)

raw_senders = Table("raw_senders", raw_meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("sender_str",   Text, nullable=False),               # raw string as-found
    Column("dataset_name", String(100), nullable=False),
    Column("first_seen_at",DateTime, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("sender_str", "dataset_name", name="uq_raw_sender"),
)

raw_recipients = Table("raw_recipients", raw_meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("raw_email_id", Integer, ForeignKey("raw_emails.id"), nullable=False),
    Column("recipient_str",Text, nullable=False),
    Column("recipient_type", String(10), default="to"),         # to / cc / bcc
)

Index("ix_raw_recipients_email_id", raw_recipients.c.raw_email_id)


# ════════════════════════════════════════════════════════════════
#  CLEANED LAYER  — normalized, deduped, parsed
# ════════════════════════════════════════════════════════════════

cleaned_senders = Table("cleaned_senders", cleaned_meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("email_address",String(320), nullable=False, unique=True),
    Column("display_name", String(200)),
    Column("domain",       String(200)),
    Column("first_seen_at",DateTime),
    Column("last_seen_at", DateTime),
    Column("email_count",  Integer, default=0),
)

Index("ix_cleaned_senders_domain", cleaned_senders.c.domain)

cleaned_emails = Table("cleaned_emails", cleaned_meta,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("raw_id",            Integer, nullable=False),
    Column("dataset_name",      String(100), nullable=False),
    Column("sender_id",         Integer, ForeignKey("cleaned_senders.id")),
    # --- Content ---
    Column("subject",           Text),
    Column("body",              Text),
    Column("cc",                Text),
    # --- Timestamp ---
    Column("timestamp",         DateTime),
    Column("year",              Integer),
    Column("month",             Integer),
    Column("day_of_week",       Integer),
    Column("hour",              Integer),
    # --- Structural ---
    Column("is_reply",          Boolean, default=False),
    Column("cc_count",          Integer, default=0),
    Column("reply_chain_depth", Integer, default=0),
    Column("email_length",      Integer, default=0),
    Column("body_length",       Integer),
    # --- Threading ---
    Column("thread_id",         String(200)),
    Column("content_hash",      String(64), unique=True),
    Column("cleaned_at",        DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

Index("ix_cleaned_emails_sender",    cleaned_emails.c.sender_id)
Index("ix_cleaned_emails_timestamp", cleaned_emails.c.timestamp)
Index("ix_cleaned_emails_dataset",   cleaned_emails.c.dataset_name)
Index("ix_cleaned_emails_thread",    cleaned_emails.c.thread_id)

cleaned_recipients = Table("cleaned_recipients", cleaned_meta,
    Column("id",             Integer, primary_key=True, autoincrement=True),
    Column("email_address",  String(320), nullable=False),
    Column("display_name",   String(200)),
    Column("domain",         String(200)),
    UniqueConstraint("email_address", name="uq_cleaned_recipient"),
)

cleaned_email_recipients = Table("cleaned_email_recipients", cleaned_meta,
    Column("id",             Integer, primary_key=True, autoincrement=True),
    Column("email_id",       Integer, ForeignKey("cleaned_emails.id"), nullable=False),
    Column("recipient_id",   Integer, ForeignKey("cleaned_recipients.id"), nullable=False),
    Column("recipient_type", String(10), default="to"),
    UniqueConstraint("email_id", "recipient_id", "recipient_type", name="uq_email_recipient"),
)

Index("ix_email_recipients_email",     cleaned_email_recipients.c.email_id)
Index("ix_email_recipients_recipient", cleaned_email_recipients.c.recipient_id)


# ════════════════════════════════════════════════════════════════
#  RESULTS LAYER  — dashboard-ready, enriched
# ════════════════════════════════════════════════════════════════

results_classifications = Table("results_classifications", results_meta,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("email_id",      Integer, nullable=False, unique=True),
    # --- Labels ---
    Column("label",         String(20)),        # primary label = priority (urgent/high/normal/low)
    Column("priority",      String(20)),
    Column("sender_type",   String(20)),
    Column("intent",        String(20)),
    Column("sentiment",     String(20)),
    Column("summary",       Text),
    # --- Label metadata ---
    Column("confidence",    Float),             # 0.0-1.0, populated by OSS model in phase 2
    Column("label_source",  String(20)),        # "claude" | "rule" | "oss_model"
    Column("reason",        Text),              # why this label was assigned
    Column("follow_up_date",String(50)),        # extracted deadline/follow-up date if any
    Column("model_used",    String(100)),
    Column("classified_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

Index("ix_results_cls_priority",    results_classifications.c.priority)
Index("ix_results_cls_sender_type", results_classifications.c.sender_type)
Index("ix_results_cls_intent",      results_classifications.c.intent)

results_sender_profiles = Table("results_sender_profiles", results_meta,
    Column("id",                  Integer, primary_key=True, autoincrement=True),
    Column("sender_id",           Integer, nullable=False, unique=True),
    Column("email_address",       String(320)),
    Column("domain",              String(200)),
    Column("dominant_sender_type",String(20)),
    Column("total_emails",        Integer, default=0),
    Column("urgent_count",        Integer, default=0),
    Column("high_count",          Integer, default=0),
    Column("avg_sentiment_score", Float),
    Column("sentiment_trend",     String(20)),   # improving/stable/declining
    Column("dominant_intent",     String(20)),
    Column("first_email_at",      DateTime),
    Column("last_email_at",       DateTime),
    Column("updated_at",          DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

results_vendor_metrics = Table("results_vendor_metrics", results_meta,
    Column("id",               Integer, primary_key=True, autoincrement=True),
    Column("vendor_domain",    String(200), nullable=False, unique=True),
    Column("email_count",      Integer, default=0),
    Column("avg_gap_hours",    Float),          # avg time between emails
    Column("min_gap_hours",    Float),
    Column("max_gap_hours",    Float),
    Column("urgent_ratio_pct", Float),
    Column("dominant_intent",  String(20)),
    Column("sla_breach",       Boolean, default=False),  # avg_gap > 48h
    Column("last_email_at",    DateTime),
    Column("updated_at",       DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

results_client_trails = Table("results_client_trails", results_meta,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("sender_id",    Integer, nullable=False),
    Column("email_id",     Integer, nullable=False),
    Column("email_address",String(320)),
    Column("subject",      Text),
    Column("summary",      Text),
    Column("priority",     String(20)),
    Column("intent",       String(20)),
    Column("sentiment",    String(20)),
    Column("timestamp",    DateTime),
    Column("sequence_num", Integer),            # position in trail (1 = oldest)
    UniqueConstraint("sender_id", "email_id", name="uq_trail_entry"),
)

Index("ix_client_trails_sender",    results_client_trails.c.sender_id)
Index("ix_client_trails_timestamp", results_client_trails.c.timestamp)


ALL_METADATA = [raw_meta, cleaned_meta, results_meta]
