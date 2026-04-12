"""
pipeline/stage1_load.py
STAGE 1 — Load
CSV → raw.raw_emails + raw.raw_senders + raw.raw_recipients

Idempotent: skips rows already loaded (by dataset_name + dataset_row).
"""

import logging
import pandas as pd
from pathlib import Path
from sqlalchemy.engine import Engine
from sqlalchemy import insert, select, text

from db.schema import raw_emails, raw_senders, raw_recipients
from pipeline.adapters import detect_adapter, BaseAdapter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500   # rows per DB insert batch


def run(engine: Engine, csv_path: str, dataset_name: str = None, adapter: BaseAdapter = None) -> int:
    """
    Load a CSV into raw.* tables.

    Args:
        engine:       SQLAlchemy engine
        csv_path:     Path to CSV file
        dataset_name: Override dataset name (else inferred from filename)
        adapter:      Override adapter (else auto-detected)

    Returns:
        Number of new rows inserted into raw_emails
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    logger.info(f"[LOAD] Read {len(df)} rows from {path.name} | columns: {list(df.columns)}")

    if adapter is None:
        adapter = detect_adapter(df)
        logger.info(f"[LOAD] Auto-detected adapter: {adapter.__class__.__name__}")

    if dataset_name:
        adapter.dataset_name = dataset_name

    inserted = _load_raw(engine, df, adapter)
    logger.info(f"[LOAD] Done — {inserted} new rows inserted into raw_emails")
    return inserted


def _load_raw(engine: Engine, df: pd.DataFrame, adapter: BaseAdapter) -> int:
    inserted_count = 0

    with engine.begin() as conn:
        # Get already-loaded row indices for this dataset (for idempotency)
        existing = set(
            row[0] for row in conn.execute(
                select(raw_emails.c.dataset_row).where(
                    raw_emails.c.dataset_name == adapter.dataset_name
                )
            ).fetchall()
        )
        logger.info(f"[LOAD] {len(existing)} rows already loaded for '{adapter.dataset_name}'")

        batch_emails    = []
        batch_senders   = []
        batch_recipients = []

        for row_dict in adapter.iter_rows(df):
            if row_dict["dataset_row"] in existing:
                continue

            batch_emails.append(row_dict)

            # Collect unique senders for raw_senders table
            if row_dict.get("sender_raw"):
                batch_senders.append({
                    "sender_str":   row_dict["sender_raw"],
                    "dataset_name": adapter.dataset_name,
                })

            if len(batch_emails) >= CHUNK_SIZE:
                inserted_count += _flush(conn, batch_emails, batch_senders, batch_recipients, adapter.dataset_name)
                batch_emails, batch_senders, batch_recipients = [], [], []

        # Final flush
        if batch_emails:
            inserted_count += _flush(conn, batch_emails, batch_senders, batch_recipients, adapter.dataset_name)

    return inserted_count


def _flush(conn, emails: list, senders: list, recipients: list, dataset_name: str) -> int:
    if not emails:
        return 0

    # Insert emails
    result = conn.execute(insert(raw_emails), emails)
    n_inserted = result.rowcount

    # Get the inserted IDs to build recipient links
    inserted_ids = conn.execute(
        select(raw_emails.c.id, raw_emails.c.dataset_row)
        .where(raw_emails.c.dataset_name == dataset_name)
        .where(raw_emails.c.dataset_row.in_([e["dataset_row"] for e in emails]))
    ).fetchall()
    id_map = {row[1]: row[0] for row in inserted_ids}

    # Insert unique senders (ignore conflicts)
    if senders:
        conn.execute(
            text("""
                INSERT OR IGNORE INTO raw_senders (sender_str, dataset_name)
                VALUES (:sender_str, :dataset_name)
            """),
            senders
        )

    # Build and insert recipient links
    recip_rows = []
    for email_dict in emails:
        email_id = id_map.get(email_dict["dataset_row"])
        if not email_id or not email_dict.get("recipients_raw"):
            continue
        for recip_str in _split_recipients(email_dict["recipients_raw"]):
            recip_rows.append({
                "raw_email_id":  email_id,
                "recipient_str": recip_str,
                "recipient_type": "to",
            })

    if recip_rows:
        conn.execute(insert(raw_recipients), recip_rows)

    return n_inserted


def _split_recipients(raw: str) -> list[str]:
    """Split comma/semicolon-separated recipient strings."""
    if not raw or raw.lower() in ("nan", "none", ""):
        return []
    import re
    parts = re.split(r"[,;]", raw)
    return [p.strip() for p in parts if p.strip()]
