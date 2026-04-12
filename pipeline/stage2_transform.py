"""
pipeline/stage2_transform.py
STAGE 2 — Transform
raw.* → cleaned.senders + cleaned.emails + cleaned.recipients + cleaned.email_recipients

Operations:
  - Parse email addresses (extract name + address from "Name <addr>" format)
  - Parse timestamps (handles messy formats via dateutil)
  - Deduplicate emails by SHA-256 of body
  - Normalize subjects (strip Re:/Fwd: prefixes, whitespace)
  - Populate derived columns: body_length, year, month, day_of_week, hour
  - Build M2M email ↔ recipient links

Idempotent: skips raw_ids already in cleaned_emails.
"""

import hashlib
import logging
import re
from email.utils import parseaddr
from typing import Optional

import pandas as pd
from dateutil import parser as dateparser
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Engine

from db.schema import (
    raw_emails,
    cleaned_emails, cleaned_senders, cleaned_recipients, cleaned_email_recipients,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE   = 500
MAX_BODY_LEN = 10_000   # store full body in DB (trim at classification time)


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_email_addr(raw: Optional[str]) -> tuple[str, str]:
    """Returns (display_name, email_address). Handles 'Name <addr>' and plain addr."""
    if not raw or str(raw).lower() in ("nan", "none", ""):
        return ("", "unknown@unknown")
    name, addr = parseaddr(str(raw).strip())
    addr = addr.lower().strip()
    if not addr or "@" not in addr:
        addr = re.sub(r"\s+", "", raw.lower())[:320]
    return (name.strip(), addr)


def _extract_domain(email_addr: str) -> str:
    if "@" in email_addr:
        return email_addr.split("@")[-1].strip()
    return "unknown"


def _parse_timestamp(raw: Optional[str]) -> Optional[object]:
    if not raw or str(raw).lower() in ("nan", "none", ""):
        return None
    try:
        return dateparser.parse(str(raw), fuzzy=True)
    except Exception:
        return None


def _normalize_subject(raw: Optional[str]) -> str:
    if not raw or str(raw).lower() in ("nan", "none", ""):
        return "(No Subject)"
    s = str(raw).strip()
    s = re.sub(r"^(re:|fwd?:|fw:)\s*", "", s, flags=re.IGNORECASE).strip()
    return s[:500] or "(No Subject)"


def _body_hash(body: Optional[str]) -> str:
    text = (body or "").strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_structural(subject: str, body: str, cc: str) -> dict:
    """Compute is_reply, cc_count, reply_chain_depth, email_length."""
    subj = subject or ""
    # Count Re: nestings
    reply_depth = len(re.findall(r"(?i)^(re:|fwd?:)\s*", subj))
    # More thorough: count all Re: occurrences in original subject before normalization
    reply_depth = len(re.findall(r"(?i)re\s*:", subj))
    is_reply    = reply_depth > 0

    cc_list  = [c.strip() for c in re.split(r"[,;]", cc or "") if c.strip()]
    cc_count = len(cc_list)

    return {
        "is_reply":          is_reply,
        "cc_count":          cc_count,
        "reply_chain_depth": reply_depth,
        "email_length":      len(body or ""),
    }


# ── Enron-specific parser ─────────────────────────────────────────────────────

def _parse_enron_message(raw_body: str) -> dict:
    """
    Parse a raw Enron message blob (may contain multiple emails).
    Returns fields for the FIRST email found.
    Splits on Message-ID: boundaries.
    """
    import email as email_lib

    # Split concatenated emails on Message-ID boundaries
    parts = re.split(r'(?=Message-ID:)', raw_body.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return {}

    # Parse the first email block
    raw = parts[0]
    try:
        msg = email_lib.message_from_string(raw)
        sender    = msg.get("From", "") or msg.get("X-From", "")
        recipient = msg.get("To", "")   or msg.get("X-To", "")
        cc        = msg.get("Cc", "")   or msg.get("X-cc", "")
        subject   = msg.get("Subject", "")
        date      = msg.get("Date", "")

        # Extract body — everything after the blank line separating headers
        if "\n\n" in raw:
            body = raw.split("\n\n", 1)[1].strip()
        else:
            body = ""

        # Clean up X- prefixed display names if real address missing
        if not sender and msg.get("X-From"):
            sender = msg.get("X-From", "")

        return {
            "sender":    sender.strip(),
            "recipient": recipient.strip(),
            "cc":        cc.strip(),
            "subject":   subject.strip(),
            "date":      date.strip(),
            "body":      body,
        }
    except Exception:
        return {}


# ── main stage ───────────────────────────────────────────────────────────────

def run(engine: Engine, dataset_name: str = None) -> int:
    """
    Transform raw emails → cleaned tables.

    Args:
        dataset_name: If set, only transform rows from this dataset.
                      If None, processes all unprocessed raw rows.
    Returns:
        Number of new cleaned_emails rows inserted.
    """
    with engine.begin() as conn:
        # Find raw_ids not yet in cleaned_emails
        already_cleaned = set(
            r[0] for r in conn.execute(select(cleaned_emails.c.raw_id)).fetchall()
        )

        q = select(raw_emails)
        if dataset_name:
            q = q.where(raw_emails.c.dataset_name == dataset_name)

        raw_rows = conn.execute(q).fetchall()
        raw_cols = raw_emails.columns.keys()

    pending = [
        dict(zip(raw_cols, row))
        for row in raw_rows
        if dict(zip(raw_cols, row))["id"] not in already_cleaned
    ]

    logger.info(f"[TRANSFORM] {len(pending)} raw rows pending | {len(already_cleaned)} already cleaned")

    if not pending:
        return 0

    inserted = 0
    for i in range(0, len(pending), CHUNK_SIZE):
        chunk = pending[i : i + CHUNK_SIZE]
        inserted += _transform_chunk(engine, chunk)
        logger.info(f"[TRANSFORM] Processed {min(i + CHUNK_SIZE, len(pending))}/{len(pending)}")

    logger.info(f"[TRANSFORM] Done — {inserted} new rows in cleaned_emails")
    return inserted


def _transform_chunk(engine: Engine, rows: list[dict]) -> int:
    inserted = 0

    with engine.begin() as conn:
        # Cache of email_address → cleaned_senders.id (to avoid repeated lookups)
        sender_cache    = {}
        recipient_cache = {}

        for row in rows:
            raw_body = (row.get("body_raw") or "")

            # ── Enron: parse headers out of raw message blob ─────
            enron_parsed = {}
            if row.get("dataset_name") == "enron" and raw_body.startswith("Message-ID:"):
                enron_parsed = _parse_enron_message(raw_body)
                raw_body = enron_parsed.get("body", raw_body)

            body = raw_body[:MAX_BODY_LEN]
            if not body.strip():
                continue   # skip empty bodies

            content_hash = _body_hash(body)

            # Skip duplicates
            exists = conn.execute(
                select(cleaned_emails.c.id).where(cleaned_emails.c.content_hash == content_hash)
            ).fetchone()
            if exists:
                continue

            # ── Sender ──────────────────────────────────────────
            raw_sender = enron_parsed.get("sender") or row.get("sender_raw")
            sender_name, sender_addr = _parse_email_addr(raw_sender)
            sender_id = _upsert_sender(conn, sender_addr, sender_name, sender_cache)

            # ── Timestamp ───────────────────────────────────────
            raw_ts = enron_parsed.get("date") or row.get("timestamp_raw")
            ts = _parse_timestamp(raw_ts)

            # ── Recipients (Enron) ───────────────────────────────
            if enron_parsed.get("recipient") and not row.get("recipients_raw"):
                row = dict(row)
                row["recipients_raw"] = enron_parsed["recipient"]

            # ── Insert cleaned email ─────────────────────────────
            subj = _normalize_subject(enron_parsed.get("subject") or row.get("subject_raw"))

            # ── CC ──────────────────────────────────────────────
            cc_raw = enron_parsed.get("cc") or ""

            # ── Structural fields ────────────────────────────────
            structural = _compute_structural(subj, body, cc_raw)

            result = conn.execute(
                insert(cleaned_emails).values(
                    raw_id             = row["id"],
                    dataset_name       = row["dataset_name"],
                    sender_id          = sender_id,
                    subject            = subj,
                    body               = body,
                    cc                 = cc_raw or None,
                    body_length        = len(body),
                    email_length       = structural["email_length"],
                    is_reply           = structural["is_reply"],
                    cc_count           = structural["cc_count"],
                    reply_chain_depth  = structural["reply_chain_depth"],
                    timestamp          = ts,
                    year               = ts.year        if ts else None,
                    month              = ts.month       if ts else None,
                    day_of_week        = ts.weekday()   if ts else None,
                    hour               = ts.hour        if ts else None,
                    thread_id          = row.get("thread_id_raw"),
                    content_hash       = content_hash,
                )
            )
            email_id = result.inserted_primary_key[0]
            inserted += 1

            # ── Recipients ──────────────────────────────────────
            recip_raw = row.get("recipients_raw") or ""
            for recip_str in _split_recipients(recip_raw):
                r_name, r_addr = _parse_email_addr(recip_str)
                r_id = _upsert_recipient(conn, r_addr, r_name, recipient_cache)
                try:
                    conn.execute(
                        insert(cleaned_email_recipients).values(
                            email_id      = email_id,
                            recipient_id  = r_id,
                            recipient_type= "to",
                        )
                    )
                except Exception:
                    pass  # ignore duplicate link

    return inserted


def _upsert_sender(conn, email_addr: str, display_name: str, cache: dict) -> int:
    if email_addr in cache:
        return cache[email_addr]

    existing = conn.execute(
        select(cleaned_senders.c.id).where(cleaned_senders.c.email_address == email_addr)
    ).fetchone()

    if existing:
        cache[email_addr] = existing[0]
        return existing[0]

    result = conn.execute(
        insert(cleaned_senders).values(
            email_address = email_addr,
            display_name  = display_name or None,
            domain        = _extract_domain(email_addr),
        )
    )
    sid = result.inserted_primary_key[0]
    cache[email_addr] = sid
    return sid


def _upsert_recipient(conn, email_addr: str, display_name: str, cache: dict) -> int:
    if email_addr in cache:
        return cache[email_addr]

    existing = conn.execute(
        select(cleaned_recipients.c.id).where(cleaned_recipients.c.email_address == email_addr)
    ).fetchone()

    if existing:
        cache[email_addr] = existing[0]
        return existing[0]

    result = conn.execute(
        insert(cleaned_recipients).values(
            email_address = email_addr,
            display_name  = display_name or None,
            domain        = _extract_domain(email_addr),
        )
    )
    rid = result.inserted_primary_key[0]
    cache[email_addr] = rid
    return rid


def _split_recipients(raw: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
