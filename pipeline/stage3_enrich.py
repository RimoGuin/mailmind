"""
pipeline/stage3_enrich.py
STAGE 3 — Enrich
cleaned.* → results.classifications + results.sender_profiles
                 + results.vendor_metrics + results.client_trails

  Part A: Claude API classification (batched, idempotent)
  Part B: Aggregate analytics written back to results.*

Idempotent: skips emails already in results_classifications.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

import anthropic
from sqlalchemy import insert, select, update, text, func
from sqlalchemy.engine import Engine

from db.schema import (
    cleaned_emails, cleaned_senders,
    results_classifications, results_sender_profiles,
    results_vendor_metrics, results_client_trails,
)

logger = logging.getLogger(__name__)

BATCH_SIZE  = 5
MAX_RETRIES = 3
MODEL       = "claude-sonnet-4-20250514"

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "urgent": -0.5}

SYSTEM_PROMPT = """You are an expert email classifier for a professional inbox intelligence system.
Classify each email in the batch and return a JSON array — one object per email, same order as input.

Each object must have exactly these fields:
  priority:       "urgent" | "high" | "normal" | "low"
  sender_type:    "client" | "vendor" | "internal" | "personal" | "spam" | "unknown"
  intent:         "request" | "update" | "complaint" | "inquiry" | "fyi" | "meeting" | "other"
  sentiment:      "positive" | "neutral" | "negative" | "urgent"
  summary:        one sentence, max 20 words
  reason:         brief explanation of why this priority was assigned (max 15 words)
  follow_up_date: "YYYY-MM-DD" if a deadline or follow-up date is mentioned, else null

Priority guide:
  urgent = action needed within hours (outages, legal, overdue payment, escalations)
  high   = action needed within 1-2 days (client requests, proposals, approvals)
  normal = routine professional communication
  low    = newsletters, notifications, non-actionable FYIs

Return ONLY valid JSON array. No markdown, no preamble."""


# ════════════════════════════════════════════════════════════════
#  PART A — Classification
# ════════════════════════════════════════════════════════════════

def run_classification(engine: Engine, dataset_name: str = None, limit: int = None) -> int:
    """
    Classify unclassified cleaned emails via Claude API.
    Returns number of newly classified emails.
    """
    client = anthropic.Anthropic()

    with engine.connect() as conn:
        already_done = set(
            r[0] for r in conn.execute(select(results_classifications.c.email_id)).fetchall()
        )

        q = select(
            cleaned_emails.c.id,
            cleaned_emails.c.subject,
            cleaned_emails.c.body,
            cleaned_senders.c.email_address.label("sender"),
        ).join(cleaned_senders, cleaned_emails.c.sender_id == cleaned_senders.c.id, isouter=True)

        if dataset_name:
            q = q.where(cleaned_emails.c.dataset_name == dataset_name)

        if limit:
            q = q.limit(limit)

        all_rows = conn.execute(q).fetchall()

    pending = [r for r in all_rows if r[0] not in already_done]
    logger.info(f"[ENRICH-A] {len(pending)} emails to classify | {len(already_done)} already done")

    if not pending:
        return 0

    classified = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        batch_dicts = [
            {"sender": r[3] or "unknown", "subject": r[1] or "", "body": (r[2] or "")[:1500]}
            for r in batch
        ]
        labels = _classify_with_retry(client, batch_dicts)

        with engine.begin() as conn:
            for row, label in zip(batch, labels):
                try:
                    conn.execute(
                        insert(results_classifications).values(
                            email_id     = row[0],
                            label        = label.get("priority", "normal"),  # primary label
                            priority     = label.get("priority",    "normal"),
                            sender_type  = label.get("sender_type", "unknown"),
                            intent       = label.get("intent",      "other"),
                            sentiment    = label.get("sentiment",   "neutral"),
                            summary      = label.get("summary",     ""),
                            reason       = label.get("reason",      ""),
                            follow_up_date = label.get("follow_up_date", None),
                            confidence   = None,          # populated in OSS phase
                            label_source = "claude",
                            model_used   = MODEL,
                        )
                    )
                    classified += 1
                except Exception as e:
                    logger.warning(f"Insert failed for email_id={row[0]}: {e}")

        logger.info(f"[ENRICH-A] Classified {min(i + BATCH_SIZE, len(pending))}/{len(pending)}")

    return classified


def _build_prompt(emails: list[dict]) -> str:
    lines = []
    for idx, e in enumerate(emails):
        lines += [f"--- EMAIL {idx+1} ---",
                  f"From: {e['sender']}",
                  f"Subject: {e['subject']}",
                  f"Body:\n{e['body']}", ""]
    return "\n".join(lines)


def _classify_with_retry(client, emails: list[dict]) -> list[dict]:
    fallback = {"priority": "normal", "sender_type": "unknown",
                "intent": "other", "sentiment": "neutral", "summary": "Classification unavailable."}
    prompt = _build_prompt(emails)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(clean)
            if isinstance(result, list) and len(result) == len(emails):
                return result
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Classification attempt {attempt} failed: {e}")
            time.sleep(2)

    return [fallback.copy() for _ in emails]


# ════════════════════════════════════════════════════════════════
#  PART B — Analytics Aggregates
# ════════════════════════════════════════════════════════════════

def run_aggregates(engine: Engine) -> None:
    """Rebuild sender profiles, vendor metrics, and client trails from results_classifications."""
    logger.info("[ENRICH-B] Building analytics aggregates...")
    _build_sender_profiles(engine)
    _build_vendor_metrics(engine)
    _build_client_trails(engine)
    logger.info("[ENRICH-B] Aggregates complete")


def _build_sender_profiles(engine: Engine):
    query = text("""
        SELECT
            cs.id                         AS sender_id,
            cs.email_address,
            cs.domain,
            COUNT(ce.id)                  AS total_emails,
            SUM(CASE WHEN rc.priority = 'urgent' THEN 1 ELSE 0 END) AS urgent_count,
            SUM(CASE WHEN rc.priority = 'high'   THEN 1 ELSE 0 END) AS high_count,
            GROUP_CONCAT(rc.sender_type)  AS sender_types,
            GROUP_CONCAT(rc.intent)       AS intents,
            GROUP_CONCAT(rc.sentiment)    AS sentiments,
            MIN(ce.timestamp)             AS first_email_at,
            MAX(ce.timestamp)             AS last_email_at
        FROM cleaned_senders cs
        JOIN cleaned_emails ce ON ce.sender_id = cs.id
        JOIN results_classifications rc ON rc.email_id = ce.id
        GROUP BY cs.id
    """)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM results_sender_profiles"))
        rows = conn.execute(query).fetchall()

        for r in rows:
            (sender_id, email_addr, domain, total, urgent, high,
             sender_types_str, intents_str, sentiments_str, first_at, last_at) = r

            dominant_type   = _mode_from_csv(sender_types_str)
            dominant_intent = _mode_from_csv(intents_str)
            sentiment_list  = [SENTIMENT_SCORE.get(s, 0) for s in (sentiments_str or "").split(",") if s]
            avg_sentiment   = sum(sentiment_list) / len(sentiment_list) if sentiment_list else 0

            # Sentiment trend: compare first half vs second half
            mid = len(sentiment_list) // 2
            if mid > 0:
                early = sum(sentiment_list[:mid]) / mid
                late  = sum(sentiment_list[mid:]) / len(sentiment_list[mid:])
                delta = late - early
                trend = "improving" if delta > 0.2 else "declining" if delta < -0.2 else "stable"
            else:
                trend = "stable"

            conn.execute(insert(results_sender_profiles).values(
                sender_id            = sender_id,
                email_address        = email_addr,
                domain               = domain,
                dominant_sender_type = dominant_type,
                total_emails         = total,
                urgent_count         = urgent,
                high_count           = high,
                avg_sentiment_score  = round(avg_sentiment, 3),
                sentiment_trend      = trend,
                dominant_intent      = dominant_intent,
                first_email_at       = first_at,
                last_email_at        = last_at,
            ))


def _build_vendor_metrics(engine: Engine):
    query = text("""
        SELECT
            cs.domain,
            COUNT(ce.id)  AS email_count,
            GROUP_CONCAT(ce.timestamp ORDER BY ce.timestamp) AS timestamps,
            AVG(CASE WHEN rc.priority = 'urgent' THEN 1.0 ELSE 0.0 END) * 100 AS urgent_pct,
            GROUP_CONCAT(rc.intent) AS intents,
            MAX(ce.timestamp) AS last_email_at
        FROM cleaned_senders cs
        JOIN cleaned_emails ce ON ce.sender_id = cs.id
        JOIN results_classifications rc ON rc.email_id = ce.id
        WHERE rc.sender_type = 'vendor'
        GROUP BY cs.domain
    """)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM results_vendor_metrics"))
        rows = conn.execute(query).fetchall()

        for r in rows:
            domain, count, timestamps_str, urgent_pct, intents_str, last_at = r

            # Calculate gaps between consecutive emails
            ts_list = _parse_timestamps(timestamps_str)
            gaps = []
            for i in range(1, len(ts_list)):
                diff_h = (ts_list[i] - ts_list[i-1]).total_seconds() / 3600
                if diff_h >= 0:
                    gaps.append(diff_h)

            avg_gap = round(sum(gaps) / len(gaps), 1) if gaps else None
            min_gap = round(min(gaps), 1) if gaps else None
            max_gap = round(max(gaps), 1) if gaps else None

            conn.execute(insert(results_vendor_metrics).values(
                vendor_domain    = domain,
                email_count      = count,
                avg_gap_hours    = avg_gap,
                min_gap_hours    = min_gap,
                max_gap_hours    = max_gap,
                urgent_ratio_pct = round(urgent_pct or 0, 1),
                dominant_intent  = _mode_from_csv(intents_str),
                sla_breach       = avg_gap is not None and avg_gap > 48,
                last_email_at    = last_at,
            ))


def _build_client_trails(engine: Engine):
    query = text("""
        SELECT
            cs.id          AS sender_id,
            ce.id          AS email_id,
            cs.email_address,
            ce.subject,
            rc.summary,
            rc.priority,
            rc.intent,
            rc.sentiment,
            ce.timestamp
        FROM cleaned_senders cs
        JOIN cleaned_emails ce ON ce.sender_id = cs.id
        JOIN results_classifications rc ON rc.email_id = ce.id
        WHERE rc.sender_type IN ('client', 'vendor')
        ORDER BY cs.id, ce.timestamp ASC
    """)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM results_client_trails"))
        rows = conn.execute(query).fetchall()

        seq_map = {}
        trail_rows = []
        for r in rows:
            sender_id = r[0]
            seq_map[sender_id] = seq_map.get(sender_id, 0) + 1
            trail_rows.append({
                "sender_id":   r[0], "email_id":   r[1],
                "email_address": r[2], "subject":  r[3],
                "summary":     r[4],  "priority":  r[5],
                "intent":      r[6],  "sentiment": r[7],
                "timestamp":   r[8],  "sequence_num": seq_map[sender_id],
            })

        if trail_rows:
            conn.execute(insert(results_client_trails), trail_rows)


# ── helpers ──────────────────────────────────────────────────────

def _mode_from_csv(csv_str: Optional[str]) -> Optional[str]:
    if not csv_str:
        return None
    items = [s.strip() for s in csv_str.split(",") if s.strip()]
    return max(set(items), key=items.count) if items else None


def _parse_timestamps(csv_str: Optional[str]) -> list:
    from dateutil import parser as dp
    if not csv_str:
        return []
    result = []
    for s in csv_str.split(","):
        try:
            result.append(dp.parse(s.strip()))
        except Exception:
            pass
    return sorted(result)


# ════════════════════════════════════════════════════════════════
#  Combined entry point
# ════════════════════════════════════════════════════════════════

def run(engine: Engine, dataset_name: str = None, limit: int = None, skip_classification: bool = False) -> dict:
    results = {}
    if not skip_classification:
        results["classified"] = run_classification(engine, dataset_name=dataset_name, limit=limit)
    run_aggregates(engine)
    return results
