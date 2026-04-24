"""
pipeline/stage3_enrich.py
STAGE 3 — Enrich
cleaned.* → results.classifications + results.sender_profiles
                 + results.vendor_metrics + results.client_trails

  Part A: Groq API classification (batched, idempotent)
  Part B: Aggregate analytics written back to results.*

Idempotent: skips emails already in results_classifications.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

from groq import Groq
import groq
from openai import OpenAI
from sqlalchemy import insert, select, update, text, func
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from db.schema import (
    cleaned_emails, cleaned_senders, raw_emails,  # <-- Added raw_emails here
    results_classifications, results_sender_profiles,
    results_vendor_metrics, results_client_trails,
)

logger = logging.getLogger(__name__)
load_dotenv()

BATCH_SIZE  = 2
MAX_RETRIES = 5
MODEL       = "llama-3.1"
# MODEL = "gemini-2.5-flash"

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "urgent": -0.5}

SYSTEM_PROMPT = """You are an expert email classifier for a professional inbox intelligence system.
Classify each email in the batch and return a JSON object containing a single key "results" which holds an array of classification objects (one per email, in the exact same order as the input).

Each object in the array must have exactly these fields:
  priority:       "urgent" | "high" | "normal" | "low"
  sender_type:    "client" | "vendor" | "internal" | "personal" | "spam" | "unknown"
  intent:         "request" | "update" | "complaint" | "inquiry" | "fyi" | "meeting" | "other"
  sentiment:      "positive" | "neutral" | "negative" | "urgent"
  summary:        one sentence, max 20 words
  reason:         brief explanation of why this priority was assigned (max 15 words)
  confidence:     "high" | "medium" | "low"
  follow_up_date: "YYYY-MM-DD" if a deadline or follow-up date is mentioned, else null

Priority guide:
  urgent = action needed within hours (outages, legal, overdue payment, escalations)
  high   = action needed within 1-2 days (client requests, proposals, approvals)
  normal = routine professional communication
  low    = newsletters, notifications, non-actionable FYIs

Output ONLY valid JSON. No markdown formatting, no preamble."""


# ════════════════════════════════════════════════════════════════
#  PART A — Retroactive Spam Correction
# ════════════════════════════════════════════════════════════════

def override_existing_spam(engine: Engine) -> int:
    """
    Overwrites existing classifications for spam by looking back at the 
    raw_emails.extra_fields column (bypassing the need to alter Stage 1 or 2).
    """
    logger.info("[ENRICH] Checking for existing spam classifications to override...")
    with engine.begin() as conn:
        # Join cleaned_emails back to raw_emails to access the extra_fields
        q = select(cleaned_emails.c.id, raw_emails.c.extra_fields).select_from(
            cleaned_emails.join(raw_emails, cleaned_emails.c.raw_id == raw_emails.c.id)
        )
        
        rows = conn.execute(q).fetchall()
        spam_ids = []

        # Extract the ground truth label from the JSON/string
        for c_id, extra_data in rows:
            if not extra_data: 
                continue
                
            try:
                if isinstance(extra_data, str):
                    extra = json.loads(extra_data.replace("'", '"')) 
                else:
                    extra = extra_data
                
                orig_label = str(extra.get("original_label", "")).strip().lower()
                
                if orig_label in ['spam', '1', 'true', 'yes']:
                    spam_ids.append(c_id)
            except Exception:
                pass 

        if not spam_ids:
            logger.info("[ENRICH] No known spam found in raw_emails to override.")
            return 0

        # Update the results table for the found spam IDs
        chunk_size = 500
        total_updated = 0
        
        for i in range(0, len(spam_ids), chunk_size):
            batch_ids = spam_ids[i:i + chunk_size]
            
            stmt = (
                update(results_classifications)
                .where(results_classifications.c.email_id.in_(batch_ids))
                .values(
                    label="spam",
                    priority="low",
                    sender_type="spam",
                    intent="other",
                    sentiment="neutral",
                    summary="Automated spam detection (Retroactive DB correction).",
                    confidence=1.0,
                    label_source="ground_truth_correction"
                )
            )
            result = conn.execute(stmt)
            total_updated += result.rowcount

        if total_updated > 0:
            logger.info(f"[ENRICH] Successfully overwrote {total_updated} existing classifications as spam.")
        return total_updated


# ════════════════════════════════════════════════════════════════
#  PART B — Classification
# ════════════════════════════════════════════════════════════════

def run_classification(engine: Engine, dataset_name: str = None, limit: int = None) -> int:
    """
    Classify unclassified cleaned emails via LLM API.
    Returns number of newly classified emails.
    """
    import os
    client = OpenAI(
        base_url="https://engage-injuries-paxil-mazda.trycloudflare.com/v1",
        api_key="ollama" 
    )
    MODEL = "llama3.1:8b"
    
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

        q = q.order_by(func.random())

        if limit:
            q = q.limit(limit)

        all_rows = conn.execute(q).fetchall()

    pending = [r for r in all_rows if r[0] not in already_done]
    logger.info(f"[ENRICH-A] {len(pending)} emails to classify | {len(already_done)} already done")
    time.sleep(0.1)
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
                conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                conf_str = str(label.get("confidence", "low")).lower()

                try:
                    conn.execute(
                        insert(results_classifications).values(
                            email_id       = row[0],
                            label          = label.get("priority", "normal"),  # primary label
                            priority       = label.get("priority",    "normal"),
                            sender_type    = label.get("sender_type", "unknown"),
                            intent         = label.get("intent",      "other"),
                            sentiment      = label.get("sentiment",   "neutral"),
                            summary        = label.get("summary",     ""),
                            reason         = label.get("reason",      ""),
                            follow_up_date = label.get("follow_up_date", None),
                            confidence     = conf_map.get(conf_str, 0.5),         # populated in OSS phase
                            label_source   ="groq_8b",
                            model_used     = MODEL,
                        )
                    )
                    classified += 1
                except Exception as e:
                    logger.warning(f"Insert failed for email_id={row[0]}: {e}")

        logger.info(f"[ENRICH-A] Classified {min(i + BATCH_SIZE, len(pending))}/{len(pending)}")

    return classified


def _build_prompt(emails: list[dict]) -> str:
    # Pass clean JSON to prevent formatting hallucinations
    return json.dumps(emails, indent=2)

def _classify_with_retry(client, emails: list[dict]) -> list[dict]:
    fallback = {"priority": "normal", "sender_type": "unknown",
                "intent": "other", "sentiment": "neutral", "summary": "Classification unavailable.", "confidence": "low"}
    
    prompt_content = _build_prompt(emails)
    MODEL = "llama3.1:8b"  
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content}
                ],
                response_format={"type": "json_object"},
                max_tokens=4000,
                temperature=0.0 # Strict determinism for labeling
            )
            
            raw = resp.choices[0].message.content

            if not raw or not raw.strip().endswith('}'):
                raise ValueError("LLM returned truncated or incomplete JSON string.")
            
            result_obj = json.loads(raw)
            results_array = result_obj.get("results", [])
            
            if len(results_array) == len(emails):
                return results_array
            else:
                logger.warning(f"Batch size mismatch. Expected {len(emails)}, got {len(results_array)}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error on attempt {attempt}: {e}. The model likely hit a token limit or hallucinated.")
            time.sleep(5) # Give the API a moment before retrying
        except Exception as e:
            logger.error(f"Classification attempt {attempt} failed: {e}")
            time.sleep(0.1)

    return [fallback.copy() for _ in emails]

# ════════════════════════════════════════════════════════════════
#  PART C — Analytics Aggregates
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

            first_at = _parse_single_dt(first_at)
            last_at  = _parse_single_dt(last_at)

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

            last_at = _parse_single_dt(last_at)

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
                "timestamp":   _parse_single_dt(r[8]),  "sequence_num": seq_map[sender_id],
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

def _parse_single_dt(dt_val):
    if not dt_val: 
        return None
    if hasattr(dt_val, "isoformat"): 
        return dt_val
    
    from dateutil import parser as dp
    try:
        return dp.parse(str(dt_val))
    except Exception:
        return None

# ════════════════════════════════════════════════════════════════
#  Combined entry point
# ════════════════════════════════════════════════════════════════

def run(engine: Engine, dataset_name: str = None, limit: int = None, skip_classification: bool = False, skip_aggregates: bool = False) -> dict:
    results = {}
    
    # 1. First, correct any existing data in the database
    results["spam_overridden"] = override_existing_spam(engine)

    # 2. Run new LLM classifications
    if not skip_classification:
        results["classified"] = run_classification(engine, dataset_name=dataset_name, limit=limit)
    
    # 3. Build aggregates using the perfectly clean data
    if not skip_aggregates:
        run_aggregates(engine)
        
    return results