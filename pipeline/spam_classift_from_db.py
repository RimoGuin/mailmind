import json
from sqlalchemy import update, select
from db.schema import cleaned_emails, raw_emails, results_classifications

def override_existing_spam(engine: Engine) -> int:
    """
    Overwrites existing classifications for spam by looking back at the 
    raw_emails.extra_fields column (bypassing the need to alter Stage 1 or 2).
    """
    with engine.begin() as conn:
        # 1. Join cleaned_emails back to raw_emails to access the extra_fields
        q = select(cleaned_emails.c.id, raw_emails.c.extra_fields).select_from(
            cleaned_emails.join(raw_emails, cleaned_emails.c.raw_id == raw_emails.c.id)
        )
        
        rows = conn.execute(q).fetchall()
        spam_ids = []

        # 2. Extract the ground truth label from the JSON/string
        for c_id, extra_data in rows:
            if not extra_data: 
                continue
                
            try:
                # Handle both SQLAlchemy JSON columns and raw stringified dicts
                if isinstance(extra_data, str):
                    # Replace single quotes with double quotes for valid JSON parsing
                    extra = json.loads(extra_data.replace("'", '"')) 
                else:
                    extra = extra_data
                
                orig_label = str(extra.get("original_label", "")).strip().lower()
                
                # Check if the label indicates spam
                if orig_label in ['spam', '1', 'true', 'yes']:
                    spam_ids.append(c_id)
            except Exception:
                pass # Skip if extra_fields is malformed

        if not spam_ids:
            logger.info("[ENRICH] No known spam found in raw_emails to override.")
            return 0

        # 3. Update the results table for the found spam IDs
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

        logger.info(f"[ENRICH] Successfully overwrote {total_updated} existing classifications as spam.")
        return total_updated