"""
pipeline/adapters.py
One adapter per Kaggle dataset format.
Each adapter maps raw CSV columns → standard dict for raw_emails insert.

Adding a new dataset = add one class here. Nothing else changes.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Iterator


class BaseAdapter(ABC):
    """Yields normalized row dicts ready for raw_emails insert."""

    dataset_name: str = "unknown"

    @abstractmethod
    def iter_rows(self, df: pd.DataFrame) -> Iterator[dict]:
        pass

    def _base_row(self, dataset_row: int, extra: dict = None) -> dict:
        return {
            "dataset_name":    self.dataset_name,
            "dataset_row":     dataset_row,
            "sender_raw":      None,
            "recipients_raw":  None,
            "subject_raw":     None,
            "body_raw":        None,
            "timestamp_raw":   None,
            "thread_id_raw":   None,
            "extra_fields":    extra or {},
        }


# ─────────────────────────────────────────────
#  Enron Email Dataset
#  Columns: Message-ID, Date, From, To, Subject, X-cc, X-bcc, body (or message)
# ─────────────────────────────────────────────
class EnronAdapter(BaseAdapter):
    dataset_name = "enron"

    def iter_rows(self, df: pd.DataFrame) -> Iterator[dict]:
        df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

        body_col    = next((c for c in df.columns if c in ("body", "message", "content", "email")), None)
        subject_col = next((c for c in df.columns if c in ("subject",)), None)
        from_col    = next((c for c in df.columns if c in ("from",)), None)
        to_col      = next((c for c in df.columns if c in ("to",)), None)
        date_col    = next((c for c in df.columns if c in ("date",)), None)
        mid_col     = next((c for c in df.columns if "message_id" in c or "id" == c), None)

        for i, row in df.iterrows():
            r = self._base_row(dataset_row=int(i))
            r["sender_raw"]     = str(row[from_col]).strip()    if from_col    else None
            r["recipients_raw"] = str(row[to_col]).strip()      if to_col      else None
            r["subject_raw"]    = str(row[subject_col]).strip() if subject_col else None
            r["body_raw"]       = str(row[body_col]).strip()    if body_col    else None
            r["timestamp_raw"]  = str(row[date_col]).strip()    if date_col    else None
            r["thread_id_raw"]  = str(row[mid_col]).strip()     if mid_col     else None
            yield r


# ─────────────────────────────────────────────
#  Spam/Ham Dataset (various Kaggle versions)
#  Columns: label/category + text/message/body
# ─────────────────────────────────────────────
class SpamHamAdapter(BaseAdapter):
    dataset_name = "spamham"

    def iter_rows(self, df: pd.DataFrame) -> Iterator[dict]:
        df.columns = [c.strip().lower() for c in df.columns]

        body_col  = next((c for c in df.columns if c in ("text", "message", "body", "email", "content")), None)
        label_col = next((c for c in df.columns if c in ("label", "category", "class", "spam", "type")), None)

        for i, row in df.iterrows():
            r = self._base_row(dataset_row=int(i))
            r["body_raw"]    = str(row[body_col]).strip() if body_col else None
            r["extra_fields"]["original_label"] = str(row[label_col]) if label_col else None
            yield r


# ─────────────────────────────────────────────
#  Email Classification Dataset
#  Common format: subject, body/message, from, to, date, category
# ─────────────────────────────────────────────
class EmailClassificationAdapter(BaseAdapter):
    dataset_name = "email_classification"

    # Column alias maps for this format
    _SENDER    = ("from", "sender", "from_email", "sender_email", "email_from")
    _RECIPIENT = ("to", "recipient", "to_email", "email_to")    
    _SUBJECT   = ("subject", "title")
    _BODY      = ("body", "message", "content", "text", "email")
    _DATE      = ("date", "datetime", "timestamp", "sent_at", "time")
    _LABEL     = ("label", "category", "class", "type")

    def _find(self, cols, aliases):
        return next((c for c in aliases if c in cols), None)

    def iter_rows(self, df: pd.DataFrame) -> Iterator[dict]:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        cols = list(df.columns)

        sender_col = self._find(cols, self._SENDER)
        recip_col  = self._find(cols, self._RECIPIENT)
        subj_col   = self._find(cols, self._SUBJECT)
        body_col   = self._find(cols, self._BODY)
        date_col   = self._find(cols, self._DATE)
        label_col  = self._find(cols, self._LABEL)

        # Columns not mapped above go into extra_fields
        mapped = {c for c in [sender_col, recip_col, subj_col, body_col, date_col, label_col] if c}

        for i, row in df.iterrows():
            r = self._base_row(dataset_row=int(i))
            r["sender_raw"]     = str(row[sender_col]).strip() if sender_col else None
            r["recipients_raw"] = str(row[recip_col]).strip()  if recip_col  else None
            r["subject_raw"]    = str(row[subj_col]).strip()   if subj_col   else None
            r["body_raw"]       = str(row[body_col]).strip()   if body_col   else None
            r["timestamp_raw"]  = str(row[date_col]).strip()   if date_col   else None
            r["extra_fields"]   = {
                c: str(row[c]) for c in cols if c not in mapped and pd.notna(row[c])
            }
            if label_col:
                r["extra_fields"]["original_label"] = str(row[label_col])
            yield r


# ─────────────────────────────────────────────
#  Auto-detect adapter from column names
# ─────────────────────────────────────────────

def detect_adapter(df: pd.DataFrame) -> BaseAdapter:
    cols = set(c.strip().lower() for c in df.columns)

    # Enron: has 'from' OR 'message-id' style columns + message/body
    if ("from" in cols or "message_id" in cols) and any(c in cols for c in ("message", "body", "email")):
        return EnronAdapter()

    # SpamHam: typically just 2 columns — label + text/message
    if len(cols) <= 3 and any(c in cols for c in ("text", "message")) and any(c in cols for c in ("label", "spam", "category")):
        return SpamHamAdapter()

    # Default: generic email classification adapter
    return EmailClassificationAdapter()
