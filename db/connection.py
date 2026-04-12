"""
db/connection.py
Single engine + connection factory for the SQLite database.
Handles SQLite-specific pragmas (WAL mode, foreign keys).
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from db.schema import ALL_METADATA
import logging

logger = logging.getLogger(__name__)

_engine = None


def get_engine(db_path: str = "email_intelligence.db") -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,                  # set True to log all SQL
        connect_args={"check_same_thread": False},
    )

    # SQLite performance + integrity pragmas
    @event.listens_for(_engine, "connect")
    def set_pragmas(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")       # safe concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")      # 64MB cache

    logger.info(f"SQLite engine ready: {db_path}")
    return _engine


def init_db(db_path: str = "email_intelligence.db"):
    """Create all tables if they don't exist."""
    engine = get_engine(db_path)
    for meta in ALL_METADATA:
        meta.create_all(engine)
    logger.info("All schema layers initialized (raw / cleaned / results)")
    return engine
