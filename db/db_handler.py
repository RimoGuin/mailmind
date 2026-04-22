# db_handler.py
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import re

Base = declarative_base()

class Email(Base):
    __tablename__ = 'emails'

    message_id = Column(String(255), primary_key=True)
    subject    = Column(String(500))
    sender     = Column(String(255))
    recipient  = Column(String(255))
    date       = Column(String(100))
    snippet    = Column(Text)
    body       = Column(Text)
    labels     = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


def get_session(db_url='sqlite:///emails.db'):
    """
    Examples:
      SQLite:     'sqlite:///emails.db'
      PostgreSQL: 'postgresql://user:password@localhost/mydb'
      MySQL:      'mysql+pymysql://user:password@localhost/mydb'
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def save_emails(session, emails):
    saved, skipped = 0, 0
    for data in emails:
        # Skip duplicates
        exists = session.query(Email).filter_by(message_id=data['message_id']).first()
        if exists:
            skipped += 1
            continue

        record = Email(**data)
        session.add(record)
        saved += 1

    session.commit()
    print(f"✅ Saved: {saved} | Skipped (duplicates): {skipped}")