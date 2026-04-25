from sqlalchemy import Column, Integer, String
from api.auth.database import Base
import os
print("ACTUAL DB PATH:", os.path.abspath("users.db"))

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)