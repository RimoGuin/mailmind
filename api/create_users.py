import sqlite3
from werkzeug.security import generate_password_hash

users = [
    ("admin", "1234"),
    ("anwesha", "pass123"),
    ("testuser", "test123"),
]

conn = sqlite3.connect("users.db")

for username, password in users:
    try:
        hashed = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed)
        )
        print(f"User {username} created")
    except sqlite3.IntegrityError:
        print(f"User {username} already exists")

conn.commit()
conn.close()