from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from passlib.context import CryptContext

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = "secret123"

CORS(app)

# -----------------------
# Flask-Login setup
# -----------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -----------------------
# User Class
# -----------------------
class User(UserMixin):
    def __init__(self, id, full_name, email, password):
        self.id = id
        self.full_name = full_name
        self.email = email
        self.password = password

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------
# DB Helper
# -----------------------
def get_db():
    return sqlite3.connect("users.db")

# -----------------------
# Create table (run once)
# -----------------------
def create_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.close()


create_table()

# -----------------------
# Load user (Flask-Login)
# -----------------------
@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()

    if user:
        return User(user[0], user[1], user[2])
    return None

# -----------------------
# Routes
# -----------------------

@app.route('/')
@login_required
def dashboard():
    return render_template("dashboard.html")

# -----------------------
# Register
# -----------------------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data:
        return {"error": "Invalid JSON"}, 400

    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not full_name or not email or not password:
        return {"error": "Missing fields"}, 400

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (full_name, email, password) VALUES (?, ?, ?)",
            (full_name, email, hashed_password)
        )
        conn.commit()
        conn.close()

        return {"message": "User registered successfully"}, 201

    except:
        return {"error": "User already exists"}, 400
# -----------------------
# Login
# -----------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found"}), 404

    stored_password = user[3]  # id, full_name, email, password

    if not check_password_hash(stored_password, password):
        return jsonify({"error": "Invalid password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "id": user[0],
            "full_name": user[1],
            "email": user[2]
        }
    })
# -----------------------
# Logout
# -----------------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)