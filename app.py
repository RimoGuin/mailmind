from flask import Flask, render_template, request, redirect, url_for
import sqlite3

from werkzeug.security import generate_password_hash, check_password_hash

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = "secret123"

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
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

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
            username TEXT UNIQUE,
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
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db()
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                         (username, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except:
            return "User already exists"

    return render_template("register.html")

# -----------------------
# Login
# -----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1], user[2])
            login_user(user_obj)
            return redirect(url_for('dashboard'))

        return "Invalid credentials"

    return render_template("login.html")

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