from fastapi import FastAPI, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from api.email_service import send_email
from sqlalchemy.orm import Session
import api.auth.models, api.auth.schemas
from api.auth.database import SessionLocal, engine
from api.auth.auth import hash_password, verify_password
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from api.auth.jwt_handler import create_access_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.auth.jwt_handler import decode_token
import os

api.auth.models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Email Dashboard API", version="1.0.0")
oauth = OAuth()

oauth.register(
    name='google',
    client_id="YOUR_GOOGLE_CLIENT_ID",
    client_secret="YOUR_GOOGLE_CLIENT_SECRET",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        'scope': 'openid email profile'
    }
)
oauth.register(
    name='microsoft',
    client_id="YOUR_MS_CLIENT_ID",
    client_secret="YOUR_MS_SECRET",
    authorize_url='https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
    access_token_url='https://login.microsoftonline.com/common/oauth2/v2.0/token',
    client_kwargs={'scope': 'openid email profile'}
)

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload
# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
        allow_origins=[
        "http://localhost:3000",  # React
        "http://localhost:8501",  # Streamlit (keep if you still use it)
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class EmailPayload(BaseModel):
    to: EmailStr
    subject: str
    body: str
    html: bool = False  # set True to send HTML emails


class EmailResponse(BaseModel):
    success: bool
    message: str

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Signup
@app.post("/signup")
def signup(user: api.auth.schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(api.auth.models.User).filter(
        api.auth.models.User.username == user.username
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = api.auth.models.User(
        username=user.username,
        password=hash_password(user.password)
    )

    db.add(new_user)
    db.commit()

    token = create_access_token({"sub": user.username})

    return {
        "message": "User created",
        "access_token": token
    }

# Login

def login(user: api.auth.schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(api.auth.models.User).filter(
        api.auth.models.User.username == user.username
    ).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": db_user.username})

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/login/google")
async def login_google(request):
    redirect_uri = "http://localhost:8000/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google(request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")

    email = user_info["email"]

    user = db.query(api.auth.models.User).filter(
        api.auth.models.User.username == email
    ).first()

    if not user:
        user = api.auth.models.User(username=email, password="")
        db.add(user)
        db.commit()

    jwt_token = create_access_token({"sub": email})

    # redirect back to frontend with token
    return RedirectResponse(
        url=f"http://localhost:3000?token={jwt_token}"
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/emails/ranked")
def get_ranked_emails():
    """
    In production: fetch from Gmail API, run scoring logic, return ranked list.
    For now returns mock data — replace with real logic.
    """
    return [
        {"priority": "critical", "score": 97, "sender": "Sarah Chen",
         "subject": "URGENT: Shipment #4820 customs hold",
         "reasoning": "Penalty clause activates in 6h", "time": "7:02 AM"},
        # ... add more or pull from Gmail API
    ]

@app.get("/vendors/health")
def get_vendor_health():
    return [
        {"name": "Apex Logistics", "health": 32, "sla": "breach", "tickets": 4},
        {"name": "CloudSync Pro",  "health": 48, "sla": "breach", "tickets": 2},
        # ...
    ]

@app.get("/threads")
def get_threads():
    return [
        {"id": 1, "subject": "Shipment #4820 customs hold",
         "summary": "...", "replies": 8},
        # ...
    ]
@app.post("/send-email", response_model=EmailResponse)
def send_email_endpoint(payload: EmailPayload):
    result = send_email(
        to=payload.to,
        subject=payload.subject,
        body=payload.body,
        html=payload.html,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result