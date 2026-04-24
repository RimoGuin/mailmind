from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from api.email_service import send_email

app = FastAPI(title="Email Dashboard API", version="1.0.0")

# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
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