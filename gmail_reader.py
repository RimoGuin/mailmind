# gmail_reader.py
import os, pickle, base64, email
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)


def get_email_body(payload):
    """Recursively extract plain text body from email payload."""
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            body += get_email_body(part)
    else:
        data = payload.get('body', {}).get('data', '')
        if data:
            decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            soup = BeautifulSoup(decoded, 'html.parser')
            body = soup.get_text()
    return body


def fetch_emails(service, max_results=20, query=''):
    """Fetch emails and return structured data."""
    results = service.users().messages().list(
        userId='me',
        maxResults=max_results,
        q=query  # e.g., 'is:unread', 'from:example@gmail.com'
    ).execute()

    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me', id=msg['id'], format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg_data['payload']['headers']}

        email_record = {
            'message_id': msg['id'],
            'subject':    headers.get('Subject', ''),
            'sender':     headers.get('From', ''),
            'recipient':  headers.get('To', ''),
            'date':       headers.get('Date', ''),
            'snippet':    msg_data.get('snippet', ''),
            'body':       get_email_body(msg_data['payload']),
            'labels':     ', '.join(msg_data.get('labelIds', [])),
        }
        emails.append(email_record)

    return emails