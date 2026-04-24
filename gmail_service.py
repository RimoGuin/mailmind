from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_service():
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    return build('gmail', 'v1', credentials=creds)


def extract_body(payload):
    """Recursively extract email body (text/plain or fallback to html)."""

    if 'parts' in payload:
        for part in payload['parts']:
            mime = part.get('mimeType')

            if mime == 'text/plain':
                data = part['body'].get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

            # fallback to html
            if mime == 'text/html':
                data = part['body'].get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

            # recursive (nested parts)
            result = extract_body(part)
            if result:
                return result

    else:
        data = payload.get('body', {}).get('data')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    return ""


def get_messages_by_category(category_label, max_results=5):
    service = get_service()

    results = service.users().messages().list(
        userId='me',
        labelIds=[category_label],
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='full'
        ).execute()

        headers = msg_data['payload'].get('headers', [])

        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')

        body = extract_body(msg_data['payload'])

        emails.append({
            "subject": subject,
            "sender": sender,
            "body": body[:200],
            "snippet": msg_data.get("snippet", "")
        })

    return emails


if __name__ == "__main__":
    categories = {
        "Primary": "CATEGORY_PERSONAL",
        "Promotions": "CATEGORY_PROMOTIONS",
        "Updates": "CATEGORY_UPDATES",
        "Social": "CATEGORY_SOCIAL"
    }

    for name, label in categories.items():
        print(f"\n===== {name} =====")
        emails = get_messages_by_category(label, max_results=5)

        for e in emails:
            print(e["subject"], "-", e["sender"])