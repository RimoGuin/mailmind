import re


def is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return bool(re.match(pattern, email.strip()))


def is_valid_subject(subject: str) -> bool:
    """Subject must be non-empty and under 200 chars."""
    return 0 < len(subject.strip()) <= 200


def is_valid_body(body: str) -> bool:
    """Body must be non-empty and under 10,000 chars."""
    return 0 < len(body.strip()) <= 10000


def validate_form(to: str, subject: str, body: str) -> list[str]:
    """
    Run all validations and return a list of error messages.
    Empty list means everything is valid.
    """
    errors = []
    if not to.strip():
        errors.append("Recipient email is required.")
    elif not is_valid_email(to):
        errors.append(f"'{to}' is not a valid email address.")

    if not subject.strip():
        errors.append("Subject is required.")
    elif not is_valid_subject(subject):
        errors.append("Subject must be between 1 and 200 characters.")

    if not body.strip():
        errors.append("Message body is required.")
    elif not is_valid_body(body):
        errors.append("Message body must be under 10,000 characters.")

    return errors


def truncate_preview(text: str, max_len: int = 80) -> str:
    """Return a short preview of a long string."""
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."