"""
Security utilities - password hashing, token generation, input sanitization.
"""

import re
import secrets
from typing import Optional


# === Password Hashing ===

def hash_password(password: str) -> str:
    """Hash a password using Argon2id."""
    from argon2 import PasswordHasher
    return PasswordHasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash (timing-safe)."""
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    try:
        return PasswordHasher().verify(password_hash, password)
    except VerifyMismatchError:
        return False


def validate_password(password: str) -> Optional[str]:
    """Validate password strength. Returns error message or None if valid."""
    if len(password) < 8:
        return "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return "Password must contain at least one digit"
    return None


# === Token Generation ===

def generate_token(nbytes: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(nbytes)


def generate_session_id() -> str:
    """Generate a session ID."""
    return generate_token(32)


def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return generate_token(32)


# === Input Sanitization ===

# Unicode control characters (excluding normal whitespace)
_CONTROL_CHARS = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
)


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input - strip control chars, enforce length."""
    if not text:
        return ""
    # Strip null bytes and control characters
    text = _CONTROL_CHARS.sub('', text)
    # Enforce max length
    return text[:max_length]


def sanitize_email(email: str) -> Optional[str]:
    """Validate and normalize an email address. Returns normalized email or None."""
    try:
        from email_validator import validate_email
        result = validate_email(email, check_deliverability=False)
        return result.normalized
    except Exception:
        return None
