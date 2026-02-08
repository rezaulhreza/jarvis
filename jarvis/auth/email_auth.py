"""
Email/password authentication - registration, login, verification, password reset.
"""

import os
from typing import Dict, Optional, Tuple

from . import db
from .security import hash_password, verify_password, validate_password, sanitize_email
from .email_service import send_verification_email, send_password_reset_email


def _requires_verification() -> bool:
    """Check if email verification is required (default: True)."""
    return os.environ.get("JARVIS_EMAIL_VERIFICATION", "true").lower() in ("1", "true", "yes")


async def register(email: str, password: str, name: str = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Register a new user with email/password.

    Returns:
        (user_dict, None) on success
        (None, error_message) on failure
    """
    # Validate email
    normalized_email = sanitize_email(email)
    if not normalized_email:
        return None, "Invalid email address"

    # Check if already exists
    existing = db.get_user_by_email(normalized_email)
    if existing:
        return None, "An account with this email already exists"

    # Validate password
    password_error = validate_password(password)
    if password_error:
        return None, password_error

    # Create user
    pw_hash = hash_password(password)
    skip_verification = not _requires_verification()
    user = db.create_user(
        email=normalized_email,
        password_hash=pw_hash,
        name=name,
        auth_provider='email',
        email_verified=skip_verification,
    )

    # Generate and send verification email (unless verification is disabled)
    if not skip_verification:
        token = db.create_verification_token(user["id"], 'email_verify', hours=24)
        try:
            await send_verification_email(normalized_email, token)
        except Exception as e:
            print(f"[Auth] Failed to send verification email: {e}")
            # User is created but email not sent - they can resend later

    return user, None


async def login(email: str, password: str, ip: str = None, user_agent: str = None) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    """
    Login with email/password.

    Returns:
        (session_id, user_dict, None) on success
        (None, None, error_message) on failure
    """
    normalized_email = sanitize_email(email)
    if not normalized_email:
        return None, None, "Invalid email address"

    user = db.get_user_by_email(normalized_email)
    if not user:
        return None, None, "Invalid email or password"

    # Check account lock
    if db.is_account_locked(user):
        return None, None, "Account temporarily locked due to too many failed attempts. Try again in 15 minutes."

    # Check email verified (skip if verification is disabled)
    if _requires_verification() and not user.get("email_verified"):
        return None, None, "Please verify your email before logging in"

    # Check password
    if not user.get("password_hash"):
        return None, None, "This account uses social login. Try Google or GitHub."

    if not verify_password(password, user["password_hash"]):
        db.increment_failed_login(user["id"])
        return None, None, "Invalid email or password"

    # Check active
    if not user.get("is_active"):
        return None, None, "Account is deactivated"

    # Success - create session
    db.update_user_login(user["id"])
    session_id = db.create_session(user["id"], ip_address=ip, user_agent=user_agent)

    safe_user = {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name"),
        "avatar_url": user.get("avatar_url"),
        "auth_provider": user.get("auth_provider"),
    }

    return session_id, safe_user, None


def verify_email(token: str) -> Tuple[bool, Optional[str]]:
    """
    Verify email with token.

    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    user_id = db.validate_verification_token(token, 'email_verify')
    if not user_id:
        return False, "Invalid or expired verification link"

    db.verify_user_email(user_id)
    return True, None


async def forgot_password(email: str) -> Tuple[bool, Optional[str]]:
    """
    Request password reset.

    Returns:
        (True, None) on success (always returns True to prevent email enumeration)
    """
    normalized_email = sanitize_email(email)
    if not normalized_email:
        return True, None  # Don't reveal email validation details

    user = db.get_user_by_email(normalized_email)
    if not user:
        return True, None  # Don't reveal whether email exists

    # Rate limit: max 3 reset emails per hour
    recent = db.count_recent_tokens(user["id"], 'password_reset', hours=1)
    if recent >= 3:
        return True, None  # Silently skip to prevent abuse

    token = db.create_verification_token(user["id"], 'password_reset', hours=1)
    try:
        await send_password_reset_email(normalized_email, token)
    except Exception as e:
        print(f"[Auth] Failed to send reset email: {e}")

    return True, None


def reset_password(token: str, new_password: str) -> Tuple[bool, Optional[str]]:
    """
    Reset password using token.

    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    # Validate new password
    password_error = validate_password(new_password)
    if password_error:
        return False, password_error

    user_id = db.validate_verification_token(token, 'password_reset')
    if not user_id:
        return False, "Invalid or expired reset link"

    # Update password and invalidate all sessions
    pw_hash = hash_password(new_password)
    db.update_user(user_id, password_hash=pw_hash)
    db.delete_user_sessions(user_id)

    return True, None


async def resend_verification(email: str) -> Tuple[bool, Optional[str]]:
    """
    Resend verification email.

    Returns:
        (True, None) on success
        (False, error_message) on failure
    """
    normalized_email = sanitize_email(email)
    if not normalized_email:
        return False, "Invalid email address"

    user = db.get_user_by_email(normalized_email)
    if not user:
        return True, None  # Don't reveal whether email exists

    if user.get("email_verified"):
        return False, "Email is already verified"

    # Rate limit: max 3 per hour
    recent = db.count_recent_tokens(user["id"], 'email_verify', hours=1)
    if recent >= 3:
        return False, "Too many verification emails sent. Try again later."

    token = db.create_verification_token(user["id"], 'email_verify', hours=24)
    try:
        await send_verification_email(normalized_email, token)
    except Exception as e:
        print(f"[Auth] Failed to resend verification: {e}")
        return False, "Failed to send verification email"

    return True, None
