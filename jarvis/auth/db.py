"""
Auth database operations - users, sessions, verification tokens.

All auth data lives in the main jarvis.db SQLite database.
Uses SQLAlchemy ORM for auth tables.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from .models import (
    Base, engine, SessionLocal, model_to_dict,
    User, Session, VerificationToken,
)
from .security import generate_session_id, generate_token


def _get_db_path():
    from .models import _get_db_path as get_path
    return get_path()


def init_auth_tables():
    """Create auth tables if they don't exist."""
    # Create all ORM-managed auth tables
    Base.metadata.create_all(engine)

    # Migrate non-ORM tables (chats, summaries) with raw SQL
    conn = sqlite3.connect(str(_get_db_path()))
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    # Migrate: add user_id to chats if not present
    try:
        cursor.execute("SELECT user_id FROM chats LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id)')

    # Migrate: add user_id to summaries if not present
    try:
        cursor.execute("SELECT user_id FROM summaries LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE summaries ADD COLUMN user_id TEXT")

    conn.commit()
    conn.close()


# === User Operations ===

def create_user(
    email: str,
    password_hash: str = None,
    name: str = None,
    auth_provider: str = 'email',
    provider_id: str = None,
    email_verified: bool = False,
    avatar_url: str = None,
) -> Dict:
    """Create a new user. Returns user dict."""
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()

    with SessionLocal() as db:
        user = User(
            id=user_id,
            email=email,
            email_verified=email_verified,
            password_hash=password_hash,
            name=name,
            avatar_url=avatar_url,
            auth_provider=auth_provider,
            provider_id=provider_id,
            created_at=now,
            updated_at=now,
        )
        db.add(user)
        db.commit()

    return {
        "id": user_id,
        "email": email,
        "email_verified": email_verified,
        "name": name,
        "avatar_url": avatar_url,
        "auth_provider": auth_provider,
    }


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        return model_to_dict(user)


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID."""
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        return model_to_dict(user)


def get_user_by_provider(provider: str, provider_id: str) -> Optional[Dict]:
    """Get user by OAuth provider and provider-specific ID."""
    with SessionLocal() as db:
        user = db.query(User).filter(
            User.auth_provider == provider,
            User.provider_id == provider_id,
        ).first()
        return model_to_dict(user)


def verify_user_email(user_id: str) -> bool:
    """Mark user's email as verified."""
    with SessionLocal() as db:
        count = db.query(User).filter(User.id == user_id).update({
            User.email_verified: True,
            User.updated_at: datetime.utcnow(),
        })
        db.commit()
        return count > 0


def update_user_login(user_id: str):
    """Update last login time and reset failed attempts."""
    now = datetime.utcnow()
    with SessionLocal() as db:
        db.query(User).filter(User.id == user_id).update({
            User.last_login_at: now,
            User.failed_login_attempts: 0,
            User.locked_until: None,
            User.updated_at: now,
        })
        db.commit()


def increment_failed_login(user_id: str) -> int:
    """Increment failed login attempts. Returns new count."""
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return 0

        user.failed_login_attempts += 1
        user.updated_at = datetime.utcnow()
        count = user.failed_login_attempts

        # Lock account after 5 failed attempts (15 min cooldown)
        if count >= 5:
            user.locked_until = datetime.utcnow() + timedelta(minutes=15)

        db.commit()
        return count


def is_account_locked(user: Dict) -> bool:
    """Check if account is locked due to failed login attempts."""
    locked_until = user.get("locked_until")
    if not locked_until:
        return False
    try:
        if isinstance(locked_until, str):
            lock_time = datetime.fromisoformat(locked_until)
        else:
            lock_time = locked_until
        return datetime.utcnow() < lock_time
    except (ValueError, TypeError):
        return False


def update_user(user_id: str, **kwargs) -> bool:
    """Update user fields."""
    allowed = {"name", "avatar_url", "password_hash", "auth_provider", "provider_id"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False

    updates["updated_at"] = datetime.utcnow()

    with SessionLocal() as db:
        # Map string keys to column attributes
        column_updates = {}
        for k, v in updates.items():
            col = getattr(User, k, None)
            if col is not None:
                column_updates[col] = v

        count = db.query(User).filter(User.id == user_id).update(column_updates)
        db.commit()
        return count > 0


# === Session Operations ===

def create_session(user_id: str, ip_address: str = None, user_agent: str = None, days: int = 30) -> str:
    """Create a new session. Returns session ID."""
    session_id = generate_session_id()
    now = datetime.utcnow()
    expires = now + timedelta(days=days)

    with SessionLocal() as db:
        sess = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=expires,
            last_active_at=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(sess)
        db.commit()

    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Get a valid (active + not expired) session."""
    now = datetime.utcnow()

    with SessionLocal() as db:
        sess = db.query(Session).filter(
            Session.id == session_id,
            Session.is_active == True,
            Session.expires_at > now,
        ).first()

        if sess:
            # Sliding window: extend session on activity
            sess.last_active_at = now
            db.commit()

        return model_to_dict(sess)


def delete_session(session_id: str) -> bool:
    """Invalidate a session."""
    with SessionLocal() as db:
        count = db.query(Session).filter(Session.id == session_id).update({
            Session.is_active: False,
        })
        db.commit()
        return count > 0


def delete_user_sessions(user_id: str) -> int:
    """Invalidate all sessions for a user (logout all devices)."""
    with SessionLocal() as db:
        count = db.query(Session).filter(Session.user_id == user_id).update({
            Session.is_active: False,
        })
        db.commit()
        return count


# === Verification Token Operations ===

def create_verification_token(user_id: str, token_type: str = 'email_verify', hours: int = 24) -> str:
    """Create a verification token. Returns token string."""
    token_id = str(uuid.uuid4())
    token = generate_token(32)
    now = datetime.utcnow()
    expires = now + timedelta(hours=hours)

    with SessionLocal() as db:
        vt = VerificationToken(
            id=token_id,
            user_id=user_id,
            token=token,
            token_type=token_type,
            created_at=now,
            expires_at=expires,
        )
        db.add(vt)
        db.commit()

    return token


def validate_verification_token(token: str, token_type: str = 'email_verify') -> Optional[str]:
    """Validate a token. Returns user_id if valid, None otherwise."""
    now = datetime.utcnow()

    with SessionLocal() as db:
        vt = db.query(VerificationToken).filter(
            VerificationToken.token == token,
            VerificationToken.token_type == token_type,
            VerificationToken.expires_at > now,
            VerificationToken.used_at == None,
        ).first()

        if vt:
            vt.used_at = now
            user_id = vt.user_id
            db.commit()
            return user_id

    return None


def count_recent_tokens(user_id: str, token_type: str, hours: int = 1) -> int:
    """Count tokens created in the last N hours (for rate limiting)."""
    since = datetime.utcnow() - timedelta(hours=hours)

    with SessionLocal() as db:
        count = db.query(VerificationToken).filter(
            VerificationToken.user_id == user_id,
            VerificationToken.token_type == token_type,
            VerificationToken.created_at > since,
        ).count()
        return count


# === Cleanup ===

def cleanup_expired():
    """Remove expired sessions and tokens. Call on startup."""
    now = datetime.utcnow()

    with SessionLocal() as db:
        db.query(Session).filter(
            (Session.expires_at < now) | (Session.is_active == False)
        ).delete(synchronize_session="fetch")

        db.query(VerificationToken).filter(
            (VerificationToken.expires_at < now) | (VerificationToken.used_at != None)
        ).delete(synchronize_session="fetch")

        db.commit()
