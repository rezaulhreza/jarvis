"""
SQLAlchemy ORM models for auth tables - users, sessions, verification tokens.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from jarvis import get_data_dir

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    email_verified = Column(Boolean, default=False)
    password_hash = Column(String, nullable=True)
    name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    auth_provider = Column(String, default="email")
    provider_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    verification_tokens = relationship("VerificationToken", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_users_provider", "auth_provider", "provider_id"),
    )


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_expires", "expires_at"),
    )


class VerificationToken(Base):
    __tablename__ = "verification_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    token_type = Column(String, default="email_verify")
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="verification_tokens")

    __table_args__ = (
        Index("idx_verify_token", "token"),
    )


def _get_db_path():
    db_path = get_data_dir() / "memory" / "jarvis.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


engine = create_engine(
    f"sqlite:///{_get_db_path()}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)


def model_to_dict(obj) -> dict | None:
    """Convert an ORM instance to a plain dict. Returns None if obj is None."""
    if obj is None:
        return None
    d = {}
    for col in obj.__table__.columns:
        val = getattr(obj, col.name)
        if isinstance(val, datetime):
            val = val.isoformat()
        if isinstance(val, bool):
            pass  # keep as bool
        d[col.name] = val
    return d
