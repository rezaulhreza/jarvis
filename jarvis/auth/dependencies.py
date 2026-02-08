"""
FastAPI dependencies for authentication.

Usage:
    from jarvis.auth.dependencies import get_current_user, get_ws_user

    @app.get("/api/something")
    async def something(user: dict = Depends(get_current_user)):
        ...
"""

from typing import Optional

from fastapi import Request, WebSocket, HTTPException, status

from . import db

SESSION_COOKIE = "jarvis_session"


async def get_current_user(request: Request) -> dict:
    """
    Extract authenticated user from session cookie.
    Raises 401 if not authenticated.
    """
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    session = db.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
        )

    user = db.get_user_by_id(session["user_id"])
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account not found or deactivated",
        )

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name"),
        "avatar_url": user.get("avatar_url"),
        "auth_provider": user.get("auth_provider"),
    }


async def get_optional_user(request: Request) -> Optional[dict]:
    """Same as get_current_user but returns None instead of raising."""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


async def get_ws_user(websocket: WebSocket) -> Optional[dict]:
    """
    Extract user from WebSocket cookies during upgrade handshake.
    Returns None if not authenticated.
    """
    session_id = websocket.cookies.get(SESSION_COOKIE)
    if not session_id:
        return None

    session = db.get_session(session_id)
    if not session:
        return None

    user = db.get_user_by_id(session["user_id"])
    if not user or not user.get("is_active"):
        return None

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name"),
        "avatar_url": user.get("avatar_url"),
        "auth_provider": user.get("auth_provider"),
    }
