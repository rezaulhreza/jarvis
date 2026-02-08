"""
Auth middleware - authentication, CSRF, rate limiting, security headers.
"""

import os
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .security import generate_csrf_token

SESSION_COOKIE = "jarvis_session"
CSRF_COOKIE = "jarvis_csrf"
CSRF_HEADER = "x-csrf-token"

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/api/auth/login",
}

# Path prefixes that don't require auth
PUBLIC_PREFIXES = (
    "/login",
    "/assets/",
    "/jarvis.jpeg",
)


def is_auth_enabled() -> bool:
    """Check if auth is enabled via env var."""
    return os.environ.get("JARVIS_AUTH_ENABLED", "").lower() in ("1", "true", "yes")


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via session cookie."""

    async def dispatch(self, request: Request, call_next):
        from . import db

        path = request.url.path

        # Skip auth for public paths
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        # Skip auth for root page (will be handled by React router)
        if path == "/":
            return await call_next(request)

        # Skip WebSocket (handled separately in the endpoint)
        if path == "/ws":
            return await call_next(request)

        # Skip non-API paths (static files, etc)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Auth check for API endpoints
        session_id = request.cookies.get(SESSION_COOKIE)
        if not session_id:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        session = db.get_session(session_id)
        if not session:
            return JSONResponse(
                status_code=401,
                content={"detail": "Session expired or invalid"},
            )

        user = db.get_user_by_id(session["user_id"])
        if not user or not user.get("is_active"):
            return JSONResponse(
                status_code=401,
                content={"detail": "Account not found or deactivated"},
            )

        # Attach user to request state
        request.state.user = {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
            "auth_provider": user.get("auth_provider"),
        }

        return await call_next(request)


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection via double-submit cookie pattern."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only check CSRF on mutating API requests
        if request.method in ("POST", "PUT", "DELETE", "PATCH") and path.startswith("/api/"):
            # Skip CSRF for auth endpoints (they need to work without prior CSRF token)
            if path.startswith("/api/auth/"):
                response = await call_next(request)
                return response

            # Check CSRF token
            csrf_cookie = request.cookies.get(CSRF_COOKIE)
            csrf_header = request.headers.get(CSRF_HEADER)

            if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed"},
                )

        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


def set_session_cookie(response: Response, session_id: str, secure: bool = False):
    """Set session cookie on response."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=30 * 24 * 3600,  # 30 days
        path="/",
    )


def set_csrf_cookie(response: Response, secure: bool = False):
    """Set CSRF cookie on response (readable by JS)."""
    csrf_token = generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE,
        value=csrf_token,
        httponly=False,  # JS needs to read this
        secure=secure,
        samesite="lax",
        max_age=30 * 24 * 3600,
        path="/",
    )


def clear_session_cookie(response: Response):
    """Clear session and CSRF cookies."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
