"""
OAuth 2.0 flows - Google and GitHub authentication.
"""

import os
from typing import Dict, Optional, Tuple

from . import db


def get_oauth_config(provider: str) -> Dict:
    """Get OAuth client config from environment."""
    provider_upper = provider.upper()
    return {
        "client_id": os.environ.get(f"{provider_upper}_CLIENT_ID", ""),
        "client_secret": os.environ.get(f"{provider_upper}_CLIENT_SECRET", ""),
    }


def is_oauth_configured(provider: str) -> bool:
    """Check if OAuth is configured for a provider."""
    config = get_oauth_config(provider)
    return bool(config["client_id"] and config["client_secret"])


async def handle_oauth_user(
    email: str,
    name: str = None,
    avatar_url: str = None,
    provider: str = 'google',
    provider_id: str = None,
    ip: str = None,
    user_agent: str = None,
) -> Tuple[str, Dict]:
    """
    Handle OAuth user - find or create, then create session.

    OAuth users are auto-verified (email confirmed by provider).

    Returns:
        (session_id, user_dict)
    """
    # Check if user exists by provider ID
    user = db.get_user_by_provider(provider, provider_id) if provider_id else None

    # Check if user exists by email
    if not user:
        user = db.get_user_by_email(email)

    if user:
        # Update OAuth info if needed (link accounts)
        if user.get("auth_provider") == "email" and provider_id:
            # Link OAuth to existing email account (only if both verified)
            db.update_user(user["id"], auth_provider=provider, provider_id=provider_id)
            if avatar_url and not user.get("avatar_url"):
                db.update_user(user["id"], avatar_url=avatar_url)
            if name and not user.get("name"):
                db.update_user(user["id"], name=name)

        # Ensure email is verified for OAuth
        if not user.get("email_verified"):
            db.verify_user_email(user["id"])

        db.update_user_login(user["id"])
    else:
        # Create new user (auto-verified)
        user = db.create_user(
            email=email,
            name=name,
            avatar_url=avatar_url,
            auth_provider=provider,
            provider_id=provider_id,
            email_verified=True,
        )

    session_id = db.create_session(user["id"], ip_address=ip, user_agent=user_agent)

    safe_user = {
        "id": user["id"],
        "email": user.get("email") or email,
        "name": user.get("name") or name,
        "avatar_url": user.get("avatar_url") or avatar_url,
        "auth_provider": provider,
    }

    return session_id, safe_user


async def google_get_user_info(code: str, redirect_uri: str) -> Optional[Dict]:
    """Exchange Google OAuth code for user info."""
    config = get_oauth_config("google")
    if not config["client_id"]:
        return None

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Exchange code for tokens
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )

            if token_resp.status_code != 200:
                print(f"[OAuth] Google token error: {token_resp.text}")
                return None

            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            # Fetch user profile
            profile_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if profile_resp.status_code != 200:
                return None

            profile = profile_resp.json()
            return {
                "email": profile.get("email"),
                "name": profile.get("name"),
                "avatar_url": profile.get("picture"),
                "provider_id": profile.get("id"),
            }
    except Exception as e:
        print(f"[OAuth] Google error: {e}")
        return None


async def github_get_user_info(code: str, redirect_uri: str) -> Optional[Dict]:
    """Exchange GitHub OAuth code for user info."""
    config = get_oauth_config("github")
    if not config["client_id"]:
        return None

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Exchange code for token
            token_resp = await client.post(
                "https://github.com/login/oauth/access_token",
                json={
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )

            if token_resp.status_code != 200:
                return None

            tokens = token_resp.json()
            access_token = tokens.get("access_token")
            if not access_token:
                return None

            headers = {"Authorization": f"Bearer {access_token}"}

            # Fetch user profile
            profile_resp = await client.get(
                "https://api.github.com/user",
                headers=headers,
            )

            if profile_resp.status_code != 200:
                return None

            profile = profile_resp.json()

            # Fetch primary email (may not be in profile)
            email = profile.get("email")
            if not email:
                emails_resp = await client.get(
                    "https://api.github.com/user/emails",
                    headers=headers,
                )
                if emails_resp.status_code == 200:
                    emails = emails_resp.json()
                    for e in emails:
                        if e.get("primary") and e.get("verified"):
                            email = e["email"]
                            break

            if not email:
                return None

            return {
                "email": email,
                "name": profile.get("name") or profile.get("login"),
                "avatar_url": profile.get("avatar_url"),
                "provider_id": str(profile.get("id")),
            }
    except Exception as e:
        print(f"[OAuth] GitHub error: {e}")
        return None
