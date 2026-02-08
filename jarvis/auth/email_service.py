"""
Email service - sends verification and password reset emails via SMTP.

Configurable via environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
  APP_URL (for email links, defaults to http://localhost:7777)
"""

import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def _get_smtp_config() -> dict:
    return {
        "host": os.environ.get("SMTP_HOST", ""),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from_addr": os.environ.get("SMTP_FROM", "noreply@jarvis.local"),
    }


def _get_app_url() -> str:
    return os.environ.get("APP_URL", "http://localhost:7777").rstrip("/")


def is_smtp_configured() -> bool:
    """Check if SMTP is configured."""
    config = _get_smtp_config()
    return bool(config["host"] and config["user"] and config["password"])


async def send_email(to: str, subject: str, html_body: str):
    """Send an email via SMTP."""
    config = _get_smtp_config()

    if not config["host"]:
        print(f"[Email] SMTP not configured. Would send to {to}: {subject}")
        return

    try:
        import aiosmtplib

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config["from_addr"]
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html"))

        await aiosmtplib.send(
            msg,
            hostname=config["host"],
            port=config["port"],
            username=config["user"],
            password=config["password"],
            start_tls=True,
        )
    except ImportError:
        print(f"[Email] aiosmtplib not installed. Would send to {to}: {subject}")
    except Exception as e:
        print(f"[Email] Failed to send to {to}: {e}")
        raise


async def send_verification_email(email: str, token: str):
    """Send email verification link."""
    app_url = _get_app_url()
    verify_url = f"{app_url}/verify-email?token={token}"

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 500px; margin: 0 auto; padding: 40px 20px;">
        <h1 style="color: #1a1a2e; font-size: 24px; margin-bottom: 8px;">Verify your email</h1>
        <p style="color: #666; font-size: 16px; line-height: 1.5; margin-bottom: 32px;">
            Click the button below to verify your email address and activate your account.
        </p>
        <a href="{verify_url}"
           style="display: inline-block; background: #6366f1; color: white; padding: 12px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px;">
            Verify Email
        </a>
        <p style="color: #999; font-size: 13px; margin-top: 32px;">
            This link expires in 24 hours. If you didn't create an account, ignore this email.
        </p>
    </div>
    """

    await send_email(email, "Verify your email - Jarvis", html)


async def send_password_reset_email(email: str, token: str):
    """Send password reset link."""
    app_url = _get_app_url()
    reset_url = f"{app_url}/reset-password?token={token}"

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 500px; margin: 0 auto; padding: 40px 20px;">
        <h1 style="color: #1a1a2e; font-size: 24px; margin-bottom: 8px;">Reset your password</h1>
        <p style="color: #666; font-size: 16px; line-height: 1.5; margin-bottom: 32px;">
            Click the button below to reset your password.
        </p>
        <a href="{reset_url}"
           style="display: inline-block; background: #6366f1; color: white; padding: 12px 32px;
                  border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px;">
            Reset Password
        </a>
        <p style="color: #999; font-size: 13px; margin-top: 32px;">
            This link expires in 1 hour. If you didn't request a reset, ignore this email.
        </p>
    </div>
    """

    await send_email(email, "Reset your password - Jarvis", html)
