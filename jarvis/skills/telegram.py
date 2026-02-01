"""Telegram bot skill for sending and receiving messages"""

import os
from typing import Optional

# Lazy import to avoid errors if not installed
telegram = None


def _get_bot():
    """Get or create Telegram bot instance."""
    global telegram
    if telegram is None:
        try:
            from telegram import Bot
            telegram = Bot
        except ImportError:
            return None

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return None

    return telegram(token=token)


async def send_message(chat_id: str, text: str) -> dict:
    """
    Send a message via Telegram.

    Args:
        chat_id: The chat ID to send to
        text: Message text

    Returns:
        Success status and message info
    """
    bot = _get_bot()
    if not bot:
        return {
            "success": False,
            "error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN in .env"
        }

    try:
        message = await bot.send_message(chat_id=chat_id, text=text)
        return {
            "success": True,
            "message_id": message.message_id,
            "chat_id": chat_id
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_updates(limit: int = 10) -> dict:
    """
    Get recent messages/updates from Telegram.

    Args:
        limit: Maximum number of updates to fetch

    Returns:
        List of recent updates
    """
    bot = _get_bot()
    if not bot:
        return {
            "success": False,
            "error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN in .env"
        }

    try:
        updates = await bot.get_updates(limit=limit)
        messages = []
        for update in updates:
            if update.message:
                messages.append({
                    "from": update.message.from_user.first_name,
                    "text": update.message.text,
                    "date": update.message.date.isoformat(),
                    "chat_id": update.message.chat_id
                })
        return {"success": True, "messages": messages}
    except Exception as e:
        return {"success": False, "error": str(e)}
