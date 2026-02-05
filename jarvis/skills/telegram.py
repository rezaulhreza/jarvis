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


async def send_photo(chat_id: str, photo_path: str, caption: str = None) -> dict:
    """
    Send a photo via Telegram.

    Args:
        chat_id: The chat ID to send to
        photo_path: Path to the image file
        caption: Optional caption for the photo

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
        from pathlib import Path
        photo = Path(photo_path)
        if not photo.exists():
            return {"success": False, "error": f"Photo not found: {photo_path}"}

        with open(photo, "rb") as f:
            message = await bot.send_photo(chat_id=chat_id, photo=f, caption=caption)

        return {
            "success": True,
            "message_id": message.message_id,
            "chat_id": chat_id,
            "type": "photo"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_video(chat_id: str, video_path: str, caption: str = None) -> dict:
    """
    Send a video via Telegram.

    Args:
        chat_id: The chat ID to send to
        video_path: Path to the video file
        caption: Optional caption for the video

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
        from pathlib import Path
        video = Path(video_path)
        if not video.exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        with open(video, "rb") as f:
            message = await bot.send_video(chat_id=chat_id, video=f, caption=caption)

        return {
            "success": True,
            "message_id": message.message_id,
            "chat_id": chat_id,
            "type": "video"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_audio(chat_id: str, audio_path: str, caption: str = None) -> dict:
    """
    Send an audio file via Telegram.

    Args:
        chat_id: The chat ID to send to
        audio_path: Path to the audio file
        caption: Optional caption for the audio

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
        from pathlib import Path
        audio = Path(audio_path)
        if not audio.exists():
            return {"success": False, "error": f"Audio not found: {audio_path}"}

        with open(audio, "rb") as f:
            message = await bot.send_audio(chat_id=chat_id, audio=f, caption=caption)

        return {
            "success": True,
            "message_id": message.message_id,
            "chat_id": chat_id,
            "type": "audio"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_media(chat_id: str, media_path: str, media_type: str = None, caption: str = None) -> dict:
    """
    Send any media type via Telegram (auto-detects type if not specified).

    Args:
        chat_id: The chat ID to send to
        media_path: Path to the media file
        media_type: Type of media (image, video, audio) - auto-detected if None
        caption: Optional caption

    Returns:
        Success status and message info
    """
    from pathlib import Path
    path = Path(media_path)

    if not path.exists():
        return {"success": False, "error": f"File not found: {media_path}"}

    # Auto-detect type if not specified
    if not media_type:
        suffix = path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            media_type = 'image'
        elif suffix in ['.mp4', '.webm', '.mov', '.avi']:
            media_type = 'video'
        elif suffix in ['.mp3', '.wav', '.ogg', '.m4a']:
            media_type = 'audio'
        else:
            media_type = 'document'

    if media_type == 'image':
        return await send_photo(chat_id, media_path, caption)
    elif media_type == 'video':
        return await send_video(chat_id, media_path, caption)
    elif media_type == 'audio':
        return await send_audio(chat_id, media_path, caption)
    else:
        # Fall back to document for unknown types
        bot = _get_bot()
        if not bot:
            return {"success": False, "error": "Telegram not configured"}
        try:
            with open(path, "rb") as f:
                message = await bot.send_document(chat_id=chat_id, document=f, caption=caption)
            return {"success": True, "message_id": message.message_id, "chat_id": chat_id, "type": "document"}
        except Exception as e:
            return {"success": False, "error": str(e)}
