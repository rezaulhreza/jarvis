"""
Telegram Bot Integration for Jarvis

This module provides a Telegram bot that:
1. Listens for incoming messages
2. Processes them through Jarvis
3. Sends responses back to users

Setup:
1. Create a bot via @BotFather on Telegram
2. Get the bot token
3. Set TELEGRAM_BOT_TOKEN in .env
4. Optionally set TELEGRAM_ALLOWED_USERS for access control
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Awaitable
from pathlib import Path

logger = logging.getLogger(__name__)

# Message handler type
MessageHandler = Callable[[str, str, str], Awaitable[str]]


class TelegramBot:
    """Telegram bot that integrates with Jarvis."""

    def __init__(
        self,
        token: Optional[str] = None,
        message_handler: Optional[MessageHandler] = None,
        allowed_users: Optional[list[str]] = None,
    ):
        """
        Initialize the Telegram bot.

        Args:
            token: Bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            message_handler: Async function(user_id, username, text) -> response
            allowed_users: List of allowed user IDs (None = allow all)
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.message_handler = message_handler
        self.allowed_users = allowed_users or self._load_allowed_users()
        self._app = None
        self._running = False

    def _load_allowed_users(self) -> Optional[list[str]]:
        """Load allowed users from env var."""
        users = os.getenv("TELEGRAM_ALLOWED_USERS", "")
        if users:
            return [u.strip() for u in users.split(",") if u.strip()]
        return None  # Allow all if not specified

    def is_user_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to use the bot."""
        if self.allowed_users is None:
            return True
        return str(user_id) in self.allowed_users

    async def start(self):
        """Start the Telegram bot."""
        if not self.token:
            logger.error("Telegram bot token not configured")
            return False

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler as TGMessageHandler,
                filters,
            )
        except ImportError:
            logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return False

        # Create application
        self._app = Application.builder().token(self.token).build()

        # Add handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("id", self._handle_id))
        self._app.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Start polling
        self._running = True
        logger.info("Starting Telegram bot...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        return True

    async def stop(self):
        """Stop the Telegram bot."""
        if self._app and self._running:
            self._running = False
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    async def _handle_start(self, update, context):
        """Handle /start command."""
        user = update.effective_user
        if not self.is_user_allowed(user.id):
            await update.message.reply_text(
                "⛔ You are not authorized to use this bot.\n"
                f"Your user ID is: `{user.id}`\n"
                "Share this with the bot admin to get access.",
                parse_mode="Markdown"
            )
            return

        await update.message.reply_text(
            f"👋 Hello {user.first_name}!\n\n"
            "I'm Jarvis, your personal AI assistant.\n\n"
            "Just send me a message and I'll help you with:\n"
            "• Answering questions\n"
            "• Web searches\n"
            "• Task management\n"
            "• And much more!\n\n"
            "Commands:\n"
            "/help - Show this message\n"
            "/id - Show your user ID"
        )

    async def _handle_help(self, update, context):
        """Handle /help command."""
        await self._handle_start(update, context)

    async def _handle_id(self, update, context):
        """Handle /id command - show user ID for whitelist."""
        user = update.effective_user
        await update.message.reply_text(
            f"Your user ID: `{user.id}`\n"
            f"Username: @{user.username or 'N/A'}",
            parse_mode="Markdown"
        )

    async def _handle_message(self, update, context):
        """Handle incoming text messages."""
        user = update.effective_user
        message = update.message

        # Check authorization
        if not self.is_user_allowed(user.id):
            await message.reply_text(
                "⛔ You are not authorized. Send /start for your user ID."
            )
            return

        text = message.text
        if not text:
            return

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action="typing"
        )

        # Process through Jarvis
        try:
            if self.message_handler:
                response = await self.message_handler(
                    str(user.id),
                    user.username or user.first_name,
                    text
                )
            else:
                response = await self._default_handler(text)

            # Send response (split if too long)
            if len(response) > 4096:
                for i in range(0, len(response), 4096):
                    await message.reply_text(response[i:i+4096])
            else:
                await message.reply_text(response)

        except Exception as e:
            logger.exception("Error processing message")
            await message.reply_text(f"❌ Error: {str(e)}")

    async def _default_handler(self, text: str) -> str:
        """Default message handler using Jarvis."""
        try:
            from jarvis.assistant import Jarvis
            import asyncio

            # Create temporary Jarvis instance
            jarvis = Jarvis()

            # process() is sync, run in executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, jarvis.process, text)
            return response or "I couldn't generate a response."
        except Exception as e:
            logger.exception("Jarvis error")
            return f"Error processing your request: {str(e)}"

    async def send_message(self, chat_id: str, text: str) -> bool:
        """Send a message to a specific chat."""
        if not self._app:
            return False

        try:
            await self._app.bot.send_message(chat_id=chat_id, text=text)
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False


async def run_telegram_bot(message_handler: Optional[MessageHandler] = None):
    """Run the Telegram bot as a standalone service."""
    bot = TelegramBot(message_handler=message_handler)

    if not await bot.start():
        print("Failed to start Telegram bot. Check your configuration.")
        return

    print("Telegram bot is running. Press Ctrl+C to stop.")

    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(run_telegram_bot())
