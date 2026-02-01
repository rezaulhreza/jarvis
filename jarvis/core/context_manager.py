"""Context management with auto-compaction"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
import tiktoken


class ContextManager:
    """
    Manages conversation context with auto-compaction.

    Features:
    - Tracks conversation history
    - Auto-compacts when token limit exceeded
    - Persists to SQLite
    - Maintains working memory for current task
    """

    def __init__(
        self,
        db_path: str = "memory/jarvis.db",
        max_tokens: int = 8000,
        keep_recent: int = 5
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

        # In-memory state
        self.messages: list[dict] = []
        self.working_memory: dict = {}
        self.current_task: Optional[str] = None

        # Token counter (approximate)
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                model TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                summary TEXT,
                message_range TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        # Rough estimate if tiktoken not available
        return len(text) // 4

    def get_context_tokens(self) -> int:
        """Get total tokens in current context."""
        total = 0
        for msg in self.messages:
            total += self.count_tokens(msg.get('content', ''))
        return total

    def add_message(self, role: str, content: str, model: str = None):
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "model": model
        }
        self.messages.append(message)

        # Persist to database
        self._save_message(message)

        # Check if compaction needed
        if self.get_context_tokens() > self.max_tokens:
            self._compact()

    def _save_message(self, message: dict):
        """Save message to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations (timestamp, role, content, model, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            message['timestamp'],
            message['role'],
            message['content'],
            message.get('model'),
            json.dumps(message.get('metadata', {}))
        ))

        conn.commit()
        conn.close()

    def _compact(self):
        """Compact conversation history when too long."""
        if len(self.messages) <= self.keep_recent:
            return

        # Keep recent messages
        recent = self.messages[-self.keep_recent:]
        old = self.messages[:-self.keep_recent]

        # Create summary of old messages
        summary = self._summarize(old)

        # Save summary to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO summaries (timestamp, summary, message_range)
            VALUES (?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            summary,
            f"{old[0]['timestamp']} to {old[-1]['timestamp']}"
        ))
        conn.commit()
        conn.close()

        # Replace messages with summary + recent
        self.messages = [
            {"role": "system", "content": f"[Previous conversation summary: {summary}]"}
        ] + recent

        print(f"\n[Context compacted: {len(old)} messages summarized]\n")

    def _summarize(self, messages: list[dict]) -> str:
        """Create a summary of messages (simple version)."""
        # In a full implementation, you'd use the LLM to summarize
        # For now, extract key points
        summary_parts = []

        for msg in messages:
            role = msg['role']
            content = msg['content'][:200]  # Truncate
            if role == 'user':
                summary_parts.append(f"User asked about: {content[:100]}")
            elif role == 'assistant':
                summary_parts.append(f"Assistant: {content[:100]}")

        return " | ".join(summary_parts[-5:])  # Keep last 5 points

    def get_messages(self) -> list[dict]:
        """Get current messages for chat."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def set_task(self, task: str):
        """Set the current task."""
        self.current_task = task
        self.working_memory['task'] = task

    def update_working_memory(self, key: str, value):
        """Update working memory with a key-value pair."""
        self.working_memory[key] = value

    def get_working_memory(self) -> dict:
        """Get current working memory."""
        return self.working_memory.copy()

    def clear(self):
        """Clear in-memory context (keeps database)."""
        self.messages = []
        self.working_memory = {}
        self.current_task = None

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get conversation history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT timestamp, role, content, model
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {"timestamp": r[0], "role": r[1], "content": r[2], "model": r[3]}
            for r in reversed(rows)
        ]
