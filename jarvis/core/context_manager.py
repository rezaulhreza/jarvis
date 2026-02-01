"""Context management with auto-compaction and chat history"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import tiktoken
import uuid


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
        self.current_chat_id: Optional[str] = None

        # Token counter (approximate)
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None

        self._init_db()
        self._migrate_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Chats table - each chat is a conversation session
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0
            )
        ''')

        # Messages table - linked to chats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                model TEXT,
                metadata TEXT,
                FOREIGN KEY (chat_id) REFERENCES chats(id)
            )
        ''')

        # Legacy conversations table (for backward compatibility)
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
                message_range TEXT,
                chat_id TEXT
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC)')

        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Migrate old conversations to new chat structure."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if migration needed (old messages exist without chat_id)
        cursor.execute('SELECT COUNT(*) FROM conversations')
        old_count = cursor.fetchone()[0]

        if old_count > 0:
            # Check if already migrated
            cursor.execute('SELECT COUNT(*) FROM chats')
            chat_count = cursor.fetchone()[0]

            if chat_count == 0:
                # Migrate old conversations to a single "History" chat
                chat_id = str(uuid.uuid4())
                now = datetime.now().isoformat()

                cursor.execute('''
                    INSERT INTO chats (id, title, created_at, updated_at, message_count)
                    VALUES (?, ?, ?, ?, ?)
                ''', (chat_id, 'Previous History', now, now, old_count))

                # Copy old messages to new structure
                cursor.execute('''
                    INSERT INTO messages (chat_id, timestamp, role, content, model, metadata)
                    SELECT ?, timestamp, role, content, model, metadata FROM conversations
                ''', (chat_id,))

                conn.commit()
                print(f"[Migration] Moved {old_count} messages to chat history")

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

    # ============== Chat Management ==============

    def create_chat(self, title: str = "New Chat") -> str:
        """Create a new chat and return its ID."""
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chats (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (chat_id, title, now, now))
        conn.commit()
        conn.close()

        self.current_chat_id = chat_id
        self.messages = []
        return chat_id

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a chat by ID with its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, title, created_at, updated_at, message_count
            FROM chats WHERE id = ?
        ''', (chat_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        chat = {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "message_count": row[4]
        }

        # Get messages for this chat
        cursor.execute('''
            SELECT id, timestamp, role, content, model
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
        ''', (chat_id,))

        chat["messages"] = [
            {"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3], "model": r[4]}
            for r in cursor.fetchall()
        ]

        conn.close()
        return chat

    def list_chats(self, limit: int = 50, search: str = None) -> List[Dict]:
        """List all chats, optionally filtered by search term."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if search:
            cursor.execute('''
                SELECT c.id, c.title, c.created_at, c.updated_at, c.message_count,
                       (SELECT content FROM messages WHERE chat_id = c.id ORDER BY id ASC LIMIT 1) as preview
                FROM chats c
                WHERE c.archived = 0 AND (c.title LIKE ? OR c.id IN (
                    SELECT DISTINCT chat_id FROM messages WHERE content LIKE ?
                ))
                ORDER BY c.updated_at DESC
                LIMIT ?
            ''', (f'%{search}%', f'%{search}%', limit))
        else:
            cursor.execute('''
                SELECT c.id, c.title, c.created_at, c.updated_at, c.message_count,
                       (SELECT content FROM messages WHERE chat_id = c.id ORDER BY id ASC LIMIT 1) as preview
                FROM chats c
                WHERE c.archived = 0
                ORDER BY c.updated_at DESC
                LIMIT ?
            ''', (limit,))

        chats = []
        for row in cursor.fetchall():
            preview = row[5][:100] + "..." if row[5] and len(row[5]) > 100 else row[5]
            chats.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4],
                "preview": preview
            })

        conn.close()
        return chats

    def update_chat(self, chat_id: str, title: str = None) -> bool:
        """Update a chat's title."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if title:
            cursor.execute('''
                UPDATE chats SET title = ?, updated_at = ?
                WHERE id = ?
            ''', (title, datetime.now().isoformat(), chat_id))

        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete messages first
        cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
        # Delete chat
        cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        # Clear current chat if it was deleted
        if self.current_chat_id == chat_id:
            self.current_chat_id = None
            self.messages = []

        return affected > 0

    def switch_chat(self, chat_id: str) -> bool:
        """Switch to a different chat, loading its messages."""
        chat = self.get_chat(chat_id)
        if not chat:
            return False

        self.current_chat_id = chat_id
        self.messages = [
            {"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
            for m in chat["messages"]
        ]
        return True

    def add_message_to_chat(self, role: str, content: str, model: str = None) -> int:
        """Add a message to the current chat."""
        if not self.current_chat_id:
            # Auto-create a new chat
            self.create_chat()

        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO messages (chat_id, timestamp, role, content, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (self.current_chat_id, now, role, content, model, '{}'))

        message_id = cursor.lastrowid

        # Update chat's updated_at and message_count
        cursor.execute('''
            UPDATE chats SET updated_at = ?, message_count = message_count + 1
            WHERE id = ?
        ''', (now, self.current_chat_id))

        conn.commit()
        conn.close()

        # Also add to in-memory messages
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": now,
            "model": model
        })

        # Also save to legacy table for backward compatibility
        self._save_message({"role": role, "content": content, "timestamp": now, "model": model})

        return message_id

    def get_current_chat_id(self) -> Optional[str]:
        """Get the current chat ID."""
        return self.current_chat_id

    def get_chat_message_count(self, chat_id: str = None) -> int:
        """Get the number of messages in a chat."""
        cid = chat_id or self.current_chat_id
        if not cid:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT message_count FROM chats WHERE id = ?', (cid,))
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else 0

    def generate_chat_title(self, chat_id: str = None) -> Optional[str]:
        """Generate a title from the first few messages (returns content for LLM to summarize)."""
        cid = chat_id or self.current_chat_id
        if not cid:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get first 3 messages
        cursor.execute('''
            SELECT role, content FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            LIMIT 3
        ''', (cid,))

        messages = cursor.fetchall()
        conn.close()

        if not messages:
            return None

        # Return conversation snippet for LLM to generate title
        snippet = "\n".join([f"{m[0]}: {m[1][:200]}" for m in messages])
        return snippet
