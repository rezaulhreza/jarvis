"""Context management with auto-compaction, chat history, and semantic retrieval"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Callable
import tiktoken
import uuid


class ContextBudget:
    """Token budget allocation for different context components.

    Adapts proportionally to the model's actual context window size.
    No hardcoded assumptions — works with 4K, 32K, 128K, or 1M+ models.
    """

    def __init__(self, total_tokens: int = 8192):
        self.total = total_tokens
        # Proportional allocation: scales with any context window size
        self.system_prompt = max(500, total_tokens // 40)       # ~2.5%
        self.tools = max(500, total_tokens // 20)               # ~5%
        self.rag_context = max(500, total_tokens // 20)         # ~5%
        self.relevant_history = max(500, total_tokens // 10)    # ~10%
        self.recent_messages = max(1000, total_tokens // 5)     # ~20%
        self.response = self.total - (
            self.system_prompt + self.tools + self.rag_context +
            self.relevant_history + self.recent_messages
        )

    def allocate(self, intent: str = "", num_tools: int = 0) -> Dict[str, int]:
        """Allocate budget dynamically based on intent."""
        budget = {
            "system_prompt": self.system_prompt,
            "tools": min(num_tools * 300 + 500, self.tools) if num_tools else self.tools,
            "rag_context": self.rag_context,
            "relevant_history": self.relevant_history,
            "recent_messages": self.recent_messages,
            "response": self.response,
        }

        # Redistribute unused tokens
        used = sum(budget.values()) - budget["response"]
        budget["response"] = self.total - used

        return budget


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

        # LLM provider for intelligent summarization (set externally)
        self._provider = None
        self._on_compact_callback: Optional[Callable[[int, str], None]] = None

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

    def get_context_stats(self) -> dict:
        """
        Get context usage statistics.

        Returns dict with:
        - tokens_used: Current token count
        - max_tokens: Maximum allowed tokens
        - percentage: Usage percentage (0-100)
        - messages: Number of messages
        - needs_compact: True if approaching limit
        """
        tokens_used = self.get_context_tokens()
        percentage = (tokens_used / self.max_tokens * 100) if self.max_tokens > 0 else 0

        return {
            "tokens_used": tokens_used,
            "max_tokens": self.max_tokens,
            "percentage": round(percentage, 1),
            "messages": len(self.messages),
            "needs_compact": percentage > 80,
            "tokens_remaining": max(0, self.max_tokens - tokens_used),
        }

    def set_max_tokens(self, max_tokens: int):
        """Update max tokens (e.g., when switching models)."""
        self.max_tokens = max_tokens
        # Check if compaction needed with new limit
        if self.get_context_tokens() > self.max_tokens:
            self._compact()

    def set_provider(self, provider):
        """Set LLM provider for intelligent summarization.

        Also queries the provider for its context window size and updates
        max_tokens accordingly, so the context budget adapts to the actual model.
        """
        self._provider = provider

        # Dynamically update max_tokens from provider's actual context length
        try:
            ctx_len = provider.get_context_length()
            if ctx_len and ctx_len > 0:
                # Use ~60% of context for history (rest for system prompt, tools, response)
                self.max_tokens = int(ctx_len * 0.6)
                self._model_context_length = ctx_len
                print(f"[Context] Model context: {ctx_len:,} tokens, history budget: {self.max_tokens:,}")
        except Exception:
            pass  # Keep default max_tokens

    def set_compact_callback(self, callback: Callable[[int, str], None]):
        """Set callback for compact notifications.

        Args:
            callback: Function(messages_compacted: int, summary: str)
        """
        self._on_compact_callback = callback

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
        """Compact conversation history when too long.

        Uses LLM-based summarization when provider is available,
        otherwise falls back to simple extraction.
        """
        if len(self.messages) <= self.keep_recent:
            return

        # Keep recent messages
        recent = self.messages[-self.keep_recent:]
        old = self.messages[:-self.keep_recent]
        num_compacted = len(old)

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

        # Notify via callback or print
        if self._on_compact_callback:
            self._on_compact_callback(num_compacted, summary)
        else:
            print(f"\n[Context compacted: {num_compacted} messages summarized]\n")

    def _summarize(self, messages: list[dict]) -> str:
        """Create a summary of messages using LLM or fallback extraction.

        When an LLM provider is available, uses it for intelligent summarization.
        Otherwise, falls back to simple key point extraction.
        """
        # Try LLM-based summarization if provider is available
        if self._provider:
            try:
                return self._summarize_with_llm(messages)
            except Exception as e:
                print(f"[Context] LLM summarization failed: {e}, using fallback")

        # Fallback: simple key point extraction
        return self._summarize_simple(messages)

    def _summarize_with_llm(self, messages: list[dict]) -> str:
        """Use LLM to create an intelligent summary of messages."""
        # Build conversation text for summarization
        conv_text = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:500]  # Limit each message
            conv_text.append(f"{role.upper()}: {content}")

        conversation = "\n".join(conv_text[-20:])  # Last 20 messages max

        prompt = f"""Summarize this conversation in 2-3 concise sentences. Focus on:
1. The main topics discussed
2. Key decisions or conclusions
3. Any important context for continuing the conversation

Conversation:
{conversation}

Summary (2-3 sentences):"""

        try:
            from ..providers import Message
            response = self._provider.chat(
                messages=[Message(role="user", content=prompt)],
                system="You are a helpful assistant that creates concise conversation summaries.",
                stream=False
            )

            # Handle generator response
            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(response)

            summary = response.strip()
            # Limit summary length
            if len(summary) > 500:
                summary = summary[:497] + "..."
            return summary

        except Exception as e:
            raise RuntimeError(f"LLM summarization error: {e}")

    def _summarize_simple(self, messages: list[dict]) -> str:
        """Simple fallback summarization by extracting key points."""
        summary_parts = []

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')[:200]  # Truncate
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

        # Embed for semantic retrieval (non-blocking, best-effort)
        try:
            self.embed_message(role, content, message_id=f"msg_{message_id}")
        except Exception:
            pass  # Don't fail message saving if embedding fails

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

    # ============== Semantic History Retrieval ==============

    def _get_history_collection(self):
        """Get or create ChromaDB collection for semantic history."""
        if not hasattr(self, '_history_collection'):
            try:
                import chromadb
                client = chromadb.Client()
                self._history_collection = client.get_or_create_collection(
                    name="conversation_history",
                    metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                self._history_collection = None
            except Exception as e:
                print(f"[Context] ChromaDB error: {e}")
                self._history_collection = None
        return self._history_collection

    def embed_message(self, role: str, content: str, message_id: str = None):
        """Embed a message for semantic retrieval."""
        collection = self._get_history_collection()
        if not collection:
            return

        if not content or len(content.strip()) < 10:
            return

        msg_id = message_id or f"msg_{datetime.now().timestamp()}"
        try:
            # Truncate for embedding
            embed_text = content[:500]
            collection.add(
                documents=[embed_text],
                ids=[msg_id],
                metadatas=[{"role": role, "timestamp": datetime.now().isoformat()}]
            )
        except Exception as e:
            print(f"[Context] Embed error: {e}")

    def retrieve_relevant_history(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve semantically relevant past messages.

        Args:
            query: Current user query
            n_results: Number of relevant messages to retrieve

        Returns:
            List of relevant messages with role, content, timestamp
        """
        collection = self._get_history_collection()
        if not collection:
            return []

        try:
            count = collection.count()
            if count == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )

            relevant = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0] * len(docs)

                for doc, meta, dist in zip(docs, metas, distances):
                    # Only include reasonably relevant results (cosine distance < 0.8)
                    if dist < 0.8:
                        relevant.append({
                            "role": meta.get("role", "user"),
                            "content": doc,
                            "timestamp": meta.get("timestamp", ""),
                            "relevance": round(1 - dist, 3),
                        })

            return relevant
        except Exception as e:
            print(f"[Context] Retrieval error: {e}")
            return []

    def get_enriched_context(
        self,
        query: str,
        recent_n: int = 10,
        relevant_n: int = 5,
        max_tokens: int = None,
    ) -> List[Dict]:
        """Get context that combines recent messages + semantically relevant history.

        This provides better context than just keeping the last N messages,
        as it retrieves relevant information from earlier in the conversation.

        Args:
            query: Current user query for semantic matching
            recent_n: Number of recent messages to always include
            relevant_n: Number of semantically relevant messages to retrieve
            max_tokens: Token budget for context (uses self.max_tokens if None)

        Returns:
            List of messages: relevant history first, then recent messages
        """
        budget = max_tokens or self.max_tokens

        # Get recent messages
        recent = self.messages[-recent_n:] if len(self.messages) > recent_n else self.messages[:]

        # Get semantically relevant messages (from earlier in conversation)
        relevant = self.retrieve_relevant_history(query, n_results=relevant_n)

        # Deduplicate: remove relevant messages that are already in recent
        recent_contents = {m.get("content", "")[:100] for m in recent}
        unique_relevant = [
            r for r in relevant
            if r["content"][:100] not in recent_contents
        ]

        # Build combined context within token budget
        context = []
        token_count = 0

        # Add relevant history first (marked)
        for msg in unique_relevant:
            tokens = self.count_tokens(msg["content"])
            if token_count + tokens > budget * 0.3:  # Use at most 30% for relevant history
                break
            context.append({
                "role": msg["role"],
                "content": f"[Earlier context] {msg['content']}"
            })
            token_count += tokens

        # Add recent messages
        for msg in recent:
            tokens = self.count_tokens(msg.get("content", ""))
            if token_count + tokens > budget * 0.8:  # Use at most 80% total
                break
            context.append({"role": msg["role"], "content": msg.get("content", "")})
            token_count += tokens

        return context

    def get_budget(self, total_tokens: int = None) -> ContextBudget:
        """Get context budget allocator using the model's actual context length."""
        tokens = total_tokens or getattr(self, '_model_context_length', None) or self.max_tokens
        return ContextBudget(tokens)
