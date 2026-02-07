# Jarvis Learning Guide (Personal Notes)

This document explains how everything works in Jarvis, module by module. Written for understanding without AI assistance.

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Module: Package Init (`jarvis/__init__.py`)](#module-package-init)
3. [Module: CLI (`jarvis/cli.py`)](#module-cli)
4. [Module: Assistant (`jarvis/assistant.py`)](#module-assistant)
5. [Module: Providers (`jarvis/providers/`)](#module-providers)
6. [Module: Core - Agent (`jarvis/core/agent.py`)](#module-core---agent)
7. [Module: Core - Intent Classifier (`jarvis/core/intent.py`)](#module-core---intent-classifier)
8. [Module: Core - Router (`jarvis/core/router.py`)](#module-core---router)
9. [Module: Core - Context Manager (`jarvis/core/context_manager.py`)](#module-core---context-manager)
10. [Module: Core - Tools (`jarvis/core/tools.py`)](#module-core---tools)
11. [Module: Core - Fact Extractor (`jarvis/core/fact_extractor.py`)](#module-core---fact-extractor)
12. [Module: Skills (`jarvis/skills/`)](#module-skills)
13. [Module: Knowledge / RAG (`jarvis/knowledge/`)](#module-knowledge--rag)
14. [Module: Web UI Backend (`jarvis/ui/app.py`)](#module-web-ui-backend)
15. [Module: Terminal UI (`jarvis/ui/terminal.py`)](#module-terminal-ui)
16. [Module: Voice (`jarvis/voice/`)](#module-voice)
17. [Module: Integrations (`jarvis/integrations/`)](#module-integrations)
18. [Module: Auth (`jarvis/auth/`)](#module-auth)
19. [Module: Frontend (`web/src/`)](#module-frontend)
20. [Module: Frontend - Hooks](#module-frontend---hooks)
21. [Module: Frontend - Components](#module-frontend---components)
22. [How It All Connects](#how-it-all-connects)
23. [Key Patterns](#key-patterns)
24. [Debugging Tips](#debugging-tips)
25. [Glossary](#glossary)

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              JARVIS                                     │
│                                                                         │
│  Interfaces:                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Terminal  │  │  Web UI  │  │ Telegram │  │  Voice   │              │
│  │  (Rich)   │  │ (React)  │  │   Bot    │  │ (Whisper)│              │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘              │
│        │              │              │              │                    │
│        └──────────────┴──────────────┴──────────────┘                   │
│                              │                                          │
│                    ┌─────────▼────────┐                                 │
│                    │   Jarvis Class   │  (assistant.py)                 │
│                    │  Config + State  │                                 │
│                    └─────────┬────────┘                                 │
│                              │                                          │
│            ┌─────────────────┼─────────────────┐                       │
│            │                 │                  │                       │
│   ┌────────▼──────┐  ┌──────▼──────┐  ┌───────▼───────┐              │
│   │    Agent      │  │   Context   │  │     RAG       │              │
│   │ (tool loop)   │  │  Manager    │  │   Engine      │              │
│   └────────┬──────┘  └─────────────┘  └───────────────┘              │
│            │                                                            │
│   ┌────────▼──────────────────────────────────┐                       │
│   │           Intent Classifier               │                       │
│   │  Message → Intent + Reasoning Level       │                       │
│   └────────┬──────────────────────────────────┘                       │
│            │                                                            │
│   ┌────────▼──────┐  ┌───────────────┐  ┌───────────────┐            │
│   │   Provider    │  │    Skills     │  │    Router     │            │
│   │ (Ollama/etc)  │  │  (35+ tools)  │  │ (intent→tool) │            │
│   └───────────────┘  └───────────────┘  └───────────────┘            │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────┐        │
│   │  Storage: SQLite (chats) + ChromaDB/Qdrant (embeddings)  │        │
│   └──────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Flow in simple terms:**
1. User sends a message (via terminal, web UI, Telegram, or voice)
2. The `Jarvis` class receives it and orchestrates everything
3. Intent classifier determines what the user wants (chat, search, code, etc.)
4. Agent decides whether to use tools or just chat
5. If tools needed: Router selects the right tool → Skill executes → Agent loops
6. RAG engine searches knowledge base for relevant context
7. Provider (Ollama/Chutes) generates the response
8. Response streams back to the interface

---

## Module: Package Init

**File:** `jarvis/__init__.py`

**Purpose:** Defines the package, version, and data directory paths.

**Key exports:**
- `__version__` = "0.1.0"
- `PACKAGE_DIR` - Where the jarvis Python package lives
- `PROJECT_ROOT` - Parent of PACKAGE_DIR (the repo root)
- `get_data_dir()` - Returns `~/.jarvis` (or `$JARVIS_DATA_DIR` if set)
- `ensure_data_dir()` - Creates the directory structure:
  ```
  ~/.jarvis/config/personas/
  ~/.jarvis/memory/
  ~/.jarvis/knowledge/documents/
  ~/.jarvis/knowledge/notes/
  ~/.jarvis/logs/
  ```

**Laravel equivalent:** Like the `bootstrap/app.php` file that sets up paths.

---

## Module: CLI

**File:** `jarvis/cli.py`

**Purpose:** Entry point for the `jarvis` command. Uses Click for command-line parsing.

**Key commands:**
```
jarvis                      # Interactive terminal (invoke_without_command=True)
jarvis --dev                # Launch web UI with hot reload
jarvis --voice              # Voice input mode
jarvis --daemon             # Background daemon
jarvis --fast / --deep      # Set reasoning level
jarvis chat "message"       # Single query
jarvis setup                # Setup wizard
jarvis models               # List Ollama models
jarvis personas             # List personas
jarvis knowledge add/list/search/remove/clear/sync
jarvis telegram setup/run/status/webhook/send
jarvis config get/set       # Configuration management
```

**How it works:**
```python
@click.group(invoke_without_command=True)
@click.option('--dev', is_flag=True)
@click.option('--fast', is_flag=True)
@click.pass_context
def main(ctx, dev, fast, ...):
    if dev:
        _launch_dev(port)           # Starts FastAPI + Vite
    elif ctx.invoked_subcommand is None:
        _launch_cli(reasoning_level)  # Interactive terminal
```

**Laravel equivalent:** Like Artisan commands. `@click.group()` = command group, `@click.option()` = `$this->option()`, `@click.argument()` = `$this->argument()`.

---

## Module: Assistant

**File:** `jarvis/assistant.py`

**Purpose:** The main orchestrator. Creates and connects all subsystems.

**Key classes:**

### `ProjectContext`
Detects the project you're working in:
- Walks up from `cwd()` looking for `.git`, `package.json`, `pyproject.toml`, etc.
- Loads `JARVIS.md` or `.jarvis/soul.md` for project-specific instructions
- Detects project type: Python, Node.js, PHP/Laravel, Rust, Go
- Reads git branch name

### `Jarvis`
The main class that ties everything together:
```python
class Jarvis:
    def __init__(self, working_dir=None, reasoning_level=None):
        self.config = load_config()          # From settings.yaml
        self.project = ProjectContext()       # Detect project
        self.provider = get_provider(...)     # Create LLM provider
        self.context = ContextManager(...)    # Chat history
        self.agent = Agent(...)              # Tool calling
        self.rag = get_rag_engine(...)       # Knowledge base
        self.fact_extractor = get_fact_extractor()  # Learn from chats
```

**Key methods:**
- `chat(message)` - Process a message through the full pipeline
- `_get_rag_context(query)` - Search knowledge base
- `_inject_user_context(system_prompt)` - Add facts/preferences
- `_build_system_prompt()` - Combine persona + project + RAG + facts
- `switch_provider(name)` - Change LLM provider at runtime
- `switch_persona(name)` - Change persona
- `switch_model(name)` - Change model

**Laravel equivalent:** Like the main `App` kernel - it bootstraps and connects all services.

---

## Module: Providers

**Directory:** `jarvis/providers/`

**Purpose:** Abstract interface for different AI backends. All providers implement the same API.

### `base.py` - BaseProvider
```python
class BaseProvider(ABC):
    name: str
    supports_streaming: bool
    supports_vision: bool
    supports_tools: bool

    def chat(messages, system) -> str           # Non-streaming
    def stream(messages, system) -> Generator    # Streaming tokens
    def chat_with_tools(messages, system, tools) # Tool calling
    def vision(messages, system, images) -> str  # Image analysis
    def discover_models() -> List[ModelInfo]     # Available models
    def get_model_for_task(task) -> str          # Task-based model selection
```

Also defines:
- `Message(role, content)` - Chat message dataclass
- `ModelInfo(id, name, capabilities, context_length)` - Model metadata

### `ollama.py` - OllamaProvider
**Local LLM provider.** Uses the `ollama` Python package.

Key features:
- `TOOL_CAPABLE_MODELS` - Maps model names to tool reliability:
  ```python
  "qwen3": {"reliability": "high", "format": "native"}
  "deepseek-r1": {"reliability": "low", "format": "prompt"}
  ```
- `VISION_MODELS` - Ordered list of vision-capable models
- `_should_use_native_tools()` - Checks if current model handles tools well
- Auto-detects available models on startup
- Extended timeouts for reasoning models (10 min vs 2 min)

### `ollama_cloud.py` - OllamaCloudProvider
**Remote Ollama server.** Same as local but connects to a URL:
```python
self.base_url = kwargs.get("base_url", "http://localhost:11434")
```

### `chutes.py` - ChutesProvider
**Comprehensive cloud AI platform.** OpenAI-compatible API.

Capabilities:
- **LLM**: Qwen3-32B, DeepSeek-V3, DeepSeek-R1, Qwen2.5-Coder, etc.
- **TTS**: Kokoro (natural), CSM-1B (conversational)
- **STT**: Whisper-large-v3
- **Image**: FLUX.1-schnell, FLUX.1-dev, SDXL, HiDream, JuggernautXL
- **Video**: Wan2.1-14B
- **Music**: DiffRhythm

Task-to-model mapping:
```python
TASK_MODELS = {
    "default": "Qwen/Qwen3-32B",
    "reasoning": "deepseek-ai/DeepSeek-V3",
    "deep": "deepseek-ai/DeepSeek-R1-TEE",
    "code": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "vision": "Qwen/Qwen2.5-VL-72B-Instruct",
    "fast": "unsloth/gemma-3-4b-it",
}
```

**Laravel equivalent:** Providers are like database drivers. `BaseProvider` = `DatabaseManager` interface, each provider = specific driver (MySQL, PostgreSQL, etc.).

---

## Module: Core - Agent

**File:** `jarvis/core/agent.py`

**Purpose:** The agentic loop - decides when and how to use tools.

**How it works:**
```
User Message
    │
    ▼
classify_intent()  ← Uses IntentClassifier
    │
    ├── Needs tools? → Enter tool loop
    │                    │
    │                    ▼
    │               Call LLM with tool definitions
    │                    │
    │                    ▼
    │               LLM returns tool call (or text)
    │                    │
    │                    ├── Tool call → Execute tool → Loop back (max 15 iterations)
    │                    └── Text → Return response
    │
    └── No tools → Direct LLM chat (fast path)
```

**Key methods:**
- `run(messages, system)` - Main entry point, runs the full agentic loop
- `classify_intent(message)` - Classify what the user wants
- `get_reasoning_level(intent)` - Determine fast/balanced/deep
- `_supports_native_tools()` - Check if model does tool calling
- `_execute_tool(name, args)` - Run a specific tool and return result
- `_detect_auto_tool(message)` - Auto-detect tools for time/weather/etc.

**Two tool modes:**
1. **Native** - Provider handles tool calls natively (structured JSON from LLM)
2. **Prompt** - Inject tool descriptions into prompt, parse JSON from response

**Laravel equivalent:** Like a Controller that decides which Service to call based on the request.

---

## Module: Core - Intent Classifier

**File:** `jarvis/core/intent.py`

**Purpose:** Smart routing - determines what the user wants without hardcoded keywords.

### Intent Enum (17 types)
```python
class Intent(Enum):
    CHAT = "chat"           # General conversation
    SEARCH = "search"       # Web search needed
    WEATHER = "weather"     # Weather query
    TIME_DATE = "time_date" # Time/date info
    FILE_OP = "file_op"     # File operations
    CODE = "code"           # Code generation
    GIT = "git"             # Git operations
    SHELL = "shell"         # Shell commands
    CALCULATE = "calculate" # Math
    RECALL = "recall"       # Memory/RAG recall
    CONTROL = "control"     # System control
    VISION = "vision"       # Image analysis
    IMAGE_GEN = "image_gen" # Image generation
    VIDEO_GEN = "video_gen" # Video generation
    MUSIC_GEN = "music_gen" # Music generation
    NEWS = "news"           # Current news
    FINANCE = "finance"     # Financial data
```

### ReasoningLevel Enum
```python
class ReasoningLevel(Enum):
    FAST = "fast"           # Simple queries, greetings
    BALANCED = "balanced"   # Most queries
    DEEP = "deep"           # Complex analysis
```

### ClassifiedIntent Dataclass
```python
@dataclass
class ClassifiedIntent:
    intent: Intent              # What they want
    confidence: float           # How sure we are (0.0-1.0)
    reasoning_level: ReasoningLevel  # How complex
    requires_tools: bool        # Need external data?
    suggested_tools: List[str]  # Which tools to use
```

### IntentClassifier Class
Two modes:
1. **LLM mode** (`intent.llm_enabled: true`) - Sends message to fast LLM, gets JSON classification
2. **Heuristic mode** (default) - Uses regex patterns and keyword matching

The LLM prompt is compact:
```
Classify this user message. Output ONLY valid JSON.
Message: "{message}"
Output: {"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning_level": "<level>", ...}
```

**Why this matters:** Before this, Jarvis had 250+ hardcoded keywords scattered across the codebase. Now one classifier handles all routing intelligently.

---

## Module: Core - Router

**File:** `jarvis/core/router.py`

**Purpose:** Maps intents to tools and extracts parameters from user messages.

### Tool Routing Priority
```
1. Intent Classification (if enabled and confident)
2. Keyword Pattern Matching (fast, reliable fallback)
3. LLM Routing (for complex/ambiguous cases)
4. Default: no tool, just chat
```

### Intent-to-Tool Mapping
```python
INTENT_TO_TOOL = {
    Intent.WEATHER: "get_weather",
    Intent.TIME_DATE: "current_time",
    Intent.CALCULATE: "calculate",
    Intent.SEARCH: "web_search",
    Intent.NEWS: "get_current_news",
    Intent.IMAGE_GEN: "generate_image",
    Intent.VIDEO_GEN: "generate_video",
    Intent.MUSIC_GEN: "generate_music",
    Intent.VISION: "analyze_image",
    ...
}
```

### Parameter Extraction
`extract_params(user_input, tool)` - Extracts params with regex:
- "weather in London" → `{"city": "London"}`
- "search for AI news" → `{"query": "AI news"}`
- "draw a sunset" → `{"prompt": "sunset"}`
- Music with `[MM:SS.ms]` timestamps → extracts lyrics separately

### Helper Functions
```python
should_use_reasoning(input)    # Returns True if deep reasoning needed
should_use_vision(input)       # Returns True if vision model needed
should_generate_image(input)   # Returns True for image gen
should_generate_video(input)   # Returns True for video gen
should_generate_music(input)   # Returns True for music gen
get_reasoning_level(input)     # Returns "fast", "balanced", or "deep"
```

**Laravel equivalent:** Like Route model binding + middleware that determines which controller method to call.

---

## Module: Core - Context Manager

**File:** `jarvis/core/context_manager.py`

**Purpose:** Manages conversation history, auto-compaction, and chat persistence.

### Features
- **Token Counting**: Uses `tiktoken` (cl100k_base encoding) to count tokens
- **Auto-Compaction**: When tokens exceed `max_tokens`, old messages get summarized
- **SQLite Storage**: Chats and messages persisted to `~/.jarvis/memory/jarvis.db`
- **Chat Sessions**: Each conversation has a unique ID and title

### Database Schema
```sql
-- Chats table
CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT 'New Chat',
    created_at TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0
);

-- Messages table
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    timestamp TEXT,
    role TEXT,
    content TEXT,
    model TEXT,
    metadata TEXT,
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
```

### Auto-Compaction Flow
```
Messages exceed max_tokens (128,000 by default)
    │
    ▼
Keep last N messages (default: 10)
    │
    ▼
Summarize older messages with LLM (or simple fallback)
    │
    ▼
Replace old messages with summary
    │
    ▼
Notify user via callback (terminal shows formatted message)
```

### Key Methods
```python
context.add_message(role, content)    # Add to history
context.get_messages()                # Get all messages
context.get_context_stats()           # {tokens_used, max_tokens, percentage, needs_compact}
context.set_provider(provider)        # Enable LLM summarization
context.set_compact_callback(fn)      # Notification when compaction happens
context.new_chat()                    # Start fresh conversation
context.list_chats()                  # All chat sessions
context.load_chat(chat_id)            # Resume previous chat
```

**Laravel equivalent:** Like the Session manager with database driver, plus automatic garbage collection.

---

## Module: Core - Tools

**File:** `jarvis/core/tools.py`

**Purpose:** Defines all 35+ tool functions that the LLM can call. These are the actual implementations.

### Tool Categories

**File Operations:**
- `read_file(path)` - Read file contents (with tracking to prevent re-reads)
- `write_file(path, content)` - Write to file (with confirmation via UI)
- `edit_file(path, old_string, new_string)` - Search and replace
- `list_files(path)` - List directory contents
- `glob_files(pattern)` - Find files by glob pattern
- `grep(pattern, path)` - Search content with regex
- `search_files(pattern, path)` - Text search in files
- `get_project_structure(path)` - Directory tree

**Git Operations:**
- `git_status()`, `git_diff()`, `git_log()`, `git_add()`, `git_commit()`, `git_branch()`, `git_stash()`

**Web & Information:**
- `web_search(query)` - DuckDuckGo/Brave search
- `web_fetch(url)` - Fetch and parse URL content
- `get_current_news(topic)` - News search
- `get_weather(city)` - Weather lookup
- `get_current_time(timezone)` - Time info
- `get_gold_price(metal)` - Gold/silver prices
- `calculate(expression)` - Math evaluation

**Memory:**
- `save_memory(key, content)` - Store in working memory
- `recall_memory(query)` - Search memory
- `github_search(query)` - GitHub code/repo search

**Tasks:**
- `task_create(title, description)`, `task_update(id, status)`, `task_list()`, `task_get(id)`

**Code Intelligence (Agent Coding Skills):**
- `apply_patch(file_path, patch)` - Apply unified diff patch to a file
- `find_definition(symbol, file_path?)` - Find where a function/class is defined (Python, JS, TS, Rust, Go)
- `find_references(symbol, file_path?)` - Find all usages of a symbol across the codebase
- `run_tests(test_path?, framework?)` - Auto-detect and run tests (pytest, jest, vitest, cargo, go test)
- `get_project_overview()` - Rich overview with tech stack detection, key files, directory tree, git history

### Tool Schema (`ALL_TOOLS`)
Each tool is defined as a dict matching OpenAI's function calling format:
```python
ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents...",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "..."}
                },
                "required": ["path"]
            }
        }
    },
    ...
]
```

This schema is sent to the LLM so it knows what tools are available.

**Laravel equivalent:** Like the `Action` classes or Service methods that actually do work.

---

## Module: Core - Fact Extractor

**File:** `jarvis/core/fact_extractor.py`

**Purpose:** Learns facts about the user from conversations.

### How It Works
1. After a conversation, extract the last 10 messages
2. Send to LLM with a fact extraction prompt
3. LLM returns facts like: "User is a software engineer", "User prefers TypeScript"
4. Deduplicate against existing facts
5. Save to `~/.jarvis/memory/facts.md`

### Fact Storage Format
```markdown
# Known Facts
- User is a full-stack developer
- User prefers Laravel for backend
- User lives in [city]
- User's name is [name]
```

These facts get injected into the system prompt for personalization.

---

## Module: Skills

**Directory:** `jarvis/skills/`

**Purpose:** Individual tool implementations organized by domain.

### `web_search.py`
- `web_search(query, max_results=5)` - DuckDuckGo search using `ddgs` library
- `get_current_news(topic)` - DuckDuckGo news search with fallback to regular search
- No API key needed (uses DuckDuckGo)

### `file_ops.py`
- `read_file(path)` - Read with encoding detection
- `list_directory(path)` - List with file types and sizes
- `write_file(path, content, create_dirs)` - Write with optional dir creation
- `edit_file(path, old_string, new_string)` - Find and replace
- `search_files(pattern, path, file_pattern)` - Recursive text search

### `shell.py`
- `shell_run(command)` - Execute shell commands
- `is_safe_command(command)` - Check against blocked commands list
- Respects `safety.blocked_commands` from settings.yaml

### `weather.py`
- `get_weather(city)` - Current weather via wttr.in (free, no key) or OpenWeather
- `get_forecast(city)` - Multi-day forecast

### `calculator.py`
- `calculate(expression)` - Safe math evaluation (no `exec`/`eval` of arbitrary code)
- `convert_units(value, from_unit, to_unit)` - Unit conversion
- `percentage(value, total)` - Percentage calculation

### `datetime_ops.py`
- `get_current_time(timezone)` - Time in any timezone
- `convert_timezone(time, from_tz, to_tz)` - Timezone conversion
- `add_time(base_time, amount, unit)` - Time arithmetic
- `time_until(target_datetime)` - Countdown

### `memory_ops.py`
- `save_fact(fact)` - Save to `~/.jarvis/memory/facts.md`
- `get_facts()` - Read all saved facts

### `github_ops.py`
- `list_repos(username)` - List user's repositories
- `repo_info(owner, repo)` - Repository details
- `list_issues(owner, repo)` - Open issues
- `create_issue(owner, repo, title, body)` - Create issue
- `list_prs(owner, repo)` - Pull requests

### `notes.py`
- `quick_note(content)` - Save quick note with timestamp
- `list_notes()` - List all notes
- `read_note(filename)` - Read specific note
- `search_notes(query)` - Search note contents

### `telegram.py`
- `send_message(chat_id, text)` - Send Telegram message
- `get_updates(limit)` - Get recent messages
- `send_photo(chat_id, photo_path)` - Send image
- `send_video(chat_id, video_path)` - Send video
- `send_audio(chat_id, audio_path)` - Send audio
- `send_media(chat_id, file_path)` - Auto-detect and send media

### `media_gen.py`
Media generation via Chutes AI:
- `generate_image(prompt, model, width, height, steps)` - Image gen (FLUX.1-schnell default)
- `generate_video(prompt, image_path, duration)` - Video gen (Wan2.1-14B)
- `generate_music(prompt, duration, lyrics)` - Music gen (DiffRhythm)
- `analyze_image(image_path, prompt, provider)` - Vision analysis (Ollama first, Chutes fallback)
- `analyze_image_ollama(image_path, prompt)` - Local Ollama vision
- `analyze_document(content, prompt)` - Document analysis
- `list_media_models()` - Available models
- `cleanup_generated_files()` - Remove old generated files
- `_auto_cleanup(max_age_hours=48, max_files=100)` - Automatic cleanup

### `multi_model_analysis.py`
- `multi_model_analyze(query, profile)` - Run query through multiple models
- `analyze_parallel(query, model_types)` - Parallel analysis with ThreadPoolExecutor
- `list_analysis_profiles()` - Available profiles: comprehensive, quick, technical, reasoning

### `skill_creator.py`
Meta-skill that creates other skills:
- `create_skill(name, description, code, parameters)` - Create new skill file
- `update_skill(name, code)` - Update existing skill
- `delete_skill(name)` - Delete skill file
- `list_user_skills()` - List custom skills
- `get_skill_code(name)` - Read skill source
- `get_skill_template()` - Template for new skills

Skills are saved to `~/.jarvis/skills/` and auto-loaded.

---

## Module: Knowledge / RAG

**File:** `jarvis/knowledge/rag.py`

**Purpose:** Retrieval Augmented Generation - search your documents by meaning.

### Architecture
```
Document Ingestion:                    Query Time:

PDF/TXT/MD                            "What are my skills?"
    │                                       │
    ▼                                       ▼
Split into chunks                      Embed query
(500 words, 50 overlap)               (nomic-embed-text)
    │                                       │
    ▼                                       ▼
Embed each chunk                       Vector search
(nomic-embed-text)                     (ChromaDB/Qdrant)
    │                                  Top 20 candidates
    ▼                                       │
Store in vector DB                          ▼
(ChromaDB local                        Cross-encoder rerank
 or Qdrant cloud)                      (ms-marco-MiniLM-L-6-v2)
                                       Top 5 relevant
                                            │
                                            ▼
                                       Inject into prompt
                                       LLM generates answer
```

### Two-Stage Retrieval
1. **Bi-encoder** (fast, approximate): Embeds query and docs separately, compares vectors
2. **Cross-encoder** (slow, accurate): Reads query + doc together, scores relevance

### RAGEngine Class
```python
class RAGEngine:
    def __init__(self, persist_dir, embedding_model="nomic-embed-text"):
        # Auto-detect backend: Qdrant if configured, else ChromaDB
        if QDRANT_URL and QDRANT_API_KEY:
            self.backend = "qdrant"
        else:
            self.backend = "chromadb"

    def add_file(path)           # Ingest a document
    def add_directory(path)      # Ingest all docs in directory
    def search(query, n_results) # Semantic search
    def get_context(query)       # Search + format for prompt injection
    def remove_source(source)    # Remove a document
    def count()                  # Total chunks stored
    def list_sources()           # All indexed documents
```

### Reranker Class
```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)
        return sorted(documents, key=score, reverse=True)[:top_k]
```

### Configuration
```yaml
memory:
  vector_store: knowledge/chroma_db
  rerank: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  relevance_threshold: 0.5
  embedding_dim: 768
```

**Laravel equivalent:** Like Laravel Scout but with semantic search instead of keyword search. ChromaDB/Qdrant = like Algolia/Meilisearch but for vectors.

---

## Module: Web UI Backend

**File:** `jarvis/ui/app.py`

**Purpose:** FastAPI server that serves the web UI and handles WebSocket communication.

### Key Components

**WebUI class** - Minimal adapter for web context:
```python
class WebUI:
    def print_tool(msg, success)    # Record tool execution
    def print_error(msg)            # Error messages
    def stream_text(text)           # Send streaming tokens
    def record_tool(name, display, duration_s, args, result)  # Tool timeline
    def confirm(msg) -> True        # Auto-confirm in web (no prompts)
```

### WebSocket Protocol
Messages are JSON with a `type` field:

**Client → Server:**
```json
{
    "type": "message",
    "content": "Hello",
    "reasoning_level": "balanced",
    "attachments": [{"name": "photo.jpg", "data": "base64..."}]
}
```

**Server → Client:**
```json
// Streaming text
{"type": "stream", "content": "Hello"}

// Response complete
{"type": "response", "content": "Full response", "done": true, "context": {...}}

// Tool execution
{"type": "tool", "name": "web_search", "display": "Searching...", "duration_s": 1.2}

// Intent detected
{"type": "intent", "intent": "search", "confidence": 0.95, "reasoning_level": "balanced"}

// Media generated
{"type": "media", "media_type": "image", "path": "/path/to/image.png", "url": "/generated/image.png"}

// RAG info
{"type": "rag", "enabled": true, "chunks": 3, "sources": ["resume.pdf"]}

// Context stats
{"type": "context", "tokens_used": 1500, "max_tokens": 128000, "percentage": 1.2}
```

### REST Endpoints
- `GET /api/settings` - Get current settings (provider, model, voices)
- `POST /api/settings` - Update settings
- `GET /api/models` - List available models
- `GET /api/providers` - List providers
- `GET /api/tts` - Text-to-speech
- `POST /api/stt` - Speech-to-text
- `GET /api/weather` - Weather data (Open-Meteo)
- `GET /api/status` - System status (psutil)
- `GET /api/chats` - List chat sessions
- `GET /api/generated/{filename}` - Serve generated media files
- `POST /api/upload` - File upload

**Laravel equivalent:** Like a Laravel WebSocket server (Reverb) combined with API controllers.

---

## Module: Terminal UI

**File:** `jarvis/ui/terminal.py`

**Purpose:** Rich-based terminal interface with gradient banner, spinners, and formatted output.

### Features
- ASCII art JARVIS banner with gradient colors (cyan → blue → purple)
- Rich panels, tables, and formatted text
- Spinner during LLM thinking
- Tool execution display with timing
- Streaming text output
- ESC key detection for stopping generation
- Context percentage in toolbar when > 50%
- `/level` command display

### Key Class: `TerminalUI`
```python
class TerminalUI:
    def print_header()          # Show banner + model info
    def print_tool(msg)         # Show tool execution
    def print_error(msg)        # Red error panel
    def show_spinner(msg)       # Animated thinking indicator
    def stream_text(text)       # Print streaming chunks
    def stream_done()           # Finalize stream
    def confirm(msg) -> bool    # Ask user yes/no
    def record_tool(name, display, duration_s, ...)  # Tool timeline entry
```

---

## Module: Voice

**File:** `jarvis/voice/voice_mode.py`

**Purpose:** Voice input mode using Whisper for speech recognition.

### Flow
```
Press Enter → Record 5 seconds audio → Whisper transcribes → Jarvis processes → System TTS speaks
```

### Dependencies
- `openai-whisper` - Speech-to-text (local)
- `sounddevice` - Audio recording
- `numpy` - Audio processing

The web UI has a more sophisticated voice system with multiple TTS/STT providers (see Frontend hooks).

---

## Module: Integrations

**File:** `jarvis/integrations/telegram_bot.py`

**Purpose:** Full Telegram bot that integrates with Jarvis.

### TelegramBot Class
```python
class TelegramBot:
    def __init__(self, token, message_handler, allowed_users):
        # token: Bot token from @BotFather
        # message_handler: async fn(user_id, username, text) -> response
        # allowed_users: Access control list

    def start()         # Start polling for messages
    def stop()          # Stop the bot
    def send_message()  # Send to a chat
    def set_webhook()   # Production webhook mode
```

### 20+ Commands
Handles: `/help`, `/model`, `/models`, `/provider`, `/providers`, `/clear`, `/export`, `/mode`, `/search`, `/knowledge`, `/remember`, `/facts`, `/status`, `/weather`, `/time`, `/settings`, `/persona`, `/personas`, `/name`, `/id`

### Access Control
```bash
# Allow specific users only
TELEGRAM_ALLOWED_USERS=12345678,87654321
```

---

## Module: Auth

**Directory:** `jarvis/auth/`

**Purpose:** Credential management for providers.

### `credentials.py`
- `get_credential(provider, key)` - Get API key from config or env
- `set_credential(provider, key, value)` - Store credential
- Reads from `settings.yaml` first, falls back to environment variables

### `claude.py` / `codex.py`
- Provider-specific auth helpers for Claude and Codex APIs

---

## Module: Frontend

**Directory:** `web/src/`

**Purpose:** React 19 + TypeScript + Tailwind CSS v4 single-page application.

### `App.tsx` - Main Application
State management:
- `mode`: 'chat' | 'voice'
- `reasoningLevel`: 'fast' | 'balanced' | 'deep' | null (auto)
- `voiceOutput`: boolean
- `showSettings`: boolean
- Provider/model/voice configuration

Key UI elements:
- Header with assistant name, reasoning level buttons (Zap/Scale/Brain)
- Context window bar (color-coded by usage percentage)
- Message list with streaming
- Unified input with file upload
- Settings panel (slide-in)
- Animated Orb for voice mode

### `types/index.ts` - TypeScript Interfaces
All shared types:
```typescript
interface Message { role, content, timestamp, tools?, media?, thinking?, thinkingDuration? }
interface ToolEvent { name, display, duration_s, id?, args?, result_preview?, success? }
interface MediaInfo { type: 'image'|'video'|'audio', path, filename, url? }
interface RagInfo { enabled, chunks, total_chunks?, sources, error? }
interface IntentInfo { detected, intent?, confidence?, reasoning_level?, requires_tools? }
interface ContextStats { tokens_used, max_tokens, percentage, messages, needs_compact }
type TTSProvider = 'browser' | 'edge' | 'elevenlabs' | 'kokoro'
type STTProvider = 'browser' | 'whisper' | 'chutes'
type ReasoningLevel = 'fast' | 'balanced' | 'deep' | null
```

---

## Module: Frontend - Hooks

### `useWebSocket.ts`
The central communication hook. Manages:
- WebSocket connection to `ws://localhost:7777/ws/chat`
- Message sending with attachments and reasoning level
- Stream parsing (accumulates chunks, detects `<think>` blocks)
- Tool event collection
- Media message handling
- RAG info tracking
- Context stats updates
- Auto-reconnection

**Thinking block parsing:**
```typescript
// Detects <think>...</think> tags across streaming chunks
if (chunk.includes('<think>')) isInsideThinkingRef.current = true
if (isInsideThinkingRef.current) streamingThinking += chunk
if (chunk.includes('</think>')) isInsideThinkingRef.current = false
```

### `useVoice.ts`
Voice I/O with multiple providers:
- **STT**: Browser SpeechRecognition API, or POST to `/api/stt` (Whisper/Chutes)
- **TTS**: Browser SpeechSynthesis, or fetch from `/api/tts` (Edge/ElevenLabs/Kokoro)
- Volume level tracking for Orb animation

### `useFileUpload.ts`
File attachment management:
- Drag-and-drop support
- File type detection (image, video, audio, document)
- Base64 encoding for upload
- Preview generation for images
- Size validation

### `useWakeWord.ts`
Wake word detection:
- Continuously listens for "Jarvis" (configurable)
- Uses Browser SpeechRecognition in continuous mode
- Triggers voice input when detected

### `useCamera.ts`
Camera access for voice+video chat:
- `startCamera()` - Request camera via `getUserMedia` (640x480, user-facing)
- `stopCamera()` - Stop all media tracks and cleanup
- `captureFrame()` - Capture current frame to canvas (320x240 JPEG, quality 0.6) → base64
- `videoRef` - Attach to `<video>` element for live preview
- Auto-cleanup on unmount

### `useSettings.ts`
Settings management:
- Fetch/update settings via REST API
- Provider and model selection
- Voice configuration

---

## Module: Frontend - Components

### Chat Components (`components/chat/`)

**`MessageBubble.tsx`** - Renders individual messages:
- User messages: right-aligned, cyan header
- Assistant messages: left-aligned, emerald header, Markdown rendering
- Thinking blocks (collapsible, purple)
- Media preview (images, videos, audio with download)
- Tool timeline (expandable, shows each tool call)
- URL extraction from text

**`MessageList.tsx`** - Scrollable message container:
- Auto-scroll to bottom on new messages
- Streaming message display with cursor

**`ThinkingBlock.tsx`** - Collapsible reasoning display:
- Purple themed with Brain icon
- Shows word count and duration when complete
- Animated dots while streaming
- Expandable/collapsible content

### Input Components (`components/input/`)

**`UnifiedInput.tsx`** - Main chat input:
- Text input with Enter to send
- File attachment button
- Voice toggle button
- Send button

**`FileUploadZone.tsx`** - Drag-and-drop file upload area
**`FilePreview.tsx`** - Preview attached files before sending

### Settings Components (`components/settings/`)

**`SettingsPanel.tsx`** - Slide-in panel with:
- Provider selection (dropdown)
- Model selection (dropdown, filtered by provider)
- Voice settings (TTS provider, voice selection, STT provider)
- System instructions editor

**`SystemInstructions.tsx`** - Edit system prompt

### Orb Components (`components/orb/`)

**`Orb.tsx`** - Animated visual indicator:
- Idle: gentle pulse
- Listening: reactive to microphone volume
- Speaking: reactive to playback volume
- Thinking: rotation animation

**`OrbRings.tsx`** - Visual ring effects around the orb

### Widget Components (`components/widgets/`)

**`WeatherWidget.tsx`** - Current weather display (from `/api/weather`)
**`TimeWidget.tsx`** - Current time with timezone
**`SystemStatusWidget.tsx`** - CPU, RAM, disk usage (from `/api/status`)
**`VoiceControlWidget.tsx`** - Voice settings quick access
**`QuickCommandsWidget.tsx`** - Common command buttons
**`UserProfileWidget.tsx`** - User info display
**`MultiModelAnalysisWidget.tsx`** - Multi-model analysis trigger

### Dashboard (`components/dashboard/`)

**`DashboardView.tsx`** - Grid layout for widgets
**`WidgetCard.tsx`** - Container for individual widgets

---

## How It All Connects

### Request Flow: Web UI → Response

```
1. User types "What's the weather in London?" in React app

2. useWebSocket sends: {"type": "message", "content": "What's the weather in London?"}

3. app.py websocket_chat() receives message

4. Jarvis.chat() is called:
   a. Intent classifier → Intent.WEATHER, confidence: 0.95
   b. Send intent info to frontend: {"type": "intent", "intent": "weather", ...}
   c. RAG engine searches knowledge base (probably no weather docs)
   d. Agent enters tool loop
   e. Router maps WEATHER → get_weather tool
   f. get_weather("London") calls wttr.in API
   g. Tool result: "London: 8°C, Cloudy"
   h. Send tool event: {"type": "tool", "name": "get_weather", "duration_s": 0.3}
   i. LLM generates response using tool result
   j. Stream tokens to frontend: {"type": "stream", "content": "The weather..."}

5. useWebSocket accumulates chunks, updates messages state

6. MessageBubble renders the response with tool timeline
```

### Request Flow: Terminal → Response

```
1. User types message in Rich terminal

2. Jarvis.chat() is called (same as web)

3. TerminalUI shows spinner while thinking

4. Tool calls displayed with Rich formatting

5. Response streamed directly to terminal
```

---

## Key Patterns

### Singleton Pattern
```python
# Used for RAG engine, fact extractor
_rag_engine = None
def get_rag_engine(config=None):
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(...)
    return _rag_engine
```

### Provider Abstraction
All providers implement `BaseProvider`. Switch at runtime:
```python
provider = get_provider("chutes", api_key=key, model="Qwen/Qwen3-32B")
```

### Lazy Imports
Heavy dependencies loaded only when needed:
```python
def _get_media_tools():
    """Lazy import of media tools."""
    from jarvis.skills.media_gen import generate_image, generate_video, ...
```

### Configuration Cascade
```
1. Command-line flags (--fast, --deep)
2. Session overrides (/level deep)
3. settings.yaml configuration
4. Environment variables (.env)
5. Hardcoded defaults
```

### Error Fallback Chain
```
Intent Classification: LLM → Heuristic patterns → Default (chat)
Provider: Ollama → Chutes → Error message
Tool Calling: Native → Prompt-based → No tools
RAG: Qdrant → ChromaDB → No context
TTS: ElevenLabs → Edge → Kokoro → Browser
```

---

## Debugging Tips

### See what intent was classified
```python
# In agent.py, after classify_intent():
print(f"Intent: {intent.intent.value}, Confidence: {intent.confidence}")
```

### Check RAG context
```bash
jarvis knowledge search "your query"
jarvis knowledge list
```

### Monitor WebSocket messages
Open browser DevTools → Network → WS → click the connection → Messages tab

### Check context usage
```bash
# In terminal
/context

# In web UI: look at the color bar below the header
```

### Test provider connectivity
```bash
# Ollama
curl http://localhost:11434/api/tags

# Chutes
curl -H "Authorization: Bearer $CHUTES_API_KEY" https://llm.chutes.ai/v1/models
```

### Enable debug logging
```yaml
# In settings.yaml
logging:
  level: "DEBUG"
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **LLM** | Large Language Model - the AI that generates text |
| **RAG** | Retrieval Augmented Generation - find docs before generating |
| **Embedding** | A list of numbers (vector) representing text meaning |
| **Vector DB** | Database that stores and searches vectors (ChromaDB, Qdrant) |
| **Chunk** | A piece of a document (~500 words) |
| **Context** | Information provided to the LLM alongside your question |
| **Token** | A word or subword - LLMs process tokens, not characters |
| **Ollama** | Tool that runs LLMs locally on your machine |
| **Chutes** | Cloud AI platform with LLM, image, video, music, TTS, STT |
| **Intent** | What the user wants to do (chat, search, generate, etc.) |
| **Reasoning Level** | How complex the response should be (fast/balanced/deep) |
| **Reranking** | Second-pass scoring of search results for accuracy |
| **Cross-encoder** | Model that reads query + document together for precise scoring |
| **Bi-encoder** | Model that encodes query and document separately (faster) |
| **Semantic Search** | Finding content by meaning, not just keywords |
| **Persona** | Personality/behavior definition for the assistant |
| **Provider** | Backend that generates AI responses (Ollama, Chutes, etc.) |
| **Tool Calling** | LLM's ability to invoke functions (search, weather, etc.) |
| **WebSocket** | Persistent connection for real-time bidirectional communication |
| **Streaming** | Sending response tokens one at a time as they're generated |

---

*Last updated: February 2026*
*This file is gitignored - your personal learning notes*
