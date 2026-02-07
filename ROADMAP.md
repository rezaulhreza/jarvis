# Jarvis Roadmap

## Vision
Transform Jarvis into a fully autonomous AI life management platform with:
- **Multi-model orchestration** - Auto-select best model per task with handover documents
- **Multi-user support** - Customizable for any user, not hardcoded
- **Enterprise security** - Authentication, encryption, audit logs
- **Real integrations** - Calendar, Email, Telegram, WhatsApp, IoT

---

## Completed

### Phase 1: Foundation (2026-02-04)

#### Core Backend
- [x] Click-based CLI with interactive mode, web UI, and voice mode
- [x] FastAPI WebSocket server for real-time communication
- [x] Rich terminal UI with gradient banner and spinners
- [x] SQLite-backed conversation history with chat sessions
- [x] YAML-based configuration system (`~/.jarvis/config/settings.yaml`)
- [x] Project context detection (JARVIS.md, CLAUDE.md, .jarvis/soul.md)
- [x] Auto-detects project type (Python, Node.js, PHP, Rust, Go)
- [x] Safety rules with destructive command confirmation

#### Provider System
- [x] BaseProvider abstract class with streaming, vision, and tool calling
- [x] OllamaProvider - local LLM with TOOL_CAPABLE_MODELS mapping
- [x] OllamaCloudProvider - remote Ollama server support
- [x] ChutesProvider - comprehensive cloud AI (LLM, TTS, STT, Image, Video, Music)
- [x] Dynamic model discovery and task-to-model mapping
- [x] Provider switching at runtime (`/provider chutes`)

#### Tool System (35+ tools)
- [x] File operations: read, write, edit, list, glob, grep, search, project structure
- [x] Git operations: status, diff, log, add, commit, branch, stash
- [x] Web: search (DuckDuckGo/Brave), fetch, news, weather, time, gold prices
- [x] Task management: create, update, list, get
- [x] Utilities: calculate, shell commands, save/recall memory, GitHub search
- [x] Native tool calling support with prompt-based fallback
- [x] Tool validation for parameter checking

#### RAG System
- [x] ChromaDB (local) and Qdrant (cloud) backends
- [x] Document ingestion (PDF, TXT, MD) with chunking
- [x] Semantic search with nomic-embed-text embeddings
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- [x] RAG integrated into both terminal and web chat loops
- [x] Knowledge CLI commands: add, list, search, remove, clear, sync
- [x] Personal knowledge directory (`~/.jarvis/knowledge/personal/`)

#### Web UI (React)
- [x] React 19 + TypeScript + Tailwind CSS v4 + Vite
- [x] WebSocket-based real-time streaming
- [x] Message bubbles with Markdown rendering (react-markdown + remark-gfm)
- [x] Tool timeline showing execution steps, duration, results
- [x] Provider and model switching in settings panel
- [x] System instructions editor
- [x] Animated orb with idle/listening/speaking/thinking states
- [x] Dashboard with weather, time, system status, quick commands widgets
- [x] File drag-and-drop upload for vision analysis

#### Telegram Integration
- [x] Full TelegramBot class with message handler
- [x] 20+ commands: /model, /models, /provider, /search, /knowledge, etc.
- [x] Per-user conversation history (SQLite)
- [x] Access control (TELEGRAM_ALLOWED_USERS)
- [x] Webhook support for production
- [x] Clean responses (stripped Rich markup)
- [x] Configurable assistant name

### Phase 2: Architecture Redesign (2026-02-04)

#### Intent Classification System
- [x] `IntentClassifier` with LLM-based classification + heuristic fallback
- [x] 17 Intent types: CHAT, SEARCH, WEATHER, TIME_DATE, FILE_OP, CODE, GIT, SHELL, CALCULATE, RECALL, CONTROL, VISION, IMAGE_GEN, VIDEO_GEN, MUSIC_GEN, NEWS, FINANCE
- [x] 3 Reasoning levels: FAST, BALANCED, DEEP
- [x] `ClassifiedIntent` dataclass with confidence scores
- [x] Configurable confidence thresholds and reasoning overrides

#### Unified Smart Mode
- [x] Auto-detects fast chat vs full agent based on intent
- [x] Reasoning level selector in web UI (Fast/Auto/Deep)
- [x] Intent info sent to frontend via WebSocket
- [x] CLI flags: `--fast`, `--deep`, `--level <fast|balanced|deep>`
- [x] Interactive `/level` command for session-level control

#### Context Management
- [x] Token counting with tiktoken (cl100k_base)
- [x] Auto-compaction with LLM-powered summarization
- [x] Context window tracking (tokens used, percentage, needs_compact)
- [x] Web UI context bar (color-coded: blue/yellow/red)
- [x] Terminal context display in toolbar when > 50%

#### Fact Extraction
- [x] LLM-based fact extraction from conversations
- [x] Persistent storage in `~/.jarvis/memory/facts.md`
- [x] Entity tracking in `~/.jarvis/memory/entities.json`
- [x] Facts injected into system prompt for personalization

### Phase 3: Multimodal Support (2026-02-05)

#### Media Generation via Chutes AI
- [x] Image generation (FLUX.1-schnell, FLUX.1-dev, SDXL, HiDream, JuggernautXL)
- [x] Video generation (Wan2.1-14B) with duration control and quality options
- [x] Music generation (DiffRhythm) with lyrics support (timestamped format)
- [x] Auto-cleanup of generated files (48h max age, 100 file limit)

#### Vision Analysis
- [x] Ollama vision models: llama3.2-vision, llava-llama3, minicpm-v, llava, moondream
- [x] Auto-detect best available vision model
- [x] Chutes cloud vision fallback (Qwen2.5-VL-72B-Instruct)
- [x] Image attachment triggers auto vision analysis

#### Thinking Block Support (2026-02-05)
- [x] Collapsible ThinkingBlock component (purple, Brain icon, word count, duration)
- [x] `<think>` tag parsing across streaming chunks
- [x] Separate accumulation: thinking vs response content
- [x] Works with DeepSeek-R1, QwQ, and other thinking models

#### Multi-Model Analysis
- [x] Run queries through multiple AI models simultaneously
- [x] Analysis profiles: comprehensive, quick, technical, reasoning
- [x] Thread pool parallel execution
- [x] Combined insight aggregation

#### Dynamic Skill Creation
- [x] `skill_creator.py` - Create skills at runtime
- [x] Save to `~/.jarvis/skills/` directory
- [x] CRUD operations: create, update, delete, list, get code

#### Voice System
- [x] Multiple TTS providers: Browser, Edge TTS, ElevenLabs, Kokoro (Chutes)
- [x] Multiple STT providers: Browser, Whisper (local), Chutes (cloud)
- [x] Wake word detection (configurable, default: "jarvis")
- [x] Push-to-talk mode in web UI
- [x] Voice control widget in dashboard

### Phase 4: Performance & Architecture Overhaul (2026-02-07)

#### Async Parallel Tool Execution
- [x] `AsyncToolExecutor` with `asyncio.gather()` and semaphore-based concurrency
- [x] Wraps synchronous tools via `asyncio.to_thread()` for non-blocking execution
- [x] `run_async()` async generator on Agent yielding progressive `AgentEvent` objects
- [x] Live tool status streaming to frontend via WebSocket (`tool_start`, `tool_complete`)

#### Dynamic Tool Selection
- [x] `TOOL_REGISTRY` mapping tools to categories, intents, and keywords
- [x] `select_tools()` sends 3-8 relevant tools per query instead of all 30+
- [x] Intent-based scoring (3.0 for intent match, 1.0 per keyword) with category sibling inclusion

#### Smart Context Management
- [x] `ContextBudget` with proportional allocation (scales to any model: 4K to 1M+)
- [x] Dynamic context length detection via `get_context_length()` on providers
- [x] Ollama provider queries actual model metadata (`{arch}.context_length`)
- [x] `embed_message()` auto-embeds all messages into ChromaDB
- [x] `get_enriched_context()` combines recent + semantically relevant past messages
- [x] Concurrent RAG search alongside tool detection

#### Multi-Agent Orchestrator
- [x] `AgentOrchestrator` decomposes complex tasks via LLM into parallel subtasks
- [x] `SubAgent` - lightweight agent with limited tools and own context
- [x] Dependency-aware execution (respects `depends_on` between subtasks)
- [x] Orchestrator events forwarded to WebSocket for real-time progress
- [x] Auto-detection via `should_orchestrate()` with complexity pattern matching

#### Document Processing Pipeline
- [x] `DocumentProcessor` for PDF, DOCX, XLSX, CSV, JSON, TXT
- [x] Paragraph-boundary-aware chunking with configurable overlap
- [x] Small docs (<15KB): direct context injection; large docs: chunk→embed→retrieve
- [x] Replaces naive 15KB truncation in `analyze_document()`

#### Frontend Performance
- [x] `ToolStatus` component with live spinners, completion checkmarks, durations
- [x] `React.memo()` on `MessageBubble` to prevent unnecessary re-renders
- [x] WebSocket hook handles `tool_status` events and clears on completion

### Phase 5: Developer Experience & Personalization (2026-02-07)

#### Customizable Assistant Identity
- [x] YAML frontmatter parsing in JARVIS.md for per-project assistant name
- [x] Priority chain: project JARVIS.md > user config > default "Jarvis"
- [x] System prompt dynamically uses configured name
- [x] Frontend receives assistant name via WebSocket connection message
- [x] `/init` command generates JARVIS.md template with name frontmatter

#### Model Switch Context Preservation
- [x] `context.set_provider()` called on model switch to update token budget
- [x] `context.set_provider()` called on provider switch
- [x] System prompt rebuilt after model/provider changes
- [x] Context stats included in model/provider change WebSocket responses

#### Auto-Memory (Fact Extraction)
- [x] Async fact extraction in web UI via `asyncio.create_task()`
- [x] Throttled to every 5 messages per session for efficiency
- [x] Runs in background thread via `asyncio.to_thread()` (non-blocking)
- [x] Integrated in all 3 response paths: fast chat, orchestrator, agent mode

#### Memory Management UI
- [x] REST API: GET/POST/PUT/DELETE `/api/memories` with facts.md parsing
- [x] `MemoryPanel.tsx` with full CRUD, search, inline editing
- [x] Category-colored badges (Job, Tech, Preference, Location, etc.)
- [x] Memory tab added to SettingsPanel

#### Agent Coding Skills (5 new tools)
- [x] `apply_patch` - Apply unified diff patches to files
- [x] `find_definition` - Find function/class definitions (Python, JS, TS, Rust, Go)
- [x] `find_references` - Find all symbol usages with word-boundary matching
- [x] `run_tests` - Auto-detect and run pytest, jest, vitest, cargo, go test
- [x] `get_project_overview` - Rich project overview with tech stack and git history
- [x] New `code` category group in TOOL_REGISTRY for code intelligence
- [x] All tools registered in agent tool maps and validation schemas

#### Voice + Video Chat
- [x] `useCamera` hook for browser camera access and frame capture (320x240 JPEG)
- [x] `sendWithVideo` WebSocket method for voice+video messages
- [x] Camera toggle and picture-in-picture video preview in voice mode
- [x] Backend `voice_with_video` handler: decode frame → vision analysis → respond
- [x] Falls back to text-only if vision fails; auto-cleanup of temp frames

---

## In Progress

### Phase 6: Polish & Stability

#### UI Improvements
- [ ] Chat history sidebar with search, edit, and auto-titles
- [ ] Responsive design alignment (desktop ↔ mobile)
- [ ] Keyboard shortcuts for common actions
- [ ] Message editing and regeneration
- [ ] Export conversations (Markdown, JSON)

#### Backend Improvements
- [ ] Anthropic provider implementation
- [ ] OpenAI provider implementation
- [ ] Gemini provider implementation
- [ ] Better error recovery and retry logic
- [ ] Connection health monitoring

#### Performance
- [ ] Lazy loading for heavy dependencies (sentence-transformers, chromadb)
- [ ] WebSocket connection pooling
- [ ] Caching for repeated intent classifications
- [ ] Batch embedding for document ingestion

---

## Planned

### Phase 7: Multi-User Support

#### Authentication
- [ ] User registration/login
- [ ] JWT token authentication
- [ ] Role-based access control (admin, user, viewer)
- [ ] Session management with refresh tokens

#### Per-User Isolation
- [ ] Per-user settings storage
- [ ] Per-user conversation history
- [ ] Per-user knowledge base
- [ ] Per-user integrations and API keys

#### Security
- [ ] API key encryption at rest
- [ ] Rate limiting per user
- [ ] Audit logging
- [ ] Input sanitization
- [ ] HTTPS enforcement

### Phase 8: Real Integrations

#### Calendar
- [ ] Google Calendar API
- [ ] Apple Calendar integration (CalDAV)
- [ ] Event creation/modification via chat
- [ ] Daily agenda briefing

#### Email
- [ ] IMAP/SMTP connection
- [ ] Gmail API integration
- [ ] Email summarization
- [ ] Draft/send via chat

#### Messaging
- [ ] WhatsApp Business API
- [ ] Slack integration
- [ ] Discord bot

#### IoT / Smart Home
- [ ] Home Assistant integration
- [ ] MQTT broker support
- [ ] Device control via chat

#### Health & Fitness
- [ ] Apple Health integration (via Shortcuts)
- [ ] Workout tracking
- [ ] Health metrics dashboard

### Phase 9: Advanced AI Features

#### Model Orchestration
- [ ] Task classification for optimal model selection
- [ ] Model capability mapping (speed, cost, quality, context)
- [ ] Handover document generation between models
- [ ] Result aggregation from multiple models
- [ ] Cost tracking and budgeting

#### Agentic Workflows
- [ ] Multi-step task planning and execution
- [ ] Sub-agent spawning for complex tasks
- [ ] Progress tracking with checkpoints
- [ ] Rollback capability for failed steps

#### Advanced RAG
- [ ] Hybrid search (vector + BM25 keyword)
- [ ] Recursive retrieval for multi-hop questions
- [ ] Document versioning and change tracking
- [ ] Auto-indexing of project files on change
- [ ] Support for more formats: DOCX, XLSX, HTML, code files

---

## Technical Decisions

| Feature | Choice | Reason |
|---------|--------|--------|
| Task Storage | SQLite | Simple, local, no external deps |
| Weather API | wttr.in / Open-Meteo | Free, no API key required |
| Telegram | python-telegram-bot | Mature, async support |
| Auth (planned) | JWT + bcrypt | Industry standard |
| Multi-model | Custom router + intent | Full control over handoffs |
| Vector DB | ChromaDB + Qdrant | Local-first with cloud option |
| Embeddings | nomic-embed-text | 768-dim, runs locally via Ollama |
| Reranking | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, good accuracy |
| Frontend | React 19 + Vite | Fast dev experience, modern tooling |
| Web Search | DuckDuckGo (ddgs) | No API key needed, privacy-first |
| Image Gen | FLUX.1-schnell | Fast, high quality via Chutes |
| Video Gen | Wan2.1-14B | Text/image to video via Chutes |
| Music Gen | DiffRhythm | Supports lyrics via Chutes |
| Intent System | LLM + heuristic hybrid | Accurate with fast fallback |

---

## Progress Log

### 2026-02-07
- Added async parallel tool executor with concurrent execution
- Added dynamic tool selection (intent-based, 3-8 tools per query)
- Added async agent loop with progressive streaming
- Added multi-agent orchestrator for complex task decomposition
- Added document processing pipeline (PDF/DOCX/XLSX chunking + RAG)
- Added dynamic context length detection from model metadata
- Added semantic history retrieval via ChromaDB embeddings
- Added proportional context budget (adapts to any model size)
- Added live tool status UI with spinners and durations
- Added React.memo optimization for message rendering
- Added customizable assistant name via JARVIS.md frontmatter
- Added model/provider switch context preservation
- Added auto-memory fact extraction in web UI (throttled, async)
- Added memory management UI panel with CRUD API
- Added 5 agent coding skills (apply_patch, find_definition, find_references, run_tests, get_project_overview)
- Added voice+video chat with camera frame capture and vision analysis

### 2026-02-05
- Fixed ThinkingBlock rendering (content appeared as plain text)
- Added multimodal support: image, video, music generation
- Added Ollama vision model auto-detection
- Added file attachment support in web UI
- Fixed music generation with lyrics support
- Improved video generation with duration control

### 2026-02-04
- Created complete architecture redesign
- Implemented intent classification system (17 intents, 3 reasoning levels)
- Unified smart mode (removed Chat/Agent toggle)
- Added context window tracking with auto-compaction
- Added reasoning level controls (CLI + web UI)
- Created Chutes provider (LLM + TTS + STT + Image + Video + Music)
- Full Telegram bot integration (20+ commands)
- Fixed weather/status endpoints (Open-Meteo, psutil)
- Fixed chat scroll, hydration errors, fake data removal
- Created RAG integration in both chat paths
- Added Ollama tool capability mapping
- Added cross-encoder reranking
