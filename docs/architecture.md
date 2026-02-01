# Jarvis Architecture & Core Concepts

## Overview
Jarvis is a local AI assistant that runs entirely on your machine. It uses Ollama for LLM inference, ChromaDB for vector storage, and provides both CLI and Web UI interfaces.

## Core Components

### 1. LLM Provider (Ollama)
- **Location**: `jarvis/providers/ollama.py`
- **Purpose**: Interface with local Ollama models
- **Features**:
  - Chat completions with streaming
  - Embeddings generation
  - Model switching
  - Tool/function calling support

### 2. RAG System (Retrieval Augmented Generation)
- **Location**: `jarvis/knowledge/rag.py`
- **Purpose**: Give the AI access to your documents and knowledge

#### How RAG Works:
1. **Ingestion**: Documents are split into chunks (~500 words)
2. **Embedding**: Each chunk is converted to a vector using `nomic-embed-text`
3. **Storage**: Vectors are stored in ChromaDB with metadata
4. **Retrieval**: When you ask a question:
   - Your query is converted to a vector
   - Similar chunks are found via cosine similarity
   - Relevant chunks are injected into the system prompt
5. **Generation**: The LLM answers using the retrieved context

#### Key Classes:
- `RAGEngine`: Main class for document management
- `get_rag_engine()`: Singleton factory function

#### CLI Commands:
```bash
jarvis knowledge add <file>      # Add document
jarvis knowledge list            # List sources
jarvis knowledge search <query>  # Semantic search
jarvis knowledge sync            # Sync all documents
```

### 3. Agent System
- **Location**: `jarvis/agent/`
- **Purpose**: Execute multi-step tasks using tools

#### How Agents Work:
1. User sends a message
2. Agent decides if tools are needed
3. If yes, agent calls appropriate tool(s)
4. Tool returns result
5. Agent continues or responds

#### Tool Calling:
Tools are functions the AI can invoke. Example tools:
- `web_search`: Search the internet
- `read_file`: Read file contents
- `write_file`: Create/modify files
- `shell_command`: Execute terminal commands
- `quick_note`: Save notes

### 4. Context Management
- **Location**: `jarvis/memory/context.py`
- **Purpose**: Manage conversation history

#### Features:
- Persistent storage in SQLite
- Auto-compaction when context exceeds limits
- Summarization of old messages
- Keeps recent N messages intact

### 5. Voice System
- **Location**: `web/src/hooks/useVoice.ts`
- **Purpose**: Voice input/output

#### Speech-to-Text (STT):
- **Browser**: Web Speech API (instant, less accurate)
- **Whisper**: OpenAI Whisper (slower, more accurate)

#### Text-to-Speech (TTS):
- **Browser**: Built-in speechSynthesis (instant)
- **Edge TTS**: Microsoft neural voices (free, good quality)
- **ElevenLabs**: Premium voices (best quality, API key required)

### 6. Web UI
- **Location**: `web/`
- **Tech Stack**: React + Vite + TypeScript + Tailwind CSS v4
- **Features**:
  - Chat interface
  - Voice mode (hands-free conversation)
  - Model/voice selection
  - Settings management

## Data Flow

### Chat Mode (Fast)
```
User Input → RAG Retrieval → System Prompt + Context → Fast Model → Response → TTS
```

### Agent Mode (Tools)
```
User Input → Agent → Tool Selection → Tool Execution → Agent → Response
```

## Configuration

### Settings File: `~/.jarvis/config/settings.yaml`
```yaml
models:
  default: qwen3:4b      # Main model
  chat: llama3.2         # Fast chat model
  code: qwen2.5-coder    # Coding tasks
  embeddings: nomic-embed-text

memory:
  vector_store: knowledge/chroma_db
  database: memory/jarvis.db

voice:
  tts_provider: browser  # browser, edge, elevenlabs
  stt_provider: browser  # browser, whisper
```

### Environment Variables: `.env`
```bash
ELEVEN_LABS_API_KEY=sk_...  # For premium TTS
```

## Key Directories

```
jarvis/
├── jarvis/              # Python backend
│   ├── agent/           # Agent system
│   ├── knowledge/       # RAG system
│   ├── memory/          # Context management
│   ├── providers/       # LLM providers (Ollama)
│   ├── skills/          # Built-in tools
│   └── ui/              # FastAPI web server
├── web/                 # React frontend
├── knowledge/           # Vector DB storage
├── memory/              # Conversation DB
└── docs/                # Documentation

~/.jarvis/               # User data (outside git)
├── config/              # User settings
├── context.db           # Conversation history
└── knowledge/           # Personal documents
    └── personal/        # Your private docs
```

## Extending Jarvis

### Adding a New Tool
1. Create function in `jarvis/skills/`
2. Register in `jarvis/skills/__init__.py`
3. Add to agent's tool list

### Adding Documents to RAG
```bash
# Add to personal knowledge (not in git)
cp mydoc.pdf ~/.jarvis/knowledge/personal/
jarvis knowledge sync

# Or programmatically
from jarvis.knowledge import get_rag_engine
rag = get_rag_engine()
rag.add_file("mydoc.pdf")
```

### Adding a New Persona
1. Create prompt in `config/personas/`
2. Update settings: `persona: my_persona`
3. Persona-specific voice/model can be configured

## Performance Tips

- Use `chat` model for voice (faster)
- Use `browser` STT for instant response
- Use `browser` TTS for zero latency
- Keep RAG chunks under 3 for speed
- Enable auto-compaction for long conversations
