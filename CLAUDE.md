# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

Jarvis is a local-first personal AI assistant powered by Ollama. It provides a privacy-respecting, self-hosted alternative to cloud-based AI assistants with support for multiple models, tool execution, persistent memory, switchable personas, a web UI, and voice input.

It can also support Ollama cloud models, OpenAI, Anthropic and so on. Subject to implementation.

## Tech Stack

- **Backend**: Python 3.10-3.12, FastAPI, Click CLI
- **Frontend**: React 19, TypeScript, Tailwind CSS v4, Vite
- **LLM Runtime**: Ollama (primary), with optional support for Claude, OpenAI, Gemini
- **Vector Database**: ChromaDB (local), Qdrant (optional cloud)
- **Storage**: SQLite (conversations), ChromaDB (embeddings)
- **Voice**: Whisper (STT), Edge TTS/ElevenLabs (TTS)

## Directory Structure

```
jarvis/                  # Python backend package
├── core/               # Agent loop, context, tools, router
├── providers/          # LLM provider abstractions (ollama, anthropic, openai, gemini)
├── skills/             # Built-in tools (web_search, file_ops, shell, etc.)
├── knowledge/          # RAG engine with ChromaDB/Qdrant
├── ui/                 # Terminal and FastAPI web server
├── voice/              # Voice I/O system
├── assistant.py        # Main Jarvis class
└── cli.py             # CLI entry point

web/                    # React frontend
├── src/
│   ├── components/    # React components
│   ├── hooks/         # Custom hooks (useVoice, etc.)
│   └── lib/           # Utilities
├── package.json
└── vite.config.ts

config/                 # Configuration templates
├── settings.yaml      # Default settings (copied to ~/.jarvis/config/)
├── rules.md           # Safety rules
└── personas/          # Pre-defined personas

docs/                   # Technical documentation
```

## Common Commands

### Python Backend

```bash
# Run CLI
jarvis                      # Interactive mode
jarvis --dev                # Web UI with hot reload
jarvis --voice              # Voice input mode
jarvis chat "message"       # Single query

# Setup and configuration
jarvis setup                # Interactive setup wizard
jarvis models               # List Ollama models
jarvis personas             # List personas

# Knowledge base (RAG)
jarvis knowledge add <file>     # Add document
jarvis knowledge add <dir> -r   # Recursive add
jarvis knowledge sync           # Sync documents
jarvis knowledge search <query> # Semantic search
```

### Frontend Development

```bash
cd web
npm install
npm run dev         # Dev server (port 3000)
npm run build       # Production build
npm run lint        # ESLint
```

### Installation

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[ui]"      # Core + web UI
pip install -e ".[all]"     # All optional dependencies
```

## Code Patterns

### Adding a New Skill

Skills are defined in `jarvis/skills/`. Each skill module exports functions decorated for tool calling. Register new skills in `jarvis/core/tools.py`.

### Adding a New Provider

Providers extend `BaseProvider` in `jarvis/providers/base.py`. Implement the required methods and register in the provider factory.

### Personas

Personas are markdown files in `config/personas/` or `~/.jarvis/config/personas/`. They define the assistant's personality and behavior.

## Configuration

User configuration lives in `~/.jarvis/config/`:
- `settings.yaml` - Model selection, context limits, integrations
- `rules.md` - Safety constraints
- `personas/` - Custom personas

## Safety Rules

The system uses `config/rules.md` to define which operations require user confirmation. Destructive file operations and shell commands are confirmed by default.

## Key Dependencies

- `ollama>=0.3.0` - Local LLM runtime
- `chromadb>=0.4.0` - Vector database (local. Can also use Qdrant)
- `rich>=13.0.0` - Terminal UI
- `click>=8.0.0` - CLI framework
- `sentence-transformers>=2.2.0` - Embeddings

## Development Notes

- Python 3.13+ is not supported due to dependency conflicts
- The web UI runs on port 7777 (backend) and 3000 (frontend dev server)
- Conversation history is stored in `memory/jarvis.db`
- Vector embeddings are stored in `knowledge/chroma_db/`

# Version Control

- NEVER expose secrets in vcs.
- NEVER add `Co Authored By Claude` in commit message.
- NEVER AUTO COMMIT. ALWAYS ASK USER TO COMMIT.
- WHEN ASKED TO COMMIT, DO ATOMIC COMMIT UNLESS USER ASKS TO COMMIT ALL.