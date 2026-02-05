<p align="center">
  <img src="assets/banner.png" alt="Jarvis" />
</p>

<p align="center">
  A local-first personal AI assistant powered by Ollama.<br>
  Features multiple providers, 29+ tools, persistent memory, switchable personas, web UI, and voice input.
</p>

## Features

- **Multi-Provider Support**: Ollama (local), Claude, OpenAI, Gemini, and Ollama Cloud (remote)
- **29+ Built-in Tools**: Web search, git operations, file ops, weather, gold prices, task management, and more
- **Smart Auto-Tools**: Automatically searches the web for current events, prices, news, and real-time data
- **Knowledge Base (RAG)**: Feed your own documents (PDF, TXT, MD) for personalized answers
- **Tool Timeline UI**: Visual execution steps showing tool calls, duration, and results
- **Chat History**: Claude-style conversation sidebar with search, edit, and auto-titles
- **Memory System**: Conversation history with auto-compaction and persistent storage
- **Personas**: Switch between different assistant modes (coder, researcher, creative, planner)
- **Web UI**: Modern browser-based interface with `jarvis --dev`
- **Voice Input**: Speech-to-text with Whisper and TTS with Edge/ElevenLabs

## Requirements

- **Python 3.10, 3.11, or 3.12** (3.13+ not yet supported due to dependency compatibility)
- [Ollama](https://ollama.ai) installed and running
- **Node.js 18+** (required for Web UI)
- 16GB RAM recommended

### Check Your Python Version

```bash
python3 --version
```

If you have Python 3.13 or 3.14, install a compatible version:

```bash
# macOS with Homebrew
brew install python@3.12

# Then use this Python for installation
/opt/homebrew/bin/python3.12 --version
```

## Installation

### Quick Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/rezaulhreza/jarvis/main/install.sh | bash
```

> **Note:** If the script detects Python 3.13+, it will fail during dependency installation. See [Troubleshooting](#troubleshooting) below.

### Manual Installation (If Quick Install Fails)

If you have Python 3.13+ or encounter dependency conflicts:

```bash
# 1. Install Python 3.12 (macOS)
brew install python@3.12

# 2. Clone the repository
git clone https://github.com/rezaulhreza/jarvis.git ~/.jarvis

# 3. Create virtual environment with Python 3.12
cd ~/.jarvis/src
/opt/homebrew/bin/python3.12 -m venv ../venv

# 4. Activate and install
source ../venv/bin/activate
pip install -e ".[ui]"
pip install python-multipart

# 5. Add alias for easy access (optional)
echo 'alias jarvis="~/.jarvis/venv/bin/jarvis"' >> ~/.zshrc
source ~/.zshrc
```

### Using pip

```bash
# Basic installation
pip install jarvis-ai-assistant

# With UI support
pip install jarvis-ai-assistant[ui]

# With voice support
pip install jarvis-ai-assistant[voice]

# Everything
pip install jarvis-ai-assistant[all]
```

### From Source

```bash
git clone https://github.com/rezaulhreza/jarvis.git
cd jarvis
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[ui]"
pip install python-multipart
```

### Install Ollama Models

```bash
# Required for chat
ollama pull llama3.2          # Fast chat model
ollama pull qwen3:4b          # General purpose

# Required for RAG (knowledge base)
ollama pull nomic-embed-text  # Text embeddings

# Optional
ollama pull deepseek-r1:8b    # Deep reasoning
ollama pull qwen2.5-coder:7b  # Code generation
ollama pull llava             # Image understanding
```

### Web UI Setup (Additional Steps)

The Web UI requires Node.js. If `jarvis --dev` shows `vite: command not found`:

```bash
# 1. Install Node.js (macOS)
brew install node

# 2. Install frontend dependencies
cd ~/.jarvis/web
npm install

# 3. Now run the dev server
jarvis --dev
```

## Usage

### CLI Mode (Default)

```bash
jarvis
```

### Web UI

```bash
jarvis --dev
# Opens at http://localhost:7777 (backend) and http://localhost:3000 (frontend)
```

### Voice Mode

```bash
jarvis --voice
# Requires: pip install jarvis-ai-assistant[voice]
```

### Telegram Bot

Full-featured Telegram integration with all capabilities:

```bash
# Interactive setup wizard
jarvis telegram setup

# Or manual setup:
jarvis config set telegram bot_token YOUR_BOT_TOKEN
jarvis telegram webhook https://your-domain.com  # Set webhook (recommended)

# Now just run your server - Telegram works automatically!
jarvis --dev
```

**Telegram Commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/model` | Show/switch AI model |
| `/models` | List available models |
| `/provider` | Show/switch provider |
| `/clear` | Clear conversation |
| `/mode <fast\|balanced\|deep>` | Set reasoning mode |
| `/search <query>` | Web search |
| `/knowledge <query>` | Search knowledge base |
| `/remember <fact>` | Save a fact about you |
| `/status` | System status |
| `/weather` | Current weather |
| `/name <new name>` | Change assistant name |

Plus all regular chat capabilities - web search, tool execution, RAG, etc.

### Web UI Features

The web UI (`jarvis --dev`) includes:
- **Chat/Agent toggle**: Fast chat mode or full agent mode with tools
- **Tool Timeline**: Expandable panel showing each tool call, arguments, duration, and results
- **Provider switching**: Change LLM provider on the fly
- **Voice input/output**: Push-to-talk and auto-speak responses
- **TTS options**: Browser, Edge TTS, or ElevenLabs voices

### Single Query

```bash
jarvis chat "What's the weather in London?"
```

### Other Commands

```bash
jarvis --help      # Show all options
jarvis setup       # Run setup wizard
jarvis models      # List Ollama models
jarvis personas    # List personas
```

## CLI Commands

Inside the interactive CLI:

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/models` | List available models |
| `/model <name>` | Switch to a different model |
| `/provider <name>` | Switch provider (ollama, anthropic, openai, gemini) |
| `/persona <name>` | Switch persona |
| `/personas` | List personas |
| `/tools` | List available tools |
| `/facts` | Show stored facts |
| `/memory` | Show working memory |
| `/clear` | Clear conversation |
| `/cls` | Clear screen |
| `/init` | Create JARVIS.md project config |
| `/history` | Show recent history |
| `/quit` | Exit |

## Troubleshooting

### `ResolutionImpossible` or dependency conflicts

This usually means your Python version is too new (3.13+). Install Python 3.12:

```bash
# macOS
brew install python@3.12

# Recreate venv with correct Python
cd ~/.jarvis/src
/opt/homebrew/bin/python3.12 -m venv ../venv --clear
source ../venv/bin/activate
pip install -e ".[ui]"
pip install python-multipart
```

### `vite: command not found`

Node.js and frontend dependencies are missing:

```bash
brew install node
cd ~/.jarvis/web
npm install
```

### `python-multipart` error

Install the missing dependency:

```bash
source ~/.jarvis/venv/bin/activate
pip install python-multipart
```

### `jarvis: command not found`

Either activate the virtual environment or add an alias:

```bash
# Option 1: Activate venv first
source ~/.jarvis/venv/bin/activate
jarvis

# Option 2: Add alias to ~/.zshrc or ~/.bashrc
echo 'alias jarvis="~/.jarvis/venv/bin/jarvis"' >> ~/.zshrc
source ~/.zshrc
```

### Backend starts but frontend doesn't load

Make sure both ports are accessible:
- Backend: http://localhost:7777
- Frontend: http://localhost:3000

Check that `npm install` completed successfully in `~/.jarvis/web`.

## Configuration

Configuration is stored in `~/.jarvis/`:

```
~/.jarvis/
├── config/
│   ├── settings.yaml     # Main config
│   ├── rules.md          # Safety rules
│   └── personas/         # Persona definitions
├── memory/
│   ├── facts.md          # User facts
│   └── jarvis.db         # SQLite history
└── knowledge/
    └── notes/            # Quick notes
```

### Environment Variables

Create `~/.jarvis/.env`:

```bash
# LLM Providers (optional - for cloud models)
ANTHROPIC_API_KEY=your_key      # Claude
OPENAI_API_KEY=your_key         # OpenAI

# Web Search (highly recommended)
BRAVE_API_KEY=your_key          # Brave Search - primary search provider
                                 # Get free key at: https://brave.com/search/api/

# APIs (optional)
GOLD_API_KEY=your_key           # GoldAPI.io - live gold/silver prices
OPENWEATHER_API_KEY=your_key    # Weather data
GITHUB_TOKEN=your_token         # GitHub integration

# Voice (optional)
ELEVEN_LABS_API_KEY=your_key    # ElevenLabs TTS

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_token
```

## Available Tools (29+)

### Web & Information
| Tool | Description |
|------|-------------|
| `web_search` | Search with Brave (primary) or DuckDuckGo (fallback) |
| `web_fetch` | Fetch and extract content from URLs |
| `get_current_news` | Get latest news on a topic |
| `get_weather` | Current weather (wttr.in or OpenWeatherMap) |
| `get_current_time` | Get time in any timezone |
| `get_gold_price` | Live gold/silver prices (GoldAPI.io) |

### File Operations
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `edit_file` | Search and replace in files |
| `list_files` | List directory contents |
| `glob_files` | Find files by pattern (e.g., `**/*.py`) |
| `grep` | Search file contents with regex |
| `search_files` | Search for text in files |
| `get_project_structure` | Get directory tree |

### Git Operations
| Tool | Description |
|------|-------------|
| `git_status` | Show working tree status |
| `git_diff` | Show changes (staged/unstaged) |
| `git_log` | Show commit history |
| `git_add` | Stage files |
| `git_commit` | Create commits |
| `git_branch` | List/create/switch branches |
| `git_stash` | Stash changes |

### Task Management
| Tool | Description |
|------|-------------|
| `task_create` | Create a new task |
| `task_update` | Update task status |
| `task_list` | List all tasks |
| `task_get` | Get task details |

### Utilities
| Tool | Description |
|------|-------------|
| `calculate` | Math expressions |
| `run_command` | Run shell commands (with confirmation) |
| `save_memory` | Save information to memory |
| `recall_memory` | Search saved memories |
| `github_search` | Search GitHub repos/code |

## Knowledge Base (RAG)

Jarvis includes a RAG (Retrieval Augmented Generation) system that lets you feed it your own documents. When you ask questions, Jarvis searches your knowledge base and uses relevant information to give accurate, personalized answers.

### How It Works

```
Your Question → Search Knowledge Base → Find Relevant Chunks → Inject into Prompt → AI Answers
```

### Quick Start

```bash
# Install embedding model (required once)
ollama pull nomic-embed-text

# Add a document
jarvis knowledge add ~/Documents/my_resume.pdf

# Add a folder of documents
jarvis knowledge add ~/Documents/work/ --recursive

# Sync all configured sources
jarvis knowledge sync --personal
```

### Knowledge Commands

| Command | Description |
|---------|-------------|
| `jarvis knowledge add <file>` | Add a file (PDF, TXT, MD) |
| `jarvis knowledge add <dir> -r` | Add directory recursively |
| `jarvis knowledge list` | List all sources and chunk counts |
| `jarvis knowledge search <query>` | Test semantic search |
| `jarvis knowledge remove <source>` | Remove a source |
| `jarvis knowledge clear` | Clear entire knowledge base |
| `jarvis knowledge sync` | Sync docs/ folder |
| `jarvis knowledge sync --personal` | Also sync ~/.jarvis/knowledge/personal/ |
| `jarvis knowledge sync --projects` | Also sync project READMEs from ~/Developer/ |

### Examples

**Add your resume and ask about your skills:**
```bash
jarvis knowledge add ~/Desktop/resume.pdf
# Added resume.pdf: 3 chunks

jarvis
> What are my skills?
# Jarvis now answers using YOUR actual resume!
```

**Add project documentation:**
```bash
jarvis knowledge add ./docs/ --recursive
# Added 5 documents

jarvis
> How does authentication work in this project?
# Answers based on your actual docs
```

**Search your knowledge base:**
```bash
jarvis knowledge search "deployment process"
# [1] deploy.md (distance: 0.3421)
#     To deploy, run ./scripts/deploy.sh...
```

### Personal Knowledge (Private)

Store personal information outside of git:

```bash
# Create personal knowledge directory
mkdir -p ~/.jarvis/knowledge/personal

# Add personal documents
cp ~/Documents/notes.md ~/.jarvis/knowledge/personal/
cp ~/Documents/cv.pdf ~/.jarvis/knowledge/personal/

# Sync personal knowledge
jarvis knowledge sync --personal
```

Files in `~/.jarvis/knowledge/personal/` are:
- Automatically indexed when you run `sync --personal`
- Never committed to git
- Perfect for personal info, credentials notes, private docs

### Supported File Types

| Type | Extension | Notes |
|------|-----------|-------|
| Text | `.txt` | Plain text files |
| Markdown | `.md` | Documentation, notes |
| PDF | `.pdf` | Resumes, reports, books |

### RAG in the Web UI

When using the web UI (`jarvis --dev`), you'll see a RAG indicator during responses:

- 🟢 **Green**: Knowledge found - shows sources used
- 🟡 **Yellow**: Knowledge base exists but no relevant match
- ⚫ **Gray**: No knowledge base configured
- 🔴 **Red**: Error retrieving knowledge

### Configuration

In `~/.jarvis/config/settings.yaml`:

```yaml
models:
  embeddings: nomic-embed-text  # Embedding model

memory:
  vector_store: knowledge/chroma_db  # ChromaDB storage path
```

### How RAG Improves Answers

**Without RAG:**
```
Q: What projects have I worked on?
A: I don't have information about your projects.
```

**With RAG (after adding your resume):**
```
Q: What projects have I worked on?
A: Based on your resume, you've worked on:
   - E-commerce platform using Laravel and Vue.js
   - CSV Importer handling million-row imports
   - Flash-toast library for Livewire/Alpine
```

## Multi-Provider Support

Jarvis supports multiple LLM providers. Configure in `~/.jarvis/config/settings.yaml`:

```yaml
# Default provider
provider: ollama

# Provider-specific settings
providers:
  ollama:
    model: llama3.2
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
  gemini:
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.0-flash
  ollama_cloud:
    base_url: https://your-ollama-server.com
    api_key: your_key  # if auth required
```

Switch providers at runtime:
```bash
/provider anthropic
/provider ollama
```

## Smart Auto-Tools

Jarvis automatically detects when to use tools without being asked:

- **Current events**: "Who is the president?" → auto web search
- **Prices**: "What's the gold price?" → auto GoldAPI lookup
- **Weather**: "Weather in London" → auto weather lookup
- **News**: "Latest tech news" → auto news search
- **Time**: "What time is it in Tokyo?" → auto time lookup

Context-aware commands also work:
```
You: Tell me about the Mars mission
Jarvis: [answers from knowledge]
You: search the web
Jarvis: [searches for "Mars mission" using previous context]
```

## Customization

### Assistant Name

The assistant name is configurable (not hardcoded as "Jarvis"):

```bash
# Via Telegram
/name Aria

# Via settings.yaml
assistant:
  name: Aria
```

### Personas

- **default**: Balanced general assistant
- **coder**: Pair programming mode
- **researcher**: Information analysis
- **creative**: Brainstorming partner
- **planner**: Productivity coach

Switch with `/persona coder` or `jarvis persona coder`.

## Adding Custom Skills

Create `~/.jarvis/skills/my_skill.py`:

```python
def my_function(param: str) -> dict:
    """Do something useful."""
    return {"success": True, "result": "..."}
```

## Adding Custom Personas

Create `~/.jarvis/config/personas/my_persona.md`:

```markdown
# My Persona

You are Jarvis in custom mode.

## Traits
- Custom trait 1
- Custom trait 2
```

## Development

### Setup

```bash
git clone https://github.com/rezaulhreza/jarvis.git
cd jarvis

# Python backend (use Python 3.10-3.12)
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
pip install python-multipart

# React frontend
cd web
npm install
cd ..
```

### Running in Dev Mode

```bash
# Full dev environment (backend + frontend with hot reload)
jarvis --dev

# This starts:
# - Backend at http://localhost:7777
# - Frontend at http://localhost:3000 (with hot reload)
```

### Building for Production

```bash
# Build React frontend
cd web && npm run build && cd ..

# Run production server
jarvis --dev
```

### Project Structure

```
jarvis/
├── jarvis/              # Python backend
│   ├── core/            # Agent, context, tools (29+ tools)
│   ├── knowledge/       # RAG system (ChromaDB/Qdrant)
│   ├── providers/       # LLM providers (Ollama, Claude, OpenAI, Gemini)
│   ├── skills/          # Additional skills
│   ├── auth/            # Provider auth helpers
│   └── ui/              # FastAPI server + terminal UI
├── web/                 # React frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── components/  # React components
│   │   └── hooks/       # WebSocket, voice hooks
│   └── dist/            # Production build
├── config/              # Default configuration templates
├── docs/                # Documentation (synced to RAG)
└── knowledge/           # Vector DB storage
```

## License

MIT
