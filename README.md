# Jarvis

A local-first personal AI assistant powered by Ollama. Features multiple models, tool execution, persistent memory, switchable personas, web UI, and voice input.

## Features

- **Multi-Model Architecture**: Automatically selects the right model for the task
- **Knowledge Base (RAG)**: Feed your own documents (PDF, TXT, MD) for personalized answers
- **Chat History**: Claude-style conversation sidebar with search, edit, and auto-titles
- **Built-in Skills**: Web search, weather, GitHub, file ops, calculator, notes, and more
- **Memory System**: Conversation history with auto-compaction and persistent storage
- **Personas**: Switch between different assistant modes (coder, researcher, creative, planner)
- **Web UI**: Browser-based interface with `jarvis --ui`
- **Voice Input**: Speech-to-text with Whisper and TTS with Edge/ElevenLabs

## Installation

### Quick Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/rezaulhreza/jarvis/main/install.sh | bash
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
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- 16GB RAM recommended

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

## Usage

### CLI Mode (Default)

```bash
jarvis
```

### Web UI

```bash
jarvis --ui
# Opens at http://localhost:7777
```

### Voice Mode

```bash
jarvis --voice
# Requires: pip install jarvis-ai-assistant[voice]
```

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
| `/persona <name>` | Switch persona |
| `/personas` | List personas |
| `/skills` | List available skills |
| `/facts` | Show stored facts |
| `/memory` | Show working memory |
| `/clear` | Clear conversation |
| `/history` | Show recent history |
| `/quit` | Exit |

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
# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_token

# Weather API (optional)
OPENWEATHER_API_KEY=your_key

# GitHub (optional)
GITHUB_TOKEN=your_token
```

## Available Skills

| Skill | Description |
|-------|-------------|
| `web_search` | Search with DuckDuckGo |
| `get_weather` | Current weather |
| `get_forecast` | Weather forecast |
| `github_repos` | List repositories |
| `github_issues` | List issues |
| `read_file` | Read file contents |
| `list_directory` | List directory |
| `shell_run` | Run safe commands |
| `calculate` | Math expressions |
| `convert_units` | Unit conversion |
| `current_time` | Get time |
| `quick_note` | Save notes |
| `search_notes` | Search notes |

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

When using the web UI (`jarvis --ui`), you'll see a RAG indicator during responses:

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

## Personas

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

```bash
git clone https://github.com/rezaulhreza/jarvis.git
cd jarvis
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

## License

MIT
