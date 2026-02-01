# Jarvis

A local-first personal AI assistant powered by Ollama. Features multiple models, tool execution, persistent memory, and switchable personas.

## Features

- **Multi-Model Architecture**: Automatically selects the right model for the task
  - General chat (qwen3)
  - Deep reasoning (deepseek-r1)
  - Vision/images (llava)
  - Code generation (qwen2.5-coder)
  - Tool routing (functiongemma)

- **Built-in Skills**
  - Web search (DuckDuckGo)
  - Weather lookup
  - GitHub operations
  - File operations
  - Shell commands (sandboxed)
  - Calculator & unit conversion
  - Note-taking
  - Date/time utilities

- **Memory System**
  - Conversation history with auto-compaction
  - Long-term fact storage
  - Entity tracking (people, projects)
  - SQLite persistence

- **Personas**
  - Default (balanced assistant)
  - Coder (pair programming)
  - Researcher (information analysis)
  - Creative (brainstorming)
  - Planner (productivity)

## Requirements

- macOS or Linux
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- 16GB RAM recommended

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:YOUR_USERNAME/jarvis.git
cd jarvis
```

### 2. Install Ollama models

```bash
# Required
ollama pull qwen3:4b
ollama pull llava
ollama pull functiongemma

# Optional (for enhanced features)
ollama pull deepseek-r1:8b      # Deep reasoning
ollama pull qwen2.5-coder:7b    # Code generation
ollama pull nomic-embed-text    # Embeddings for RAG
```

### 3. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

### 5. Run

```bash
python main.py
```

## Usage

### Basic Chat

```
You: What's the weather in London?
Jarvis: [Uses weather skill to fetch current conditions]

You: Explain how async/await works in Python
Jarvis: [Provides technical explanation]

You: /Users/me/screenshot.jpg what's in this image?
Jarvis: [Uses llava to analyze the image]
```

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/models` | List installed Ollama models |
| `/persona <name>` | Switch persona (default, coder, researcher, creative, planner) |
| `/facts` | Show stored facts about you |
| `/memory` | Show current working memory |
| `/clear` | Clear conversation history |
| `/history` | Show recent conversation |
| `/skills` | List available skills |
| `/quit` | Exit Jarvis |

### Switching Personas

```
You: /persona coder
Jarvis: Switched to coder persona.

You: Review this function for performance issues
Jarvis: [Responds with code-focused analysis]
```

## Project Structure

```
jarvis/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
│
├── config/
│   ├── settings.yaml         # Main configuration
│   ├── rules.md              # Safety rules
│   └── personas/             # Persona definitions
│       ├── default.md
│       ├── coder.md
│       ├── researcher.md
│       ├── creative.md
│       └── planner.md
│
├── core/
│   ├── ollama_client.py      # Ollama API wrapper
│   ├── context_manager.py    # Memory & context
│   └── router.py             # Tool selection
│
├── memory/
│   ├── facts.md              # User facts
│   └── entities.json         # Tracked entities
│
├── skills/                   # Tool implementations
│   ├── web_search.py
│   ├── weather.py
│   ├── github_ops.py
│   ├── shell.py
│   ├── file_ops.py
│   ├── calculator.py
│   ├── datetime_ops.py
│   ├── notes.py
│   └── telegram.py
│
└── knowledge/                # RAG storage
    ├── documents/            # Your documents
    └── notes/                # Quick notes
```

## Configuration

### Settings (config/settings.yaml)

```yaml
models:
  default: "qwen3:4b"         # Change default model
  reasoning: "deepseek-r1:8b" # For complex problems

context:
  max_tokens: 8000            # Auto-compact threshold
  keep_recent_messages: 5     # Messages to keep after compaction

persona: "default"            # Starting persona
```

### Environment Variables (.env)

```bash
# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_token

# Weather (optional - falls back to wttr.in)
OPENWEATHER_API_KEY=your_key

# GitHub (optional - uses gh CLI auth)
GITHUB_TOKEN=your_token
```

## Adding Custom Skills

1. Create a new file in `skills/`:

```python
# skills/my_skill.py

def my_function(param: str) -> dict:
    """Do something useful."""
    return {"success": True, "result": "..."}
```

2. Register in `skills/__init__.py`:

```python
from .my_skill import my_function

AVAILABLE_SKILLS["my_skill"] = {
    "function": my_function,
    "description": "Does something useful",
    "parameters": {"param": "string - description"}
}
```

## Adding Custom Personas

Create a new file in `config/personas/`:

```markdown
# My Persona

You are Jarvis in [mode] mode.

## Core Traits
- Trait 1
- Trait 2

## Communication Style
- Style guidelines
```

Then use: `/persona my_persona`

## Roadmap

- [ ] Voice input/output (Whisper + TTS)
- [ ] Telegram bot integration
- [ ] RAG with local documents
- [ ] Calendar integration
- [ ] Scheduled tasks
- [ ] Web UI option

## Troubleshooting

### Ollama connection failed
```bash
# Make sure Ollama is running
ollama serve
```

### Model not found
```bash
# Pull the required model
ollama pull model_name
```

### Out of memory
- Use smaller models (3b/4b variants)
- Reduce `max_tokens` in settings.yaml
- Close other applications

## License

MIT
