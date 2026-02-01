# Jarvis

A local-first personal AI assistant powered by Ollama. Features multiple models, tool execution, persistent memory, switchable personas, web UI, and voice input.

## Features

- **Multi-Model Architecture**: Automatically selects the right model for the task
- **Built-in Skills**: Web search, weather, GitHub, file ops, calculator, notes, and more
- **Memory System**: Conversation history with auto-compaction and persistent storage
- **Personas**: Switch between different assistant modes (coder, researcher, creative, planner)
- **Web UI**: Browser-based interface with `jarvis --ui`
- **Voice Input**: Speech-to-text with Whisper using `jarvis --voice`

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
# Required
ollama pull qwen3:4b
ollama pull llava
ollama pull functiongemma

# Optional
ollama pull deepseek-r1:8b    # Deep reasoning
ollama pull qwen2.5-coder:7b  # Code generation
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
