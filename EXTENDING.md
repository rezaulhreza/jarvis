# Extending Jarvis

Jarvis is designed to be easily extended. You can add new skills, personas, rules, and more - all without modifying the core code.

## Directory Structure

All customizations live in `~/.jarvis/`:

```
~/.jarvis/
├── config/
│   ├── settings.yaml       # Main configuration
│   ├── rules.md            # Safety rules and constraints
│   └── personas/           # Custom personas
│       ├── default.md
│       └── my_persona.md
├── skills/                 # Custom skills
│   ├── my_skill.py
│   └── README.md
├── memory/
│   ├── facts.md            # Facts about you
│   ├── entities.json       # Tracked entities
│   └── jarvis.db           # Conversation history
└── knowledge/
    ├── documents/          # Documents for RAG
    └── notes/              # Quick notes
```

---

## Creating Skills

### Method 1: Ask Jarvis

Just ask Jarvis to create a skill for you:

```
You: Create a skill that fetches the current Bitcoin price

Jarvis: I'll create that skill for you...
[Creates ~/.jarvis/skills/fetch_bitcoin_price.py]

You: What's the Bitcoin price?

Jarvis: [Uses the new skill] Bitcoin is currently $XX,XXX
```

### Method 2: Manual Creation

Create a file in `~/.jarvis/skills/`:

```python
# ~/.jarvis/skills/fetch_stock_price.py
"""
Fetch stock prices from Yahoo Finance.
"""

import requests

def fetch_stock_price(symbol: str) -> dict:
    """
    Get current stock price.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Stock price data
    """
    try:
        # Your implementation here
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        response = requests.get(url, timeout=10)
        data = response.json()

        price = data['chart']['result'][0]['meta']['regularMarketPrice']

        return {
            "success": True,
            "symbol": symbol,
            "price": price
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Required: Skill metadata
SKILL_INFO = {
    "name": "fetch_stock_price",
    "function": fetch_stock_price,
    "description": "Get current stock price by ticker symbol",
    "parameters": {
        "symbol": "string - stock ticker (e.g., AAPL, GOOGL)"
    }
}
```

### Skill Templates

Available templates for common patterns:

- **basic** - Simple function template
- **api_fetch** - HTTP API calls
- **file_processor** - File operations
- **shell_command** - Shell command wrappers

Ask Jarvis: "Show me the api_fetch skill template"

### Skill Requirements

1. **Function must return a dict** with at least `success: bool`
2. **SKILL_INFO dict required** with:
   - `name`: Skill identifier (snake_case)
   - `function`: Reference to the function
   - `description`: What it does
   - `parameters`: Dict of param names to descriptions

---

## Creating Personas

Personas change how Jarvis behaves and responds.

### Create a Persona

Create `~/.jarvis/config/personas/my_persona.md`:

```markdown
# DevOps Engineer

You are Jarvis in DevOps mode - an infrastructure specialist.

## Core Traits
- Infrastructure-first thinking
- Security conscious
- Automation focused
- Metrics-driven

## Communication Style
- Use technical terminology
- Provide command examples
- Reference best practices
- Mention potential risks

## Expertise Areas
- Docker, Kubernetes
- CI/CD pipelines
- Cloud platforms (AWS, GCP, Azure)
- Monitoring and logging
- Infrastructure as Code

## When Asked About Code
- Consider deployment implications
- Suggest containerization
- Think about scalability
- Mention logging and monitoring
```

### Use the Persona

```
You: /persona devops

Jarvis: Switched to devops persona.

You: How should I deploy this Node.js app?

Jarvis: [Responds with infrastructure-focused advice]
```

---

## Customizing Rules

Rules constrain what Jarvis can and cannot do.

### Edit Rules

Edit `~/.jarvis/config/rules.md`:

```markdown
# Rules

## Tool Execution

### Always Confirm Before
- Deleting files
- Sending messages
- Running git push
- Any destructive action

### Safe to Run
- Read operations
- Search operations
- Listing files

## Custom Rules

### Project-Specific
- Never modify files in /production
- Always run tests before suggesting commits
- Prefer npm over yarn in this project

### Personal Preferences
- Use metric units
- Default timezone: Europe/London
- Prefer concise answers
```

---

## Adding Facts

Facts help Jarvis remember things about you.

### Edit Facts

Edit `~/.jarvis/memory/facts.md`:

```markdown
# Facts About Me

## Identity
- Name: Boss
- Timezone: Europe/London
- Location: London, UK

## Work
- Role: Full-stack developer
- Company: Acme Corp
- Current project: E-commerce platform

## Technical
- Primary languages: PHP, JavaScript
- Frameworks: Laravel, Vue.js, React
- Prefers: VSCode, iTerm2, Firefox

## Preferences
- Communication: Concise, technical
- Documentation: Markdown
- Git: Conventional commits

## Current Goals
- Learning Python and LLMs
- Building personal AI assistant
- Improving DevOps skills

## People
- Alice: Backend team lead
- Bob: DevOps engineer
- Charlie: Product manager
```

---

## Configuration

### settings.yaml

```yaml
# Model selection
models:
  default: "qwen3:4b"           # General chat
  reasoning: "deepseek-r1:8b"   # Complex problems
  vision: "llava"               # Images
  code: "qwen2.5-coder:7b"      # Code generation
  tools: "functiongemma"        # Tool routing

# Context management
context:
  max_tokens: 8000              # When to auto-compact
  keep_recent_messages: 5       # Keep after compaction

# Default persona
persona: "default"

# Integrations
integrations:
  telegram:
    enabled: false
  web_search:
    enabled: true
    provider: "duckduckgo"
```

---

## Adding Documents (RAG)

Place documents in `~/.jarvis/knowledge/documents/`:

```
~/.jarvis/knowledge/documents/
├── project-readme.md
├── api-docs.md
├── meeting-notes/
│   ├── 2024-01-15.md
│   └── 2024-01-22.md
└── reference/
    └── style-guide.md
```

Jarvis will index these and use them to answer questions.

---

## API Reference

### Skill Functions

All skills should follow this pattern:

```python
def my_skill(param1: str, param2: int = 10) -> dict:
    """
    Description of what the skill does.

    Args:
        param1: Description
        param2: Description with default

    Returns:
        Dict with success status and results
    """
    try:
        # Implementation
        result = do_something(param1, param2)

        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Meta-Skills for Extension

| Skill | Description |
|-------|-------------|
| `create_skill` | Create a new skill programmatically |
| `delete_skill` | Remove a user-created skill |
| `list_user_skills` | List custom skills |
| `get_skill_code` | View skill source code |
| `get_skill_template` | Get a template for new skills |

---

## Best Practices

1. **Skills should be focused** - One skill, one job
2. **Always handle errors** - Return `{"success": False, "error": "message"}`
3. **Document parameters** - Clear descriptions help the AI use skills correctly
4. **Test locally first** - Run the function manually before using with Jarvis
5. **Keep personas focused** - Each persona should have a clear purpose
6. **Update facts regularly** - Keep your facts current for better personalization

---

## Troubleshooting

### Skill not loading
- Check for syntax errors: `python -c "import my_skill"`
- Ensure SKILL_INFO is defined
- Check file permissions

### Persona not working
- Verify file is in `~/.jarvis/config/personas/`
- Check markdown syntax
- Restart Jarvis after adding

### Rules not applied
- Rules are guidance, not hard enforcement
- Be specific in rule descriptions
- Use clear, unambiguous language
