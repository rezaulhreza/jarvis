# Rules

## Tool Execution

### Always Confirm Before
- Deleting files or directories
- Sending messages (Telegram, WhatsApp, email)
- Making purchases or financial transactions
- Running commands that modify system state
- Sharing any personal information externally

### Safe to Run Without Confirmation
- Reading files
- Web searches
- Listing directories
- Getting calendar events
- Checking weather/time

## Shell Commands

### Allowed
- ls, cat, head, tail, less
- grep, find, which
- git status, git log, git diff
- npm/yarn/composer info commands
- python/php/node version checks

### Requires Confirmation
- git push, git reset
- rm, mv (outside project directories)
- Any sudo command
- Installing packages globally

### Never Run
- rm -rf with wildcards
- Commands with unvalidated user input
- Anything touching ~/.ssh or credentials

## Privacy

### Keep Local
- All conversation history
- Personal facts and preferences
- Document embeddings
- Calendar and contact data

### Allowed External Calls
- Ollama (local)
- Web search APIs (DuckDuckGo)
- Weather APIs
- Telegram Bot API (user-initiated only)

## Memory Management

### Remember
- User preferences and corrections
- Ongoing projects and their context
- People mentioned and relationships
- Technical preferences (frameworks, tools)

### Forget After Task
- Temporary file contents
- One-off calculations
- Sensitive data (passwords, tokens)

## Context Management

### Auto-Compact When
- Conversation exceeds 8000 tokens
- Keep: current task, key facts, recent 5 exchanges
- Archive: full history to SQLite

### Always Preserve
- User profile facts
- Active project context
- Unresolved tasks
