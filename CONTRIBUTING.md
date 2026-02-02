# Contributing Guidelines

## Commit Messages

### Do NOT include AI co-author attribution

Commits should **never** include `Co-Authored-By: Claude` or similar AI attribution lines.

**Wrong:**
```
Fix authentication bug

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Correct:**
```
Fix authentication bug
```

### Commit message format

- Use imperative mood in the subject line ("Add feature" not "Added feature")
- Keep the subject line under 72 characters
- Separate subject from body with a blank line
- Use the body to explain *what* and *why*, not *how*
