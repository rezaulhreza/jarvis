"""
Tools for Jarvis - Used with native tool calling.

These functions are passed to the LLM's tools parameter.
Includes all skills that are useful for agentic behavior.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

# Global project root and UI reference
_PROJECT_ROOT: Path = Path.cwd()
_UI = None
_PENDING_WRITES = {}  # filepath -> content, for confirmation
_READ_FILES = set()  # Track files that have been read this session


def set_project_root(path: Path):
    """Set the project root for all tools."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = path


def set_ui(ui):
    """Set the UI instance for confirmations."""
    global _UI
    _UI = ui


def get_pending_writes():
    """Get pending write operations."""
    return _PENDING_WRITES


def clear_pending_writes():
    """Clear pending writes after confirmation."""
    global _PENDING_WRITES
    _PENDING_WRITES = {}


def clear_read_files():
    """Clear the set of read files (call at start of new conversation)."""
    global _READ_FILES
    _READ_FILES = set()


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def read_file(path: str) -> str:
    """Read a file's contents. Use this to examine code, configs, or any text file.

    Args:
        path: File path relative to project root, or absolute path

    Returns:
        The file contents, or error message
    """
    global _PROJECT_ROOT, _READ_FILES

    try:
        if path.startswith('/'):
            file_path = Path(path)
        else:
            file_path = _PROJECT_ROOT / path

        if not file_path.exists():
            matches = list(_PROJECT_ROOT.rglob(f"**/{Path(path).name}"))
            if matches:
                file_path = matches[0]
            else:
                return f"Error: File not found: {path}"

        content = file_path.read_text(errors='replace')
        if len(content) > 15000:
            content = content[:15000] + f"\n\n... [TRUNCATED - file is {len(content)} chars, showing first 15000]"

        # Track that this file was read
        _READ_FILES.add(str(file_path.resolve()))

        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files in a directory. Use this to explore project structure.

    Args:
        path: Directory path relative to project root
        pattern: Glob pattern like '*.py' or '**/*.js'

    Returns:
        List of files and directories
    """
    global _PROJECT_ROOT

    try:
        if not path:
            path = "."

        if path.startswith('/'):
            dir_path = Path(path)
        else:
            dir_path = _PROJECT_ROOT / path

        if not dir_path.exists():
            return f"Error: Directory not found: {path}"

        if pattern == "*":
            items = list(dir_path.iterdir())
        else:
            items = list(dir_path.rglob(pattern))

        result = []
        excludes = ['.git', 'node_modules', '__pycache__', 'vendor', '.next', 'dist', 'build', '.idea', '.vscode']

        for item in sorted(items)[:100]:
            if any(x in str(item) for x in excludes):
                continue
            if item.name.startswith('.'):
                continue

            try:
                rel = item.relative_to(_PROJECT_ROOT)
            except ValueError:
                rel = item

            suffix = "/" if item.is_dir() else ""
            result.append(f"{rel}{suffix}")

        return "\n".join(result[:50]) if result else "No files found"
    except Exception as e:
        return f"Error: {e}"


def search_files(query: str, path: str = ".", file_type: str = "") -> str:
    """Search for text in files. Use this to find code, functions, classes, or patterns.

    Args:
        query: Text or pattern to search for
        path: Directory to search in (default: project root)
        file_type: File extension like 'py', 'js', 'php' (optional)

    Returns:
        Matching lines with file paths and line numbers
    """
    global _PROJECT_ROOT

    if not query:
        return "Error: No search query provided"

    try:
        if not path:
            path = "."

        search_path = _PROJECT_ROOT / path if not path.startswith('/') else Path(path)

        cmd = ["grep", "-rn", "-I"]  # -I ignores binary files

        if file_type:
            cmd.extend(["--include", f"*.{file_type}"])

        for exclude in ['node_modules', '.git', '__pycache__', 'vendor', '.next', 'dist', 'build']:
            cmd.extend(["--exclude-dir", exclude])

        cmd.extend([query, str(search_path)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.stdout:
            # Limit output and format nicely
            lines = result.stdout.strip().split('\n')[:30]
            return "\n".join(lines)
        else:
            return f"No matches found for '{query}'"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error: {e}"


def run_command(command: str) -> str:
    """Run a shell command. Use for git, npm, pip, tests, builds, etc.

    Args:
        command: The shell command to run

    Returns:
        Command output (stdout and stderr)
    """
    global _PROJECT_ROOT, _UI

    # Safety check
    dangerous = ['rm -rf /', 'sudo rm', 'mkfs', '> /dev/sd', 'dd if=/dev/zero', 'chmod -R 777 /']
    if any(d in command for d in dangerous):
        return "Error: Command blocked for safety"

    # Commands that need confirmation
    needs_confirm = ['rm ', 'git push', 'git reset', 'drop table', 'delete from']

    if _UI and any(c in command.lower() for c in needs_confirm):
        _UI.console.print(f"[yellow]Run command: {command}[/yellow]")
        _UI.console.print("[dim]  y = yes, n = no[/dim]")
        try:
            response = input("> ").strip().lower()
            if response not in ('y', 'yes'):
                return "Command cancelled by user"
        except (EOFError, KeyboardInterrupt):
            return "Command cancelled"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout + result.stderr
        return output[:5000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s limit)"
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file. Use this to create or overwrite files.

    Args:
        path: File path relative to project root
        content: The COMPLETE file content to write

    Returns:
        Success or error message
    """
    global _PROJECT_ROOT, _UI, _READ_FILES

    try:
        file_path = _PROJECT_ROOT / path if not path.startswith('/') else Path(path)
        resolved = str(file_path.resolve())

        # For existing files, require that they were read first
        if file_path.exists() and resolved not in _READ_FILES:
            return f"Error: You must read_file('{path}') first before writing to it. Read the file to understand its current content."

        # If UI is available, show diff and confirm
        if _UI:
            from ..ui.diff import show_file_change, apply_file_change

            approved, action = show_file_change(_UI.console, path, content, _PROJECT_ROOT)

            if not approved:
                return f"Write to {path} cancelled"

            if action == "skip":
                return f"No changes to {path}"

            success = apply_file_change(path, content, _PROJECT_ROOT, action)

            if success:
                return f"✓ Wrote {path}"
            else:
                return f"Error writing {path}"

        # No UI - just write directly
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Successfully wrote to {path}"

    except Exception as e:
        return f"Error writing {path}: {e}"


def get_project_structure() -> str:
    """Get an overview of the project's file structure.

    Returns:
        Tree-like view of the project files
    """
    global _PROJECT_ROOT

    result = [f"Project: {_PROJECT_ROOT.name}", ""]
    excludes = ['node_modules', '__pycache__', 'vendor', '.git', '.next', 'dist', 'build', '.idea', '.vscode']

    def add_tree(path: Path, prefix: str = "", depth: int = 0):
        if depth > 3:
            return

        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return

        dirs = [i for i in items if i.is_dir() and not i.name.startswith('.') and i.name not in excludes]
        files = [i for i in items if i.is_file() and not i.name.startswith('.')]

        for f in files[:10]:
            result.append(f"{prefix}{f.name}")

        for d in dirs[:5]:
            result.append(f"{prefix}{d.name}/")
            add_tree(d, prefix + "  ", depth + 1)

    add_tree(_PROJECT_ROOT)
    return "\n".join(result[:80])


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing a specific string. Use for small, targeted changes.

    Args:
        path: File path relative to project root
        old_string: The exact text to find and replace (must be unique in file)
        new_string: The text to replace it with

    Returns:
        Success or error message
    """
    global _PROJECT_ROOT, _UI, _READ_FILES

    try:
        file_path = _PROJECT_ROOT / path if not path.startswith('/') else Path(path)

        if not file_path.exists():
            return f"Error: File not found: {path}"

        # Require that the file was read first
        resolved = str(file_path.resolve())
        if resolved not in _READ_FILES:
            return f"Error: You must read_file('{path}') first before editing it. Read the file to see its actual content."

        content = file_path.read_text()

        if old_string not in content:
            return f"Error: Could not find the text to replace in {path}"

        # Count occurrences
        count = content.count(old_string)
        if count > 1:
            return f"Error: Found {count} occurrences of the text. Please provide more context to make it unique."

        # Create new content
        new_content = content.replace(old_string, new_string, 1)

        # Show diff and confirm if UI available
        if _UI:
            from ..ui.diff import show_file_change, apply_file_change

            approved, action = show_file_change(_UI.console, path, new_content, _PROJECT_ROOT)

            if not approved:
                return f"Edit to {path} cancelled"

            if action == "skip":
                return f"No changes to {path}"

            success = apply_file_change(path, new_content, _PROJECT_ROOT, action)

            if success:
                return f"✓ Edited {path}"
            else:
                return f"Error editing {path}"

        # No UI - just write directly
        file_path.write_text(new_content)
        return f"Successfully edited {path}"

    except Exception as e:
        return f"Error editing {path}: {e}"


# =============================================================================
# WEB SEARCH
# =============================================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information. Use for current events, facts, documentation, etc.

    Args:
        query: The search query
        max_results: Maximum number of results (default 5)

    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body'][:200]}...")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {e}"


def get_current_news(topic: str) -> str:
    """Get current news about a topic. Use for recent events, breaking news, etc.

    Args:
        topic: The topic to get news about

    Returns:
        Recent news articles about the topic
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.news(topic, max_results=5))

        if not results:
            # Fallback to regular search
            return web_search(f"{topic} latest news", max_results=5)

        formatted = [f"Latest news about: {topic}\n"]
        for i, r in enumerate(results, 1):
            date = r.get('date', 'Recent')
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   Date: {date}\n"
                f"   Source: {r.get('source', 'Unknown')}\n"
                f"   {r['body'][:200]}...\n"
                f"   URL: {r['url']}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"News search failed: {e}. Try web_search instead."


# =============================================================================
# WEATHER
# =============================================================================

def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "London", "New York")

    Returns:
        Current weather conditions
    """
    try:
        import requests

        url = f"https://wttr.in/{city}?format=j1"
        headers = {"User-Agent": "curl/7.68.0"}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]

        return (
            f"Weather in {city}:\n"
            f"  Temperature: {current['temp_C']}°C ({current['temp_F']}°F)\n"
            f"  Feels like: {current['FeelsLikeC']}°C\n"
            f"  Condition: {current['weatherDesc'][0]['value']}\n"
            f"  Humidity: {current['humidity']}%\n"
            f"  Wind: {current['windspeedKmph']} km/h {current['winddir16Point']}"
        )

    except Exception as e:
        return f"Weather lookup failed: {e}"


# =============================================================================
# DATE/TIME
# =============================================================================

def get_current_time(timezone: str = "UTC") -> str:
    """Get current date and time.

    Args:
        timezone: Timezone name (e.g., "UTC", "America/New_York", "Europe/London")

    Returns:
        Current date and time
    """
    try:
        from datetime import datetime
        import zoneinfo

        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)

        return (
            f"Current time ({timezone}):\n"
            f"  Date: {now.strftime('%Y-%m-%d')}\n"
            f"  Time: {now.strftime('%H:%M:%S')}\n"
            f"  Day: {now.strftime('%A')}\n"
            f"  ISO: {now.isoformat()}"
        )

    except Exception as e:
        return f"Time lookup failed: {e}"


# =============================================================================
# CALCULATOR
# =============================================================================

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")

    Returns:
        Calculation result
    """
    import math
    import re

    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
        "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
        "pi": math.pi, "e": math.e, "tau": math.tau,
    }

    try:
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\^a-zA-Z_]+$', expression):
            return "Error: Invalid characters in expression"

        expression = expression.replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return f"{expression} = {result}"

    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


# =============================================================================
# MEMORY / NOTES
# =============================================================================

def save_memory(content: str, category: str = "general") -> str:
    """Save information to memory for later recall.

    Args:
        content: Information to remember
        category: Category (e.g., "user_preferences", "project_notes", "general")

    Returns:
        Confirmation message
    """
    try:
        from datetime import datetime

        memory_dir = Path(__file__).parent.parent / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_file = memory_dir / "memories.md"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{category}] {timestamp}\n{content}\n"

        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(entry)

        return f"Saved to memory under '{category}'"

    except Exception as e:
        return f"Failed to save memory: {e}"


def recall_memory(query: str = "") -> str:
    """Recall saved memories, optionally filtered by search query.

    Args:
        query: Optional search term to filter memories

    Returns:
        Matching memories or all memories if no query
    """
    try:
        memory_file = Path(__file__).parent.parent / "memory" / "memories.md"

        if not memory_file.exists():
            return "No memories saved yet."

        content = memory_file.read_text()

        if query:
            # Filter to matching sections
            sections = content.split("\n## ")
            matches = [s for s in sections if query.lower() in s.lower()]
            if matches:
                return "## " + "\n\n## ".join(matches[-10:])  # Last 10 matches
            return f"No memories matching '{query}'"

        # Return last 2000 chars
        if len(content) > 2000:
            return "...(earlier memories truncated)...\n" + content[-2000:]
        return content

    except Exception as e:
        return f"Failed to recall memory: {e}"


# =============================================================================
# GITHUB
# =============================================================================

def github_search(query: str, search_type: str = "repos") -> str:
    """Search GitHub for repositories, code, issues, or users.

    Args:
        query: Search query
        search_type: Type of search - "repos", "code", "issues", "users"

    Returns:
        Search results
    """
    try:
        result = subprocess.run(
            ["gh", "search", search_type, query, "--limit", "5"],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0 and result.stdout:
            return result.stdout
        elif result.stderr:
            return f"GitHub search error: {result.stderr}"
        else:
            return f"No {search_type} found for: {query}"

    except FileNotFoundError:
        return "GitHub CLI (gh) not installed. Install with: brew install gh"
    except Exception as e:
        return f"GitHub search failed: {e}"


# =============================================================================
# ALL TOOLS LIST
# =============================================================================

ALL_TOOLS = [
    # File operations
    read_file,
    list_files,
    search_files,
    write_file,
    edit_file,
    get_project_structure,
    # Shell
    run_command,
    # Web
    web_search,
    get_current_news,
    # Weather
    get_weather,
    # Time
    get_current_time,
    # Math
    calculate,
    # Memory
    save_memory,
    recall_memory,
    # GitHub
    github_search,
]
