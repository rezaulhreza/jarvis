"""
Tools for Jarvis - Used with native tool calling.

These functions are passed to the LLM's tools parameter.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Callable

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
            content = content[:15000] + "\n\n... (truncated, file too large)"

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
    """Write content to a file. YOU MUST USE THIS when asked to write, save, create, or refactor a file.

    IMPORTANT: When the user asks you to write/save/create/modify a file, call this tool.
    Do NOT just output code in your response - use this tool to actually write it.

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

    Use this tool when asked to modify, update, or fix a specific part of a file.
    For full file rewrites, use write_file instead.

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


# All tools for LLM
ALL_TOOLS = [
    read_file,
    list_files,
    search_files,
    run_command,
    write_file,
    edit_file,
    get_project_structure,
]
