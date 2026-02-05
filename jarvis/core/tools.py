"""
Tools for Jarvis - Used with native tool calling.

These functions are passed to the LLM's tools parameter.
Includes all skills that are useful for agentic behavior.
"""

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global project root and UI reference
_PROJECT_ROOT: Path = Path.cwd()
_UI = None
_PENDING_WRITES = {}  # filepath -> content, for confirmation
_READ_FILES = set()  # Track files that have been read this session

# Task management storage
_TASKS: dict = {}
_TASK_COUNTER: int = 0


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


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit a file by replacing a specific string. Use for small, targeted changes.

    Args:
        path: File path relative to project root
        old_string: The exact text to find and replace
        new_string: The text to replace it with
        replace_all: If True, replace all occurrences (default False requires unique match)

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

        if count > 1 and not replace_all:
            return f"Error: Found {count} occurrences. Use replace_all=True or provide more context to make it unique."

        # Create new content
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
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
                msg = f"✓ Edited {path}"
                if replace_all and count > 1:
                    msg += f" ({count} replacements)"
                return msg
            else:
                return f"Error editing {path}"

        # No UI - just write directly
        file_path.write_text(new_content)
        msg = f"Successfully edited {path}"
        if replace_all and count > 1:
            msg += f" ({count} replacements)"
        return msg

    except Exception as e:
        return f"Error editing {path}: {e}"


# =============================================================================
# GIT OPERATIONS
# =============================================================================

def git_status() -> str:
    """Get git repository status including branch, modified, staged, and untracked files.

    Returns:
        Formatted git status with branch info and file changes
    """
    global _PROJECT_ROOT

    try:
        # Get branch info
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        branch = branch_result.stdout.strip() or "HEAD detached"

        # Get status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error: {result.stderr or 'Not a git repository'}"

        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        staged = []
        modified = []
        untracked = []

        for line in lines:
            if not line:
                continue
            status = line[:2]
            filename = line[3:]

            if status[0] in 'MADRC':
                staged.append(f"  {filename}")
            if status[1] == 'M':
                modified.append(f"  {filename}")
            elif status == '??':
                untracked.append(f"  {filename}")

        output = [f"On branch: {branch}", ""]

        if staged:
            output.append("Staged changes:")
            output.extend(staged[:20])
            if len(staged) > 20:
                output.append(f"  ... and {len(staged) - 20} more")

        if modified:
            output.append("\nModified (not staged):")
            output.extend(modified[:20])
            if len(modified) > 20:
                output.append(f"  ... and {len(modified) - 20} more")

        if untracked:
            output.append("\nUntracked files:")
            output.extend(untracked[:10])
            if len(untracked) > 10:
                output.append(f"  ... and {len(untracked) - 10} more")

        if not staged and not modified and not untracked:
            output.append("Working tree clean")

        return "\n".join(output)

    except Exception as e:
        return f"Error: {e}"


def git_diff(staged: bool = False, file: str = "") -> str:
    """Show git diff of uncommitted changes.

    Args:
        staged: If True, show staged changes only (default: False shows unstaged)
        file: Optional specific file to diff

    Returns:
        Unified diff output showing changes
    """
    global _PROJECT_ROOT

    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if file:
            cmd.extend(["--", file])

        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        diff = result.stdout.strip()
        if not diff:
            return "No changes" + (" staged" if staged else "")

        # Truncate if too long
        if len(diff) > 10000:
            diff = diff[:10000] + "\n\n... [TRUNCATED - diff is large]"

        return diff

    except Exception as e:
        return f"Error: {e}"


def git_log(count: int = 10, oneline: bool = True) -> str:
    """Show recent git commit history.

    Args:
        count: Number of commits to show (default 10)
        oneline: Use compact format (default True)

    Returns:
        Commit history with hashes, authors, dates, and messages
    """
    global _PROJECT_ROOT

    try:
        if oneline:
            cmd = ["git", "log", f"-{count}", "--oneline", "--decorate"]
        else:
            cmd = ["git", "log", f"-{count}", "--format=%h %s (%an, %ar)"]

        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stdout.strip() or "No commits found"

    except Exception as e:
        return f"Error: {e}"


def git_commit(message: str, files: str = "") -> str:
    """Create a git commit with staged changes.

    Args:
        message: Commit message (required)
        files: Optional space-separated list of files to add before commit

    Returns:
        Success message with commit hash, or error
    """
    global _PROJECT_ROOT, _UI

    try:
        # Add files if specified
        if files:
            file_list = files.split()
            add_result = subprocess.run(
                ["git", "add"] + file_list,
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )
            if add_result.returncode != 0:
                return f"Error adding files: {add_result.stderr}"

        # Check if there are staged changes
        status = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            timeout=10
        )

        if status.returncode == 0:
            return "Error: No changes staged for commit. Use git_add first."

        # Confirm with user if UI available
        if _UI:
            _UI.console.print(f"[yellow]Commit: {message[:60]}{'...' if len(message) > 60 else ''}[/yellow]")
            _UI.console.print("[dim]  y = yes, n = no[/dim]")
            try:
                response = input("> ").strip().lower()
                if response not in ('y', 'yes'):
                    return "Commit cancelled by user"
            except (EOFError, KeyboardInterrupt):
                return "Commit cancelled"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Get the commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        commit_hash = hash_result.stdout.strip()

        return f"✓ Committed {commit_hash}: {message}"

    except Exception as e:
        return f"Error: {e}"


def git_add(files: str = ".") -> str:
    """Stage files for commit.

    Args:
        files: Files to stage - space-separated paths, or "." for all (default ".")

    Returns:
        Success message with staged files
    """
    global _PROJECT_ROOT

    try:
        file_list = files.split() if files != "." else ["."]

        result = subprocess.run(
            ["git", "add"] + file_list,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Show what was staged
        status = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )

        staged = status.stdout.strip().split('\n') if status.stdout.strip() else []

        if staged:
            return f"✓ Staged {len(staged)} file(s):\n" + "\n".join(f"  {f}" for f in staged[:20])
        return "No changes to stage"

    except Exception as e:
        return f"Error: {e}"


def git_branch(name: str = "", create: bool = False, switch: bool = False) -> str:
    """List, create, or switch git branches.

    Args:
        name: Branch name (required for create/switch, optional for list)
        create: Create new branch if True
        switch: Switch to branch if True

    Returns:
        Branch list or operation result
    """
    global _PROJECT_ROOT

    try:
        if create and name:
            result = subprocess.run(
                ["git", "checkout", "-b", name],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return f"✓ Created and switched to branch: {name}"

        elif switch and name:
            result = subprocess.run(
                ["git", "checkout", name],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return f"✓ Switched to branch: {name}"

        else:
            # List branches
            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return result.stdout.strip() or "No branches found"

    except Exception as e:
        return f"Error: {e}"


def git_stash(action: str = "push", message: str = "") -> str:
    """Stash or restore uncommitted changes.

    Args:
        action: "push" (save), "pop" (restore & remove), "list", or "apply" (restore & keep)
        message: Optional stash message (for push only)

    Returns:
        Stash operation result
    """
    global _PROJECT_ROOT

    try:
        if action == "push":
            cmd = ["git", "stash", "push"]
            if message:
                cmd.extend(["-m", message])
        elif action == "pop":
            cmd = ["git", "stash", "pop"]
        elif action == "apply":
            cmd = ["git", "stash", "apply"]
        elif action == "list":
            cmd = ["git", "stash", "list"]
        else:
            return f"Error: Unknown action '{action}'. Use: push, pop, apply, list"

        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            if "No stash entries" in result.stderr or "No local changes" in result.stderr:
                return "No stash entries found" if action in ("pop", "apply", "list") else "No changes to stash"
            return f"Error: {result.stderr}"

        output = result.stdout.strip() or result.stderr.strip()

        if action == "push":
            return f"✓ Stashed changes" + (f": {message}" if message else "")
        elif action == "pop":
            return "✓ Applied and removed stash"
        elif action == "apply":
            return "✓ Applied stash (kept in stash list)"
        else:
            return output or "No stash entries"

    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# GLOB AND ENHANCED SEARCH
# =============================================================================

def glob_files(pattern: str, path: str = "") -> str:
    """Find files matching a glob pattern, sorted by modification time.

    Args:
        pattern: Glob pattern like "**/*.py", "src/**/*.ts", "*.md"
        path: Base directory (default: project root)

    Returns:
        Newline-separated list of matching file paths, most recently modified first
    """
    global _PROJECT_ROOT

    try:
        base = Path(path) if path and path.startswith('/') else _PROJECT_ROOT / (path or "")

        if not base.exists():
            return f"Error: Path not found: {path or str(_PROJECT_ROOT)}"

        # Exclude common ignored directories
        excludes = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', '.next', 'dist', 'build', '.idea', '.vscode', 'site-packages'}

        matches = []
        for p in base.glob(pattern):
            if any(ex in p.parts for ex in excludes):
                continue
            if p.is_file():
                try:
                    matches.append((p, p.stat().st_mtime))
                except OSError:
                    continue

        # Sort by modification time (newest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return relative paths
        result = []
        for p, _ in matches[:100]:  # Limit to 100 results
            try:
                result.append(str(p.relative_to(_PROJECT_ROOT)))
            except ValueError:
                result.append(str(p))

        if not result:
            return f"No files matching '{pattern}'"

        return "\n".join(result)

    except Exception as e:
        return f"Error: {e}"


def grep(
    pattern: str,
    path: str = ".",
    file_type: str = "",
    glob_pattern: str = "",
    context: int = 0,
    ignore_case: bool = False,
    max_results: int = 50
) -> str:
    """Search for patterns in files using regex.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in (default: ".")
        file_type: File extension filter like "py", "js", "ts"
        glob_pattern: Glob pattern filter like "*.py", "**/*.tsx"
        context: Lines of context before and after each match
        ignore_case: Case-insensitive search (default False)
        max_results: Maximum results to return (default 50)

    Returns:
        Search results with file paths and line numbers
    """
    global _PROJECT_ROOT

    try:
        # Compile regex
        flags = re.IGNORECASE if ignore_case else 0

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        search_path = _PROJECT_ROOT / path if not path.startswith('/') else Path(path)
        excludes = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build', '.next', 'site-packages'}

        # Determine file pattern
        if glob_pattern:
            files = search_path.glob(glob_pattern) if '**' in glob_pattern else search_path.rglob(glob_pattern)
        elif file_type:
            files = search_path.rglob(f"*.{file_type}")
        else:
            files = search_path.rglob("*")

        results = []
        match_count = 0

        for filepath in files:
            if not filepath.is_file():
                continue
            if any(ex in filepath.parts for ex in excludes):
                continue
            # Skip binary files
            if filepath.suffix in {'.pyc', '.exe', '.dll', '.so', '.dylib', '.png', '.jpg', '.gif', '.ico', '.pdf', '.zip'}:
                continue

            try:
                content = filepath.read_text(errors='ignore')
                lines = content.splitlines()

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        try:
                            rel_path = str(filepath.relative_to(_PROJECT_ROOT))
                        except ValueError:
                            rel_path = str(filepath)

                        if context > 0:
                            ctx_lines = []
                            start = max(0, line_num - 1 - context)
                            end = min(len(lines), line_num + context)
                            for i in range(start, end):
                                prefix = ">" if i == line_num - 1 else " "
                                ctx_lines.append(f"  {prefix} {i+1}: {lines[i][:200]}")
                            results.append(f"{rel_path}:\n" + "\n".join(ctx_lines))
                        else:
                            results.append(f"{rel_path}:{line_num}: {line[:200]}")

                        match_count += 1
                        if match_count >= max_results:
                            break

            except Exception:
                continue

            if match_count >= max_results:
                break

        if not results:
            return f"No matches for '{pattern}'"

        output = "\n\n".join(results) if context > 0 else "\n".join(results)
        if match_count >= max_results:
            output += f"\n\n... (limited to {max_results} results)"

        return output

    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TASK MANAGEMENT
# =============================================================================

def task_create(subject: str, description: str = "") -> str:
    """Create a new task to track work.

    Args:
        subject: Brief task title (imperative form, e.g., "Fix auth bug")
        description: Detailed description of what needs to be done

    Returns:
        Task ID and confirmation
    """
    global _TASKS, _TASK_COUNTER

    _TASK_COUNTER += 1
    task_id = str(_TASK_COUNTER)

    _TASKS[task_id] = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    return f"✓ Created task #{task_id}: {subject}"


def task_update(task_id: str, status: str = "", description: str = "") -> str:
    """Update an existing task.

    Args:
        task_id: The task ID to update
        status: New status: "pending", "in_progress", "completed", or "deleted"
        description: Updated description (optional)

    Returns:
        Updated task details
    """
    global _TASKS

    if task_id not in _TASKS:
        return f"Error: Task #{task_id} not found"

    task = _TASKS[task_id]

    if status:
        valid = {"pending", "in_progress", "completed", "deleted"}
        if status not in valid:
            return f"Error: Invalid status. Use: {', '.join(valid)}"

        if status == "deleted":
            del _TASKS[task_id]
            return f"✓ Deleted task #{task_id}"

        task["status"] = status

    if description:
        task["description"] = description

    task["updated_at"] = datetime.now().isoformat()

    return f"✓ Updated task #{task_id}: [{task['status']}] {task['subject']}"


def task_list() -> str:
    """List all tasks with their status.

    Returns:
        Formatted list of tasks with IDs, subjects, and statuses
    """
    global _TASKS

    if not _TASKS:
        return "No tasks. Use task_create to add tasks."

    lines = ["Tasks:"]
    for task_id, task in sorted(_TASKS.items(), key=lambda x: int(x[0])):
        status_icon = {
            "pending": "○",
            "in_progress": "◐",
            "completed": "●"
        }.get(task["status"], "?")
        lines.append(f"  {status_icon} #{task_id}: {task['subject']} [{task['status']}]")

    return "\n".join(lines)


def task_get(task_id: str) -> str:
    """Get full details of a specific task.

    Args:
        task_id: The task ID to retrieve

    Returns:
        Full task details including description
    """
    global _TASKS

    if task_id not in _TASKS:
        return f"Error: Task #{task_id} not found"

    task = _TASKS[task_id]

    return f"""Task #{task['id']}
Subject: {task['subject']}
Status: {task['status']}
Created: {task['created_at']}
Updated: {task['updated_at']}

Description:
{task['description'] or '(no description)'}"""


# =============================================================================
# WEB SEARCH
# =============================================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information using Brave Search (primary) or DuckDuckGo (fallback).

    Args:
        query: The search query
        max_results: Maximum number of results (default 5)

    Returns:
        Search results with titles, URLs, and snippets
    """
    import os
    import httpx

    today = datetime.now()
    current_year = today.year

    # Add date-awareness for current events queries
    # This helps get fresh results for time-sensitive queries
    current_keywords = [
        'president', 'current', 'now', 'today', 'latest', 'recent',
        'prime minister', 'ceo', 'leader', 'chairman', 'governor',
        'price', 'rate', 'stock', 'market', 'breaking', 'news'
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in current_keywords):
        # Add current year if not already present
        if str(current_year) not in query and str(current_year - 1) not in query:
            query = f"{query} {current_year}"

    # === BRAVE SEARCH (Primary) ===
    brave_key = os.getenv("BRAVE_API_KEY")
    if brave_key:
        print(f"[web_search] Using Brave Search for: {query[:50]}...")
        try:
            resp = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results, "text_decorations": False},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": brave_key
                },
                timeout=15.0,
            )
            print(f"[web_search] Brave response status: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                web_results = (data.get("web") or {}).get("results") or []
                if web_results:
                    formatted = [f"Search results (Brave, {today.strftime('%Y-%m-%d')}):\n"]
                    for i, r in enumerate(web_results[:max_results], 1):
                        title = str(r.get("title") or "Untitled")
                        url = str(r.get("url") or "")
                        desc = str(r.get("description") or "")
                        if len(desc) > 300:
                            desc = desc[:300]
                        formatted.append(f"{i}. {title}\n   URL: {url}\n   {desc}")
                    return "\n\n".join(formatted)
                else:
                    print("[web_search] Brave returned no results")
            elif resp.status_code == 401:
                print("[web_search] Brave API: Invalid API key")
            elif resp.status_code == 429:
                print("[web_search] Brave API: Rate limited")
            else:
                error_text = str(resp.text or "")[:200] if resp.text else ""
                print(f"[web_search] Brave API error: {resp.status_code} - {error_text}")
        except Exception as e:
            print(f"[web_search] Brave Search failed: {e}")

    # === DUCKDUCKGO (Fallback) ===
    print(f"[web_search] Falling back to DuckDuckGo for: {query[:50]}...")
    try:
        from ddgs import DDGS
        import time

        # Ensure max_results is an integer (fixes type errors in ddgs library)
        max_results = int(max_results) if max_results else 5

        results = []
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    # Use text() - ddgs v9 API
                    search_results = ddgs.text(query, max_results=max_results)
                    results = list(search_results) if search_results else []
                    if results:
                        break
                if attempt < 1:
                    time.sleep(0.5)
            except Exception as e:
                print(f"[web_search] DuckDuckGo attempt {attempt+1} failed: {e}")
                continue

        if results:
            formatted = [f"Search results (DuckDuckGo, {today.strftime('%Y-%m-%d')}):\n"]
            for i, r in enumerate(results, 1):
                try:
                    title = str(r.get('title', 'Untitled') if isinstance(r, dict) else 'Untitled')
                    url = str(r.get('href', r.get('url', '')) if isinstance(r, dict) else '')
                    body = str(r.get('body', r.get('description', '')) if isinstance(r, dict) else '')
                    if len(body) > 300:
                        body = body[:300]
                    formatted.append(f"{i}. {title}\n   URL: {url}\n   {body}")
                except Exception:
                    continue
            if len(formatted) > 1:
                return "\n\n".join(formatted)

    except ImportError:
        print("[web_search] ddgs not installed")
    except Exception as e:
        print(f"[web_search] DuckDuckGo error: {e}")

    # === WIKIPEDIA (Last resort for factual queries) ===
    print(f"[web_search] Trying Wikipedia for: {query[:50]}...")
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3},
            timeout=10.0
        )
        if resp.status_code == 200:
            wiki_results = resp.json().get("query", {}).get("search", [])
            if wiki_results:
                formatted = [f"Wikipedia results for '{query}':\n"]
                for i, r in enumerate(wiki_results, 1):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")[:250]
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    formatted.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
                return "\n\n".join(formatted)
    except Exception as e:
        print(f"[web_search] Wikipedia error: {e}")

    return f"Search failed: No results found for '{query}'. Please try a different query."


# =============================================================================
# GOLD PRICE (GoldAPI.io)
# =============================================================================

def get_gold_price(currency: str = "USD") -> str:
    """Get current gold spot price using GoldAPI.io.

    Args:
        currency: Quote currency (e.g., "USD", "GBP", "EUR")

    Returns:
        Current gold price and timestamp or an error message
    """
    try:
        import os
        import requests

        api_key = os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY")
        if not api_key:
            return "Error: GOLDAPI_KEY not configured"

        cur = (currency or "USD").upper()
        url = f"https://www.goldapi.io/api/XAU/{cur}"
        headers = {"x-access-token": api_key}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 401:
            return "Error: GoldAPI.io unauthorized (check GOLDAPI_KEY)"
        if response.status_code == 429:
            return "Error: GoldAPI.io rate limit exceeded"
        response.raise_for_status()
        data = response.json()

        price = data.get("price")
        ts = data.get("timestamp")
        if price is None:
            return "Error: GoldAPI.io response missing price"

        # Timestamp is Unix seconds
        ts_str = None
        try:
            from datetime import datetime, timezone
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
        except Exception:
            ts_str = None

        return (
            f"Gold spot price (XAU/{cur}): {price}\n"
            + (f"Timestamp (UTC): {ts_str}" if ts_str else "")
        ).strip()

    except Exception as e:
        return f"Gold price lookup failed: {e}"


def web_fetch(url: str) -> str:
    """Fetch content from a URL and convert HTML to readable text.

    Args:
        url: The URL to fetch

    Returns:
        Page content converted to markdown/text format
    """
    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')

        if 'text/html' in content_type:
            # Try to use html2text if available
            try:
                from html2text import HTML2Text
                h = HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                content = h.handle(response.text)
            except ImportError:
                # Fallback: basic HTML tag removal
                import re
                text = response.text
                # Remove script and style elements
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                content = text.strip()
        else:
            content = response.text

        # Truncate if too long
        if len(content) > 15000:
            content = content[:15000] + "\n\n... [TRUNCATED]"

        return f"Content from {url}:\n\n{content}"

    except requests.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error: {e}"


def get_current_news(topic: str) -> str:
    """Get current news about a topic. Use for recent events, breaking news, etc.

    Args:
        topic: The topic to get news about

    Returns:
        Recent news articles about the topic
    """
    import time

    if not topic or not topic.strip():
        return "Error: No topic provided for news search."

    topic = topic.strip()
    results = []

    # Try DuckDuckGo news
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            # ddgs v9 API
            news_results = ddgs.news(topic, max_results=5)
            results = list(news_results) if news_results else []

    except Exception as e:
        print(f"[get_current_news] DuckDuckGo news error: {e}")

    if results:
        today = datetime.now()
        formatted = [f"Latest news about '{topic}' (as of {today.strftime('%Y-%m-%d')}):\n"]
        for i, r in enumerate(results, 1):
            date = r.get('date', 'Recent')
            title = r.get('title', 'Untitled')
            source = r.get('source', 'Unknown')
            body = r.get('body', '')[:200]
            url = r.get('url', '')
            formatted.append(
                f"{i}. {title}\n"
                f"   Date: {date}\n"
                f"   Source: {source}\n"
                f"   {body}...\n"
                f"   URL: {url}"
            )
        return "\n\n".join(formatted)

    # Fallback to regular web search with news qualifier
    time.sleep(0.3)  # Small delay
    return web_search(f"{topic} latest news {datetime.now().year}", max_results=5)


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
        import os
        import requests
        from jarvis.assistant import load_config

        config = load_config()
        weather_cfg = (config.get("integrations", {}) or {}).get("weather", {})
        provider = weather_cfg.get("provider", "wttr")
        units = weather_cfg.get("units", "metric")
        api_key = os.getenv("OPENWEATHER_API_KEY")

        if provider == "openweather" and api_key:
            # OpenWeatherMap API (optional)
            params = {
                "q": city,
                "appid": api_key,
                "units": "metric" if units == "metric" else "imperial",
            }
            response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            temp_c = data["main"]["temp"]
            feels_c = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_kph = round(data["wind"]["speed"] * 3.6, 1)  # m/s to km/h
            desc = data["weather"][0]["description"]
            return (
                f"Weather in {city}:\n"
                f"  Temperature: {temp_c}°C\n"
                f"  Feels like: {feels_c}°C\n"
                f"  Condition: {desc}\n"
                f"  Humidity: {humidity}%\n"
                f"  Wind: {wind_kph} km/h"
            )

        # Default: free wttr.in
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
    glob_files,
    grep,
    # Git operations
    git_status,
    git_diff,
    git_log,
    git_commit,
    git_add,
    git_branch,
    git_stash,
    # Shell
    run_command,
    # Web
    web_search,
    web_fetch,
    get_current_news,
    get_gold_price,
    # Weather
    get_weather,
    # Time
    get_current_time,
    # Math
    calculate,
    # Memory
    save_memory,
    recall_memory,
    # Task management
    task_create,
    task_update,
    task_list,
    task_get,
    # GitHub
    github_search,
]
