"""
Diff view for file changes - like Claude Code.
"""

import difflib
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text


def get_file_extension(path: str) -> str:
    """Get file extension for syntax highlighting."""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.css': 'css',
        '.scss': 'scss',
        '.html': 'html',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
    }
    suffix = Path(path).suffix.lower()
    return ext_map.get(suffix, 'text')


def create_diff(old_content: str, new_content: str, filename: str = "file") -> list:
    """Create a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    ))

    return diff


def print_diff(console: Console, old_content: str, new_content: str, filepath: str):
    """Print a colored diff."""

    filename = Path(filepath).name
    diff_lines = create_diff(old_content, new_content, filename)

    if not diff_lines:
        console.print("[dim]No changes[/dim]")
        return

    # Build colored diff output
    output = Text()

    for line in diff_lines:
        line = line.rstrip('\n')

        if line.startswith('---'):
            output.append(line + "\n", style="bold red")
        elif line.startswith('+++'):
            output.append(line + "\n", style="bold green")
        elif line.startswith('@@'):
            output.append(line + "\n", style="cyan")
        elif line.startswith('-'):
            output.append(line + "\n", style="red")
        elif line.startswith('+'):
            output.append(line + "\n", style="green")
        else:
            output.append(line + "\n", style="dim")

    # Print in a panel
    console.print(Panel(
        output,
        title=f"[bold]Changes to {filepath}[/bold]",
        border_style="blue",
        padding=(0, 1)
    ))


def print_new_file(console: Console, content: str, filepath: str):
    """Print a new file being created."""
    lang = get_file_extension(filepath)
    lines = len(content.splitlines())

    # Show preview (first 20 lines)
    preview_lines = content.splitlines()[:20]
    preview = "\n".join(preview_lines)
    if len(content.splitlines()) > 20:
        preview += f"\n... ({lines - 20} more lines)"

    syntax = Syntax(preview, lang, theme="monokai", line_numbers=True)

    console.print(Panel(
        syntax,
        title=f"[bold green]New file: {filepath}[/bold green] ({lines} lines)",
        border_style="green",
        padding=(0, 1)
    ))


def show_file_change(console: Console, filepath: str, new_content: str, project_root: Path) -> Tuple[bool, str]:
    """
    Show file change and ask for confirmation.

    Returns:
        (approved, action) - whether approved and what action ("write", "skip", "edit")
    """
    full_path = project_root / filepath if not filepath.startswith('/') else Path(filepath)

    # Check if file exists
    if full_path.exists():
        try:
            old_content = full_path.read_text()

            # Check if content is same
            if old_content == new_content:
                console.print(f"[dim]No changes to {filepath}[/dim]")
                return True, "skip"

            print_diff(console, old_content, new_content, filepath)

        except Exception as e:
            console.print(f"[red]Error reading {filepath}: {e}[/red]")
            return False, "error"
    else:
        print_new_file(console, new_content, filepath)

    # Ask for confirmation
    console.print()
    console.print("[bold]Apply this change?[/bold]")
    console.print("[dim]  y = yes, n = no, e = edit in $EDITOR[/dim]")

    try:
        response = input("> ").strip().lower()

        if response in ('y', 'yes', ''):
            return True, "write"
        elif response in ('e', 'edit'):
            return True, "edit"
        else:
            return False, "skip"

    except (EOFError, KeyboardInterrupt):
        console.print()
        return False, "skip"


def apply_file_change(filepath: str, content: str, project_root: Path, action: str) -> bool:
    """Apply the file change based on action."""
    import os
    import tempfile
    import subprocess

    full_path = project_root / filepath if not filepath.startswith('/') else Path(filepath)

    if action == "skip":
        return False

    if action == "edit":
        # Write to temp file and open in editor
        editor = os.environ.get('EDITOR', 'nano')

        with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filepath).suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            subprocess.run([editor, temp_path])

            # Read edited content
            with open(temp_path) as f:
                content = f.read()
        finally:
            os.unlink(temp_path)

    # Write the file
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return True
    except Exception:
        return False
