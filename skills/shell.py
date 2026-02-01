"""Shell command skill with safety restrictions"""

import subprocess
import shlex

# Commands that are safe to run without confirmation
SAFE_COMMANDS = {
    'ls', 'cat', 'head', 'tail', 'less', 'more',
    'grep', 'find', 'which', 'whereis', 'file',
    'pwd', 'echo', 'date', 'cal', 'whoami',
    'git status', 'git log', 'git diff', 'git branch',
    'npm list', 'yarn list', 'composer show',
    'python --version', 'php --version', 'node --version',
    'ollama list', 'ollama ps',
}

# Commands that are never allowed
BLOCKED_PATTERNS = [
    'rm -rf',
    'sudo',
    '> /dev',
    'mkfs',
    'dd if=',
    ':(){',  # Fork bomb
    'chmod 777',
    '| sh',
    '| bash',
]


def is_safe_command(command: str) -> tuple[bool, str]:
    """Check if a command is safe to run."""
    cmd_lower = command.lower().strip()

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return False, f"Blocked: contains '{pattern}'"

    # Check if it's a known safe command
    base_cmd = cmd_lower.split()[0] if cmd_lower else ""

    # Git commands need special handling
    if base_cmd == 'git':
        git_action = ' '.join(cmd_lower.split()[:2])
        if git_action in ['git status', 'git log', 'git diff', 'git branch']:
            return True, "Safe git command"
        return False, "Git command needs confirmation"

    if base_cmd in {'ls', 'cat', 'head', 'tail', 'pwd', 'echo', 'date', 'which', 'file', 'ollama'}:
        return True, "Safe command"

    return False, "Command needs confirmation"


def shell_run(command: str, timeout: int = 30, force: bool = False) -> dict:
    """
    Run a shell command with safety checks.

    Args:
        command: The command to run
        timeout: Maximum seconds to wait
        force: Skip safety check (use with caution)

    Returns:
        Dict with stdout, stderr, return_code, and safety info
    """
    safe, reason = is_safe_command(command)

    if not safe and not force:
        return {
            "success": False,
            "needs_confirmation": True,
            "reason": reason,
            "command": command
        }

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None  # Uses current directory
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "needs_confirmation": False
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "needs_confirmation": False
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "needs_confirmation": False
        }
