"""GitHub operations skill"""

import os
import subprocess
from typing import Optional


def _run_gh(args: list[str]) -> dict:
    """Run a gh CLI command."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GH_TOKEN": os.getenv("GITHUB_TOKEN", "")}
        )

        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip()}
        else:
            return {"success": False, "error": result.stderr.strip()}

    except FileNotFoundError:
        return {"success": False, "error": "GitHub CLI (gh) not installed. Run: brew install gh"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_repos(limit: int = 10) -> dict:
    """List your GitHub repositories."""
    return _run_gh(["repo", "list", "--limit", str(limit)])


def repo_info(repo: str) -> dict:
    """Get info about a repository."""
    return _run_gh(["repo", "view", repo])


def list_issues(repo: str, state: str = "open", limit: int = 10) -> dict:
    """List issues for a repository."""
    return _run_gh(["issue", "list", "-R", repo, "--state", state, "--limit", str(limit)])


def create_issue(repo: str, title: str, body: str = "") -> dict:
    """Create a new issue."""
    args = ["issue", "create", "-R", repo, "--title", title]
    if body:
        args.extend(["--body", body])
    return _run_gh(args)


def list_prs(repo: str, state: str = "open", limit: int = 10) -> dict:
    """List pull requests for a repository."""
    return _run_gh(["pr", "list", "-R", repo, "--state", state, "--limit", str(limit)])


def pr_status(repo: str) -> dict:
    """Get status of pull requests in current repo."""
    return _run_gh(["pr", "status", "-R", repo])


def clone_repo(repo: str, directory: Optional[str] = None) -> dict:
    """Clone a repository."""
    args = ["repo", "clone", repo]
    if directory:
        args.append(directory)
    return _run_gh(args)


def create_repo(name: str, private: bool = True, description: str = "") -> dict:
    """Create a new repository."""
    args = ["repo", "create", name]
    if private:
        args.append("--private")
    else:
        args.append("--public")
    if description:
        args.extend(["--description", description])
    return _run_gh(args)
