# Skills module
# Each skill is a tool the assistant can use

from .web_search import web_search
from .shell import shell_run, is_safe_command
from .file_ops import read_file, list_directory
from .memory_ops import save_fact, get_facts
from .weather import get_weather, get_forecast
from .github_ops import list_repos, repo_info, list_issues, create_issue, list_prs
from .datetime_ops import get_current_time, convert_timezone, add_time, time_until
from .calculator import calculate, convert_units, percentage
from .notes import quick_note, list_notes, read_note, search_notes

# Telegram requires async - import separately when needed
# from .telegram import send_message, get_updates

AVAILABLE_SKILLS = {
    # Web
    "web_search": {
        "function": web_search,
        "description": "Search the web for information",
        "parameters": {"query": "string - the search query"}
    },

    # Shell
    "shell_run": {
        "function": shell_run,
        "description": "Run a safe shell command",
        "parameters": {"command": "string - the command to run"}
    },

    # Files
    "read_file": {
        "function": read_file,
        "description": "Read contents of a file",
        "parameters": {"path": "string - path to the file"}
    },
    "list_directory": {
        "function": list_directory,
        "description": "List files in a directory",
        "parameters": {"path": "string - path to the directory"}
    },

    # Memory
    "save_fact": {
        "function": save_fact,
        "description": "Save a fact about the user to memory",
        "parameters": {"fact": "string - the fact to remember"}
    },
    "get_facts": {
        "function": get_facts,
        "description": "Retrieve saved facts about the user",
        "parameters": {}
    },

    # Weather
    "get_weather": {
        "function": get_weather,
        "description": "Get current weather for a city",
        "parameters": {"city": "string - city name"}
    },
    "get_forecast": {
        "function": get_forecast,
        "description": "Get weather forecast",
        "parameters": {"city": "string - city name", "days": "int - number of days (1-5)"}
    },

    # GitHub
    "github_repos": {
        "function": list_repos,
        "description": "List your GitHub repositories",
        "parameters": {"limit": "int - max repos to show"}
    },
    "github_issues": {
        "function": list_issues,
        "description": "List issues for a repository",
        "parameters": {"repo": "string - repo name (owner/repo)"}
    },
    "github_prs": {
        "function": list_prs,
        "description": "List pull requests for a repository",
        "parameters": {"repo": "string - repo name (owner/repo)"}
    },

    # Date/Time
    "current_time": {
        "function": get_current_time,
        "description": "Get current date and time",
        "parameters": {"timezone": "string - timezone (default: Europe/London)"}
    },
    "convert_timezone": {
        "function": convert_timezone,
        "description": "Convert time between timezones",
        "parameters": {
            "time_str": "string - time to convert",
            "from_tz": "string - source timezone",
            "to_tz": "string - target timezone"
        }
    },
    "time_until": {
        "function": time_until,
        "description": "Calculate time until a date",
        "parameters": {"target": "string - target date (YYYY-MM-DD)"}
    },

    # Calculator
    "calculate": {
        "function": calculate,
        "description": "Evaluate a math expression",
        "parameters": {"expression": "string - math expression"}
    },
    "convert_units": {
        "function": convert_units,
        "description": "Convert between units",
        "parameters": {
            "value": "float - value to convert",
            "from_unit": "string - source unit",
            "to_unit": "string - target unit"
        }
    },

    # Notes
    "quick_note": {
        "function": quick_note,
        "description": "Save a quick note",
        "parameters": {"content": "string - note content", "tags": "list - optional tags"}
    },
    "list_notes": {
        "function": list_notes,
        "description": "List recent notes",
        "parameters": {"limit": "int - max notes to show"}
    },
    "search_notes": {
        "function": search_notes,
        "description": "Search through notes",
        "parameters": {"query": "string - search term"}
    },
}


def get_skill(name: str):
    """Get a skill by name."""
    if name in AVAILABLE_SKILLS:
        return AVAILABLE_SKILLS[name]["function"]
    return None


def list_skills() -> list[str]:
    """List all available skill names."""
    return list(AVAILABLE_SKILLS.keys())


def get_skills_schema() -> str:
    """Get schema of all skills for the router."""
    lines = ["Available tools:"]
    for i, (name, info) in enumerate(AVAILABLE_SKILLS.items(), 1):
        params = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        lines.append(f"{i}. {name}({params}) - {info['description']}")
    return "\n".join(lines)
