# Skills module
# Each skill is a tool the assistant can use
#
# Skills can be:
# 1. Built-in (this package)
# 2. User-created (~/.jarvis/skills/)

import importlib.util
from pathlib import Path

from .web_search import web_search, get_current_news
from .shell import shell_run, is_safe_command
from .file_ops import read_file, list_directory, write_file, edit_file, search_files
from .memory_ops import save_fact, get_facts
from .weather import get_weather, get_forecast
from .github_ops import list_repos, repo_info, list_issues, create_issue, list_prs
from .datetime_ops import get_current_time, convert_timezone, add_time, time_until
from .calculator import calculate, convert_units, percentage
from .notes import quick_note, list_notes, read_note, search_notes
from .skill_creator import (
    create_skill, update_skill, delete_skill,
    list_user_skills, get_skill_code, get_skill_template
)
from .multi_model_analysis import (
    multi_model_analyze, list_analysis_profiles, analyze_parallel
)
from .media_gen import (
    generate_image, generate_video, generate_music, analyze_image,
    analyze_image_ollama, analyze_document, cleanup_generated_files, list_media_models
)

# Built-in skills
BUILT_IN_SKILLS = {
    # Web
    "web_search": {
        "function": web_search,
        "description": "Search the web for information",
        "parameters": {"query": "string - the search query"}
    },
    "get_current_news": {
        "function": get_current_news,
        "description": "Get current news and recent information about a topic. Use for current events, politics, sports, celebrities, recent news.",
        "parameters": {"topic": "string - the topic to get news about"}
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
    "write_file": {
        "function": write_file,
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "path": "string - path to the file",
            "content": "string - content to write",
            "create_dirs": "bool - create parent directories if needed (optional)"
        }
    },
    "edit_file": {
        "function": edit_file,
        "description": "Edit a file by replacing a specific string with new content",
        "parameters": {
            "path": "string - path to the file",
            "old_string": "string - exact string to find and replace",
            "new_string": "string - replacement string"
        }
    },
    "search_files": {
        "function": search_files,
        "description": "Search for a pattern in files",
        "parameters": {
            "pattern": "string - text to search for",
            "path": "string - directory to search (default: current)",
            "file_pattern": "string - file glob pattern (e.g., *.py)"
        }
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

    # Skill Management (Meta-skills)
    "create_skill": {
        "function": create_skill,
        "description": "Create a new skill (provide name, description, and Python code)",
        "parameters": {
            "name": "string - skill name (snake_case)",
            "description": "string - what the skill does",
            "code": "string - Python function code",
            "parameters": "dict - parameter descriptions (optional)"
        }
    },
    "delete_skill": {
        "function": delete_skill,
        "description": "Delete a user-created skill",
        "parameters": {"name": "string - skill name to delete"}
    },
    "list_user_skills": {
        "function": list_user_skills,
        "description": "List all user-created custom skills",
        "parameters": {}
    },
    "get_skill_code": {
        "function": get_skill_code,
        "description": "View the source code of a skill",
        "parameters": {"name": "string - skill name"}
    },
    "get_skill_template": {
        "function": get_skill_template,
        "description": "Get a template for creating skills",
        "parameters": {"template_type": "string - basic, api_fetch, file_processor, shell_command"}
    },

    # Multi-Model Analysis
    "multi_model_analyze": {
        "function": multi_model_analyze,
        "description": "Analyze a query using multiple AI models (fast, reasoning, code, thinking) for comprehensive insights",
        "parameters": {
            "query": "string - the question or topic to analyze",
            "profile": "string - analysis profile: comprehensive, quick, technical, reasoning (default: comprehensive)"
        }
    },
    "list_analysis_profiles": {
        "function": list_analysis_profiles,
        "description": "List available multi-model analysis profiles",
        "parameters": {}
    },

    # Media Generation
    "generate_image": {
        "function": generate_image,
        "description": "Generate an image from a text description. Use for drawing, creating art, logos, pictures.",
        "parameters": {
            "prompt": "string - description of the image to generate",
            "model": "string - model (flux-schnell, flux-dev, sdxl, hidream, juggernaut) (optional)",
            "width": "int - image width (default 1024) (optional)",
            "height": "int - image height (default 1024) (optional)"
        }
    },
    "generate_video": {
        "function": generate_video,
        "description": "Generate a video from an image (image-to-video). Animates a still image based on the prompt.",
        "parameters": {
            "prompt": "string - description of what should happen in the video",
            "image_path": "string - path to source image (required)",
            "frames": "int - number of frames, 21-140, default 81 (~5 seconds) (optional)",
            "resolution": "string - '480p' or '720p' (optional)"
        }
    },
    "generate_music": {
        "function": generate_music,
        "description": "Generate music from a text description. Use for soundtracks, jingles, background music.",
        "parameters": {
            "prompt": "string - description of the music to generate",
            "duration": "float - duration in seconds (default 30) (optional)"
        }
    },
    "analyze_image": {
        "function": analyze_image,
        "description": "Analyze an image and describe its contents. Use for image understanding, OCR, visual Q&A. Auto-detects Ollama or Chutes.",
        "parameters": {
            "image_path": "string - path to the image file",
            "prompt": "string - question or instruction about the image (optional)",
            "provider": "string - 'auto', 'ollama', or 'chutes' (optional, default: auto)"
        }
    },
    "analyze_image_ollama": {
        "function": analyze_image_ollama,
        "description": "Analyze an image using local Ollama vision model (llava, llama3.2-vision, etc.)",
        "parameters": {
            "image_path": "string - path to the image file",
            "prompt": "string - question or instruction about the image (optional)"
        }
    },
    "list_media_models": {
        "function": list_media_models,
        "description": "List available media generation models for images, video, and music",
        "parameters": {}
    },
    "analyze_document": {
        "function": analyze_document,
        "description": "Analyze a document (PDF, DOCX, XLSX, CSV, TXT). Extract text and answer questions about it.",
        "parameters": {
            "doc_path": "string - path to the document file",
            "prompt": "string - question or instruction about the document (optional)"
        }
    },
    "cleanup_generated_files": {
        "function": cleanup_generated_files,
        "description": "Clean up old generated media files to free disk space",
        "parameters": {
            "max_age_hours": "int - delete files older than this many hours (default 24)"
        }
    },
}


def _load_user_skills() -> dict:
    """Load user-created skills from ~/.jarvis/skills/"""
    from .. import get_data_dir

    user_skills = {}
    skills_dir = get_data_dir() / "skills"

    if not skills_dir.exists():
        return user_skills

    for skill_file in skills_dir.glob("*.py"):
        if skill_file.name.startswith("_"):
            continue

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                skill_file.stem, skill_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get skill info
            if hasattr(module, 'SKILL_INFO'):
                info = module.SKILL_INFO
                user_skills[info['name']] = {
                    "function": info['function'],
                    "description": info.get('description', 'User-created skill'),
                    "parameters": info.get('parameters', {}),
                    "user_created": True
                }
        except Exception as e:
            # Skip broken skills
            print(f"Warning: Could not load skill {skill_file.name}: {e}")

    return user_skills


def get_all_skills() -> dict:
    """Get all skills (built-in + user-created)."""
    all_skills = BUILT_IN_SKILLS.copy()
    all_skills.update(_load_user_skills())
    return all_skills


# This is what gets imported - includes user skills
AVAILABLE_SKILLS = get_all_skills()


def reload_skills():
    """Reload skills (call after creating new skills)."""
    global AVAILABLE_SKILLS
    AVAILABLE_SKILLS = get_all_skills()


def get_skill(name: str):
    """Get a skill by name."""
    skills = get_all_skills()
    if name in skills:
        return skills[name]["function"]
    return None


def list_skills() -> list[str]:
    """List all available skill names."""
    return list(get_all_skills().keys())


def get_skills_schema() -> str:
    """Get schema of all skills for the router."""
    skills = get_all_skills()
    lines = ["Available tools:"]
    for i, (name, info) in enumerate(skills.items(), 1):
        params = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        marker = " [custom]" if info.get("user_created") else ""
        lines.append(f"{i}. {name}({params}) - {info['description']}{marker}")
    return "\n".join(lines)
