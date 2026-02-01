"""
Skill Creator - Allows Jarvis to create new skills dynamically

This is a meta-skill that creates other skills.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from .. import get_data_dir


def get_user_skills_dir() -> Path:
    """Get the user's custom skills directory."""
    skills_dir = get_data_dir() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def create_skill(
    name: str,
    description: str,
    code: str,
    parameters: Optional[dict] = None
) -> dict:
    """
    Create a new skill and save it to the user's skills directory.

    Args:
        name: Skill name (snake_case, e.g., 'fetch_stock_price')
        description: What the skill does
        code: Python code for the skill function
        parameters: Dict of parameter names to descriptions

    Returns:
        Success status and file path
    """
    # Validate name
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return {
            "success": False,
            "error": "Skill name must be snake_case (e.g., 'my_skill')"
        }

    skills_dir = get_user_skills_dir()
    skill_file = skills_dir / f"{name}.py"

    # Check if already exists
    if skill_file.exists():
        return {
            "success": False,
            "error": f"Skill '{name}' already exists. Use update_skill to modify."
        }

    # Build the skill file content
    params_doc = ""
    if parameters:
        params_doc = "\n    Args:\n"
        for param, desc in parameters.items():
            params_doc += f"        {param}: {desc}\n"

    file_content = f'''"""
{description}

Auto-generated skill.
Created: {datetime.now().isoformat()}
"""

{code}


# Skill metadata for registration
SKILL_INFO = {{
    "name": "{name}",
    "function": {name},
    "description": "{description}",
    "parameters": {parameters or {}}
}}
'''

    try:
        with open(skill_file, 'w', encoding='utf-8') as f:
            f.write(file_content)

        # Also create/update the skills index
        _update_skills_index(name, description)

        return {
            "success": True,
            "message": f"Skill '{name}' created successfully",
            "path": str(skill_file),
            "usage": f"The skill is now available as '{name}'"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def update_skill(name: str, code: str = None, description: str = None) -> dict:
    """
    Update an existing skill.

    Args:
        name: Skill name
        code: New code (optional)
        description: New description (optional)

    Returns:
        Success status
    """
    skills_dir = get_user_skills_dir()
    skill_file = skills_dir / f"{name}.py"

    if not skill_file.exists():
        return {
            "success": False,
            "error": f"Skill '{name}' not found. Use create_skill first."
        }

    try:
        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if description:
            # Update docstring
            content = re.sub(
                r'"""[\s\S]*?"""',
                f'"""\n{description}\n\nAuto-generated skill.\nUpdated: {datetime.now().isoformat()}\n"""',
                content,
                count=1
            )

        if code:
            # This is tricky - for now, require full replacement
            return {
                "success": False,
                "error": "Code updates require recreating the skill. Delete and recreate."
            }

        with open(skill_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "message": f"Skill '{name}' updated",
            "path": str(skill_file)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_skill(name: str) -> dict:
    """
    Delete a user-created skill.

    Args:
        name: Skill name to delete

    Returns:
        Success status
    """
    skills_dir = get_user_skills_dir()
    skill_file = skills_dir / f"{name}.py"

    if not skill_file.exists():
        return {
            "success": False,
            "error": f"Skill '{name}' not found"
        }

    try:
        skill_file.unlink()
        _update_skills_index(name, None, remove=True)

        return {
            "success": True,
            "message": f"Skill '{name}' deleted"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_user_skills() -> dict:
    """
    List all user-created skills.

    Returns:
        List of skill names and descriptions
    """
    skills_dir = get_user_skills_dir()

    skills = []
    for skill_file in skills_dir.glob("*.py"):
        if skill_file.name.startswith("_"):
            continue

        name = skill_file.stem
        description = ""

        # Try to extract description from docstring
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'"""([\s\S]*?)"""', content)
                if match:
                    description = match.group(1).strip().split('\n')[0]
        except:
            pass

        skills.append({
            "name": name,
            "description": description,
            "path": str(skill_file)
        })

    return {
        "success": True,
        "skills": skills,
        "count": len(skills),
        "directory": str(skills_dir)
    }


def get_skill_code(name: str) -> dict:
    """
    Get the source code of a skill.

    Args:
        name: Skill name

    Returns:
        Skill source code
    """
    skills_dir = get_user_skills_dir()
    skill_file = skills_dir / f"{name}.py"

    if not skill_file.exists():
        return {
            "success": False,
            "error": f"Skill '{name}' not found"
        }

    try:
        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "success": True,
            "name": name,
            "code": content,
            "path": str(skill_file)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _update_skills_index(name: str, description: str, remove: bool = False):
    """Update the skills index file."""
    skills_dir = get_user_skills_dir()
    index_file = skills_dir / "README.md"

    # Load or create index
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# Custom Skills\n\nUser-created skills for Jarvis.\n\n## Skills\n\n"

    if remove:
        # Remove skill from index
        content = re.sub(rf'\n- \*\*{name}\*\*:.*\n', '\n', content)
    else:
        # Add or update skill in index
        skill_entry = f"\n- **{name}**: {description}\n"
        if f"**{name}**" in content:
            content = re.sub(rf'\n- \*\*{name}\*\*:.*\n', skill_entry, content)
        else:
            content = content.rstrip() + skill_entry

    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(content)


# Template for common skill patterns
SKILL_TEMPLATES = {
    "api_fetch": '''
def {name}({params}) -> dict:
    """
    {description}
    """
    import requests

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {{"success": True, "data": response.json()}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
''',

    "file_processor": '''
def {name}(file_path: str) -> dict:
    """
    {description}
    """
    from pathlib import Path

    try:
        path = Path(file_path).expanduser()
        if not path.exists():
            return {{"success": False, "error": "File not found"}}

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Process content here
        result = content

        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
''',

    "shell_command": '''
def {name}({params}) -> dict:
    """
    {description}
    """
    import subprocess

    try:
        result = subprocess.run(
            [{command}],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {{
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
''',

    "basic": '''
def {name}({params}) -> dict:
    """
    {description}
    """
    try:
        # Your logic here
        result = None

        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''
}


def get_skill_template(template_type: str = "basic") -> dict:
    """
    Get a skill template to help create new skills.

    Args:
        template_type: One of 'basic', 'api_fetch', 'file_processor', 'shell_command'

    Returns:
        Template code
    """
    if template_type not in SKILL_TEMPLATES:
        return {
            "success": False,
            "error": f"Unknown template. Available: {list(SKILL_TEMPLATES.keys())}"
        }

    return {
        "success": True,
        "template": SKILL_TEMPLATES[template_type],
        "available_templates": list(SKILL_TEMPLATES.keys())
    }
