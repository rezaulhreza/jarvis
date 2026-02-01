"""Memory operations - save and retrieve facts"""

import os
from pathlib import Path
from datetime import datetime

# Path to facts file
FACTS_FILE = Path(__file__).parent.parent / "memory" / "facts.md"


def save_fact(fact: str, category: str = "Notes") -> dict:
    """
    Append a fact to the facts file.

    Args:
        fact: The fact to save
        category: Category to save under (default: Notes)

    Returns:
        Success status
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- {fact} (added: {timestamp})\n"

        with open(FACTS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n## {category}\n{entry}" if category else entry)

        return {
            "success": True,
            "message": f"Saved fact under {category}",
            "fact": fact
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_facts() -> dict:
    """
    Read all facts from the facts file.

    Returns:
        Dict with facts content
    """
    try:
        if not FACTS_FILE.exists():
            return {"success": True, "content": "No facts stored yet."}

        with open(FACTS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "success": True,
            "content": content,
            "path": str(FACTS_FILE)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
