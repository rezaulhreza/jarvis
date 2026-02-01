"""Note-taking and quick capture skill"""

import os
from pathlib import Path
from datetime import datetime
import json

NOTES_DIR = Path(__file__).parent.parent / "knowledge" / "notes"


def _ensure_notes_dir():
    """Ensure notes directory exists."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


def quick_note(content: str, tags: list[str] = None) -> dict:
    """
    Save a quick note with timestamp.

    Args:
        content: Note content
        tags: Optional tags for categorization

    Returns:
        Success status and note info
    """
    _ensure_notes_dir()

    timestamp = datetime.now()
    filename = timestamp.strftime("%Y%m%d_%H%M%S") + ".md"
    filepath = NOTES_DIR / filename

    # Format note with metadata
    note_content = f"""---
date: {timestamp.isoformat()}
tags: {tags or []}
---

{content}
"""

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)

        return {
            "success": True,
            "message": "Note saved",
            "file": str(filepath),
            "timestamp": timestamp.isoformat()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_notes(limit: int = 10, tag: str = None) -> dict:
    """
    List recent notes.

    Args:
        limit: Maximum notes to return
        tag: Filter by tag

    Returns:
        List of notes
    """
    _ensure_notes_dir()

    try:
        notes = []
        for filepath in sorted(NOTES_DIR.glob("*.md"), reverse=True)[:limit * 2]:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract first line of actual content (after metadata)
            lines = content.split('\n')
            preview = ""
            in_metadata = False
            for line in lines:
                if line.strip() == "---":
                    in_metadata = not in_metadata
                    continue
                if not in_metadata and line.strip():
                    preview = line.strip()[:100]
                    break

            # Check tag filter
            if tag and f"tags:" in content:
                if tag.lower() not in content.lower():
                    continue

            notes.append({
                "file": filepath.name,
                "preview": preview,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            })

            if len(notes) >= limit:
                break

        return {"success": True, "notes": notes, "count": len(notes)}

    except Exception as e:
        return {"success": False, "error": str(e)}


def read_note(filename: str) -> dict:
    """
    Read a specific note.

    Args:
        filename: Note filename

    Returns:
        Note content
    """
    _ensure_notes_dir()

    filepath = NOTES_DIR / filename
    if not filepath.exists():
        return {"success": False, "error": f"Note not found: {filename}"}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "success": True,
            "file": filename,
            "content": content
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def search_notes(query: str) -> dict:
    """
    Search notes for a query.

    Args:
        query: Search term

    Returns:
        Matching notes
    """
    _ensure_notes_dir()

    try:
        matches = []
        query_lower = query.lower()

        for filepath in NOTES_DIR.glob("*.md"):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if query_lower in content.lower():
                # Find matching line
                for line in content.split('\n'):
                    if query_lower in line.lower():
                        matches.append({
                            "file": filepath.name,
                            "match": line.strip()[:100]
                        })
                        break

        return {
            "success": True,
            "query": query,
            "matches": matches,
            "count": len(matches)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def append_to_note(filename: str, content: str) -> dict:
    """
    Append content to an existing note.

    Args:
        filename: Note filename
        content: Content to append

    Returns:
        Success status
    """
    _ensure_notes_dir()

    filepath = NOTES_DIR / filename
    if not filepath.exists():
        return {"success": False, "error": f"Note not found: {filename}"}

    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n\n---\n*Added {datetime.now().isoformat()}*\n\n{content}")

        return {
            "success": True,
            "message": f"Appended to {filename}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
