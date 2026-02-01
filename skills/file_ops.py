"""File operation skills"""

import os
from pathlib import Path


def read_file(path: str, max_lines: int = 500) -> dict:
    """
    Read contents of a file.

    Args:
        path: Path to the file
        max_lines: Maximum lines to read (default 500)

    Returns:
        Dict with content and metadata
    """
    try:
        filepath = Path(path).expanduser().resolve()

        if not filepath.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not filepath.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        # Check file size
        size = filepath.stat().st_size
        if size > 1_000_000:  # 1MB limit
            return {
                "success": False,
                "error": f"File too large ({size} bytes). Use head/tail for large files."
            }

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        truncated = len(lines) > max_lines
        content = ''.join(lines[:max_lines])

        return {
            "success": True,
            "content": content,
            "path": str(filepath),
            "lines": min(len(lines), max_lines),
            "truncated": truncated,
            "total_lines": len(lines)
        }

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_directory(path: str = ".", show_hidden: bool = False) -> dict:
    """
    List contents of a directory.

    Args:
        path: Path to directory (default current)
        show_hidden: Include hidden files

    Returns:
        Dict with directory listing
    """
    try:
        dirpath = Path(path).expanduser().resolve()

        if not dirpath.exists():
            return {"success": False, "error": f"Directory not found: {path}"}

        if not dirpath.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        items = []
        for item in sorted(dirpath.iterdir()):
            if not show_hidden and item.name.startswith('.'):
                continue

            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })

        return {
            "success": True,
            "path": str(dirpath),
            "items": items,
            "count": len(items)
        }

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
