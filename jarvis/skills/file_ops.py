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


def write_file(path: str, content: str, create_dirs: bool = False) -> dict:
    """
    Write content to a file.

    Args:
        path: Path to the file
        content: Content to write
        create_dirs: Create parent directories if they don't exist

    Returns:
        Dict with success status and file info
    """
    try:
        filepath = Path(path).expanduser().resolve()

        # Safety check - don't write to sensitive locations
        sensitive_paths = ['.ssh', '.gnupg', '.aws', 'credentials', '.env']
        if any(s in str(filepath).lower() for s in sensitive_paths):
            return {
                "success": False,
                "error": f"Cannot write to sensitive path: {path}"
            }

        # Create parent directories if requested
        if create_dirs and not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check if parent directory exists
        if not filepath.parent.exists():
            return {
                "success": False,
                "error": f"Parent directory does not exist: {filepath.parent}"
            }

        # Write the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "path": str(filepath),
            "bytes_written": len(content.encode('utf-8')),
            "lines": content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        }

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """
    Edit a file by replacing old_string with new_string.

    Args:
        path: Path to the file
        old_string: The exact string to find and replace
        new_string: The replacement string

    Returns:
        Dict with success status and diff info
    """
    try:
        filepath = Path(path).expanduser().resolve()

        if not filepath.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not filepath.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        # Safety check
        sensitive_paths = ['.ssh', '.gnupg', '.aws', 'credentials', '.env']
        if any(s in str(filepath).lower() for s in sensitive_paths):
            return {"success": False, "error": f"Cannot edit sensitive path: {path}"}

        # Read current content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if old_string exists
        if old_string not in content:
            return {
                "success": False,
                "error": f"String not found in file. Make sure to match exact whitespace and indentation."
            }

        # Count occurrences
        count = content.count(old_string)
        if count > 1:
            return {
                "success": False,
                "error": f"Found {count} occurrences. Please provide more context to uniquely identify the string."
            }

        # Perform replacement
        new_content = content.replace(old_string, new_string, 1)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Generate simple diff
        old_lines = old_string.split('\n')
        new_lines = new_string.split('\n')

        return {
            "success": True,
            "path": str(filepath),
            "lines_removed": len(old_lines),
            "lines_added": len(new_lines),
            "diff": f"- {old_string[:100]}{'...' if len(old_string) > 100 else ''}\n+ {new_string[:100]}{'...' if len(new_string) > 100 else ''}"
        }

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_files(pattern: str, path: str = ".", file_pattern: str = "*") -> dict:
    """
    Search for a pattern in files.

    Args:
        pattern: Text pattern to search for
        path: Directory to search in
        file_pattern: Glob pattern for files (e.g., "*.py")

    Returns:
        Dict with matching files and lines
    """
    try:
        search_path = Path(path).expanduser().resolve()

        if not search_path.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        results = []
        files_searched = 0

        for filepath in search_path.rglob(file_pattern):
            if filepath.is_file() and not any(p.startswith('.') for p in filepath.parts):
                files_searched += 1
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                results.append({
                                    "file": str(filepath.relative_to(search_path)),
                                    "line": line_num,
                                    "content": line.strip()[:200]
                                })
                                if len(results) >= 50:
                                    break
                except:
                    pass

            if len(results) >= 50:
                break

        return {
            "success": True,
            "pattern": pattern,
            "files_searched": files_searched,
            "matches": results,
            "total_matches": len(results),
            "truncated": len(results) >= 50
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
