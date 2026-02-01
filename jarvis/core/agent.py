"""
Agentic loop with tool calling.

Supports both native tool calling and prompt-based fallback.
"""

import re
import json
from typing import List
from pathlib import Path

from .tools import (
    read_file, list_files, search_files, run_command,
    write_file, edit_file, get_project_structure, set_project_root, set_ui,
    clear_read_files, ALL_TOOLS
)

# Models known to support native tool calling well
# Note: Some models output JSON as text instead - the agent handles this as fallback
NATIVE_TOOL_MODELS = [
    "qwen3", "llama3.1", "llama3.2", "llama3.3",
    "mistral", "mixtral", "command-r", "firefunction",
    "claude", "gpt-4", "gpt-3.5", "gemini",
]

# Models that claim tool support but often output JSON as text
# These will use native tools but fallback to text parsing works too
PARTIAL_TOOL_MODELS = [
    "qwen2.5", "qwen2.5-coder", "qwen3-coder",
    "granite", "granite3", "granite4",
]


class Agent:
    """Agent with tool calling - native or prompt-based."""

    def __init__(self, provider, project_root: Path, ui=None):
        self.provider = provider
        self.project_root = project_root
        self.ui = ui
        self.max_iterations = 15
        set_project_root(project_root)
        set_ui(ui)  # Pass UI to tools for confirmations

    def _supports_native_tools(self) -> bool:
        """Check if current model supports native tool calling."""
        model = self.provider.model.lower()
        all_tool_models = NATIVE_TOOL_MODELS + PARTIAL_TOOL_MODELS
        return any(t in model for t in all_tool_models)

    def _get_tools_prompt(self) -> str:
        """Get prompt describing available tools for non-native models."""
        return """
MANDATORY: You MUST use tools. NEVER make up code or guess.

To use a tool, respond with ONLY this JSON format:
{"tool": "tool_name", "param": "value"}

AVAILABLE TOOLS:

READING:
- {"tool": "read_file", "path": "path/to/file.php"}
- {"tool": "search_files", "query": "function name", "file_type": "php"}
- {"tool": "list_files", "path": "app/Models", "pattern": "*.php"}
- {"tool": "get_project_structure"}

WRITING (USE THESE WHEN ASKED TO WRITE/SAVE/CREATE/MODIFY FILES):
- {"tool": "write_file", "path": "path/to/file.php", "content": "full file content here"}
- {"tool": "edit_file", "path": "path/to/file.php", "old_string": "text to find", "new_string": "replacement text"}

STRICT RULES:
1. ALWAYS read files BEFORE answering code questions
2. NEVER generate fake code - only quote actual code from files
3. When asked to WRITE/SAVE/REFACTOR a file: USE write_file or edit_file tool
4. NEVER just output code when asked to write it - USE THE TOOL
5. Output ONLY the JSON when calling a tool, nothing else
"""

    def _parse_tool_call_from_text(self, text: str) -> tuple:
        """Try to extract a tool call from model's text output."""
        if not text:
            return None, None

        # Try to find JSON object with balanced braces
        # Start from first { and find matching }
        start = text.find('{')
        if start == -1:
            return None, None

        # Count braces to find the matching closing brace
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            return None, None

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
            tool_name = data.get("tool") or data.get("name")
            if tool_name:
                # Get arguments - could be in "arguments" or directly in data
                args = data.get("arguments", {})
                # Handle case where arguments is a JSON string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if not args:
                    args = {k: v for k, v in data.items() if k not in ["tool", "name", "arguments"]}
                return tool_name, args
        except json.JSONDecodeError:
            pass

        return None, None

    def _format_tool_display(self, tool_name: str, args: dict, result: str = None) -> str:
        """Format tool call for display like Claude Code."""
        if tool_name == "read_file":
            path = args.get("path", "file")
            if result and not result.startswith("Error"):
                lines = len(result.split('\n'))
                return f"Read {path} ({lines} lines)"
            return f"Read {path}"

        elif tool_name == "list_files":
            path = args.get("path", ".") or "."
            pattern = args.get("pattern", "*")
            if pattern and pattern != "*":
                return f"List {path} ({pattern})"
            return f"List {path}"

        elif tool_name == "search_files":
            query = args.get("query", "")
            file_type = args.get("file_type", "")
            if file_type:
                return f"Search '{query}' in *.{file_type}"
            return f"Search '{query}'"

        elif tool_name == "run_command":
            cmd = args.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return f"Run `{cmd}`"

        elif tool_name == "write_file":
            path = args.get("path", "file")
            return f"Write {path}"

        elif tool_name == "edit_file":
            path = args.get("path", "file")
            return f"Edit {path}"

        elif tool_name == "get_project_structure":
            return "Get project structure"

        return f"{tool_name}()"

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        tool_map = {
            "read_file": read_file,
            "list_files": list_files,
            "search_files": search_files,
            "run_command": run_command,
            "write_file": write_file,
            "edit_file": edit_file,
            "get_project_structure": get_project_structure,
        }

        if tool_name not in tool_map:
            return f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}"

        try:
            return tool_map[tool_name](**args)
        except Exception as e:
            return f"Error: {e}"

    def _call_model_with_timeout(self, messages, system_prompt, tools, timeout=120):
        """Call model with timeout using threading."""
        import threading
        import time

        result = {"response": None, "error": None}

        def call():
            try:
                result["response"] = self.provider.chat_with_tools(
                    messages=messages,
                    system=system_prompt,
                    tools=tools
                )
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=call)
        thread.daemon = True
        thread.start()

        # Wait with periodic checks for interrupt
        start = time.time()
        while thread.is_alive():
            if self.ui and self.ui.stop_requested:
                return None  # Interrupted
            if time.time() - start > timeout:
                return None  # Timeout
            thread.join(timeout=0.5)  # Check every 0.5s

        if result["error"]:
            raise result["error"]
        return result["response"]

    def run(self, user_message: str, system_prompt: str, history: List = None) -> str:
        """Run agentic loop with tool calling."""
        import time
        start_time = time.time()
        tool_count = 0

        # Check if model supports native tools
        use_native = self._supports_native_tools()

        if not use_native:
            # Add tools description to system prompt for prompt-based approach
            system_prompt = system_prompt + "\n\n" + self._get_tools_prompt()

        messages = []
        if history:
            for msg in history:
                if hasattr(msg, 'role'):
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            if self.ui and self.ui.stop_requested:
                return "[dim]Stopped[/dim]"

            try:
                tools = ALL_TOOLS if use_native else None

                # Call model with timeout (interruptible)
                if self.ui:
                    self.ui.console.print("[dim]  Thinking...[/dim]", end="\r")

                response = self._call_model_with_timeout(messages, system_prompt, tools)

                # Clear the "Thinking..." line
                if self.ui:
                    self.ui.console.print("             ", end="\r")

                if response is None:
                    return "[dim]Stopped[/dim]"

                if self.ui and self.ui.stop_requested:
                    return "[dim]Stopped[/dim]"

                # Parse response
                msg = response.message if hasattr(response, 'message') else response.get('message', {})
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '') or ''
                tool_calls = msg.tool_calls if hasattr(msg, 'tool_calls') else msg.get('tool_calls')

                # For native tool calling
                if use_native and tool_calls:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for call in tool_calls:
                        if self.ui and self.ui.stop_requested:
                            return "[dim]Stopped[/dim]"

                        if hasattr(call, 'function'):
                            tool_name = call.function.name
                            args = call.function.arguments or {}
                        else:
                            func = call.get('function', {})
                            tool_name = func.get('name', '')
                            args = func.get('arguments', {})

                        result = self._execute_tool(tool_name, args)
                        tool_count += 1
                        if self.ui:
                            self.ui.print_tool(self._format_tool_display(tool_name, args, result))
                        messages.append({"role": "tool", "content": result})

                # Check if content contains JSON tool call (fallback for models that output JSON as text)
                elif content and ('"name"' in content or '"tool"' in content):
                    tool_name, args = self._parse_tool_call_from_text(content)

                    if tool_name:
                        result = self._execute_tool(tool_name, args)
                        tool_count += 1
                        if self.ui:
                            self.ui.print_tool(self._format_tool_display(tool_name, args, result))

                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nNow answer the original question based on this information."})
                    else:
                        # Couldn't parse JSON - remove JSON blob and return rest, or show error
                        # Find and remove the JSON object (handles nested braces)
                        clean = content
                        start = content.find('{')
                        if start != -1:
                            depth = 0
                            end = start
                            for i, char in enumerate(content[start:], start):
                                if char == '{':
                                    depth += 1
                                elif char == '}':
                                    depth -= 1
                                    if depth == 0:
                                        end = i + 1
                                        break
                            clean = (content[:start] + content[end:]).strip()

                        if clean:
                            final_response = clean
                        else:
                            final_response = "[dim]Model output malformed tool call. Try rephrasing your request.[/dim]"
                        break

                # No tool calls - final response
                elif content:
                    final_response = self._clean_content(content)
                    break

                else:
                    break

            except Exception as e:
                return f"[red]Error: {e}[/red]"

        if iteration >= self.max_iterations:
            final_response += "\n[dim](max iterations)[/dim]"

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Format timing info
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.0f}s"

        # Add stats footer
        stats = f"\n[dim]({time_str}"
        if tool_count > 0:
            stats += f" · {tool_count} tool{'s' if tool_count > 1 else ''}"
        stats += ")[/dim]"

        response = final_response if final_response else "[dim]No response[/dim]"
        return response + stats

    def _clean_content(self, text: str) -> str:
        """Remove thinking tags and clean response."""
        if not text:
            return ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()
