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
    clear_read_files, ALL_TOOLS,
    # Additional tools
    web_search, get_current_news, get_weather, get_current_time,
    calculate, save_memory, recall_memory, github_search,
)

# Models known to support native tool calling well
# Note: Some models output JSON as text instead - the agent handles this as fallback
NATIVE_TOOL_MODELS = [
    # OpenAI
    "gpt-4", "gpt-3.5", "gpt-oss",
    # Anthropic
    "claude",
    # Google
    "gemini",
    # Meta Llama
    "llama3.1", "llama3.2", "llama3.3",
    # Qwen (Alibaba)
    "qwen3",
    # Mistral
    "mistral", "mixtral",
    # Others with good tool support
    "command-r", "firefunction", "glm-4", "glm4",
]

# Models that claim tool support but often output JSON as text
# These will use native tools but fallback to text parsing works too
PARTIAL_TOOL_MODELS = [
    "qwen2.5", "qwen2.5-coder", "qwen3-coder",
    "granite", "granite3", "granite4",
]

# Reasoning models that need longer timeouts (they think a lot)
REASONING_MODELS = [
    "deepseek-r1", "o1", "o3", "qwq", "gpt-oss",
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

    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model that needs longer timeout."""
        model = self.provider.model.lower()
        return any(r in model for r in REASONING_MODELS)

    def _get_timeout(self) -> int:
        """Get appropriate timeout based on model type."""
        if self._is_reasoning_model():
            return 300  # 5 minutes for reasoning models
        return 120  # 2 minutes default

    def _get_tools_prompt(self) -> str:
        """Get prompt describing available tools for non-native models."""
        return """
You have access to tools. To use a tool, respond with JSON:
{"tool": "tool_name", "param": "value"}

AVAILABLE TOOLS:

FILES:
- {"tool": "read_file", "path": "path/to/file"}
- {"tool": "search_files", "query": "pattern", "file_type": "py"}
- {"tool": "list_files", "path": "dir", "pattern": "*.py"}
- {"tool": "get_project_structure"}
- {"tool": "write_file", "path": "file", "content": "content"}
- {"tool": "edit_file", "path": "file", "old_string": "find", "new_string": "replace"}

WEB & INFO:
- {"tool": "web_search", "query": "search query"}
- {"tool": "get_current_news", "topic": "topic"}
- {"tool": "get_weather", "city": "London"}
- {"tool": "get_current_time", "timezone": "UTC"}

UTILITIES:
- {"tool": "calculate", "expression": "2 + 2"}
- {"tool": "run_command", "command": "git status"}
- {"tool": "github_search", "query": "term", "search_type": "repos"}

MEMORY:
- {"tool": "save_memory", "content": "info to remember", "category": "general"}
- {"tool": "recall_memory", "query": "search term"}

RULES:
1. For current events/news/weather: USE web_search, get_current_news, or get_weather
2. For code questions: read_file FIRST, never guess
3. For writing files: USE write_file or edit_file
4. Output ONLY the JSON when calling a tool
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

        elif tool_name == "web_search":
            query = args.get("query", "")
            if len(query) > 40:
                query = query[:37] + "..."
            return f"Search web: '{query}'"

        elif tool_name == "get_current_news":
            topic = args.get("topic", "")
            if len(topic) > 40:
                topic = topic[:37] + "..."
            return f"Get news: '{topic}'"

        elif tool_name == "get_weather":
            city = args.get("city", "")
            return f"Weather: {city}"

        elif tool_name == "get_current_time":
            tz = args.get("timezone", "UTC")
            return f"Time: {tz}"

        elif tool_name == "calculate":
            expr = args.get("expression", "")
            if len(expr) > 30:
                expr = expr[:27] + "..."
            return f"Calculate: {expr}"

        elif tool_name == "save_memory":
            cat = args.get("category", "general")
            return f"Save memory: [{cat}]"

        elif tool_name == "recall_memory":
            query = args.get("query", "")
            if query:
                return f"Recall: '{query}'"
            return "Recall memory"

        elif tool_name == "github_search":
            query = args.get("query", "")
            stype = args.get("search_type", "repos")
            return f"GitHub: {stype} '{query}'"

        return f"{tool_name}()"

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        tool_map = {
            # File operations
            "read_file": read_file,
            "list_files": list_files,
            "search_files": search_files,
            "write_file": write_file,
            "edit_file": edit_file,
            "get_project_structure": get_project_structure,
            # Shell
            "run_command": run_command,
            # Web
            "web_search": web_search,
            "get_current_news": get_current_news,
            # Weather
            "get_weather": get_weather,
            # Time
            "get_current_time": get_current_time,
            # Math
            "calculate": calculate,
            # Memory
            "save_memory": save_memory,
            "recall_memory": recall_memory,
            # GitHub
            "github_search": github_search,
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

                timeout = self._get_timeout()
                response = self._call_model_with_timeout(messages, system_prompt, tools, timeout=timeout)

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
