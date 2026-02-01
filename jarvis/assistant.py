#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant
"""

import sys
import os
import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .core.ollama_client import OllamaClient
from .core.context_manager import ContextManager
from .core.router import ToolRouter, should_use_reasoning, should_use_vision
from .skills import AVAILABLE_SKILLS, get_skills_schema, reload_skills, get_all_skills
from . import get_data_dir, ensure_data_dir, PACKAGE_DIR

load_dotenv()

console = Console()


def _get_config_dir() -> Path:
    """Get config directory, initializing from defaults if needed."""
    data_dir = ensure_data_dir()
    config_dir = data_dir / "config"

    default_config = PACKAGE_DIR.parent / "config"
    if default_config.exists():
        for item in default_config.iterdir():
            dest = config_dir / item.name
            if not dest.exists():
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    return config_dir


def _get_memory_dir() -> Path:
    """Get memory directory, initializing from defaults if needed."""
    data_dir = ensure_data_dir()
    memory_dir = data_dir / "memory"

    default_memory = PACKAGE_DIR.parent / "memory"
    if default_memory.exists():
        for item in default_memory.iterdir():
            dest = memory_dir / item.name
            if not dest.exists():
                shutil.copy2(item, dest)

    return memory_dir


CONFIG_DIR = _get_config_dir()
MEMORY_DIR = _get_memory_dir()
PERSONAS_DIR = CONFIG_DIR / "personas"


def load_config() -> dict:
    config_path = CONFIG_DIR / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_persona(name: str = "default") -> str:
    persona_path = PERSONAS_DIR / f"{name}.md"
    if persona_path.exists():
        with open(persona_path) as f:
            return f.read()

    default_path = PERSONAS_DIR / "default.md"
    if default_path.exists():
        with open(default_path) as f:
            return f.read()

    return "You are Jarvis, a helpful AI assistant. Be concise and direct."


def load_rules() -> str:
    rules_path = CONFIG_DIR / "rules.md"
    if rules_path.exists():
        with open(rules_path) as f:
            return f.read()
    return ""


def load_facts() -> str:
    facts_path = MEMORY_DIR / "facts.md"
    if facts_path.exists():
        with open(facts_path) as f:
            return f.read()
    return ""


def list_personas() -> list[str]:
    if not PERSONAS_DIR.exists():
        return ["default"]
    return [p.stem for p in PERSONAS_DIR.glob("*.md")]


class Jarvis:
    """Main assistant class."""

    def __init__(self):
        self.config = load_config()
        self.current_persona = self.config.get("persona", "default")
        self.persona_prompt = load_persona(self.current_persona)
        self.rules = load_rules()
        self.facts = load_facts()

        model_config = self.config.get('models', {})
        self.ollama = OllamaClient(default_model=model_config.get('default', 'qwen3:4b'))
        self.context = ContextManager(
            db_path=str(MEMORY_DIR / "jarvis.db"),
            max_tokens=self.config.get('context', {}).get('max_tokens', 8000)
        )
        self.router = ToolRouter(self.ollama, router_model=model_config.get('tools', 'functiongemma'))

        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build concise system prompt."""
        self.system_prompt = f"""You are Jarvis, a personal AI assistant.

IMPORTANT RULES:
- Be concise and direct. No emojis. No unnecessary pleasantries.
- When given tool output, summarize it naturally for the user.
- Never say you're going to use a tool - tools are handled automatically.
- If you don't know something, say so briefly.

User context:
{self.facts}
"""

    def switch_persona(self, name: str) -> str:
        available = list_personas()
        if name not in available:
            return f"Unknown persona: {name}. Available: {', '.join(available)}"

        self.current_persona = name
        self.persona_prompt = load_persona(name)
        self._build_system_prompt()
        return f"Switched to {name} persona."

    def process(self, user_input: str) -> str:
        """Process user input."""

        # Handle commands
        if user_input.startswith('/'):
            return self._handle_command(user_input)

        # Check for vision
        if should_use_vision(user_input):
            return self._handle_vision(user_input)

        # Route to tool
        route_result = self.router.route(user_input, self.context.get_working_memory())
        tool_name = route_result.get('tool')
        tool_output = None

        if tool_name and tool_name != 'none':
            tool_output = self._execute_tool(route_result)

        # Add user message to context
        self.context.add_message("user", user_input)

        # Generate response
        return self._generate_response(tool_output)

    def _generate_response(self, tool_output: str = None) -> str:
        """Generate response, incorporating tool output if any."""
        messages = self.context.get_messages()

        # If we have tool output, tell the model to use it
        if tool_output:
            messages = messages.copy()
            messages.append({
                "role": "system",
                "content": f"Tool result (present this to the user naturally):\n{tool_output}"
            })

        response_text = ""
        try:
            for chunk in self.ollama.chat(
                messages=messages,
                system=self.system_prompt,
                stream=True
            ):
                console.print(chunk, end="")
                response_text += chunk
        except Exception as e:
            response_text = f"Error: {e}"
            console.print(response_text)

        console.print()
        self.context.add_message("assistant", response_text)
        return response_text

    def _handle_vision(self, user_input: str) -> str:
        """Handle image analysis."""
        import re
        path_match = re.search(
            r'([/~][^\s]+\.(jpg|jpeg|png|gif|webp|bmp))',
            user_input,
            re.IGNORECASE
        )

        if not path_match:
            console.print("Provide an image path. Example: /path/to/image.jpg what is this?")
            return ""

        image_path = os.path.expanduser(path_match.group(1))

        if not os.path.exists(image_path):
            console.print(f"Image not found: {image_path}")
            return ""

        console.print(f"[dim]Analyzing image...[/dim]")

        try:
            result = self.ollama.vision(image_path, user_input)
            self.context.add_message("user", user_input)
            self.context.add_message("assistant", result)
            console.print(result)
            return result
        except Exception as e:
            error_msg = f"Error: {e}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def _execute_tool(self, route_result: dict) -> str:
        """Execute a tool and return output."""
        tool_name = route_result.get('tool')
        params = route_result.get('params', {})

        skills = get_all_skills()

        if tool_name not in skills:
            return None

        console.print(f"[dim]Using {tool_name}...[/dim]")

        try:
            tool_func = skills[tool_name]['function']
            params = {k: v for k, v in params.items() if v}
            result = tool_func(**params)

            # Reload skills if we created/deleted one
            if tool_name in ['create_skill', 'delete_skill']:
                reload_skills()
                self._build_system_prompt()

            return str(result)
        except Exception as e:
            console.print(f"[red]Tool error: {e}[/red]")
            return f"Error: {e}"

    def _handle_command(self, command: str) -> str:
        """Handle slash commands."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            '/help': self._cmd_help,
            '/commands': self._cmd_help,
            '/models': self._cmd_models,
            '/persona': lambda: self._cmd_persona(args),
            '/personas': self._cmd_personas,
            '/skills': self._cmd_skills,
            '/facts': self._cmd_facts,
            '/memory': self._cmd_memory,
            '/clear': self._cmd_clear,
            '/history': self._cmd_history,
            '/quit': self._cmd_quit,
            '/exit': self._cmd_quit,
            '/q': self._cmd_quit,
        }

        if cmd in commands:
            return commands[cmd]()
        else:
            console.print(f"Unknown command: {cmd}")
            return self._cmd_help()

    def _cmd_help(self) -> str:
        help_text = """
## Commands

| Command | Description |
|---------|-------------|
| /help | Show this help |
| /models | List Ollama models |
| /persona <name> | Switch persona |
| /personas | List personas |
| /skills | List available skills |
| /facts | Show stored facts |
| /memory | Show working memory |
| /clear | Clear conversation |
| /history | Show recent messages |
| /quit | Exit |
"""
        console.print(Markdown(help_text))
        return ""

    def _cmd_models(self) -> str:
        try:
            models = self.ollama.list_models()
            console.print("\n[bold]Models:[/bold]")
            for m in models:
                console.print(f"  {m}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        return ""

    def _cmd_persona(self, name: str) -> str:
        if not name:
            console.print(f"Current: {self.current_persona}")
            console.print(f"Available: {', '.join(list_personas())}")
            return ""
        result = self.switch_persona(name)
        console.print(result)
        return ""

    def _cmd_personas(self) -> str:
        console.print("\n[bold]Personas:[/bold]")
        for p in list_personas():
            marker = " (active)" if p == self.current_persona else ""
            console.print(f"  {p}{marker}")
        return ""

    def _cmd_skills(self) -> str:
        skills = get_all_skills()
        console.print("\n[bold]Skills:[/bold]")
        for name, info in skills.items():
            marker = " [custom]" if info.get("user_created") else ""
            console.print(f"  [cyan]{name}[/cyan]: {info['description']}{marker}")
        return ""

    def _cmd_facts(self) -> str:
        console.print(Markdown(f"## Facts\n\n{self.facts}"))
        return ""

    def _cmd_memory(self) -> str:
        mem = self.context.get_working_memory()
        console.print(f"\n[bold]Working Memory:[/bold]\n{mem}")
        return ""

    def _cmd_clear(self) -> str:
        self.context.clear()
        console.print("Cleared.")
        return ""

    def _cmd_history(self) -> str:
        history = self.context.get_history(limit=10)
        if not history:
            console.print("No history.")
            return ""
        console.print("\n[bold]History:[/bold]\n")
        for msg in history:
            role = msg['role'].upper()
            content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            console.print(f"[bold]{role}:[/bold] {content}\n")
        return ""

    def _cmd_quit(self) -> str:
        console.print("Goodbye.")
        sys.exit(0)


def run_cli():
    """Run interactive CLI."""
    console.print(Panel.fit(
        "[bold blue]JARVIS[/bold blue]\n"
        "[dim]/help for commands, /quit to exit[/dim]",
        border_style="blue"
    ))

    try:
        jarvis = Jarvis()
    except Exception as e:
        console.print(f"[red]Failed to start: {e}[/red]")
        console.print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    console.print(f"[dim]Persona: {jarvis.current_persona} | Model: {jarvis.ollama.default_model}[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            console.print("\n[bold blue]Jarvis[/bold blue]:", end=" ")
            jarvis.process(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]/quit to exit[/dim]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    run_cli()
