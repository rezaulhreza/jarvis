#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant
A local-first AI assistant with memory, tools, and multiple models.
"""

import sys
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from core.ollama_client import OllamaClient
from core.context_manager import ContextManager
from core.router import ToolRouter, should_use_reasoning, should_use_vision
from skills import AVAILABLE_SKILLS, get_skills_schema, list_skills

# Load environment variables
load_dotenv()

# Initialize rich console for nice output
console = Console()

# Paths
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
MEMORY_DIR = BASE_DIR / "memory"
PERSONAS_DIR = CONFIG_DIR / "personas"


def load_config() -> dict:
    """Load configuration from settings.yaml"""
    config_path = CONFIG_DIR / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_persona(name: str = "default") -> str:
    """Load a persona prompt"""
    persona_path = PERSONAS_DIR / f"{name}.md"
    if persona_path.exists():
        with open(persona_path) as f:
            return f.read()

    # Fallback to default
    default_path = PERSONAS_DIR / "default.md"
    if default_path.exists():
        with open(default_path) as f:
            return f.read()

    return "You are a helpful assistant."


def load_rules() -> str:
    """Load safety rules"""
    rules_path = CONFIG_DIR / "rules.md"
    if rules_path.exists():
        with open(rules_path) as f:
            return f.read()
    return ""


def load_facts() -> str:
    """Load user facts"""
    facts_path = MEMORY_DIR / "facts.md"
    if facts_path.exists():
        with open(facts_path) as f:
            return f.read()
    return ""


def list_personas() -> list[str]:
    """List available personas."""
    if not PERSONAS_DIR.exists():
        return ["default"]
    return [p.stem for p in PERSONAS_DIR.glob("*.md")]


class Jarvis:
    """Main assistant class that orchestrates everything."""

    def __init__(self):
        self.config = load_config()
        self.current_persona = self.config.get("persona", "default")
        self.persona_prompt = load_persona(self.current_persona)
        self.rules = load_rules()
        self.facts = load_facts()

        # Initialize components
        model_config = self.config.get('models', {})
        self.ollama = OllamaClient(default_model=model_config.get('default', 'qwen3:4b'))
        self.context = ContextManager(
            db_path=str(BASE_DIR / "memory" / "jarvis.db"),
            max_tokens=self.config.get('context', {}).get('max_tokens', 8000)
        )
        self.router = ToolRouter(self.ollama, router_model=model_config.get('tools', 'functiongemma'))

        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build the system prompt from persona, facts, and rules."""
        self.system_prompt = f"""{self.persona_prompt}

## User Facts
{self.facts}

## Available Tools
{get_skills_schema()}

## Rules
{self.rules}

When you need to use a tool, I'll do so automatically based on the user's request.
"""

    def switch_persona(self, name: str) -> str:
        """Switch to a different persona."""
        available = list_personas()
        if name not in available:
            return f"Unknown persona: {name}. Available: {', '.join(available)}"

        self.current_persona = name
        self.persona_prompt = load_persona(name)
        self._build_system_prompt()
        return f"Switched to {name} persona."

    def process(self, user_input: str) -> str:
        """Process user input and generate response."""

        # Check for special commands
        if user_input.startswith('/'):
            return self._handle_command(user_input)

        # Add user message to context
        self.context.add_message("user", user_input)

        # Determine if we need special handling
        if should_use_vision(user_input):
            return self._handle_vision(user_input)

        if should_use_reasoning(user_input):
            return self._handle_reasoning(user_input)

        # Route to appropriate tool
        route_result = self.router.route(user_input, self.context.get_working_memory())

        if route_result.get('tool') and route_result['tool'] != 'none':
            tool_output = self._execute_tool(route_result)
            if tool_output:
                # Include tool output in response generation
                self.context.update_working_memory("last_tool", route_result['tool'])
                self.context.update_working_memory("tool_output", tool_output)

        # Generate response
        return self._generate_response()

    def _generate_response(self) -> str:
        """Generate a response using the default model."""
        messages = self.context.get_messages()

        # Add tool output to context if available
        tool_output = self.context.get_working_memory().get("tool_output")
        if tool_output:
            messages.append({
                "role": "system",
                "content": f"Tool output:\n{tool_output}"
            })
            # Clear tool output after using
            self.context.working_memory.pop("tool_output", None)

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
            response_text = f"Error generating response: {e}"
            console.print(response_text)

        console.print()  # Newline after streaming

        # Save assistant response to context
        self.context.add_message("assistant", response_text)

        return response_text

    def _handle_vision(self, user_input: str) -> str:
        """Handle image-related requests."""
        import re
        # Match various image path patterns
        path_match = re.search(
            r'([/~][^\s]+\.(jpg|jpeg|png|gif|webp|bmp))',
            user_input,
            re.IGNORECASE
        )

        if not path_match:
            return "Please provide an image path. Example: describe /path/to/image.jpg"

        image_path = path_match.group(1)
        image_path = os.path.expanduser(image_path)

        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"

        console.print(f"[dim]Analyzing image: {image_path}[/dim]")

        try:
            result = self.ollama.vision(image_path, user_input)
            self.context.add_message("assistant", result)
            console.print(result)
            return result
        except Exception as e:
            error_msg = f"Error analyzing image: {e}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def _handle_reasoning(self, user_input: str) -> str:
        """Handle requests that need deep reasoning."""
        reasoning_model = self.config.get('models', {}).get('reasoning', 'deepseek-r1:8b')

        # Check if model is available
        available_models = self.ollama.list_models()
        if reasoning_model not in available_models:
            console.print(f"[yellow]Reasoning model {reasoning_model} not installed, using default[/yellow]")
            return self._generate_response()

        console.print(f"[dim]Using reasoning model: {reasoning_model}[/dim]\n")

        messages = self.context.get_messages()
        response_text = ""

        try:
            for chunk in self.ollama.chat(
                messages=messages,
                model=reasoning_model,
                system=self.system_prompt,
                stream=True
            ):
                console.print(chunk, end="")
                response_text += chunk
        except Exception as e:
            console.print(f"\n[yellow]Reasoning model failed, using default[/yellow]")
            return self._generate_response()

        console.print()
        self.context.add_message("assistant", response_text, model=reasoning_model)

        return response_text

    def _execute_tool(self, route_result: dict) -> str:
        """Execute a tool and return its output."""
        tool_name = route_result.get('tool')
        params = route_result.get('params', {})

        if tool_name not in AVAILABLE_SKILLS:
            return ""

        console.print(f"[dim]Using tool: {tool_name}[/dim]")

        try:
            tool_func = AVAILABLE_SKILLS[tool_name]['function']
            # Filter out empty params
            params = {k: v for k, v in params.items() if v}
            result = tool_func(**params)
            return str(result)
        except Exception as e:
            console.print(f"[red]Tool error: {e}[/red]")
            return f"Tool error: {e}"

    def _handle_command(self, command: str) -> str:
        """Handle slash commands."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == '/help':
            help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help |
| `/models` | List available Ollama models |
| `/persona <name>` | Switch persona |
| `/personas` | List available personas |
| `/skills` | List available skills |
| `/facts` | Show stored facts |
| `/memory` | Show working memory |
| `/clear` | Clear conversation history |
| `/history` | Show recent conversation |
| `/quit` | Exit Jarvis |
"""
            console.print(Markdown(help_text))
            return ""

        elif cmd == '/models':
            try:
                models = self.ollama.list_models()
                console.print("\n[bold]Available models:[/bold]")
                for m in models:
                    console.print(f"  - {m}")
            except Exception as e:
                console.print(f"[red]Error listing models: {e}[/red]")
            return ""

        elif cmd == '/persona':
            if not args:
                console.print(f"Current persona: [bold]{self.current_persona}[/bold]")
                console.print(f"Available: {', '.join(list_personas())}")
                return ""
            result = self.switch_persona(args)
            console.print(result)
            return ""

        elif cmd == '/personas':
            personas = list_personas()
            console.print("\n[bold]Available personas:[/bold]")
            for p in personas:
                marker = " (active)" if p == self.current_persona else ""
                console.print(f"  - {p}{marker}")
            return ""

        elif cmd == '/skills':
            console.print("\n[bold]Available skills:[/bold]")
            for name, info in AVAILABLE_SKILLS.items():
                console.print(f"  - [cyan]{name}[/cyan]: {info['description']}")
            return ""

        elif cmd == '/clear':
            self.context.clear()
            console.print("Conversation cleared.")
            return ""

        elif cmd == '/facts':
            console.print(Markdown(f"## Stored Facts\n\n{self.facts}"))
            return ""

        elif cmd == '/memory':
            mem = self.context.get_working_memory()
            console.print(f"\n[bold]Working Memory:[/bold]\n{mem}")
            return ""

        elif cmd == '/history':
            history = self.context.get_history(limit=10)
            if not history:
                console.print("No history yet.")
                return ""
            console.print("\n[bold]Recent History:[/bold]\n")
            for msg in history:
                role = msg['role'].upper()
                content = msg['content'][:100]
                if len(msg['content']) > 100:
                    content += "..."
                console.print(f"[bold]{role}:[/bold] {content}\n")
            return ""

        elif cmd in ['/quit', '/exit', '/q']:
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)

        else:
            console.print(f"Unknown command: {cmd}. Try /help")
            return ""


def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold blue]JARVIS[/bold blue] - Personal AI Assistant\n"
        "[dim]Type /help for commands, /quit to exit[/dim]",
        border_style="blue"
    ))

    try:
        jarvis = Jarvis()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
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
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
