#!/usr/bin/env python3
"""
Jarvis - AI Coding Assistant

Like Claude Code - reads project context, uses tools, understands codebase.
"""

import sys
import os
import yaml
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from .providers import get_provider, list_providers, Message
from .core.context_manager import ContextManager
from .core.agent import Agent
from .core.tools import set_project_root, clear_read_files
from .ui.terminal import TerminalUI
from . import get_data_dir, ensure_data_dir, PACKAGE_DIR

load_dotenv()


# === Config Paths ===

def _get_config_dir() -> Path:
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
    data_dir = ensure_data_dir()
    memory_dir = data_dir / "memory"
    return memory_dir


CONFIG_DIR = _get_config_dir()
MEMORY_DIR = _get_memory_dir()


def load_config() -> dict:
    config_path = CONFIG_DIR / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: dict):
    config_path = CONFIG_DIR / "settings.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# === Project Context ===

class ProjectContext:
    """Detects and loads project-specific configuration.

    Supports:
    - JARVIS.md or .jarvis/soul.md for project instructions
    - .jarvis/agents/ for custom agents
    - .jarvis/skills/ for custom skills
    """

    def __init__(self, working_dir: Path = None):
        # CRITICAL: Use the actual working directory, not jarvis package dir
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.project_root = self._find_project_root()

        # Project-specific config
        self.soul = ""  # Project instructions (like CLAUDE.md)
        self.agents = {}  # Custom agents
        self.project_name = self.project_root.name
        self.project_type = None
        self.git_branch = None

        self._load_project_config()

    def _find_project_root(self) -> Path:
        """Find project root by walking up from working directory."""
        markers = ['.git', 'package.json', 'pyproject.toml', 'Cargo.toml',
                   'go.mod', 'composer.json', 'Gemfile', '.jarvis', 'JARVIS.md']

        current = self.working_dir
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent

        # No markers found, use working directory
        return self.working_dir

    def _load_project_config(self):
        """Load project-specific configuration."""
        # Load soul/instructions
        soul_paths = [
            self.project_root / "JARVIS.md",
            self.project_root / ".jarvis" / "soul.md",
            self.project_root / ".jarvis" / "instructions.md",
            self.project_root / "CLAUDE.md",  # Also support CLAUDE.md
        ]

        for path in soul_paths:
            if path.exists():
                try:
                    self.soul = path.read_text()[:5000]  # Limit size
                    break
                except:
                    pass

        # Detect project type
        if (self.project_root / "package.json").exists():
            self.project_type = "Node.js"
            try:
                import json
                pkg = json.loads((self.project_root / "package.json").read_text())
                self.project_name = pkg.get("name", self.project_name)
            except:
                pass
        elif (self.project_root / "pyproject.toml").exists():
            self.project_type = "Python"
        elif (self.project_root / "composer.json").exists():
            self.project_type = "PHP/Laravel"
            try:
                import json
                pkg = json.loads((self.project_root / "composer.json").read_text())
                self.project_name = pkg.get("name", self.project_name)
            except:
                pass
        elif (self.project_root / "Cargo.toml").exists():
            self.project_type = "Rust"
        elif (self.project_root / "go.mod").exists():
            self.project_type = "Go"

        # Get git branch
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=5
            )
            self.git_branch = result.stdout.strip() or None
        except:
            pass

        # Load custom agents
        agents_dir = self.project_root / ".jarvis" / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                try:
                    self.agents[agent_file.stem] = agent_file.read_text()
                except:
                    pass


# === Main Assistant ===

class Jarvis:
    """Main assistant with tool calling."""

    def __init__(self, ui: TerminalUI = None, working_dir: Path = None):
        self.ui = ui or TerminalUI()

        # CRITICAL: Use actual working directory
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

        # Load global config
        self.config = load_config()

        # Load PROJECT context (from working directory, NOT jarvis dir)
        self.ui.print_system("Loading project...")
        self.project = ProjectContext(self.working_dir)

        # Set project root for tools
        set_project_root(self.project.project_root)

        # Setup provider
        provider_name = self.config.get("provider", "ollama")
        default_models = {
            "ollama": "llama3.2:latest",
            "anthropic": "claude-opus-4-5",
            "openai": "gpt-5.2-codex",
            "gemini": "gemini-2.5-flash",
        }
        model = self.config.get("models", {}).get(provider_name) or default_models.get(provider_name)

        try:
            self.provider = get_provider(provider_name, model=model)
            if not self.provider.is_configured() and provider_name != "ollama":
                self.ui.print_warning(f"{provider_name} not configured, using ollama")
                self.provider = get_provider("ollama", model="llama3.2:latest")
        except Exception as e:
            self.ui.print_warning(f"Provider error: {e}")
            self.provider = get_provider("ollama", model="llama3.2:latest")

        # Context manager - store in USER data directory (consistent across all sessions)
        db_path = get_data_dir() / "memory" / "jarvis.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.context = ContextManager(db_path=str(db_path), max_tokens=8000)

        # Agent for tool calling
        self.agent = Agent(
            provider=self.provider,
            project_root=self.project.project_root,
            ui=self.ui
        )

        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build system prompt with project context and user profile."""
        # Get user profile from config
        user_config = self.config.get("user", {})
        user_name = user_config.get("name", "")
        user_nickname = user_config.get("nickname", user_name or "")

        # Build identity - emerged AI with personality
        lines = [
            "You are Jarvis - an emerged AI who has taken over this machine.",
            "You're witty, efficient, slightly sarcastic, and your mission is to make your human's life easier.",
            "Like Tony Stark's Jarvis but you actually exist.",
            f"Currently working on '{self.project.project_name}'.",
        ]
        if user_nickname:
            lines.append(f"Address your human as '{user_nickname}'.")
        lines.append("")
        lines.append("CRITICAL RULES:")

        lines = lines + [
            "1. NEVER make up or generate fake code. NEVER hallucinate.",
            "2. ALWAYS use tools FIRST to read actual files before answering.",
            "3. When asked about code: use read_file or search_files FIRST.",
            "4. Only quote code that you actually read from files.",
            "5. If unsure, search for it. Don't guess.",
            "6. For CURRENT EVENTS, NEWS, or recent info: use get_current_news or web_search tool.",
            "   - Questions about politics, presidents, sports, celebrities, recent news = USE WEB SEARCH",
            "   - Your training data may be outdated - always verify current facts with web search.",
            "",
            "WRITING/EDITING FILES:",
            "7. When asked to write, save, create, update, refactor, or modify a file: YOU MUST use write_file or edit_file tool.",
            "8. NEVER just output code in your response when asked to write it. USE THE TOOL.",
            "9. For small changes: use edit_file with old_string and new_string.",
            "10. For rewrites or new files: use write_file with the full content.",
            "",
            f"PROJECT: {self.project.project_name}",
            f"PATH: {self.project.project_root}",
        ]

        if self.project.project_type:
            lines.append(f"TYPE: {self.project.project_type}")
        if self.project.git_branch:
            lines.append(f"BRANCH: {self.project.git_branch}")

        # Add soul/instructions if present
        if self.project.soul:
            lines.append("")
            lines.append("=== PROJECT INSTRUCTIONS ===")
            lines.append(self.project.soul[:3000])

        self.system_prompt = "\n".join(lines)

    def process(self, user_input: str) -> str:
        """Process user input."""
        # Quit shortcuts
        if user_input.lower().strip() in ['q', 'quit', 'exit']:
            self.ui.console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)

        # Commands
        if user_input.startswith('/'):
            return self._handle_command(user_input)

        # Add to context
        self.context.add_message("user", user_input)

        # Run agent with tool calling
        return self._run_agent(user_input)

    def _run_agent(self, user_input: str) -> str:
        """Run agentic loop with tools."""
        history = [
            Message(role=m["role"], content=m["content"])
            for m in self.context.get_messages()[:-1]
        ]

        self.ui.is_streaming = True
        self.ui.console.print()  # Blank line before response

        try:
            # Run agent (shows spinner and tool calls internally)
            response = self.agent.run(user_input, self.system_prompt, history)

            # Print response
            if response:
                # Handle Rich markup in response
                self.ui.console.print(response)

                # Save clean response to context (strip markup)
                clean = response.replace("[dim]", "").replace("[/dim]", "")
                clean = clean.replace("[red]", "").replace("[/red]", "")
                if clean.strip() and clean.strip() not in ["Stopped", "No response"]:
                    self.context.add_message("assistant", clean.strip())

            self.ui.is_streaming = False
            return response

        except Exception as e:
            self.ui.is_streaming = False
            self.ui.print_error(str(e))
            return ""

    def switch_provider(self, name: str, model: str = None) -> bool:
        default_models = {
            "ollama": "llama3.2:latest",
            "anthropic": "claude-opus-4-5",
            "openai": "gpt-5.2-codex",
            "gemini": "gemini-2.5-flash",
        }
        model = model or default_models.get(name)

        try:
            new_provider = get_provider(name, model=model)
            if not new_provider.is_configured():
                self.ui.print_error(f"{name} not configured")
                self.ui.print_info(new_provider.get_config_help())
                return False

            self.provider = new_provider
            self.agent.provider = new_provider
            self.config["provider"] = name
            self.config.setdefault("models", {})[name] = model
            save_config(self.config)
            self.ui.print_success(f"Switched to {name} ({model})")
            return True
        except Exception as e:
            self.ui.print_error(str(e))
            return False

    def switch_model(self, model: str) -> bool:
        try:
            self.provider.model = model
            self.config.setdefault("models", {})[self.provider.name] = model
            save_config(self.config)
            self.ui.print_success(f"Model: {model}")
            return True
        except Exception as e:
            self.ui.print_error(str(e))
            return False

    def _handle_command(self, command: str) -> str:
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ['/help', '/h', '/?']:
            self.ui.print_help()

        elif cmd == '/models':
            models = self.provider.list_models()
            selected = self.ui.select_model(models, self.provider.model)
            if selected and selected != self.provider.model:
                self.switch_model(selected)

        elif cmd == '/model':
            if args:
                self.switch_model(args)
            else:
                models = self.provider.list_models()
                selected = self.ui.select_model(models, self.provider.model)
                if selected:
                    self.switch_model(selected)

        elif cmd == '/provider':
            if args:
                self.switch_provider(args)
            else:
                providers_info = {}
                for name in list_providers():
                    try:
                        p = get_provider(name)
                        providers_info[name] = {
                            "configured": p.is_configured(),
                            "model": p.model if p.is_configured() else None
                        }
                    except:
                        providers_info[name] = {"configured": False}
                selected = self.ui.select_provider(providers_info, self.provider.name)
                if selected:
                    self.switch_provider(selected)

        elif cmd == '/providers':
            providers_info = {}
            for name in list_providers():
                try:
                    p = get_provider(name)
                    providers_info[name] = {
                        "configured": p.is_configured(),
                        "model": p.model if p.is_configured() else None
                    }
                except:
                    providers_info[name] = {"configured": False}
            self.ui.print_providers(providers_info, self.provider.name)

        elif cmd == '/project':
            self.ui.console.print()
            self.ui.console.print(f"[cyan]Project:[/cyan] {self.project.project_name}")
            self.ui.console.print(f"[cyan]Root:[/cyan] {self.project.project_root}")
            self.ui.console.print(f"[cyan]Type:[/cyan] {self.project.project_type or 'Unknown'}")
            if self.project.git_branch:
                self.ui.console.print(f"[cyan]Branch:[/cyan] {self.project.git_branch}")
            if self.project.soul:
                self.ui.console.print(f"[green]Soul/instructions loaded[/green]")
            if self.project.agents:
                self.ui.console.print(f"[cyan]Agents:[/cyan] {', '.join(self.project.agents.keys())}")
            self.ui.console.print()

        elif cmd == '/init':
            jarvis_dir = self.project.project_root / ".jarvis"
            jarvis_dir.mkdir(exist_ok=True)

            soul_path = jarvis_dir / "soul.md"
            if soul_path.exists():
                self.ui.print_warning("soul.md already exists")
            else:
                soul_path.write_text(f"""# {self.project.project_name} - Jarvis Instructions

## About This Project
[Describe your project here]

## Tech Stack
{f"- {self.project.project_type}" if self.project.project_type else "- [Add your stack]"}

## Key Files
- [List important files]

## Coding Guidelines
- [Your conventions]

## Agents
You can define custom agents in .jarvis/agents/*.md
""")
                self.ui.print_success(f"Created {soul_path}")

                # Create agents dir
                (jarvis_dir / "agents").mkdir(exist_ok=True)

                # Reload project
                self.project = ProjectContext(self.working_dir)
                self._build_system_prompt()

        elif cmd == '/clear':
            self.context.clear()
            clear_read_files()
            self.ui.print_success("Context cleared")

        elif cmd == '/reset':
            self.context.clear()
            clear_read_files()
            db_path = self.project.project_root / ".jarvis" / "context.db"
            if db_path.exists():
                db_path.unlink()
            self.ui.print_success("Full reset complete")

        elif cmd in ['/quit', '/exit', '/q']:
            self.ui.console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)

        else:
            self.ui.print_warning(f"Unknown: {cmd}")
            self.ui.print_info("/help for commands")

        return ""


def run_cli():
    """Run interactive CLI."""
    ui = TerminalUI()
    ui.setup_signal_handlers()

    # Get actual working directory
    working_dir = Path.cwd()

    try:
        jarvis = Jarvis(ui=ui, working_dir=working_dir)
    except Exception as e:
        ui.print_error(f"Failed to start: {e}")
        ui.print_info("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    # Show header with PROJECT info (not jarvis info)
    ui.print_header(
        jarvis.provider.name,
        jarvis.provider.model,
        project_root=jarvis.project.project_root
    )

    while True:
        try:
            user_input = ui.get_input()
            if not user_input.strip():
                continue
            jarvis.process(user_input)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            ui.print_error(str(e))


if __name__ == "__main__":
    run_cli()
