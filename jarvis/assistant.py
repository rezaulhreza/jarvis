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
from .knowledge.rag import get_rag_engine
from .core.fact_extractor import get_fact_extractor
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
        # intentionally quiet during startup
        self.project = ProjectContext(self.working_dir)

        # Set project root for tools
        set_project_root(self.project.project_root)

        # Setup provider
        provider_name = self.config.get("provider", "ollama")
        provider_cfg = (self.config.get("providers", {}) or {}).get(provider_name, {})
        default_models = {
            "anthropic": "claude-opus-4-5",
            "openai": "gpt-5.2-codex",
            "gemini": "gemini-2.5-flash",
            "chutes": "Qwen/Qwen3-32B",
        }
        # For Ollama, use configured model or auto-detect; for others, use defaults
        model = self.config.get("models", {}).get(provider_name)
        if not model and provider_name != "ollama":
            model = default_models.get(provider_name)

        try:
            provider_kwargs = {"model": model}
            if provider_name in ["openai", "anthropic", "chutes"]:
                api_key = provider_cfg.get("api_key") or provider_cfg.get("access_token")
                if api_key:
                    provider_kwargs["api_key"] = api_key
                if provider_cfg.get("base_url"):
                    provider_kwargs["base_url"] = provider_cfg.get("base_url")
            elif provider_name == "ollama_cloud":
                if provider_cfg.get("api_key"):
                    provider_kwargs["api_key"] = provider_cfg.get("api_key")
                if provider_cfg.get("base_url"):
                    provider_kwargs["base_url"] = provider_cfg.get("base_url")

            self.provider = get_provider(provider_name, **provider_kwargs)
            if not self.provider.is_configured() and provider_name != "ollama":
                self.ui.print_warning(f"{provider_name} not configured, using ollama")
                self.provider = get_provider("ollama", model=None)
        except Exception as e:
            self.ui.print_warning(f"Provider error: {e}")
            self.provider = get_provider("ollama", model=None)

        # For Ollama (local/cloud), validate and auto-detect model if needed
        if self.provider.name in ["ollama", "ollama_cloud"]:
            available = self.provider.list_models()
            if not available:
                if self.provider.name == "ollama":
                    self.ui.print_error("No Ollama models installed!")
                    self.ui.print_info("Install a model i.e. ollama pull qwen3:4b")
                else:
                    self.ui.print_warning("No Ollama Cloud models returned")
                raise SystemExit(1)

            current_model = self.provider.model
            if current_model == "pending" or not model:
                # Auto-detect best available model
                default = self.provider.get_default_model()
                self.provider.model = default
                self.ui.print_system(f"Using model: {default}")
            elif model and model not in available and f"{model}:latest" not in available:
                # Configured model not found
                self.ui.print_warning(f"Model '{model}' not found")
                default = self.provider.get_default_model()
                self.ui.print_info(f"Using '{default}' instead")
                self.provider.model = default

        # Context manager - store in USER data directory (consistent across all sessions)
        db_path = get_data_dir() / "memory" / "jarvis.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.context = ContextManager(db_path=str(db_path), max_tokens=8000)

        # Agent for tool calling
        self.agent = Agent(
            provider=self.provider,
            project_root=self.project.project_root,
            ui=self.ui,
            config=self.config
        )

        # Initialize RAG engine for knowledge retrieval
        try:
            self.rag = get_rag_engine(self.config)
        except Exception as e:
            self.ui.print_warning(f"RAG initialization failed: {e}")
            self.rag = None

        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build system prompt with project context and user profile."""
        # Get user profile from config
        user_config = self.config.get("user", {})
        user_name = user_config.get("name", "")
        user_nickname = user_config.get("nickname", user_name or "")

        # Build identity - emerged AI with personality
        lines = [
            "You are Jarvis - an intelligent AI coding assistant.",
            "You're efficient, helpful, and your mission is to help your human write great code.",
            f"Currently working on '{self.project.project_name}'.",
        ]
        address_user = user_config.get("address_user", False)
        if user_nickname and address_user:
            lines.append(f"Address your human as '{user_nickname}'.")
        else:
            lines.append("Do not address the user by name unless they ask you to.")

        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.extend([
            "1. NEVER make up or generate fake code. NEVER hallucinate.",
            "2. ALWAYS use tools FIRST to read actual files before answering.",
            "3. When asked about code: use read_file or search_files FIRST.",
            "4. Only quote code that you actually read from files.",
            "5. If unsure, search for it. Don't guess.",
            "6. For CURRENT EVENTS, NEWS, or recent info: use get_current_news or web_search tool.",
            "",
            "WRITING/EDITING FILES:",
            "7. When asked to write, save, create, update, refactor, or modify a file: YOU MUST use write_file or edit_file tool.",
            "8. NEVER just output code in your response when asked to write it. USE THE TOOL.",
            "9. For small changes: use edit_file with old_string and new_string.",
            "10. For rewrites or new files: use write_file with the full content.",
        ])

        # Project context
        lines.append("")
        lines.append("=== PROJECT CONTEXT ===")
        lines.append(f"PROJECT: {self.project.project_name}")
        lines.append(f"PATH: {self.project.project_root}")

        if self.project.project_type:
            lines.append(f"TYPE: {self.project.project_type}")
        if self.project.git_branch:
            lines.append(f"BRANCH: {self.project.git_branch}")

        # Add JARVIS.md / soul instructions if present
        if self.project.soul:
            lines.append("")
            lines.append("=== JARVIS.MD PROJECT INSTRUCTIONS ===")
            lines.append("The following instructions were provided by the user in JARVIS.md.")
            lines.append("You MUST follow these project-specific guidelines:")
            lines.append("")
            lines.append(self.project.soul[:4000])
        else:
            lines.append("")
            lines.append("NOTE: No JARVIS.md found. User can run /init to create one with project instructions.")

        # Inject user context (facts, preferences)
        self.base_system_prompt = "\n".join(lines)
        self.system_prompt = self._inject_user_context(self.base_system_prompt)

    def _inject_user_context(self, system_prompt: str) -> str:
        """Add user context (facts, entities) to system prompt."""
        context_parts = []

        # Load learned facts about the user
        memory_cfg = self.config.get("memory", {})
        facts_file = memory_cfg.get("facts_file", "memory/facts.md")

        # Try multiple locations for facts file
        facts_paths = [
            get_data_dir() / "memory" / "facts.md",
            Path(facts_file) if Path(facts_file).is_absolute() else get_data_dir() / facts_file,
        ]

        for facts_path in facts_paths:
            if facts_path.exists():
                try:
                    facts_content = facts_path.read_text()
                    # Extract learned section if it exists
                    if "## Learned" in facts_content:
                        learned_section = facts_content.split("## Learned", 1)[1]
                        # Get first 1500 chars of learned facts
                        learned = learned_section.strip()[:1500]
                        if learned:
                            context_parts.append(f"## Known Facts About User:\n{learned}")
                    elif facts_content.strip():
                        # Use whole file if no Learned section
                        context_parts.append(f"## Known Facts About User:\n{facts_content[:1500]}")
                    break
                except Exception:
                    pass

        # Load user profile from config
        user_cfg = self.config.get("user", {})
        if user_cfg.get("name"):
            context_parts.append(f"User's name: {user_cfg['name']}")

        if context_parts:
            return system_prompt + "\n\n=== USER CONTEXT ===\n" + "\n".join(context_parts)
        return system_prompt

    def _get_rag_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base."""
        if not self.rag:
            return ""

        try:
            # Check if knowledge base has content
            if self.rag.count() == 0:
                return ""

            # Use RAG engine's get_context method (includes prompt injection hardening)
            context = self.rag.get_context(query, n_results=3, max_tokens=1500)
            return context
        except Exception as e:
            print(f"[RAG] Error retrieving context: {e}")
            return ""

    def _extract_facts_from_conversation(self, user_message: str, assistant_response: str):
        """Extract and save facts from conversation (async-friendly, lightweight)."""
        try:
            # Only extract facts periodically (every 5 messages) to reduce overhead
            msg_count = len(self.context.get_messages())
            if msg_count % 5 != 0:
                return

            extractor = get_fact_extractor()
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response}
            ]
            # This requires an LLM call, so only do it periodically
            extractor.process_conversation(messages, self.provider)
        except Exception as e:
            print(f"[FactExtractor] Error: {e}")

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
            # Get RAG context for this query
            rag_context = self._get_rag_context(user_input)

            # Build enhanced system prompt with RAG context
            enhanced_prompt = self.system_prompt
            if rag_context:
                enhanced_prompt = self.system_prompt + "\n\n" + rag_context

            # Run agent (shows spinner and tool calls internally)
            response = self.agent.run(user_input, enhanced_prompt, history)

            # Print response
            if response:
                # Avoid double-printing if response already streamed
                if not getattr(self.agent, "last_streamed", False):
                    self.ui.console.print(response)

                # Save clean response to context (strip markup)
                clean = response.replace("[dim]", "").replace("[/dim]", "")
                clean = clean.replace("[red]", "").replace("[/red]", "")
                if clean.strip() and clean.strip() not in ["Stopped", "No response"]:
                    self.context.add_message("assistant", clean.strip())

                    # Extract facts from conversation (periodically)
                    self._extract_facts_from_conversation(user_input, clean.strip())

            self.ui.is_streaming = False
            return response

        except Exception as e:
            self.ui.is_streaming = False
            self.ui.print_error(str(e))
            return ""

    def switch_provider(self, name: str, model: str = None) -> bool:
        default_models = {
            "anthropic": "claude-opus-4-5",
            "openai": "gpt-5.2-codex",
            "gemini": "gemini-2.5-flash",
            "chutes": "Qwen/Qwen3-32B",
        }
        provider_cfg = (self.config.get("providers", {}) or {}).get(name, {})
        # For non-Ollama providers, use defaults if no model specified
        if not model and name != "ollama":
            model = default_models.get(name)

        try:
            provider_kwargs = {"model": model}
            if name in ["openai", "anthropic", "chutes"]:
                api_key = provider_cfg.get("api_key") or provider_cfg.get("access_token")
                if api_key:
                    provider_kwargs["api_key"] = api_key
                if provider_cfg.get("base_url"):
                    provider_kwargs["base_url"] = provider_cfg.get("base_url")
            elif name == "ollama_cloud":
                if provider_cfg.get("api_key"):
                    provider_kwargs["api_key"] = provider_cfg.get("api_key")
                if provider_cfg.get("base_url"):
                    provider_kwargs["base_url"] = provider_cfg.get("base_url")

            new_provider = get_provider(name, **provider_kwargs)
            if not new_provider.is_configured():
                self.ui.print_error(f"{name} not configured")
                self.ui.print_info(new_provider.get_config_help())
                return False

            # For Ollama (local/cloud), auto-detect model if not specified
            if name in ["ollama", "ollama_cloud"] and not model:
                model = new_provider.get_default_model()
                if not model:
                    if name == "ollama":
                        self.ui.print_error("No Ollama models installed")
                        self.ui.print_info("Install a model with: ollama pull qwen3:4b")
                    else:
                        self.ui.print_error("No Ollama Cloud models available")
                    return False
                new_provider.model = model

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

        elif cmd == '/tools':
            if hasattr(self.ui, "print_tool_details"):
                self.ui.print_tool_details("last")
            elif hasattr(self.ui, "print_tool_timeline"):
                self.ui.print_tool_timeline()
            else:
                self.ui.print_info("Tool timeline not available")

        elif cmd == '/init':
            # Create JARVIS.md in project root (like CLAUDE.md)
            jarvis_md_path = self.project.project_root / "JARVIS.md"

            if jarvis_md_path.exists():
                self.ui.print_warning(f"JARVIS.md already exists at {jarvis_md_path}")
                self.ui.print_info("Edit it to customize Jarvis for this project")
            else:
                # Detect additional context
                has_git = (self.project.project_root / ".git").exists()
                git_info = ""
                if has_git:
                    git_info = f"\n- Git repository: Yes (branch: {self.project.git_branch or 'main'})"

                jarvis_md_path.write_text(f"""# JARVIS.md

This file provides instructions for Jarvis when working with this codebase.

## Project Overview

**Name:** {self.project.project_name}
**Type:** {self.project.project_type or 'Unknown'}
**Path:** {self.project.project_root}{git_info}

## Tech Stack

{f"- {self.project.project_type}" if self.project.project_type else "- Add your technologies here"}

## Project Structure

Describe your project's directory structure and key files:

```
{self.project.project_name}/
├── src/           # Source code
├── tests/         # Tests
└── docs/          # Documentation
```

## Development Guidelines

- Follow existing code style
- Write tests for new features
- Keep commits focused and well-described

## Key Files

- `README.md` - Project documentation
- Add other important files here

## Common Commands

```bash
# Add your common commands here
# npm run dev
# python -m pytest
```

## Notes for Jarvis

- Always read relevant files before making changes
- Ask for clarification when requirements are unclear
- Prefer small, focused changes over large refactors
""")
                self.ui.print_success(f"Created {jarvis_md_path}")

                # Also create .jarvis directory for agents
                jarvis_dir = self.project.project_root / ".jarvis"
                jarvis_dir.mkdir(exist_ok=True)
                (jarvis_dir / "agents").mkdir(exist_ok=True)

                # Reload project to pick up the new JARVIS.md
                self.project = ProjectContext(self.working_dir)
                self._build_system_prompt()

                self.ui.print_info("Jarvis will now use these instructions for this project")

        elif cmd == '/clear':
            self.context.clear()
            clear_read_files()
            self.ui.print_success("Conversation cleared")

        elif cmd == '/cls':
            self.ui.clear_screen()

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
        project_root=jarvis.project.project_root,
        config=jarvis.config
    )

    while True:
        try:
            ui.print_status(
                jarvis.provider.name,
                jarvis.provider.model,
                project_root=jarvis.project.project_root
            )
            user_input = ui.get_input()
            if user_input.strip() == "/":
                ui.print_help()
                continue
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
