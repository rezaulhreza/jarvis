"""
Terminal UI for Jarvis - Claude Code-like interface.
"""

import sys
import os
import signal
import threading
import time
from typing import Optional, Callable, List
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

try:
    import questionary
    from questionary import Style as QStyle
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


class TerminalUI:
    """Terminal interface like Claude Code."""

    def __init__(self):
        self.console = Console()
        self.is_streaming = False
        self.stop_requested = False
        self._interrupt_count = 0
        self._last_interrupt_time = 0

        # Colors
        self.colors = {
            "user": "bold green",
            "assistant": "bold blue",
            "system": "dim",
            "error": "bold red",
            "warning": "yellow",
            "success": "green",
            "info": "cyan",
            "tool": "dim",
        }

        # Questionary style
        if HAS_QUESTIONARY:
            self.qstyle = QStyle([
                ('qmark', 'fg:cyan bold'),
                ('question', 'bold'),
                ('answer', 'fg:cyan'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
            ])

        # Setup prompt with history
        self.session = None
        if HAS_PROMPT_TOOLKIT:
            try:
                history_file = Path.home() / ".jarvis" / "history"
                history_file.parent.mkdir(parents=True, exist_ok=True)
                self.session = PromptSession(history=FileHistory(str(history_file)))
            except Exception:
                pass

    def setup_signal_handlers(self):
        """Set up Ctrl+C handling."""
        def handler(signum, frame):
            current_time = time.time()

            # If streaming, just request stop
            if self.is_streaming:
                self.stop_requested = True
                print()  # New line
                return

            # Double Ctrl+C to exit (within 1.5 seconds)
            if current_time - self._last_interrupt_time < 1.5:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 1
            self._last_interrupt_time = current_time

            if self._interrupt_count >= 2:
                print()  # New line
                self.console.print("[yellow]Goodbye![/yellow]")
                sys.exit(0)
            else:
                print()  # New line
                self.console.print("[dim]Press Ctrl+C again to exit[/dim]")

        signal.signal(signal.SIGINT, handler)

    def print_header(self, provider: str, model: str, project_root: Path = None):
        """Print startup header."""
        self.console.print()
        banner = """[bold cyan]     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝[/bold cyan]"""
        self.console.print(banner)
        self.console.print()
        self.console.print(f"[dim]Provider:[/dim] [cyan]{provider}[/cyan]  [dim]Model:[/dim] [cyan]{model}[/cyan]")
        if project_root:
            self.console.print(f"[dim]Project:[/dim] [magenta]{project_root.name}[/magenta]  [dim]Path:[/dim] {project_root}")
        self.console.print()
        self.console.print("[dim]Type /help • Ctrl+C to stop • Ctrl+C×2 to exit[/dim]")
        self.console.print()

    def get_input(self, prompt: str = ">") -> str:
        """Get user input."""
        try:
            self.stop_requested = False

            if self.session:
                return self.session.prompt(f"{prompt} ")
            else:
                self.console.print(f"[green]{prompt}[/green] ", end="")
                return input()

        except EOFError:
            return "/quit"

        except KeyboardInterrupt:
            current_time = time.time()

            # Check for double Ctrl+C
            if current_time - self._last_interrupt_time < 1.5:
                self._interrupt_count += 1
            else:
                self._interrupt_count = 1
            self._last_interrupt_time = current_time

            if self._interrupt_count >= 2:
                print()
                self.console.print("[yellow]Goodbye![/yellow]")
                sys.exit(0)
            else:
                print()
                self.console.print("[dim]Ctrl+C again to exit[/dim]")
                return ""

    def print_tool(self, message: str):
        """Print tool activity in Claude style."""
        self.console.print(f"[dim]  {message}[/dim]")

    def show_spinner(self, message: str = "Thinking"):
        """Show a spinner. Returns a context manager."""
        return Live(
            Spinner("dots", text=f" [dim]{message}...[/dim]", style="cyan"),
            console=self.console,
            refresh_per_second=10,
            transient=True
        )

    def print_error(self, message: str):
        self.console.print(f"[red]Error: {message}[/red]")

    def print_warning(self, message: str):
        self.console.print(f"[yellow]{message}[/yellow]")

    def print_success(self, message: str):
        self.console.print(f"[green]✓ {message}[/green]")

    def print_info(self, message: str):
        self.console.print(f"[cyan]{message}[/cyan]")

    def print_system(self, message: str):
        self.console.print(f"[dim]{message}[/dim]")

    def print_help(self):
        """Print help."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        commands = [
            ("/help", "Show this help"),
            ("/models", "Select model interactively"),
            ("/model <name>", "Switch to specific model"),
            ("/provider", "Switch provider (ollama/anthropic/openai/gemini)"),
            ("/project", "Show project info"),
            ("/clear", "Clear conversation"),
            ("/quit", "Exit"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def select_model(self, models: List[str], current: str) -> Optional[str]:
        """Interactive model selection."""
        if not HAS_QUESTIONARY:
            self.print_models(models, current)
            return None

        choices = []
        for m in models[:20]:
            if m == current:
                choices.append(f"● {m} (current)")
            else:
                choices.append(f"  {m}")

        try:
            result = questionary.select(
                "Select model:",
                choices=choices,
                style=self.qstyle,
            ).ask()

            if result:
                return result.replace("● ", "").replace(" (current)", "").strip()
        except Exception:
            pass
        return None

    def select_provider(self, providers: dict, current: str) -> Optional[str]:
        """Interactive provider selection."""
        if not HAS_QUESTIONARY:
            self.print_providers(providers, current)
            return None

        choices = []
        for name, info in providers.items():
            status = "✓" if info.get("configured") else "✗"
            marker = "●" if name == current else " "
            choices.append(f"{marker} {name} [{status}]")

        try:
            result = questionary.select(
                "Select provider:",
                choices=choices,
                style=self.qstyle,
            ).ask()

            if result:
                parts = result.split()
                return parts[1] if len(parts) > 1 else None
        except Exception:
            pass
        return None

    def print_models(self, models: list, current: str):
        self.console.print()
        for model in models[:15]:
            marker = " [cyan]●[/cyan]" if model == current else "  "
            self.console.print(f"{marker} {model}")
        self.console.print()

    def print_providers(self, providers: dict, current: str):
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Model")

        for name, info in providers.items():
            marker = "[cyan]●[/cyan]" if name == current else " "
            status = "[green]✓[/green]" if info["configured"] else "[red]✗[/red]"
            model = info.get("model") or "-"
            table.add_row(f"{marker} {name}", status, model)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def confirm(self, message: str) -> bool:
        if HAS_QUESTIONARY:
            return questionary.confirm(message).ask() or False
        self.console.print(f"{message} [dim](y/n)[/dim] ", end="")
        return input().lower().strip() in ('y', 'yes')
