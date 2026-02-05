"""
Terminal UI for Jarvis - Interactive AI assistant interface.
"""

import sys
import os
import signal
import threading
import time
import select
from typing import Optional, Callable, List
from pathlib import Path

# For non-blocking keyboard input (ESC detection)
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.padding import Padding

# Gradient colors for the banner (cyan -> blue -> purple)
GRADIENT_COLORS = [
    "#00d4ff",  # Bright cyan
    "#00b4e6",  # Cyan-blue
    "#0094cc",  # Blue
    "#0074b3",  # Darker blue
    "#5454a6",  # Blue-purple
    "#8844aa",  # Purple
]

# JARVIS ASCII Banner with gradient support
JARVIS_BANNER = [
    "     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗",
    "     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝",
    "     ██║███████║██████╔╝██║   ██║██║███████╗",
    "██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║",
    "╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║",
    " ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝",
]

# Small robot icon for compact display
JARVIS_ICON = [
    "  ┌─────┐",
    "  │◉   ◉│",
    "  │  ▽  │",
    "  └──┬──┘",
    "   ╭─┴─╮",
    "   │░░░│",
    "   ╰───╯",
]

try:
    import questionary
    from questionary import Style as QStyle
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from .. import __version__

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.document import Document
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# Valid commands for highlighting
VALID_COMMANDS = {
    "/help", "/init", "/models", "/model", "/provider", "/providers",
    "/project", "/tools", "/context", "/level", "/clear", "/cls", "/quit", "/exit", "/q", "/reset",
    "/analyze"
}


class TerminalUI:
    """Terminal interface for Jarvis AI assistant."""

    def __init__(self):
        self.console = Console()
        self.is_streaming = False
        self.stop_requested = False
        self._interrupt_count = 0
        self._last_interrupt_time = 0
        self.project_root: Optional[Path] = None
        self._file_cache: List[str] = []
        self._file_cache_ts = 0.0
        self._tool_log: List[str] = []
        self._tool_turn: List[dict] = []

        # Track current provider/model for status display
        self._current_provider: str = ""
        self._current_model: str = ""

        # Track context stats
        self._context_stats: dict = {}

        # Track operation counts for expandable summaries
        self._op_counts: dict = {}  # {"Read": 2, "Search": 1, ...}

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
                key_bindings = KeyBindings()

                @key_bindings.add("enter")
                def _(event):
                    if event.app.current_buffer.complete_state:
                        event.app.current_buffer.complete_state = None
                    event.app.current_buffer.validate_and_handle()

                @key_bindings.add("escape", "enter")  # Alt+Enter for newline
                def handle_newline(event):
                    event.app.current_buffer.insert_text("\n")

                @key_bindings.add("c-c")  # Ctrl+C clears input
                def handle_ctrl_c(event):
                    buf = event.app.current_buffer
                    if buf.text:
                        # Clear the input buffer
                        buf.reset()
                    else:
                        # If already empty, raise KeyboardInterrupt for exit handling
                        event.app.exit(exception=KeyboardInterrupt)

                @key_bindings.add("tab")
                def _(event):
                    buf = event.app.current_buffer
                    if buf.complete_state:
                        buf.complete_next()
                    else:
                        buf.start_completion(select_first=True)

                @key_bindings.add("@")
                def handle_at(event):
                    buf = event.app.current_buffer
                    buf.insert_text("@")
                    buf.start_completion(select_first=False)

                # Add key binding for / to trigger command completion
                @key_bindings.add("/")
                def handle_slash(event):
                    buf = event.app.current_buffer
                    # Only trigger completion at start of line or after whitespace
                    text_before = buf.document.text_before_cursor
                    buf.insert_text("/")
                    if not text_before or text_before.endswith(" ") or text_before.endswith("\n"):
                        buf.start_completion(select_first=False)

                style = Style.from_dict({
                    "prompt": "#e07a5f bold",  # Warm orange/coral prompt
                    "placeholder": "#555555",  # Subtle gray placeholder
                    # Command highlighting styles
                    "command": "#a855f7 bold",  # Purple for valid commands
                    "file-mention": "#60a5fa",  # Blue for @file mentions
                    # Completion menu
                    "completion-menu": "bg:#1e1e1e #cccccc",
                    "completion-menu.completion": "bg:#1e1e1e #aaaaaa",
                    "completion-menu.completion.current": "bg:#3a3a3a #ffffff bold",
                    "completion-menu.meta.completion": "bg:#1e1e1e #666666",
                    "completion-menu.meta.completion.current": "bg:#3a3a3a #888888",
                    # Bottom toolbar
                    "bottom-toolbar": "bg:#1a1a1a #666666",
                    "bottom-toolbar.text": "#666666",
                })

                self.session = PromptSession(
                    history=FileHistory(str(history_file)),
                    completer=_JarvisCompleter(self),
                    complete_while_typing=True,  # Show completions while typing
                    key_bindings=key_bindings,
                    multiline=True,
                    auto_suggest=AutoSuggestFromHistory(),
                    complete_in_thread=True,
                    complete_style=CompleteStyle.MULTI_COLUMN,
                    style=style,
                    lexer=_JarvisLexer(),
                )
            except Exception as e:
                # Log the error for debugging
                import sys
                print(f"Warning: Could not initialize prompt session: {e}", file=sys.stderr)

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

    def _render_gradient_banner(self) -> Text:
        """Render JARVIS banner with gradient colors."""
        banner_text = Text()
        for i, line in enumerate(JARVIS_BANNER):
            # Cycle through gradient colors
            color = GRADIENT_COLORS[i % len(GRADIENT_COLORS)]
            banner_text.append(line + "\n", style=f"bold {color}")
        return banner_text

    def print_header(
        self,
        provider: str,
        model: str,
        project_root: Path = None,
        user_name: str = None,
        config: dict = None
    ):
        """Print startup header with gradient ASCII banner."""
        if project_root:
            self.project_root = project_root

        # Store for status updates
        self._current_provider = provider
        self._current_model = model

        # Get user name from config if not provided
        if not user_name and config:
            user_config = config.get("user", {})
            user_name = user_config.get("nickname") or user_config.get("name")

        # Print version header
        self.console.print()
        self.console.print(f"[bold cyan]Jarvis[/bold cyan] [dim]v{__version__}[/dim]")

        # Print gradient banner
        banner = self._render_gradient_banner()
        self.console.print(banner)

        # Greeting and info
        greeting = f"Welcome back{', ' + user_name if user_name else ''}!"
        self.console.print(f"  [bold white]{greeting}[/bold white]")
        self.console.print()

        # Info line
        info_parts = [provider, model]
        if project_root:
            info_parts.append(project_root.name)
        info_line = " · ".join(info_parts)
        self.console.print(f"  [dim]{info_line}[/dim]")

        # Tips
        self.console.print()
        self.console.print("[dim]  /help for commands • ESC to stop response • Ctrl+C to clear input[/dim]")
        self.console.print()

    def print_status(self, provider: str = None, model: str = None, project_root: Path = None, context_stats: dict = None):
        """Update stored status values."""
        if provider:
            self._current_provider = provider
        if model:
            self._current_model = model
        if project_root:
            self.project_root = project_root
        if context_stats:
            self._context_stats = context_stats

    def _get_toolbar_text(self):
        """Generate toolbar text for display."""
        parts = ["? help", "ESC stop", self._current_provider, self._current_model]

        # Add context stats if available
        if self._context_stats:
            pct = self._context_stats.get("percentage", 0)
            if pct > 0:
                tokens_k = self._context_stats.get("tokens_used", 0) / 1000
                max_k = self._context_stats.get("max_tokens", 8000) / 1000
                if pct > 80:
                    parts.append(f"⚠ {pct:.0f}% ({tokens_k:.1f}k/{max_k:.0f}k)")
                elif pct > 50:
                    parts.append(f"ctx {pct:.0f}%")

        return " · ".join(parts)

    def get_input(self, prompt: str = ">", placeholder: str = None) -> str:
        """Get user input with bordered input field."""
        try:
            self.stop_requested = False

            # Print top border
            width = self.console.width or 80
            self.console.print(f"[dim]{'─' * width}[/dim]")

            if self.session:
                result = self.session.prompt(
                    HTML(f"<prompt>{prompt}</prompt> "),
                    placeholder=HTML('<style fg="#555555">Ask anything...</style>') if placeholder is None else placeholder,
                    bottom_toolbar=HTML(f'<style fg="#666666">{self._get_toolbar_text()}</style>'),
                )
                # Print bottom border after input
                self.console.print(f"[dim]{'─' * width}[/dim]")
                return result
            else:
                self.console.print(f"[bold #e07a5f]{prompt}[/bold #e07a5f] ", end="")
                result = input()
                self.console.print(f"[dim]{'─' * width}[/dim]")
                return result

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

    def print_tool(self, message: str, success: bool = True):
        """Print tool activity with status indicator."""
        indicator = "[green]●[/green]" if success else "[red]●[/red]"
        self._tool_log.append({"message": message, "success": success})
        self.console.print(f"  {indicator} [dim]{message}[/dim]")

    def print_tool_timeline(self, limit: int = 20):
        """Show recent tool calls."""
        self.console.print(Rule("Tools", style="dim"))
        if not self._tool_log:
            self.console.print("[dim]No tools used yet.[/dim]")
            return
        for item in self._tool_log[-limit:]:
            if isinstance(item, dict):
                msg = item.get("message", str(item))
                success = item.get("success", True)
            else:
                msg = item
                success = True
            indicator = "[green]●[/green]" if success else "[red]●[/red]"
            self.console.print(f"  {indicator} [dim]{msg}[/dim]")

    def begin_turn(self):
        """Reset tool list and operation counts for current turn."""
        self._tool_turn = []
        self._op_counts = {}

    def record_tool(
        self,
        name: str,
        display: str,
        duration_s: float,
        tool_call_id: str | None = None,
        args: dict | None = None,
        result: str | None = None,
        success: bool = True,
    ):
        """Record a tool usage for the current turn with success/failure tracking."""
        preview = None
        if result:
            preview = result.strip().replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:177] + "..."

        # Determine if tool failed based on result
        if result and (result.startswith("Error:") or result.startswith("error:")):
            success = False

        self._tool_turn.append({
            "name": name,
            "display": display,
            "duration_s": duration_s,
            "id": tool_call_id,
            "args": args or {},
            "result_preview": preview,
            "success": success,
        })

        # Track operation counts by category
        op_category = self._get_op_category(name)
        self._op_counts[op_category] = self._op_counts.get(op_category, 0) + 1

    def _get_op_category(self, tool_name: str) -> str:
        """Map tool names to operation categories."""
        categories = {
            "read_file": "Read",
            "write_file": "Write",
            "edit_file": "Edit",
            "list_files": "List",
            "search_files": "Search",
            "run_command": "Run",
            "web_search": "Search",
            "get_project_structure": "Read",
        }
        return categories.get(tool_name, tool_name.replace("_", " ").title())

    def print_tool_section(self):
        """Print a compact tools section with operation summaries."""
        if not self._tool_turn:
            return

        # Count successes and failures
        success_count = sum(1 for t in self._tool_turn if t.get("success", True))
        fail_count = len(self._tool_turn) - success_count

        # Build summary line with operation counts
        summary_parts = []
        for op, count in self._op_counts.items():
            if count == 1:
                summary_parts.append(f"{op} 1 file")
            else:
                summary_parts.append(f"{op} {count} files")

        # Overall indicator
        if fail_count > 0:
            indicator = "[red]●[/red]"
        else:
            indicator = "[green]●[/green]"

        # Print summary line
        if summary_parts:
            summary = ", ".join(summary_parts)
            self.console.print(f"  {indicator} {summary} [dim](type /tools to expand)[/dim]")
        else:
            self.console.print(f"  {indicator} [dim]{len(self._tool_turn)} operations (/tools to expand)[/dim]")

    def print_tool_details(self, scope: str = "last"):
        """Print detailed tool info with success/failure indicators."""
        self.console.print(Rule("Tool Details", style="dim"))
        if scope == "last":
            turns = self._tool_turn
        else:
            turns = []
            for item in self._tool_log:
                if isinstance(item, dict):
                    turns.append(item)
                else:
                    turns.append({"display": item, "success": True})

        if not turns:
            self.console.print("[dim]No tool details available.[/dim]")
            return

        for idx, t in enumerate(turns, 1):
            success = t.get("success", True)
            indicator = "[green]●[/green]" if success else "[red]●[/red]"
            display = t.get("display") or t.get("message", str(t))

            dur_ms = int(t.get("duration_s", 0) * 1000)
            dur_str = f" [{dur_ms}ms]" if dur_ms > 0 else ""

            self.console.print(f"  {indicator} {idx}. {display}{dur_str}")

            if t.get("args"):
                self.console.print(f"     [dim]args: {t['args']}[/dim]")
            if t.get("result_preview"):
                style = "dim" if success else "dim red"
                self.console.print(f"     [{style}]result: {t['result_preview']}[/{style}]")

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

    def clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()

    def check_escape_pressed(self) -> bool:
        """Check if ESC key was pressed (non-blocking). Returns True if ESC detected."""
        if not HAS_TERMIOS or not sys.stdin.isatty():
            return False

        try:
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Set terminal to raw mode for non-blocking read
                tty.setraw(sys.stdin.fileno())

                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':  # ESC character
                        return True
                return False
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            return False

    def start_response_mode(self):
        """Enter response mode - enable ESC detection."""
        self.is_streaming = True
        self.stop_requested = False

    def end_response_mode(self):
        """Exit response mode."""
        self.is_streaming = False

    def print_system(self, message: str):
        self.console.print(f"[dim]{message}[/dim]")

    def print_media(self, media_type: str, path: str, filename: str = None):
        """Print media generation result with clean formatting."""
        icons = {
            "image": "🖼️",
            "video": "🎬",
            "audio": "🎵",
            "music": "🎵",
        }
        icon = icons.get(media_type, "📁")
        name = filename or Path(path).name

        self.console.print()
        self.console.print(f"[bold green]{icon} Generated {media_type}:[/bold green] [cyan]{name}[/cyan]")
        self.console.print(f"[dim]Saved to:[/dim] {path}")
        self.console.print()

    def stream_text(self, text: str):
        """Stream text to the terminal without newlines."""
        if not text:
            return
        try:
            self.console.print(text, end="")
        except Exception:
            sys.stdout.write(text)
            sys.stdout.flush()

    def stream_done(self):
        """Finalize a streamed response with a newline."""
        self.console.print()

    def print_help(self):
        """Print help."""
        self.console.print(Rule(style="dim"))
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        commands = [
            ("/help", "Show this help"),
            ("/init", "Create JARVIS.md with project instructions"),
            ("/models", "Select model interactively"),
            ("/model <name>", "Switch to specific model"),
            ("/provider", "Switch provider"),
            ("/providers", "List providers"),
            ("/project", "Show project info"),
            ("/tools", "Show recent tool calls (expand details)"),
            ("/context", "Show context window usage"),
            ("/level [level]", "Set reasoning level (fast/balanced/deep/auto)"),
            ("/analyze <query>", "Multi-model AI analysis"),
            ("/clear", "Clear conversation history"),
            ("/cls", "Clear terminal screen"),
            ("/quit", "Exit"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print("[dim]Tip: Use /provider to change between local and cloud providers.[/dim]")
        self.console.print(Rule(style="dim"))

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
        """Print available providers with their status."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Model")

        for name, info in providers.items():
            marker = "[cyan]●[/cyan]" if name == current else " "
            status = "[green]✓[/green]" if info.get("configured") else "[red]✗[/red]"
            model = info.get("model") or "-"
            table.add_row(f"{marker} {name}", status, model)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def confirm(self, message: str) -> bool:
        """Ask for confirmation."""
        if HAS_QUESTIONARY:
            return questionary.confirm(message).ask() or False
        self.console.print(f"{message} [dim](y/n)[/dim] ", end="")
        return input().lower().strip() in ('y', 'yes')


class _JarvisLexer(Lexer):
    """Syntax highlighter for Jarvis input - highlights commands and file mentions."""

    def lex_document(self, document):
        """Return a callable that returns tokens for a line."""
        def get_line_tokens(line_number):
            line = document.lines[line_number]
            tokens = []
            i = 0

            while i < len(line):
                # Check for commands at start of line
                if i == 0 and line.startswith("/"):
                    # Find end of command word
                    end = i + 1
                    while end < len(line) and not line[end].isspace():
                        end += 1
                    cmd = line[i:end]
                    # Check if it's a valid command (match prefix)
                    cmd_base = cmd.split()[0] if cmd else cmd
                    is_valid = any(cmd_base == c or c.startswith(cmd_base) for c in VALID_COMMANDS)
                    if is_valid:
                        tokens.append(("class:command", cmd))
                    else:
                        tokens.append(("", cmd))
                    i = end
                # Check for @file mentions
                elif line[i] == "@" and i + 1 < len(line) and not line[i + 1].isspace():
                    end = i + 1
                    while end < len(line) and not line[end].isspace():
                        end += 1
                    tokens.append(("class:file-mention", line[i:end]))
                    i = end
                else:
                    # Regular character
                    tokens.append(("", line[i]))
                    i += 1

            return tokens

        return get_line_tokens


class _JarvisCompleter(Completer):
    """Autocomplete for commands and @file mentions."""

    COMMANDS = [
        "/help",
        "/init",
        "/models",
        "/model",
        "/provider",
        "/providers",
        "/project",
        "/tools",
        "/context",
        "/level",
        "/analyze",
        "/clear",
        "/cls",
        "/reset",
        "/quit",
        "/exit",
        "/q",
    ]

    def __init__(self, ui: TerminalUI):
        self.ui = ui

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)  # Include special chars

        # Command completion - check if we're at start of input or after newline
        line_start = text.rfind("\n") + 1
        current_line = text[line_start:]

        if current_line.startswith("/"):
            # Get the partial command
            partial = current_line.split()[0] if current_line else "/"
            for cmd in self.COMMANDS:
                if cmd.startswith(partial):
                    yield Completion(
                        cmd,
                        start_position=-len(partial),
                        display_meta="command"
                    )
            return

        # @file completion
        at_index = text.rfind("@")
        if at_index != -1:
            # Make sure @ is not inside a word
            if at_index == 0 or text[at_index - 1].isspace():
                prefix = text[at_index + 1:]
                # Remove any trailing spaces from prefix
                if " " in prefix:
                    return

                root = self.ui.project_root
                if root and root.exists():
                    try:
                        now = time.time()
                        if now - self.ui._file_cache_ts > 10 or not self.ui._file_cache:
                            files = []
                            for path in root.rglob("*"):
                                if path.is_dir():
                                    continue
                                # Skip hidden files and common ignore patterns
                                rel = str(path.relative_to(root))
                                if not any(part.startswith(".") for part in rel.split("/")):
                                    if not any(skip in rel for skip in ["node_modules", "__pycache__", ".git", "venv"]):
                                        files.append(rel)
                            self.ui._file_cache = sorted(files)[:500]  # Limit to 500 files
                            self.ui._file_cache_ts = now

                        for rel in self.ui._file_cache:
                            if prefix == "" or rel.lower().startswith(prefix.lower()):
                                yield Completion(
                                    f"@{rel}",
                                    start_position=-len(prefix) - 1,
                                    display_meta="file"
                                )
                    except Exception:
                        return
