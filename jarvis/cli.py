#!/usr/bin/env python3
"""
Jarvis CLI - Main entry point for the jarvis command
"""

import click
import sys
from pathlib import Path

from . import __version__, ensure_data_dir


@click.group(invoke_without_command=True)
@click.option('--version', '-v', is_flag=True, help='Show version')
@click.option('--ui', is_flag=True, help='Launch web UI')
@click.option('--voice', is_flag=True, help='Enable voice input mode')
@click.option('--port', default=7777, help='Port for web UI (default: 7777)')
@click.option('--daemon', is_flag=True, help='Run as background daemon')
@click.pass_context
def main(ctx, version, ui, voice, port, daemon):
    """
    Jarvis - Your local AI assistant.

    Run without arguments for interactive CLI mode.

    \b
    Examples:
        jarvis              # Interactive CLI
        jarvis --ui         # Web UI at localhost:7777
        jarvis --voice      # Voice input mode
        jarvis chat "hello" # Single query
    """
    if version:
        click.echo(f"Jarvis v{__version__}")
        return

    # Ensure data directory exists
    ensure_data_dir()

    if ui:
        _launch_ui(port)
    elif voice:
        _launch_voice_mode()
    elif daemon:
        _launch_daemon()
    elif ctx.invoked_subcommand is None:
        # Default: interactive CLI
        _launch_cli()


@main.command()
@click.argument('message')
def chat(message):
    """Send a single message and get a response."""
    from .assistant import Jarvis

    jarvis = Jarvis()
    response = jarvis.process(message)
    # Response is already printed by the assistant


@main.command()
def setup():
    """Interactive setup wizard."""
    click.echo("Jarvis Setup Wizard")
    click.echo("=" * 40)

    data_dir = ensure_data_dir()
    click.echo(f"\nData directory: {data_dir}")

    # Check Ollama
    click.echo("\nChecking Ollama...")
    try:
        import ollama
        models = ollama.list()
        click.echo(f"  ✓ Ollama running, {len(models.get('models', []))} models installed")
    except Exception as e:
        click.echo(f"  ✗ Ollama not available: {e}")
        click.echo("    Install from: https://ollama.ai")
        return

    # Recommend models
    click.echo("\nRecommended models:")
    recommended = [
        ("qwen3:4b", "General chat"),
        ("llava", "Image understanding"),
        ("functiongemma", "Tool routing"),
    ]

    for model, desc in recommended:
        click.echo(f"  ollama pull {model}  # {desc}")

    click.echo("\n✓ Setup complete! Run 'jarvis' to start.")


@main.command()
def models():
    """List available Ollama models."""
    try:
        import ollama
        result = ollama.list()

        if not result.get('models'):
            click.echo("No models installed.")
            click.echo("Install with: ollama pull <model>")
            return

        click.echo("Available models:")
        for model in result['models']:
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024**3)  # Convert to GB
            click.echo(f"  {name:<30} {size:.1f} GB")

    except Exception as e:
        click.echo(f"Error: {e}")
        click.echo("Make sure Ollama is running: ollama serve")


@main.command()
@click.argument('name')
def persona(name):
    """Switch to a different persona."""
    from .assistant import Jarvis, list_personas

    available = list_personas()
    if name not in available:
        click.echo(f"Unknown persona: {name}")
        click.echo(f"Available: {', '.join(available)}")
        return

    click.echo(f"Persona set to: {name}")
    click.echo("This will take effect in the next session.")


@main.command()
def personas():
    """List available personas."""
    from .assistant import list_personas

    click.echo("Available personas:")
    for p in list_personas():
        click.echo(f"  - {p}")


def _launch_cli():
    """Launch interactive CLI mode."""
    from .assistant import run_cli
    run_cli()


def _launch_ui(port: int):
    """Launch web UI."""
    try:
        from .ui import create_app
        import uvicorn

        click.echo(f"Starting Jarvis UI at http://localhost:{port}")
        click.echo("Press Ctrl+C to stop")

        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    except ImportError:
        click.echo("UI dependencies not installed.")
        click.echo("Install with: pip install jarvis-ai-assistant[ui]")
        sys.exit(1)


def _launch_voice_mode():
    """Launch voice input mode."""
    try:
        from .voice import run_voice_mode
        run_voice_mode()
    except ImportError:
        click.echo("Voice dependencies not installed.")
        click.echo("Install with: pip install jarvis-ai-assistant[voice]")
        sys.exit(1)


def _launch_daemon():
    """Launch background daemon."""
    click.echo("Daemon mode not yet implemented.")
    click.echo("Coming soon: Telegram bot, scheduled tasks, etc.")
    sys.exit(1)


if __name__ == "__main__":
    main()
