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
@click.option('--ui', is_flag=True, help='Launch web UI (production build)')
@click.option('--dev', is_flag=True, help='Launch full dev environment (backend + frontend)')
@click.option('--voice', is_flag=True, help='Enable voice input mode')
@click.option('--port', default=7777, help='Port for web UI (default: 7777)')
@click.option('--reload', is_flag=True, help='Enable hot reload for development')
@click.option('--daemon', is_flag=True, help='Run as background daemon')
@click.pass_context
def main(ctx, version, ui, dev, voice, port, reload, daemon):
    """
    Jarvis - Your local AI assistant.

    Run without arguments for interactive CLI mode.

    \b
    Examples:
        jarvis              # Interactive CLI
        jarvis --dev        # Full dev mode (backend + frontend)
        jarvis --ui         # Web UI at localhost:7777
        jarvis --voice      # Voice input mode
        jarvis chat "hello" # Single query
    """
    if version:
        click.echo(f"Jarvis v{__version__}")
        return

    # Ensure data directory exists
    ensure_data_dir()

    if dev:
        _launch_dev(port)
    elif ui:
        _launch_ui(port, reload=reload)
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

        # Handle both dict and object response formats
        models_list = []
        if hasattr(result, 'models'):
            models_list = result.models
        elif isinstance(result, dict) and 'models' in result:
            models_list = result['models']

        if not models_list:
            click.echo("No models installed.")
            click.echo("Install with: ollama pull <model>")
            return

        click.echo("Available models:")
        for model in models_list:
            # Handle both dict and object model formats
            if hasattr(model, 'model'):
                name = model.model
                size = getattr(model, 'size', 0)
            else:
                name = model.get('model', model.get('name', 'unknown'))
                size = model.get('size', 0)
            size_gb = size / (1024**3)  # Convert to GB
            click.echo(f"  {name:<30} {size_gb:.1f} GB")

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


def _launch_dev(port: int):
    """Launch full dev environment - backend + frontend together."""
    import subprocess
    import signal
    import os

    # Find web directory
    web_dir = Path(__file__).parent.parent / "web"
    if not web_dir.exists():
        click.echo("Error: web/ directory not found")
        sys.exit(1)

    click.echo("🚀 Starting Jarvis Dev Environment")
    click.echo(f"   Backend:  http://localhost:{port}")
    click.echo(f"   Frontend: http://localhost:3000")
    click.echo("   Press Ctrl+C to stop\n")

    processes = []

    try:
        # Start backend
        backend_env = os.environ.copy()
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "jarvis.ui:create_app",
             "--host", "0.0.0.0", "--port", str(port),
             "--reload", "--reload-dir", "jarvis", "--factory"],
            env=backend_env,
            cwd=Path(__file__).parent.parent
        )
        processes.append(backend)

        # Start frontend
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=web_dir,
            env=os.environ.copy()
        )
        processes.append(frontend)

        # Wait for either to exit
        while all(p.poll() is None for p in processes):
            try:
                processes[0].wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    except KeyboardInterrupt:
        click.echo("\n\nShutting down...")
    finally:
        # Clean up all processes
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        click.echo("Done.")


def _launch_ui(port: int, reload: bool = False):
    """Launch web UI."""
    try:
        import uvicorn

        click.echo(f"Starting Jarvis UI at http://localhost:{port}")
        if reload:
            click.echo("Hot reload enabled - changes will auto-refresh")
        click.echo("Press Ctrl+C to stop")

        if reload:
            # Use string path for reload support
            uvicorn.run(
                "jarvis.ui:create_app",
                host="0.0.0.0",
                port=port,
                reload=True,
                reload_dirs=["jarvis"],
                factory=True,
                log_level="info"
            )
        else:
            from .ui import create_app
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


# ============== Knowledge Base Commands ==============

@main.group()
def knowledge():
    """Manage the RAG knowledge base."""
    pass


@knowledge.command("add")
@click.argument("path")
@click.option("--recursive", "-r", is_flag=True, help="Recursively add directory contents")
def knowledge_add(path, recursive):
    """Add a file or directory to the knowledge base.

    \b
    Examples:
        jarvis knowledge add document.pdf
        jarvis knowledge add ./docs --recursive
    """
    from .knowledge import get_rag_engine
    from pathlib import Path

    rag = get_rag_engine()
    p = Path(path)

    if not p.exists():
        click.echo(f"Error: Path not found: {path}")
        sys.exit(1)

    if p.is_file():
        try:
            chunks = rag.add_file(str(p))
            click.echo(f"Added {p.name}: {chunks} chunks")
        except Exception as e:
            click.echo(f"Error: {e}")
            sys.exit(1)
    elif p.is_dir():
        if not recursive:
            click.echo("Use --recursive to add directory contents")
            sys.exit(1)
        results = rag.add_directory(str(p))
        success = sum(1 for r in results.values() if r["status"] == "success")
        click.echo(f"Added {success}/{len(results)} files")
        for file, result in results.items():
            status = "OK" if result["status"] == "success" else f"Error: {result.get('error', 'unknown')}"
            click.echo(f"  {Path(file).name}: {status}")


@knowledge.command("list")
def knowledge_list():
    """List all sources in the knowledge base."""
    from .knowledge import get_rag_engine

    rag = get_rag_engine()
    sources = rag.list_sources()

    if not sources:
        click.echo("Knowledge base is empty.")
        return

    total = rag.count()
    click.echo(f"Knowledge base: {total} chunks from {len(sources)} sources\n")

    for src in sources:
        click.echo(f"  {src['source']}: {src['chunks']} chunks")


@knowledge.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
def knowledge_search(query, limit):
    """Search the knowledge base.

    \b
    Examples:
        jarvis knowledge search "how to deploy"
        jarvis knowledge search "API authentication" -n 10
    """
    from .knowledge import get_rag_engine

    rag = get_rag_engine()
    results = rag.search(query, n_results=limit)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        click.echo(f"[{i}] {doc['source']} (distance: {doc['distance']:.4f})")
        # Show first 200 chars
        preview = doc["content"][:200].replace("\n", " ")
        click.echo(f"    {preview}...")
        click.echo()


@knowledge.command("remove")
@click.argument("source")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def knowledge_remove(source, yes):
    """Remove a source from the knowledge base.

    \b
    Examples:
        jarvis knowledge remove document.pdf
    """
    from .knowledge import get_rag_engine

    rag = get_rag_engine()

    if not yes:
        confirm = click.confirm(f"Remove all chunks from '{source}'?")
        if not confirm:
            click.echo("Cancelled.")
            return

    deleted = rag.delete_source(source)
    click.echo(f"Deleted {deleted} chunks from {source}")


@knowledge.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def knowledge_clear(yes):
    """Clear the entire knowledge base."""
    from .knowledge import get_rag_engine

    if not yes:
        confirm = click.confirm("Clear ALL documents from knowledge base?")
        if not confirm:
            click.echo("Cancelled.")
            return

    rag = get_rag_engine()
    deleted = rag.clear()
    click.echo(f"Cleared {deleted} chunks from knowledge base")


if __name__ == "__main__":
    main()
