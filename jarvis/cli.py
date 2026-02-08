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
@click.option('--dev', is_flag=True, help='Launch web UI with hot reload')
@click.option('--voice', is_flag=True, help='Enable voice input mode')
@click.option('--port', default=7777, help='Port for web UI (default: 7777)')
@click.option('--daemon', is_flag=True, help='Run as background daemon')
@click.option('--fast', is_flag=True, help='Use fast reasoning (quick responses)')
@click.option('--deep', is_flag=True, help='Use deep reasoning (complex analysis)')
@click.option('--level', type=click.Choice(['fast', 'balanced', 'deep']), help='Set reasoning level')
@click.pass_context
def main(ctx, version, dev, voice, port, daemon, fast, deep, level):
    """
    Jarvis - Your local AI assistant.

    Run without arguments for interactive CLI mode.

    \b
    Examples:
        jarvis              # Interactive CLI
        jarvis --dev        # Web UI at localhost:7777
        jarvis --voice      # Voice input mode
        jarvis chat "hello" # Single query
    """
    if version:
        click.echo(f"Jarvis v{__version__}")
        return

    # Ensure data directory exists
    ensure_data_dir()

    # Determine reasoning level from flags
    reasoning_level = level
    if fast:
        reasoning_level = 'fast'
    elif deep:
        reasoning_level = 'deep'

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['reasoning_level'] = reasoning_level

    if dev:
        _launch_dev(port)
    elif voice:
        _launch_voice_mode()
    elif daemon:
        _launch_daemon()
    elif ctx.invoked_subcommand is None:
        # Default: interactive CLI
        _launch_cli(reasoning_level=reasoning_level)


@main.command()
@click.argument('message')
@click.option('--fast', is_flag=True, help='Use fast reasoning')
@click.option('--deep', is_flag=True, help='Use deep reasoning')
@click.option('--level', type=click.Choice(['fast', 'balanced', 'deep']), help='Set reasoning level')
@click.pass_context
def chat(ctx, message, fast, deep, level):
    """Send a single message and get a response.

    \b
    Examples:
        jarvis chat "hello"              # Default balanced
        jarvis chat "quick summary" --fast
        jarvis chat "explain in detail" --deep
    """
    from .assistant import Jarvis

    # Determine reasoning level (command options > parent context)
    reasoning_level = level
    if fast:
        reasoning_level = 'fast'
    elif deep:
        reasoning_level = 'deep'
    elif ctx.obj and ctx.obj.get('reasoning_level'):
        reasoning_level = ctx.obj['reasoning_level']

    jarvis = Jarvis()

    # Set reasoning level if specified
    if reasoning_level and hasattr(jarvis, 'agent') and hasattr(jarvis.agent, 'set_reasoning_level'):
        jarvis.agent.set_reasoning_level(reasoning_level)

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


@main.group()
def config():
    """Manage API credentials (stored in ~/.jarvis/credentials.json)."""
    pass


@config.command("set")
@click.argument("provider")
@click.argument("key", default="api_key")
@click.argument("value", required=False)
def config_set(provider, key, value):
    """Set an API credential.

    \b
    Examples:
        jarvis config set chutes api_key YOUR_KEY
        jarvis config set brave api_key YOUR_KEY
        jarvis config set goldapi api_key YOUR_KEY
    """
    from .auth.credentials import set_credential

    if not value:
        # Read from stdin or prompt
        value = click.prompt(f"Enter {provider} {key}", hide_input=True)

    set_credential(provider, key, value)
    click.echo(f"✓ Saved {provider}.{key}")


@config.command("get")
@click.argument("provider", required=False)
def config_get(provider):
    """Show configured credentials (masked).

    \b
    Examples:
        jarvis config get          # Show all
        jarvis config get chutes   # Show chutes only
    """
    from .auth.credentials import get_all_credentials, list_configured_providers

    if provider:
        creds = get_all_credentials()
        if provider in creds:
            click.echo(f"{provider}:")
            for k, v in creds[provider].items():
                click.echo(f"  {k}: {v}")
        else:
            click.echo(f"No credentials for {provider}")
    else:
        providers = list_configured_providers()
        if not providers:
            click.echo("No credentials configured.")
            click.echo("\nSet credentials with:")
            click.echo("  jarvis config set chutes api_key YOUR_KEY")
            return

        creds = get_all_credentials()
        click.echo("Configured credentials:")
        for p in providers:
            click.echo(f"  {p}:")
            for k, v in creds.get(p, {}).items():
                click.echo(f"    {k}: {v}")


@config.command("delete")
@click.argument("provider")
@click.argument("key", required=False)
def config_delete(provider, key):
    """Delete a credential.

    \b
    Examples:
        jarvis config delete chutes           # Delete all chutes creds
        jarvis config delete chutes api_key   # Delete specific key
    """
    from .auth.credentials import delete_credential

    delete_credential(provider, key)
    if key:
        click.echo(f"✓ Deleted {provider}.{key}")
    else:
        click.echo(f"✓ Deleted all {provider} credentials")


@config.command("migrate")
def config_migrate():
    """Migrate API keys from environment variables to credentials.json."""
    from .auth.credentials import migrate_from_env

    migrated = migrate_from_env()
    if migrated:
        click.echo("✓ Migrated credentials:")
        for item in migrated:
            click.echo(f"  {item}")
    else:
        click.echo("No environment variables to migrate.")

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


@main.command()
def facts():
    """Show stored facts about you."""
    from . import get_data_dir

    facts_path = get_data_dir() / "memory" / "facts.md"
    if facts_path.exists():
        click.echo(facts_path.read_text())
    else:
        click.echo("No facts stored yet.")
        click.echo(f"Facts will be saved to: {facts_path}")


# ============== User Management Commands ==============

@main.group()
def user():
    """Manage user accounts."""
    pass


@user.command("create")
@click.option("--email", "-e", prompt="Email", help="User email address")
@click.option("--password", "-p", prompt="Password", hide_input=True, confirmation_prompt=True, help="User password")
@click.option("--name", "-n", prompt="Name", default="", help="Display name (optional)")
def user_create(email, password, name):
    """Create a new user account.

    \b
    Examples:
        jarvis user create
        jarvis user create -e user@example.com -p secret -n "John"
    """
    from .auth.db import init_auth_tables, create_user, get_user_by_email
    from .auth.security import hash_password

    init_auth_tables()

    # Check if user already exists
    existing = get_user_by_email(email)
    if existing:
        click.echo(f"Error: User with email '{email}' already exists.")
        sys.exit(1)

    # Basic email validation
    if "@" not in email or "." not in email.split("@")[-1]:
        click.echo("Error: Invalid email address.")
        sys.exit(1)

    # Create user with email_verified=True
    password_hash = hash_password(password)
    user_data = create_user(
        email=email,
        password_hash=password_hash,
        name=name or None,
        auth_provider="email",
        email_verified=True,
    )

    click.echo(f"\nUser created successfully.")
    click.echo(f"  Email:    {user_data['email']}")
    click.echo(f"  Name:     {user_data.get('name') or '(none)'}")
    click.echo(f"  ID:       {user_data['id']}")
    click.echo(f"  Verified: Yes")


@user.command("list")
def user_list():
    """List all user accounts."""
    from .auth.db import init_auth_tables
    from .auth.models import SessionLocal, User

    init_auth_tables()

    with SessionLocal() as db:
        users = db.query(User).order_by(User.created_at).all()
        if not users:
            click.echo("No users found.")
            return

        click.echo(f"{'Email':<35} {'Name':<20} {'Verified':<10} {'Created'}")
        click.echo("-" * 90)
        for u in users:
            verified = "Yes" if u.email_verified else "No"
            created = u.created_at.strftime("%Y-%m-%d") if u.created_at else "?"
            click.echo(f"{u.email:<35} {(u.name or ''):<20} {verified:<10} {created}")


@user.command("delete")
@click.argument("email")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def user_delete(email, yes):
    """Delete a user account by email.

    \b
    Example:
        jarvis user delete user@example.com
    """
    from .auth.db import init_auth_tables, get_user_by_email
    from .auth.models import SessionLocal, User

    init_auth_tables()

    user_data = get_user_by_email(email)
    if not user_data:
        click.echo(f"Error: No user found with email '{email}'.")
        sys.exit(1)

    if not yes:
        if not click.confirm(f"Delete user '{email}'? This cannot be undone."):
            click.echo("Cancelled.")
            return

    with SessionLocal() as db:
        db.query(User).filter(User.email == email).delete()
        db.commit()

    click.echo(f"User '{email}' deleted.")


@user.command("passwd")
@click.argument("email")
@click.option("--password", "-p", prompt="New password", hide_input=True, confirmation_prompt=True, help="New password")
def user_passwd(email, password):
    """Reset a user's password.

    \b
    Examples:
        jarvis user passwd user@example.com
        jarvis user passwd user@example.com -p newpass
    """
    from .auth.db import init_auth_tables, get_user_by_email
    from .auth.security import hash_password
    from .auth.models import SessionLocal, User

    init_auth_tables()

    existing = get_user_by_email(email)
    if not existing:
        click.echo(f"Error: No user found with email '{email}'.")
        sys.exit(1)

    hashed = hash_password(password)
    with SessionLocal() as db:
        db.query(User).filter(User.email == email).update({"password_hash": hashed})
        db.commit()

    click.echo(f"Password updated for '{email}'.")


@user.command("rename")
@click.argument("email")
@click.option("--name", "-n", prompt="New name", help="New display name")
def user_rename(email, name):
    """Change a user's display name.

    \b
    Examples:
        jarvis user rename user@example.com
        jarvis user rename user@example.com -n "New Name"
    """
    from .auth.db import init_auth_tables, get_user_by_email
    from .auth.models import SessionLocal, User

    init_auth_tables()

    existing = get_user_by_email(email)
    if not existing:
        click.echo(f"Error: No user found with email '{email}'.")
        sys.exit(1)

    with SessionLocal() as db:
        db.query(User).filter(User.email == email).update({"name": name})
        db.commit()

    click.echo(f"Name updated to '{name}' for '{email}'.")


@user.command("email")
@click.argument("old_email")
@click.option("--new-email", "-e", prompt="New email", help="New email address")
def user_email(old_email, new_email):
    """Change a user's email address.

    \b
    Examples:
        jarvis user email old@example.com
        jarvis user email old@example.com -e new@example.com
    """
    from .auth.db import init_auth_tables, get_user_by_email
    from .auth.models import SessionLocal, User

    init_auth_tables()

    existing = get_user_by_email(old_email)
    if not existing:
        click.echo(f"Error: No user found with email '{old_email}'.")
        sys.exit(1)

    # Check new email isn't taken
    conflict = get_user_by_email(new_email)
    if conflict:
        click.echo(f"Error: Email '{new_email}' is already in use.")
        sys.exit(1)

    if "@" not in new_email or "." not in new_email.split("@")[-1]:
        click.echo("Error: Invalid email address.")
        sys.exit(1)

    with SessionLocal() as db:
        db.query(User).filter(User.email == old_email).update({"email": new_email})
        db.commit()

    click.echo(f"Email changed from '{old_email}' to '{new_email}'.")


def _launch_cli(reasoning_level: str = None):
    """Launch interactive CLI mode."""
    from .assistant import run_cli
    run_cli(reasoning_level=reasoning_level)


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

    # Kill any existing processes on the ports first
    subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True)
    subprocess.run("lsof -ti:3000 | xargs kill -9 2>/dev/null", shell=True)

    click.echo("🚀 Starting Jarvis Dev Environment")
    click.echo(f"   Backend:  http://localhost:{port}")
    click.echo(f"   Frontend: http://localhost:3000")
    click.echo("   Press Ctrl+C to stop\n")

    processes = []

    def cleanup():
        """Kill all processes by port - most reliable on macOS."""
        # Kill by port
        subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True)
        subprocess.run("lsof -ti:3000 | xargs kill -9 2>/dev/null", shell=True)
        # Also terminate our tracked processes
        for p in processes:
            if p.poll() is None:
                p.kill()

    try:
        # Start backend
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "jarvis.ui:create_app",
             "--host", "0.0.0.0", "--port", str(port),
             "--reload", "--reload-dir", "jarvis", "--factory"],
            cwd=Path(__file__).parent.parent
        )
        processes.append(backend)

        # Start frontend
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=web_dir
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
        cleanup()
        click.echo("Done.")


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


@knowledge.command("sync")
@click.option("--projects", is_flag=True, help="Sync project README files from Developer folder")
@click.option("--personal", is_flag=True, help="Sync personal documents from ~/.jarvis/knowledge/personal/")
@click.option("--docs", is_flag=True, help="Sync Jarvis docs and shared documents")
def knowledge_sync(projects, personal, docs):
    """Sync knowledge base with configured sources.

    \b
    Examples:
        jarvis knowledge sync              # Sync docs (default)
        jarvis knowledge sync --personal   # Sync only personal knowledge
        jarvis knowledge sync --projects   # Sync only project READMEs
        jarvis knowledge sync --docs --personal  # Sync both
    """
    from .knowledge import get_rag_engine
    from pathlib import Path

    rag = get_rag_engine()

    # If no flags specified, default to docs
    if not projects and not personal and not docs:
        docs = True

    # Sync Jarvis documentation (architecture, AI concepts)
    if docs:
        docs_dir = Path(__file__).parent.parent / "docs"
        if docs_dir.exists():
            click.echo(f"Syncing Jarvis docs from {docs_dir}...")
            results = rag.add_directory(str(docs_dir))
            success = sum(1 for r in results.values() if r["status"] == "success")
            click.echo(f"  Added {success} documentation files")

        # Sync knowledge/documents folder (shared, in git)
        shared_docs = Path(__file__).parent.parent / "knowledge" / "documents"
        if shared_docs.exists():
            click.echo(f"Syncing shared documents from {shared_docs}...")
            results = rag.add_directory(str(shared_docs))
            success = sum(1 for r in results.values() if r["status"] == "success")
            click.echo(f"  Added {success} shared documents")

    # Sync personal documents (outside git, in ~/.jarvis/)
    if personal:
        personal_dir = Path.home() / ".jarvis" / "knowledge" / "personal"
        if personal_dir.exists():
            click.echo(f"Syncing personal knowledge from {personal_dir}...")
            results = rag.add_directory(str(personal_dir))
            success = sum(1 for r in results.values() if r["status"] == "success")
            click.echo(f"  Added {success} personal documents")
        else:
            click.echo(f"Personal knowledge directory not found: {personal_dir}")
            click.echo("  Create it and add .txt/.md/.pdf files to sync personal knowledge")

    # Sync project READMEs
    if projects:
        dev_dir = Path.home() / "Developer"
        if dev_dir.exists():
            click.echo(f"Scanning projects in {dev_dir}...")
            readme_count = 0
            for project_dir in dev_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                readme = project_dir / "README.md"
                if readme.exists():
                    try:
                        content = readme.read_text()
                        if len(content) > 100:  # Skip tiny READMEs
                            rag.add_document(
                                content,
                                f"project:{project_dir.name}",
                                {"type": "project", "path": str(project_dir)}
                            )
                            readme_count += 1
                    except Exception:
                        pass
            click.echo(f"  Added {readme_count} project READMEs")

    click.echo(f"\nTotal: {rag.count()} chunks in knowledge base")


# ============== Integration Commands ==============

@main.group()
def telegram():
    """Telegram bot integration commands."""
    pass


@telegram.command("run")
def telegram_run():
    """Run the Telegram bot.

    \b
    Prerequisites:
        1. Create a bot via @BotFather on Telegram
        2. Set TELEGRAM_BOT_TOKEN in .env or via:
           jarvis config set telegram bot_token YOUR_TOKEN

    \b
    Optional security:
        Set TELEGRAM_ALLOWED_USERS to comma-separated user IDs
        to restrict who can use the bot.

    \b
    Example:
        jarvis telegram run
    """
    import asyncio
    from .integrations.telegram_bot import TelegramBot
    from .assistant import Jarvis

    # Check if token is configured
    import os
    from .auth.credentials import get_credential

    token = get_credential("telegram", "bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        click.echo("❌ Telegram bot token not configured.")
        click.echo("\nSet it with:")
        click.echo("  jarvis config set telegram bot_token YOUR_BOT_TOKEN")
        click.echo("\nOr add to .env:")
        click.echo("  TELEGRAM_BOT_TOKEN=your_token_here")
        return

    # Create message handler that uses Jarvis
    async def handle_message(user_id: str, username: str, text: str) -> str:
        """Process message through Jarvis."""
        import asyncio
        jarvis = Jarvis()
        # process() is sync, run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, jarvis.process, text)
        return response or "I couldn't generate a response."

    async def run():
        bot = TelegramBot(token=token, message_handler=handle_message)
        if not await bot.start():
            click.echo("❌ Failed to start Telegram bot")
            return

        click.echo("✅ Telegram bot is running!")
        click.echo("   Press Ctrl+C to stop\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            click.echo("\n\nStopping...")
        finally:
            await bot.stop()

    asyncio.run(run())


@telegram.command("status")
def telegram_status():
    """Check Telegram bot configuration status."""
    import os
    from .auth.credentials import get_credential

    token = get_credential("telegram", "bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    allowed = get_credential("telegram", "allowed_users") or os.getenv("TELEGRAM_ALLOWED_USERS")
    chat_id = get_credential("telegram", "chat_id")

    if token:
        masked = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else "***"
        click.echo(f"✅ Bot token: {masked}")
    else:
        click.echo("❌ Bot token: Not configured")

    if chat_id:
        click.echo(f"✅ Default chat ID: {chat_id}")
    else:
        click.echo("⚠️  Default chat ID: Not set")

    if allowed:
        click.echo(f"✅ Allowed users: {allowed}")
    else:
        click.echo("⚠️  Allowed users: All (not restricted)")

    click.echo("\nRun 'jarvis telegram setup' for interactive configuration")


@telegram.command("setup")
def telegram_setup():
    """Interactive setup wizard for Telegram bot.

    This will guide you through:
    1. Creating a bot via @BotFather
    2. Setting up the bot token
    3. Getting your chat ID
    4. Configuring allowed users
    """
    import httpx
    from .auth.credentials import set_credential, get_credential

    click.echo("=" * 50)
    click.echo("🤖 Telegram Bot Setup Wizard")
    click.echo("=" * 50)
    click.echo()

    # Step 1: Bot Token
    click.echo("STEP 1: Bot Token")
    click.echo("-" * 30)

    existing_token = get_credential("telegram", "bot_token")
    if existing_token:
        masked = f"{existing_token[:8]}...{existing_token[-4:]}"
        click.echo(f"Current token: {masked}")
        if not click.confirm("Do you want to change it?", default=False):
            token = existing_token
        else:
            token = None
    else:
        token = None

    if not token:
        click.echo()
        click.echo("To get a bot token:")
        click.echo("  1. Open Telegram and search for @BotFather")
        click.echo("  2. Send /newbot")
        click.echo("  3. Follow the prompts to name your bot")
        click.echo("  4. Copy the token (looks like: 123456789:ABCdef...)")
        click.echo()
        token = click.prompt("Paste your bot token", hide_input=True)

    # Test the token
    click.echo("\nTesting token...", nl=False)
    try:
        import httpx
        resp = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                bot_info = data.get("result", {})
                click.echo(f" ✅ Valid!")
                click.echo(f"   Bot name: {bot_info.get('first_name')}")
                click.echo(f"   Username: @{bot_info.get('username')}")
                set_credential("telegram", "bot_token", token)
            else:
                click.echo(" ❌ Invalid token")
                return
        else:
            click.echo(f" ❌ Error: {resp.status_code}")
            return
    except Exception as e:
        click.echo(f" ❌ Error: {e}")
        return

    # Step 2: Get Chat ID
    click.echo()
    click.echo("STEP 2: Get Your Chat ID")
    click.echo("-" * 30)
    click.echo()
    click.echo(f"Now message your bot (@{bot_info.get('username')}) on Telegram.")
    click.echo("Send any message (like 'hello') to register your chat.")
    click.echo()

    if click.confirm("Have you sent a message to your bot?", default=True):
        click.echo("\nFetching your chat ID...", nl=False)
        try:
            resp = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                updates = data.get("result", [])
                if updates:
                    # Get the most recent message
                    for update in reversed(updates):
                        msg = update.get("message", {})
                        if msg:
                            chat_id = str(msg.get("chat", {}).get("id"))
                            user = msg.get("from", {})
                            username = user.get("username") or user.get("first_name")
                            click.echo(f" ✅ Found!")
                            click.echo(f"   Chat ID: {chat_id}")
                            click.echo(f"   User: {username}")

                            if click.confirm(f"\nSave {chat_id} as your default chat ID?", default=True):
                                set_credential("telegram", "chat_id", chat_id)
                                click.echo("   ✅ Saved!")

                            if click.confirm(f"Restrict bot to only your user ID ({chat_id})?", default=True):
                                set_credential("telegram", "allowed_users", chat_id)
                                click.echo("   ✅ Access restricted to you only!")
                            break
                else:
                    click.echo(" ⚠️  No messages found")
                    click.echo("   Make sure you sent a message to the bot, then try again.")
            else:
                click.echo(f" ❌ Error: {resp.status_code}")
        except Exception as e:
            click.echo(f" ❌ Error: {e}")
    else:
        click.echo("\nYou can get your chat ID later by:")
        click.echo("  1. Message the bot")
        click.echo("  2. Run: jarvis telegram setup")

    # Step 3: Done
    click.echo()
    click.echo("=" * 50)
    click.echo("✅ Setup Complete!")
    click.echo("=" * 50)
    click.echo()
    click.echo("To start the bot:")
    click.echo("  jarvis telegram run")
    click.echo()
    click.echo("To check status:")
    click.echo("  jarvis telegram status")


@telegram.command("set-chat")
@click.argument("chat_id")
def telegram_set_chat(chat_id):
    """Set the default chat ID for sending messages.

    \b
    Example:
        jarvis telegram set-chat 123456789
    """
    from .auth.credentials import set_credential
    set_credential("telegram", "chat_id", chat_id)
    click.echo(f"✅ Default chat ID set to: {chat_id}")


@telegram.command("webhook")
@click.argument("url", required=False)
@click.option("--remove", is_flag=True, help="Remove webhook (use polling instead)")
def telegram_webhook(url, remove):
    """Set up webhook for seamless Telegram integration.

    When webhook is active, messages are received automatically
    through your web server - no separate bot process needed!

    \b
    Examples:
        jarvis telegram webhook https://your-domain.com
        jarvis telegram webhook --remove
    """
    import httpx
    from .auth.credentials import get_credential, set_credential, delete_credential

    token = get_credential("telegram", "bot_token")
    if not token:
        click.echo("❌ Bot token not configured. Run: jarvis telegram setup")
        return

    if remove:
        click.echo("Removing webhook...", nl=False)
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{token}/deleteWebhook",
                timeout=10
            )
            if resp.json().get("ok"):
                delete_credential("telegram", "webhook_url")
                click.echo(" ✅ Removed")
                click.echo("   Now use: jarvis telegram run (polling mode)")
            else:
                click.echo(f" ❌ {resp.json().get('description')}")
        except Exception as e:
            click.echo(f" ❌ {e}")
        return

    if not url:
        # Show current status
        current = get_credential("telegram", "webhook_url")
        if current:
            click.echo(f"✅ Webhook active: {current}")
        else:
            click.echo("⚠️  No webhook configured")
            click.echo("\nSet up with:")
            click.echo("  jarvis telegram webhook https://your-domain.com")
            click.echo("\nFor local testing with ngrok:")
            click.echo("  ngrok http 7777")
            click.echo("  jarvis telegram webhook https://abc123.ngrok.io")
        return

    # Set up webhook
    webhook_url = f"{url.rstrip('/')}/api/telegram/webhook"
    click.echo(f"Setting webhook to: {webhook_url}...", nl=False)

    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/setWebhook",
            json={"url": webhook_url},
            timeout=10
        )
        result = resp.json()

        if result.get("ok"):
            set_credential("telegram", "webhook_url", webhook_url)
            click.echo(" ✅ Success!")
            click.echo("\n🎉 Telegram is now integrated!")
            click.echo("   Messages will be handled automatically by your web server.")
            click.echo("   Just run: jarvis --dev")
        else:
            click.echo(f" ❌ {result.get('description')}")
    except Exception as e:
        click.echo(f" ❌ {e}")


@telegram.command("start")
def telegram_start_daemon():
    """Start Telegram bot as a background daemon.

    The bot will run in the background and auto-restart if it crashes.
    Logs are written to ~/.jarvis/logs/telegram.log
    """
    import subprocess
    import os

    # Create log directory
    log_dir = Path.home() / ".jarvis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "telegram.log"
    pid_file = log_dir / "telegram.pid"

    # Check if already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            click.echo(f"⚠️  Telegram bot already running (PID: {pid})")
            click.echo(f"   Stop with: jarvis telegram stop")
            return
        except (ProcessLookupError, ValueError):
            pid_file.unlink()  # Clean up stale pid file

    # Start in background
    jarvis_path = Path(__file__).parent.parent
    venv_python = sys.executable

    process = subprocess.Popen(
        [venv_python, "-m", "jarvis.cli", "telegram", "run"],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        cwd=jarvis_path,
        start_new_session=True,
    )

    # Save PID
    pid_file.write_text(str(process.pid))

    click.echo(f"✅ Telegram bot started (PID: {process.pid})")
    click.echo(f"   Logs: {log_file}")
    click.echo(f"   Stop: jarvis telegram stop")


@telegram.command("stop")
def telegram_stop_daemon():
    """Stop the Telegram bot daemon."""
    import os
    import signal

    pid_file = Path.home() / ".jarvis" / "logs" / "telegram.pid"

    if not pid_file.exists():
        click.echo("⚠️  No running Telegram bot found")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
        click.echo(f"✅ Telegram bot stopped (PID: {pid})")
    except ProcessLookupError:
        pid_file.unlink()
        click.echo("⚠️  Bot was not running (cleaned up stale PID)")
    except Exception as e:
        click.echo(f"❌ Error stopping bot: {e}")


@telegram.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def telegram_logs(follow, lines):
    """View Telegram bot logs."""
    import subprocess

    log_file = Path.home() / ".jarvis" / "logs" / "telegram.log"

    if not log_file.exists():
        click.echo("No logs found. Start the bot first: jarvis telegram start")
        return

    if follow:
        subprocess.run(["tail", "-f", str(log_file)])
    else:
        subprocess.run(["tail", "-n", str(lines), str(log_file)])


@telegram.command("send")
@click.argument("message")
@click.option("--chat", "-c", help="Chat ID (uses default if not specified)")
def telegram_send(message, chat):
    """Send a message via Telegram.

    \b
    Examples:
        jarvis telegram send "Hello from Jarvis!"
        jarvis telegram send "Alert!" --chat 123456789
    """
    import httpx
    from .auth.credentials import get_credential

    token = get_credential("telegram", "bot_token")
    if not token:
        click.echo("❌ Bot token not configured. Run: jarvis telegram setup")
        return

    chat_id = chat or get_credential("telegram", "chat_id")
    if not chat_id:
        click.echo("❌ No chat ID specified and no default set.")
        click.echo("   Use --chat or run: jarvis telegram set-chat YOUR_CHAT_ID")
        return

    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            click.echo(f"✅ Message sent to chat {chat_id}")
        else:
            error = resp.json().get("description", "Unknown error")
            click.echo(f"❌ Failed: {error}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
