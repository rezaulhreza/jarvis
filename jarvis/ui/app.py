"""
Jarvis Web UI - Modern interface with voice, model selection, and diff view
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import subprocess
import base64
import io
from pathlib import Path
import json
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# React build directory
REACT_BUILD_DIR = Path(__file__).parent.parent.parent / "web" / "dist"


class DummyConsole:
    """Mock console that captures output."""
    def __init__(self, messages_list):
        self._messages = messages_list

    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        self._messages.append(("console", text))


class WebUI:
    """Minimal UI adapter for web context."""
    def __init__(self):
        self._messages = []
        self.console = DummyConsole(self._messages)
        self.is_streaming = False
        self.stop_requested = False

    def print_tool(self, msg): self._messages.append(("tool", msg))
    def print_error(self, msg): self._messages.append(("error", msg))
    def print_warning(self, msg): self._messages.append(("warning", msg))
    def print_success(self, msg): self._messages.append(("success", msg))
    def print_info(self, msg): self._messages.append(("info", msg))
    def print_system(self, msg): self._messages.append(("system", msg))
    def show_spinner(self, msg="Thinking"): return DummySpinner()
    def setup_signal_handlers(self): pass
    def print_header(self, *args, **kwargs): pass
    def confirm(self, msg): return True  # Auto-confirm in web UI

    def get_messages(self):
        msgs = self._messages.copy()
        self._messages.clear()
        return msgs


class DummySpinner:
    def __enter__(self): return self
    def __exit__(self, *args): pass


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Jarvis", description="Personal AI Assistant")

    connections: dict[str, WebSocket] = {}
    instances: dict[str, any] = {}

    # Shared context manager for HTTP endpoints (chat history API)
    # Use consistent path with Jarvis instances
    from jarvis.core.context_manager import ContextManager
    from jarvis import get_data_dir
    db_path = get_data_dir() / "memory" / "jarvis.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shared_context = ContextManager(db_path=str(db_path))

    # Serve React build if available
    if REACT_BUILD_DIR.exists():
        app.mount("/assets", StaticFiles(directory=REACT_BUILD_DIR / "assets"), name="assets")

        # Serve root-level static files (jarvis.jpeg, etc.)
        @app.get("/jarvis.jpeg")
        async def serve_avatar():
            avatar_path = REACT_BUILD_DIR / "jarvis.jpeg"
            if avatar_path.exists():
                return FileResponse(avatar_path)
            return {"error": "Image not found"}, 404

    @app.get("/", response_class=HTMLResponse)
    async def index():
        # Serve React build if it exists
        react_index = REACT_BUILD_DIR / "index.html"
        if react_index.exists():
            return FileResponse(react_index)
        # Fallback to inline HTML
        return get_html_template()

    @app.get("/api/personas")
    async def get_personas():
        """Get available personas."""
        # For now, just return default. Can be expanded later.
        return {"personas": ["default"]}

    @app.get("/api/voices")
    async def get_voices():
        """Get available TTS voices."""
        return {
            "voices": [
                {"id": "en-GB-SoniaNeural", "name": "Sonia (British Female)", "lang": "en-GB"},
                {"id": "en-GB-RyanNeural", "name": "Ryan (British Male)", "lang": "en-GB"},
                {"id": "en-US-JennyNeural", "name": "Jenny (US Female)", "lang": "en-US"},
                {"id": "en-US-ChristopherNeural", "name": "Christopher (US Male)", "lang": "en-US"},
                {"id": "en-US-GuyNeural", "name": "Guy (US Male)", "lang": "en-US"},
                {"id": "en-US-AriaNeural", "name": "Aria (US Female)", "lang": "en-US"},
                {"id": "en-AU-WilliamNeural", "name": "William (Australian Male)", "lang": "en-AU"},
                {"id": "en-AU-NatashaNeural", "name": "Natasha (Australian Female)", "lang": "en-AU"},
            ]
        }

    @app.get("/api/settings/voice")
    async def get_voice_settings():
        """Get current voice settings."""
        from jarvis.assistant import load_config
        config = load_config()
        voice_config = config.get("voice", {})
        return {
            "tts_provider": voice_config.get("tts_provider", "browser"),
            "tts_voice": voice_config.get("tts_voice", "en-GB-SoniaNeural"),
            "stt_provider": voice_config.get("stt_provider", "browser"),
        }

    @app.post("/api/settings/voice")
    async def set_voice(data: dict):
        """Update voice settings."""
        from jarvis.assistant import load_config, save_config
        config = load_config()
        if "voice" not in config:
            config["voice"] = {}

        if "voice" in data:
            config["voice"]["tts_voice"] = data["voice"]
        if "tts_provider" in data:
            config["voice"]["tts_provider"] = data["tts_provider"]
        if "stt_provider" in data:
            config["voice"]["stt_provider"] = data["stt_provider"]

        save_config(config)
        return {"success": True}

    @app.post("/api/settings/elevenlabs")
    async def set_elevenlabs(data: dict):
        """Configure ElevenLabs API key and voice."""
        from jarvis.assistant import load_config, save_config
        config = load_config()
        if "voice" not in config:
            config["voice"] = {}

        if "api_key" in data:
            config["voice"]["elevenlabs_key"] = data["api_key"]
        if "voice_id" in data:
            config["voice"]["elevenlabs_voice"] = data["voice_id"]
        if "provider" in data:
            config["voice"]["tts_provider"] = data["provider"]  # "elevenlabs", "edge", or "browser"

        save_config(config)
        return {"success": True}

    def get_elevenlabs_key():
        """Get ElevenLabs API key from .env or config."""
        import os
        # Check .env first (more secure)
        key = os.environ.get("ELEVEN_LABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
        if key:
            return key
        # Fall back to config
        from jarvis.assistant import load_config
        config = load_config()
        return config.get("voice", {}).get("elevenlabs_key")

    @app.get("/api/elevenlabs/status")
    async def elevenlabs_status():
        """Check if ElevenLabs API key is configured."""
        import os
        key = get_elevenlabs_key()
        source = None
        if os.environ.get("ELEVEN_LABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY"):
            source = ".env"
        elif key:
            source = "settings"
        return {
            "configured": bool(key),
            "source": source,
            "key_preview": f"{key[:8]}...{key[-4:]}" if key and len(key) > 12 else None
        }

    @app.get("/api/elevenlabs/voices")
    async def get_elevenlabs_voices():
        """Get available ElevenLabs voices."""
        api_key = get_elevenlabs_key()

        if not api_key:
            return {"voices": [], "error": "No API key configured"}

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": api_key}
                )
                if res.status_code == 200:
                    data = res.json()
                    voices = [{"id": v["voice_id"], "name": v["name"]} for v in data.get("voices", [])]
                    return {"voices": voices}
        except Exception as e:
            return {"voices": [], "error": str(e)}

        return {"voices": []}

    @app.post("/api/tts/elevenlabs")
    async def elevenlabs_tts(data: dict):
        """Stream TTS from ElevenLabs - very low latency."""
        text = data.get("text", "")
        if not text:
            return {"error": "No text"}

        # Strip emojis and markdown characters
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', ' ', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'`+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return {"error": "No text"}

        api_key = get_elevenlabs_key()
        from jarvis.assistant import load_config
        config = load_config()
        voice_id = config.get("voice", {}).get("elevenlabs_voice", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel

        if not api_key:
            return {"error": "No ElevenLabs API key configured. Add ELEVEN_LABS_API_KEY to .env"}

        try:
            import httpx

            async def stream_audio():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                        headers={
                            "xi-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "text": text[:1000],  # Limit for speed
                            "model_id": "eleven_turbo_v2_5",  # Fastest model
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75,
                            }
                        },
                        timeout=30.0
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                stream_audio(),
                media_type="audio/mpeg",
                headers={"Cache-Control": "no-cache"}
            )
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/models")
    async def get_models():
        """Get available chat models from Ollama (excludes embedding models)."""
        # Embedding models can't do chat - filter them out
        EMBEDDING_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm", "snowflake-arctic-embed"}
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            models = []
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip():
                    name = line.split()[0]
                    # Filter out embedding models
                    base_name = name.split(":")[0]  # Remove :latest tag
                    if base_name not in EMBEDDING_MODELS:
                        models.append(name)
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    # ============== Chat History API ==============

    @app.get("/api/chats")
    async def list_chats(search: str = None, limit: int = 50):
        """List all chats, optionally filtered by search."""
        chats = shared_context.list_chats(limit=limit, search=search)
        return {"chats": chats}

    @app.post("/api/chats")
    async def create_chat(title: str = "New Chat"):
        """Create a new chat. Cleans up empty chats first."""
        # Clean up empty chats (no messages) before creating new one
        existing_chats = shared_context.list_chats(limit=10)
        for chat in existing_chats:
            if chat.get("message_count", 0) == 0:
                shared_context.delete_chat(chat["id"])

        chat_id = shared_context.create_chat(title)
        return {"id": chat_id, "title": title}

    @app.get("/api/chats/{chat_id}")
    async def get_chat(chat_id: str):
        """Get a chat with its messages."""
        chat = shared_context.get_chat(chat_id)
        if not chat:
            return {"error": "Chat not found"}, 404
        return chat

    @app.patch("/api/chats/{chat_id}")
    async def update_chat(chat_id: str, title: str = None):
        """Update a chat's title."""
        success = shared_context.update_chat(chat_id, title=title)
        if not success:
            return {"error": "Chat not found"}, 404
        return {"success": True}

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        """Delete a chat."""
        success = shared_context.delete_chat(chat_id)
        if not success:
            return {"error": "Chat not found"}, 404
        return {"success": True}

    @app.post("/api/chats/{chat_id}/switch")
    async def switch_chat(chat_id: str):
        """Switch to a different chat."""
        success = shared_context.switch_chat(chat_id)
        if not success:
            return {"error": "Chat not found"}, 404
        return {"success": True, "messages": shared_context.messages}

    @app.post("/api/chats/{chat_id}/generate-title")
    async def generate_chat_title(chat_id: str):
        """Generate a title for a chat using AI."""
        snippet = shared_context.generate_chat_title(chat_id)
        if not snippet:
            return {"error": "No messages to generate title from"}

        # Use LLM to generate a short title
        try:
            from jarvis.assistant import load_config
            from jarvis.providers import get_provider

            config = load_config()
            provider = get_provider(config)

            prompt = f"Generate a short title (3-6 words, no quotes) for this conversation:\n\n{snippet}\n\nTitle:"
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are a helpful assistant. Generate only a short, descriptive title. No quotes, no punctuation at the end.",
                stream=False,
                options={"num_predict": 20}
            )

            # Get the response text
            title = ""
            for chunk in response:
                title += chunk
            title = title.strip().strip('"\'').strip()[:50]  # Clean up and limit length

            if title:
                shared_context.update_chat(chat_id, title=title)
                return {"title": title}
            else:
                return {"error": "Failed to generate title"}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/chats/current")
    async def get_current_chat():
        """Get the current chat ID."""
        return {"chat_id": shared_context.get_current_chat_id()}

    @app.post("/api/transcribe")
    async def transcribe_audio(audio: UploadFile = File(...)):
        """Transcribe audio using Whisper."""
        tmp_path = None
        try:
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                content = await audio.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Convert webm to wav using ffmpeg
            wav_path = tmp_path.replace(".webm", ".wav")
            conv_result = subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
                capture_output=True, timeout=10
            )

            if not Path(wav_path).exists():
                return {"transcript": "", "error": "ffmpeg conversion failed", "use_browser": True}

            # Try mlx-whisper (fast on Mac)
            try:
                result = subprocess.run(
                    ["mlx_whisper", wav_path, "--model", "mlx-community/whisper-base-mlx"],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0 and result.stdout.strip():
                    return {"transcript": result.stdout.strip()}
            except FileNotFoundError:
                pass

            # Try openai-whisper CLI
            try:
                result = subprocess.run(
                    ["whisper", wav_path, "--model", "base", "--output_format", "txt", "--output_dir", "/tmp"],
                    capture_output=True, text=True, timeout=60
                )
                txt_path = wav_path.replace(".wav", ".txt")
                if Path(txt_path).exists():
                    transcript = Path(txt_path).read_text().strip()
                    Path(txt_path).unlink(missing_ok=True)
                    return {"transcript": transcript}
            except FileNotFoundError:
                pass

            # Try Python whisper library directly
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(wav_path)
                return {"transcript": result["text"].strip()}
            except ImportError:
                pass

            # No whisper available - tell frontend to use browser
            return {"transcript": "", "error": "No whisper installed", "use_browser": True}

        except Exception as e:
            return {"transcript": "", "error": str(e), "use_browser": True}
        finally:
            # Cleanup temp files
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
                Path(tmp_path.replace(".webm", ".wav")).unlink(missing_ok=True)

    @app.post("/api/tts")
    async def text_to_speech(data: dict):
        """Convert text to speech - streams audio chunks as they're generated."""
        text = data.get("text", "")
        fast_mode = data.get("fast", False)  # Use browser TTS for speed

        if not text:
            return {"error": "No text provided"}

        # Strip emojis and markdown characters that TTS reads literally
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        # Strip markdown: *bold*, _italic_, #headers, ```code```, etc.
        text = re.sub(r'\*+', '', text)  # asterisks
        text = re.sub(r'_+', ' ', text)  # underscores
        text = re.sub(r'#+\s*', '', text)  # headers
        text = re.sub(r'`+', '', text)  # code ticks
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [link](url) -> link
        text = re.sub(r'\s+', ' ', text).strip()  # collapse whitespace

        if not text:
            return {"error": "No text"}

        # Fast mode - tell frontend to use browser TTS
        if fast_mode:
            return {"use_browser": True, "text": text}

        # Load voice settings
        from jarvis.assistant import load_config
        config = load_config()
        tts_voice = config.get("voice", {}).get("tts_voice", "en-GB-SoniaNeural")

        try:
            import edge_tts

            # Limit text length for speed
            short_text = text[:500]

            communicate = edge_tts.Communicate(short_text, tts_voice)

            # Collect all audio chunks
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return {"error": "No audio generated"}

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=speech.mp3",
                }
            )
        except Exception as e:
            print(f"[TTS] Edge TTS failed: {e}")

            # Option 2: macOS say command (fallback)
            import platform
            print(f"[TTS] Falling back to macOS say")
            if platform.system() == "Darwin":
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                    aiff_path = tmp.name
                mp3_path = aiff_path.replace(".aiff", ".mp3")

                result = subprocess.run(
                    ["say", "-v", "Samantha", "-o", aiff_path, text[:1000]],
                    capture_output=True, timeout=30
                )

                if Path(aiff_path).exists():
                    subprocess.run(
                        ["ffmpeg", "-i", aiff_path, "-y", "-q:a", "2", mp3_path],
                        capture_output=True, timeout=30
                    )
                    Path(aiff_path).unlink(missing_ok=True)

                    if Path(mp3_path).exists():
                        audio_data = Path(mp3_path).read_bytes()
                        Path(mp3_path).unlink()
                        return StreamingResponse(
                            io.BytesIO(audio_data),
                            media_type="audio/mpeg",
                            headers={"Content-Disposition": "inline; filename=speech.mp3"}
                        )

            return {"error": "No TTS available"}

        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/commands")
    async def get_commands():
        """Get available slash commands."""
        return {
            "commands": [
                {"cmd": "/help", "desc": "Show help"},
                {"cmd": "/models", "desc": "List models"},
                {"cmd": "/model", "desc": "Change model"},
                {"cmd": "/provider", "desc": "Change provider"},
                {"cmd": "/project", "desc": "Project info"},
                {"cmd": "/clear", "desc": "Clear chat"},
                {"cmd": "/init", "desc": "Init project config"},
            ]
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        session_id = str(id(websocket))
        connections[session_id] = websocket

        jarvis = None
        try:
            # Import here to avoid circular imports
            from jarvis.assistant import Jarvis, load_config
            from jarvis.providers import get_provider

            # Create WebUI adapter
            ui = WebUI()

            # Get working directory from query params or use home
            working_dir = Path.home()

            # Create Jarvis with web UI
            jarvis = Jarvis(ui=ui, working_dir=working_dir)
            instances[session_id] = jarvis

            await websocket.send_json({
                "type": "connected",
                "provider": jarvis.provider.name,
                "model": jarvis.provider.model,
                "project": jarvis.project.project_name,
                "projectRoot": str(jarvis.project.project_root),
            })

        except Exception as e:
            import traceback
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize: {e}\n{traceback.format_exc()}"
            })
            await websocket.close()
            return

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "message")

                if msg_type == "message":
                    user_input = data.get("content", "").strip()
                    chat_mode = data.get("chat_mode", False)
                    if user_input:
                        await process_message(websocket, jarvis, user_input, chat_mode)

                elif msg_type == "switch_model":
                    model = data.get("model")
                    if model and jarvis:
                        jarvis.switch_model(model)
                        await websocket.send_json({
                            "type": "model_changed",
                            "model": model
                        })

                elif msg_type == "switch_provider":
                    provider = data.get("provider")
                    if provider and jarvis:
                        jarvis.switch_provider(provider)
                        await websocket.send_json({
                            "type": "provider_changed",
                            "provider": provider,
                            "model": jarvis.provider.model
                        })

                elif msg_type == "clear":
                    if jarvis:
                        jarvis.context.clear()
                    await websocket.send_json({"type": "cleared"})

                elif msg_type == "set_working_dir":
                    # Reinitialize with new working directory
                    new_dir = Path(data.get("path", "")).expanduser()
                    if new_dir.exists() and new_dir.is_dir():
                        ui = WebUI()
                        jarvis = Jarvis(ui=ui, working_dir=new_dir)
                        instances[session_id] = jarvis
                        await websocket.send_json({
                            "type": "project_changed",
                            "project": jarvis.project.project_name,
                            "projectRoot": str(jarvis.project.project_root),
                        })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            connections.pop(session_id, None)
            instances.pop(session_id, None)

    async def process_message(websocket: WebSocket, jarvis, user_input: str, chat_mode: bool = False):
        """Process a message and stream the response."""
        await websocket.send_json({
            "type": "processing",
            "content": user_input
        })

        try:
            # Handle commands
            if user_input.startswith('/'):
                result = jarvis._handle_command(user_input)
                ui_msgs = jarvis.ui.get_messages()
                await websocket.send_json({
                    "type": "response",
                    "content": result or "\n".join(m[1] for m in ui_msgs) or "Done.",
                    "done": True
                })
                return

            # Add to context (creates chat if needed)
            jarvis.context.add_message_to_chat("user", user_input)

            # Build history
            from jarvis.providers import Message
            history = []
            for m in jarvis.context.get_messages()[:-1]:
                history.append(Message(role=m["role"], content=m["content"]))

            if chat_mode:
                # CHAT MODE: Ultra-fast, minimal overhead, NO THINKING

                # Load user facts for context
                user_facts = ""
                try:
                    from jarvis import get_data_dir
                    facts_path = get_data_dir() / "memory" / "facts.md"
                    if facts_path.exists():
                        user_facts = facts_path.read_text()[:1500]  # Limit size
                except Exception:
                    pass

                # Short system prompt - less tokens = faster
                chat_system = "You are Jarvis, a witty AI assistant. Be concise and direct. No thinking out loud. Never address the user by name."

                # Add user facts if available
                if user_facts:
                    chat_system += f"\n\nUser context:\n{user_facts}"

                # RAG: Retrieve relevant context from knowledge base
                rag_context = ""
                rag_info = {"enabled": False, "chunks": 0, "sources": []}
                try:
                    from jarvis.knowledge import get_rag_engine
                    rag = get_rag_engine(jarvis.config)
                    count = rag.count()
                    print(f"[RAG] Knowledge base has {count} chunks")
                    if count > 0:
                        rag_info["enabled"] = True
                        rag_info["total_chunks"] = count
                        # Get search results with source info
                        results = rag.search(user_input, n_results=5)
                        if results:
                            # Filter by relevance - only keep chunks with distance < 1.2
                            # Lower distance = more relevant
                            RELEVANCE_THRESHOLD = 1.2
                            relevant_results = [r for r in results if r.get("distance", 2.0) < RELEVANCE_THRESHOLD]

                            if relevant_results:
                                rag_info["chunks"] = len(relevant_results)
                                rag_info["sources"] = list(set(r.get("source", "unknown") for r in relevant_results))
                                # Build context from relevant results only
                                context_parts = []
                                for doc in relevant_results:
                                    source = doc.get("source", "unknown")
                                    content = doc.get("content", "").strip()
                                    distance = doc.get("distance", 0)
                                    context_parts.append(f"[From: {source}]\n{content}")
                                    print(f"[RAG] Including {source} (distance: {distance:.3f})")
                                rag_context = "Relevant knowledge:\n\n" + "\n\n---\n\n".join(context_parts)
                                chat_system += f"\n\n{rag_context}"
                            else:
                                # Results found but none relevant enough
                                print(f"[RAG] Found {len(results)} chunks but none relevant (all distance > {RELEVANCE_THRESHOLD})")
                                for r in results[:3]:
                                    print(f"[RAG]   - {r.get('source', '?')}: distance {r.get('distance', 0):.3f}")
                        else:
                            print("[RAG] No context found")
                except Exception as e:
                    import traceback
                    print(f"[RAG] Error retrieving context: {e}")
                    traceback.print_exc()
                    rag_info["error"] = str(e)

                # Send RAG status to frontend
                await websocket.send_json({
                    "type": "rag_status",
                    "rag": rag_info
                })

                # Minimal context - just last 2 exchanges for speed
                recent = history[-2:] if len(history) > 2 else history
                all_messages = recent + [Message(role="user", content=user_input)]

                # Use fast chat model
                chat_model = jarvis.config.get("models", {}).get("chat", "llama3.2")
                original_model = jarvis.provider.model
                jarvis.provider.model = chat_model

                # Stream with options for speed
                response_text = ""
                try:
                    stream = jarvis.provider.chat(
                        messages=all_messages,
                        system=chat_system,
                        stream=True,
                        options={"num_predict": 500}  # Reasonable response length
                    )
                except Exception as e:
                    jarvis.provider.model = original_model
                    raise e

                for chunk in stream:
                    response_text += chunk
                    await websocket.send_json({
                        "type": "stream",
                        "content": chunk,
                        "done": False
                    })

                # Restore original model
                jarvis.provider.model = original_model

                await websocket.send_json({
                    "type": "response",
                    "content": response_text,
                    "done": True
                })

                if response_text.strip():
                    jarvis.context.add_message_to_chat("assistant", response_text.strip())

            else:
                # AGENT MODE: Use tools for coding tasks
                response = jarvis.agent.run(user_input, jarvis.system_prompt, history)

                if response:
                    if "```diff" in response or "wrote:" in response.lower():
                        await websocket.send_json({
                            "type": "diff",
                            "content": response,
                            "done": False
                        })

                    await websocket.send_json({
                        "type": "response",
                        "content": response,
                        "done": True
                    })

                    clean = response.replace("[dim]", "").replace("[/dim]", "")
                    clean = clean.replace("[red]", "").replace("[/red]", "")
                    if clean.strip():
                        jarvis.context.add_message_to_chat("assistant", clean.strip())

        except Exception as e:
            import traceback
            await websocket.send_json({
                "type": "error",
                "message": f"{e}\n{traceback.format_exc()}"
            })

    return app



def get_html_template() -> str:
    """Fallback when React build is not available."""
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Jarvis - Build Required</title>
    <style>
        body { font-family: system-ui; background: #0a0a0f; color: #e4e4e7; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .container { text-align: center; padding: 2rem; }
        h1 { color: #3b82f6; }
        code { background: #1a1a24; padding: 0.5rem 1rem; border-radius: 0.5rem; display: block; margin: 1rem 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Jarvis</h1>
        <p>React build not found. Please build the frontend first:</p>
        <code>cd web && npm install && npm run build</code>
        <p>Or run in dev mode:</p>
        <code>jarvis --dev</code>
    </div>
</body>
</html>'''

def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
