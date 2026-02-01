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

    # Serve React build if available
    if REACT_BUILD_DIR.exists():
        app.mount("/assets", StaticFiles(directory=REACT_BUILD_DIR / "assets"), name="assets")

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

            # Add to context
            jarvis.context.add_message("user", user_input)

            # Build history
            from jarvis.providers import Message
            history = []
            for m in jarvis.context.get_messages()[:-1]:
                history.append(Message(role=m["role"], content=m["content"]))

            if chat_mode:
                # CHAT MODE: Ultra-fast, minimal overhead, NO THINKING
                user_nickname = jarvis.config.get("user", {}).get("nickname", "")

                # Short system prompt - less tokens = faster
                chat_system = f"You are Jarvis, a witty AI assistant. Be concise and direct. No thinking out loud."
                if user_nickname:
                    chat_system += f" Call user '{user_nickname}'."

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
                        results = rag.search(user_input, n_results=3)
                        if results:
                            rag_info["chunks"] = len(results)
                            rag_info["sources"] = list(set(r.get("source", "unknown") for r in results))
                            # Build context from results
                            context_parts = []
                            for doc in results:
                                source = doc.get("source", "unknown")
                                content = doc.get("content", "").strip()
                                context_parts.append(f"[From: {source}]\n{content}")
                            rag_context = "Relevant knowledge:\n\n" + "\n\n---\n\n".join(context_parts)
                            print(f"[RAG] Found {len(results)} chunks from: {rag_info['sources']}")
                            chat_system += f"\n\n{rag_context}"
                        else:
                            print("[RAG] No relevant context found")
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
                    jarvis.context.add_message("assistant", response_text.strip())

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
                        jarvis.context.add_message("assistant", clean.strip())

        except Exception as e:
            import traceback
            await websocket.send_json({
                "type": "error",
                "message": f"{e}\n{traceback.format_exc()}"
            })

    return app


def get_html_template() -> str:
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jarvis</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #79c0ff;
            --success: #3fb950;
            --error: #f85149;
            --warning: #d29922;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: var(--bg-secondary);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            gap: 16px;
            flex-wrap: wrap;
        }
        .logo { display: flex; align-items: center; gap: 10px; }
        .logo h1 { font-size: 1.25rem; color: var(--accent); }
        .header-controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        .status {
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 12px;
            background: var(--bg-tertiary);
        }
        .status.connected { color: var(--success); }
        .status.disconnected { color: var(--error); }
        select, button {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 0.875rem;
            cursor: pointer;
        }
        select:hover, button:hover { border-color: var(--accent); }
        button.primary {
            background: var(--accent);
            color: #000;
            border: none;
        }
        button.primary:hover { background: var(--accent-hover); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        main {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .messages { max-width: 900px; margin: 0 auto; }
        .message {
            margin-bottom: 16px;
            padding: 16px;
            border-radius: 8px;
            animation: fadeIn 0.2s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user {
            background: var(--bg-tertiary);
            margin-left: 15%;
            border: 1px solid var(--border);
        }
        .message.assistant {
            background: var(--bg-secondary);
            margin-right: 15%;
            border: 1px solid var(--border);
        }
        .message.system {
            background: transparent;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
            padding: 8px;
        }
        .message .role {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .message .content {
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .message pre {
            background: var(--bg-primary);
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 8px 0;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.875rem;
        }
        .message code {
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
        }
        /* Diff styling */
        .diff-add { color: var(--success); background: rgba(63, 185, 80, 0.15); }
        .diff-remove { color: var(--error); background: rgba(248, 81, 73, 0.15); }
        .diff-header { color: var(--accent); }
        footer {
            background: var(--bg-secondary);
            padding: 16px 20px;
            border-top: 1px solid var(--border);
        }
        .input-area { max-width: 900px; margin: 0 auto; }
        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        textarea {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 1rem;
            resize: none;
            min-height: 48px;
            max-height: 200px;
            font-family: inherit;
        }
        textarea:focus { outline: none; border-color: var(--accent); }
        .suggestions {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 4px;
            display: none;
            max-height: 200px;
            overflow-y: auto;
        }
        .suggestions.active { display: block; }
        .suggestion {
            padding: 10px 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
        }
        .suggestion:hover, .suggestion.selected {
            background: var(--bg-tertiary);
        }
        .suggestion .cmd { color: var(--accent); font-family: monospace; }
        .suggestion .desc { color: var(--text-secondary); font-size: 0.875rem; }
        .voice-btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }
        .voice-btn.recording {
            background: var(--error);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .typing {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            color: var(--text-secondary);
        }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            display: inline-block;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .project-badge {
            background: var(--bg-tertiary);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span style="font-size:1.5rem">🤖</span>
            <h1>Jarvis</h1>
            <span class="project-badge" id="project">Loading...</span>
        </div>
        <div class="header-controls">
            <select id="model" title="Select model">
                <option>Loading models...</option>
            </select>
            <button id="chatMode" title="Chat mode (no tools)" style="opacity:0.5">💬</button>
            <button id="tts" title="Voice output">🔇</button>
            <button id="clear">Clear</button>
            <span class="status disconnected" id="status">Connecting...</span>
        </div>
    </header>

    <main>
        <div class="messages" id="messages"></div>
    </main>

    <footer>
        <div class="input-area">
            <div class="input-container">
                <div class="input-wrapper">
                    <div class="suggestions" id="suggestions"></div>
                    <textarea id="input" placeholder="Type a message or / for commands..." rows="1"></textarea>
                </div>
                <button class="voice-btn" id="voice" title="Voice input">🎤</button>
                <button class="primary" id="send" disabled>Send</button>
            </div>
        </div>
    </footer>

    <script>
        const $ = id => document.getElementById(id);
        const messagesEl = $('messages');
        const inputEl = $('input');
        const sendBtn = $('send');
        const statusEl = $('status');
        const modelEl = $('model');
        const clearBtn = $('clear');
        const voiceBtn = $('voice');
        const suggestionsEl = $('suggestions');
        const projectEl = $('project');

        let ws = null;
        let commands = [];
        let selectedSuggestion = -1;
        let isRecording = false;
        let recognition = null;
        let ttsEnabled = false;
        let chatMode = false;
        let currentModel = null;
        const ttsBtn = $('tts');
        const chatModeBtn = $('chatMode');

        // Chat mode toggle
        chatModeBtn.addEventListener('click', () => {
            chatMode = !chatMode;
            chatModeBtn.style.opacity = chatMode ? '1' : '0.5';
            chatModeBtn.title = chatMode ? 'Chat mode ON (no tools)' : 'Chat mode OFF (uses tools)';
            addSystemMessage(chatMode ? '💬 Chat mode - just conversation, no tools' : '🔧 Agent mode - can use tools');
        });

        // Text-to-Speech using backend API (better quality) with browser fallback
        let currentAudio = null;

        async function speak(text) {
            if (!ttsEnabled) return;

            // Stop any ongoing audio
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
            speechSynthesis.cancel();

            // Clean text - remove code blocks, markdown, etc.
            let cleanText = text
                .replace(/```[\s\S]*?```/g, '')
                .replace(/`[^`]+`/g, '')
                .replace(/\[.*?\]\(.*?\)/g, '')
                .replace(/[#*_~]/g, '')
                .replace(/\n+/g, '. ')
                .trim();

            if (!cleanText || cleanText.length < 2) return;

            // Try backend TTS first (better quality)
            try {
                const res = await fetch('/api/tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: cleanText })
                });

                if (res.ok && res.headers.get('content-type')?.includes('audio')) {
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    currentAudio = new Audio(url);
                    currentAudio.play();
                    return;
                }
            } catch (e) {
                console.log('Backend TTS failed, using browser:', e);
            }

            // Fallback to browser speech synthesis
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;

            const voices = speechSynthesis.getVoices();
            const preferredVoice = voices.find(v =>
                v.name.includes('Samantha') ||
                v.name.includes('Daniel') ||
                v.lang.startsWith('en')
            );
            if (preferredVoice) utterance.voice = preferredVoice;

            speechSynthesis.speak(utterance);
        }

        ttsBtn.addEventListener('click', () => {
            ttsEnabled = !ttsEnabled;
            ttsBtn.textContent = ttsEnabled ? '🔊' : '🔇';
            ttsBtn.title = ttsEnabled ? 'Voice ON' : 'Voice OFF';
            if (ttsEnabled) {
                speak('Voice enabled');
            } else {
                if (currentAudio) currentAudio.pause();
                speechSynthesis.cancel();
            }
        });

        // Load voices when available
        if ('speechSynthesis' in window) {
            speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
        }

        // Auto-resize textarea
        inputEl.addEventListener('input', () => {
            inputEl.style.height = 'auto';
            inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
            handleSlashCommands();
        });

        // Slash command suggestions
        function handleSlashCommands() {
            const val = inputEl.value;
            if (val.startsWith('/') && !val.includes(' ')) {
                const query = val.slice(1).toLowerCase();
                const filtered = commands.filter(c =>
                    c.cmd.slice(1).toLowerCase().startsWith(query)
                );
                if (filtered.length > 0) {
                    suggestionsEl.innerHTML = filtered.map((c, i) => `
                        <div class="suggestion ${i === selectedSuggestion ? 'selected' : ''}" data-cmd="${c.cmd}">
                            <span class="cmd">${c.cmd}</span>
                            <span class="desc">${c.desc}</span>
                        </div>
                    `).join('');
                    suggestionsEl.classList.add('active');
                    return;
                }
            }
            suggestionsEl.classList.remove('active');
            selectedSuggestion = -1;
        }

        suggestionsEl.addEventListener('click', e => {
            const suggestion = e.target.closest('.suggestion');
            if (suggestion) {
                inputEl.value = suggestion.dataset.cmd + ' ';
                suggestionsEl.classList.remove('active');
                inputEl.focus();
            }
        });

        inputEl.addEventListener('keydown', e => {
            if (suggestionsEl.classList.contains('active')) {
                const items = suggestionsEl.querySelectorAll('.suggestion');
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    selectedSuggestion = Math.min(selectedSuggestion + 1, items.length - 1);
                    updateSelectedSuggestion(items);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    selectedSuggestion = Math.max(selectedSuggestion - 1, 0);
                    updateSelectedSuggestion(items);
                } else if (e.key === 'Tab' || e.key === 'Enter') {
                    if (selectedSuggestion >= 0 && items[selectedSuggestion]) {
                        e.preventDefault();
                        inputEl.value = items[selectedSuggestion].dataset.cmd + ' ';
                        suggestionsEl.classList.remove('active');
                    }
                } else if (e.key === 'Escape') {
                    suggestionsEl.classList.remove('active');
                }
            } else if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        });

        function updateSelectedSuggestion(items) {
            items.forEach((el, i) => {
                el.classList.toggle('selected', i === selectedSuggestion);
            });
        }

        // Voice input using MediaRecorder + backend Whisper (with browser fallback)
        let mediaRecorder = null;
        let audioChunks = [];
        let useBrowserTranscription = false;
        let browserRecognition = null;

        // Setup browser speech recognition as fallback
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            browserRecognition = new SpeechRecognition();
            browserRecognition.continuous = false;
            browserRecognition.interimResults = false;

            browserRecognition.onresult = e => {
                const transcript = e.results[0][0].transcript;
                inputEl.value = transcript;
                inputEl.dispatchEvent(new Event('input'));
                setTimeout(send, 300);
            };

            browserRecognition.onend = () => {
                isRecording = false;
                voiceBtn.classList.remove('recording');
                voiceBtn.textContent = '🎤';
            };
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];

                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                    // Show transcribing status
                    inputEl.placeholder = 'Transcribing...';
                    voiceBtn.disabled = true;

                    // Send to backend for Whisper transcription
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.webm');

                    try {
                        const res = await fetch('/api/transcribe', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await res.json();

                        if (data.transcript) {
                            inputEl.value = data.transcript;
                            inputEl.dispatchEvent(new Event('input'));
                            // Auto-send
                            setTimeout(send, 300);
                        } else if (data.use_browser) {
                            // Fallback to browser speech recognition
                            addSystemMessage('Using browser transcription (install whisper for better results)');
                            useBrowserTranscription = true;
                        } else if (data.error) {
                            addSystemMessage('Transcription: ' + data.error);
                        }
                    } catch (e) {
                        addSystemMessage('Transcription failed, using browser fallback');
                        useBrowserTranscription = true;
                    }

                    inputEl.placeholder = 'Type a message or / for commands...';
                    voiceBtn.disabled = false;
                };

                mediaRecorder.start();
                isRecording = true;
                voiceBtn.classList.add('recording');
                voiceBtn.textContent = '⏹️';
            } catch (e) {
                addSystemMessage('Microphone access denied');
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            isRecording = false;
            voiceBtn.classList.remove('recording');
            voiceBtn.textContent = '🎤';
        }

        voiceBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                // Use browser recognition if backend whisper not available
                if (useBrowserTranscription && browserRecognition) {
                    browserRecognition.start();
                    isRecording = true;
                    voiceBtn.classList.add('recording');
                    voiceBtn.textContent = '⏹️';
                } else {
                    startRecording();
                }
            }
        });

        // WebSocket connection
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status connected';
                sendBtn.disabled = false;
                loadModels();
                loadCommands();
            };

            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status disconnected';
                sendBtn.disabled = true;
                setTimeout(connect, 3000);
            };

            ws.onerror = () => {
                addSystemMessage('Connection error. Retrying...');
            };

            ws.onmessage = e => {
                const data = JSON.parse(e.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'connected':
                    projectEl.textContent = data.project || 'No project';
                    currentModel = data.model;
                    // Set dropdown after a short delay to ensure models are loaded
                    setTimeout(() => {
                        if (currentModel) modelEl.value = currentModel;
                    }, 500);
                    addSystemMessage(`Connected to ${data.provider} (${data.model})`);
                    break;

                case 'processing':
                    // Message already shown immediately in send()
                    break;

                case 'stream':
                    // Streaming chunk - show immediately
                    hideTyping();
                    appendToLastMessage(data.content);
                    break;

                case 'response':
                    hideTyping();
                    if (data.done) {
                        finalizeLastMessage(data.content);
                        speak(data.content);
                    }
                    break;

                case 'diff':
                    break;

                case 'error':
                    hideTyping();
                    addMessage('system', '❌ ' + data.message);
                    break;

                case 'model_changed':
                    addSystemMessage(`Model changed to ${data.model}`);
                    break;

                case 'project_changed':
                    projectEl.textContent = data.project;
                    addSystemMessage(`Project: ${data.project}`);
                    break;

                case 'cleared':
                    messagesEl.innerHTML = '';
                    addSystemMessage('Chat cleared');
                    break;
            }
        }

        async function loadModels() {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                if (data.models && data.models.length > 0) {
                    modelEl.innerHTML = data.models.map(m =>
                        `<option value="${m}">${m}</option>`
                    ).join('');
                    // Set to current model if known
                    if (currentModel) modelEl.value = currentModel;
                }
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        }

        async function loadCommands() {
            try {
                const res = await fetch('/api/commands');
                const data = await res.json();
                commands = data.commands || [];
            } catch (e) {
                console.error('Failed to load commands:', e);
            }
        }

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = `message ${role}`;

            // Format content (basic markdown)
            let formatted = escapeHtml(content);
            // Code blocks
            formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
                // Check for diff
                if (lang === 'diff') {
                    code = code.split('\n').map(line => {
                        if (line.startsWith('+')) return `<span class="diff-add">${line}</span>`;
                        if (line.startsWith('-')) return `<span class="diff-remove">${line}</span>`;
                        if (line.startsWith('@@')) return `<span class="diff-header">${line}</span>`;
                        return line;
                    }).join('\n');
                }
                return `<pre><code>${code}</code></pre>`;
            });
            // Inline code
            formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

            if (role === 'system') {
                div.innerHTML = formatted;
            } else {
                div.innerHTML = `
                    <div class="role">${role === 'user' ? 'You' : 'Jarvis'}</div>
                    <div class="content">${formatted}</div>
                `;
            }

            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function addSystemMessage(content) {
            addMessage('system', content);
        }

        // Streaming support
        let streamingDiv = null;
        let streamingText = '';

        function appendToLastMessage(chunk) {
            if (!streamingDiv) {
                // Create new streaming message
                streamingDiv = document.createElement('div');
                streamingDiv.className = 'message assistant';
                streamingDiv.innerHTML = `
                    <div class="role">Jarvis</div>
                    <div class="content"></div>
                `;
                messagesEl.appendChild(streamingDiv);
                streamingText = '';
            }
            streamingText += chunk;
            const contentEl = streamingDiv.querySelector('.content');
            contentEl.textContent = streamingText;
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function finalizeLastMessage(fullContent) {
            if (streamingDiv) {
                // Format the final content properly
                let formatted = escapeHtml(fullContent);
                formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => `<pre><code>${code}</code></pre>`);
                formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
                const contentEl = streamingDiv.querySelector('.content');
                contentEl.innerHTML = formatted;
            }
            streamingDiv = null;
            streamingText = '';
        }

        function showTyping() {
            if (!document.getElementById('typing')) {
                const div = document.createElement('div');
                div.id = 'typing';
                div.className = 'typing';
                div.innerHTML = `
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                    <span>Jarvis is thinking...</span>
                `;
                messagesEl.appendChild(div);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }
        }

        function hideTyping() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function send() {
            const text = inputEl.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

            // Show message immediately
            addMessage('user', text);
            showTyping();

            ws.send(JSON.stringify({ type: 'message', content: text, chat_mode: chatMode }));
            inputEl.value = '';
            inputEl.style.height = 'auto';
            suggestionsEl.classList.remove('active');
        }

        sendBtn.addEventListener('click', send);

        modelEl.addEventListener('change', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'switch_model', model: modelEl.value }));
            }
        });

        clearBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'clear' }));
            }
        });

        connect();
    </script>
</body>
</html>'''


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
