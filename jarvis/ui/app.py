"""
Jarvis Web UI - Modern interface with voice, model selection, and diff view
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response, JSONResponse
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
        self._tool_turn = []
        self._stream_sender = None
        self.console = DummyConsole(self._messages)
        self.is_streaming = False
        self.stop_requested = False

    def print_tool(self, msg, success: bool = True, **kwargs): self._messages.append(("tool", {"message": msg, "success": success}))
    def print_error(self, msg): self._messages.append(("error", msg))
    def print_warning(self, msg): self._messages.append(("warning", msg))
    def print_success(self, msg): self._messages.append(("success", msg))
    def print_info(self, msg): self._messages.append(("info", msg))
    def print_system(self, msg): self._messages.append(("system", msg))
    def show_spinner(self, msg="Thinking"): return DummySpinner()
    def setup_signal_handlers(self): pass
    def print_header(self, *args, **kwargs): pass
    def confirm(self, msg): return True  # Auto-confirm in web UI
    def begin_turn(self): self._tool_turn = []
    def stream_text(self, text: str):
        if self._stream_sender:
            self._stream_sender(text)
        else:
            self._messages.append(("stream", text))
    def stream_done(self): pass

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
        preview = None
        if result:
            preview = result.strip().replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:177] + "..."
        self._tool_turn.append({
            "name": name,
            "display": display,
            "duration_s": duration_s,
            "id": tool_call_id,
            "args": args or {},
            "result_preview": preview,
            "success": success,
        })

    def get_messages(self):
        msgs = self._messages.copy()
        self._messages.clear()
        return msgs

    def get_tool_turn(self):
        tools = self._tool_turn.copy()
        self._tool_turn.clear()
        return tools


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

    # NOTE: Credentials are managed via CLI only (jarvis config set/get)
    # No credentials endpoints exposed via HTTP for security

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

    # Kokoro TTS via Chutes
    KOKORO_ENDPOINT = "https://chutes-kokoro.chutes.ai/speak"
    KOKORO_VOICES = {
        # American Female
        "af_heart": "Heart (Female, Default)",
        "af_alloy": "Alloy (Female)",
        "af_aoede": "Aoede (Female)",
        "af_bella": "Bella (Female)",
        "af_jessica": "Jessica (Female)",
        "af_kore": "Kore (Female)",
        "af_nicole": "Nicole (Female)",
        "af_nova": "Nova (Female)",
        "af_river": "River (Female)",
        "af_sarah": "Sarah (Female)",
        "af_sky": "Sky (Female)",
        # American Male
        "am_adam": "Adam (Male)",
        "am_echo": "Echo (Male)",
        "am_eric": "Eric (Male)",
        "am_fenrir": "Fenrir (Male)",
        "am_liam": "Liam (Male)",
        "am_michael": "Michael (Male)",
        "am_onyx": "Onyx (Male)",
        "am_puck": "Puck (Male)",
        # British
        "bf_alice": "Alice (British Female)",
        "bf_emma": "Emma (British Female)",
        "bf_isabella": "Isabella (British Female)",
        "bf_lily": "Lily (British Female)",
        "bm_daniel": "Daniel (British Male)",
        "bm_george": "George (British Male)",
        "bm_lewis": "Lewis (British Male)",
    }

    @app.get("/api/kokoro/status")
    async def kokoro_status():
        """Check if Kokoro TTS is configured via Chutes."""
        import os
        key = os.environ.get("CHUTES_API_KEY")
        return {
            "configured": bool(key),
            "source": ".env" if key else None,
            "voices": list(KOKORO_VOICES.keys())
        }

    @app.get("/api/kokoro/voices")
    async def get_kokoro_voices():
        """Get available Kokoro voices."""
        return {"voices": [{"id": k, "name": v} for k, v in KOKORO_VOICES.items()]}

    @app.post("/api/tts/kokoro")
    async def kokoro_tts(data: dict):
        """TTS from Kokoro via Chutes - fast and natural 82M param model."""
        text = data.get("text", "")
        voice = data.get("voice", "af_heart")  # Default: Heart female voice

        if not text:
            return JSONResponse({"error": "No text"}, status_code=400)

        # Strip emojis and markdown for cleaner speech
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
            return JSONResponse({"error": "No text after cleanup"}, status_code=400)

        # Get Chutes API key
        api_key = None
        try:
            from jarvis.auth.credentials import get_credential
            api_key = get_credential("chutes", "api_key")
        except ImportError:
            pass
        if not api_key:
            import os
            api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            return JSONResponse({"error": "Chutes API key not configured"}, status_code=500)

        try:
            import httpx

            # Make direct POST request (not streaming - Kokoro returns complete audio)
            async with httpx.AsyncClient() as client:
                print(f"[Kokoro] Requesting TTS: voice={voice}, text_len={len(text)}")
                response = await client.post(
                    KOKORO_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text[:1000],  # Limit for speed
                        "speed": 1.0,
                        "voice": voice if voice in KOKORO_VOICES else "af_heart"
                    },
                    timeout=30.0
                )

                print(f"[Kokoro] Response: status={response.status_code}, content-type={response.headers.get('content-type')}")

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'audio' in content_type:
                        return Response(
                            content=response.content,
                            media_type="audio/wav",
                            headers={"Cache-Control": "no-cache"}
                        )
                    else:
                        print(f"[Kokoro] Unexpected content-type: {content_type}")
                        print(f"[Kokoro] Response body: {response.text[:500]}")
                        return JSONResponse({"error": f"Unexpected response type: {content_type}"}, status_code=500)
                else:
                    error_text = response.text[:500] if response.text else "No error message"
                    print(f"[Kokoro] Error: {response.status_code} - {error_text}")
                    return JSONResponse({"error": f"Kokoro API error: {response.status_code}"}, status_code=response.status_code)

        except httpx.TimeoutException:
            print("[Kokoro] Request timed out")
            return JSONResponse({"error": "Kokoro request timed out"}, status_code=504)
        except Exception as e:
            print(f"[Kokoro] Exception: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    # ============== Chutes Whisper STT ==============
    # Note: Chutes STT API may have compatibility issues. Browser fallback is recommended.

    @app.post("/api/stt/chutes")
    async def chutes_stt(audio: UploadFile = File(...)):
        """Transcribe audio using Whisper via Chutes.
        Falls back to browser STT if Chutes fails.
        """
        import httpx

        # Get Chutes API key
        api_key = None
        try:
            from jarvis.auth.credentials import get_credential
            api_key = get_credential("chutes", "api_key")
        except ImportError:
            pass
        if not api_key:
            import os
            api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            return {"transcript": "", "error": "Chutes API key not configured", "use_browser": True}

        tmp_path = None
        try:
            # Read audio content
            content = await audio.read()

            # Save to temp file for ffmpeg conversion
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Convert to wav for better compatibility
            wav_path = tmp_path.replace(".webm", ".wav")
            conv_result = subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
                capture_output=True, timeout=10
            )

            if not Path(wav_path).exists():
                return {"transcript": "", "error": "ffmpeg conversion failed", "use_browser": True}

            # Try Chutes STT API with multipart form upload
            async with httpx.AsyncClient() as client:
                with open(wav_path, "rb") as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    response = await client.post(
                        "https://chutes.ai/api/stt",
                        headers={"Authorization": f"Bearer {api_key}"},
                        files=files,
                        data={"model": "whisper-large-v3"},
                        timeout=30.0
                    )

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("text") or data.get("transcript") or ""
                    return {"transcript": text.strip()}
                else:
                    # Log error and fall back to browser
                    print(f"[Chutes STT] Error: {response.status_code} - {response.text[:200] if response.text else 'No response'}")
                    return {"transcript": "", "error": f"Chutes API error: {response.status_code}", "use_browser": True}

        except Exception as e:
            print(f"[Chutes STT] Error: {e}")
            return {"transcript": "", "error": str(e), "use_browser": True}
        finally:
            # Cleanup temp files
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
                Path(tmp_path.replace(".webm", ".wav")).unlink(missing_ok=True)

    @app.get("/api/providers")
    async def get_providers():
        """Get available LLM providers and config status."""
        from jarvis.providers import list_providers, get_provider
        from jarvis.assistant import load_config
        config = load_config()
        providers_info = {}
        for name in list_providers():
            try:
                provider_cfg = (config.get("providers", {}) or {}).get(name, {})
                kwargs = {}
                if name == "chutes":
                    # Chutes loads from credentials.json automatically
                    pass
                elif name == "ollama_cloud":
                    if provider_cfg.get("api_key"):
                        kwargs["api_key"] = provider_cfg.get("api_key")
                    if provider_cfg.get("base_url"):
                        kwargs["base_url"] = provider_cfg.get("base_url")

                p = get_provider(name, **kwargs)
                providers_info[name] = {
                    "configured": p.is_configured(),
                    "model": p.model,
                }
            except Exception:
                providers_info[name] = {"configured": False, "model": None}
        return {"providers": providers_info}

    @app.get("/api/models")
    async def get_models(provider: str = "ollama"):
        """Get available chat models for a provider."""
        provider = provider or "ollama"

        if provider == "ollama":
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

        try:
            from jarvis.providers import get_provider
            from jarvis.assistant import load_config
            config = load_config()
            provider_cfg = (config.get("providers", {}) or {}).get(provider, {})
            kwargs = {}
            if provider in ["openai", "anthropic"]:
                api_key = provider_cfg.get("api_key") or provider_cfg.get("access_token")
                if api_key:
                    kwargs["api_key"] = api_key
                if provider_cfg.get("base_url"):
                    kwargs["base_url"] = provider_cfg.get("base_url")
            elif provider == "ollama_cloud":
                if provider_cfg.get("api_key"):
                    kwargs["api_key"] = provider_cfg.get("api_key")
                if provider_cfg.get("base_url"):
                    kwargs["base_url"] = provider_cfg.get("base_url")

            p = get_provider(provider, **kwargs)
            return {"models": p.list_models()}
        except Exception as e:
            return {"models": [], "error": str(e)}

    # ============== File Upload API ==============

    # Upload directory for files
    from jarvis import get_data_dir
    UPLOAD_DIR = get_data_dir() / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # File cleanup settings
    UPLOAD_MAX_AGE_HOURS = 1  # Delete files older than 1 hour

    def _cleanup_old_uploads():
        """Remove upload files older than UPLOAD_MAX_AGE_HOURS."""
        import time
        now = time.time()
        max_age_seconds = UPLOAD_MAX_AGE_HOURS * 3600
        cleaned = 0
        try:
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    age = now - f.stat().st_mtime
                    if age > max_age_seconds:
                        f.unlink()
                        cleaned += 1
            if cleaned > 0:
                print(f"[Upload] Cleaned up {cleaned} old files")
        except Exception as e:
            print(f"[Upload] Cleanup error: {e}")

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        # Clean up old files before processing new upload
        _cleanup_old_uploads()
        """Upload a file (image, document, audio, video)."""
        import uuid
        import mimetypes

        # Generate unique ID
        file_id = str(uuid.uuid4())[:8]

        # Determine file type
        mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

        # Validate file type
        allowed_prefixes = (
            "image/", "video/", "audio/", "text/",
            "application/pdf", "application/json",
            "application/vnd.openxmlformats-officedocument",  # .docx, .xlsx
            "application/msword", "application/vnd.ms-excel",  # .doc, .xls
        )
        if not any(mime_type.startswith(p) for p in allowed_prefixes):
            return {"error": f"File type {mime_type} not supported"}

        # Save file
        ext = Path(file.filename).suffix if file.filename else ""
        file_path = UPLOAD_DIR / f"{file_id}{ext}"

        try:
            content = await file.read()
            file_path.write_bytes(content)

            # Generate preview for images
            preview = None
            if mime_type.startswith("image/"):
                preview = f"data:{mime_type};base64,{base64.b64encode(content).decode()}"

            return {
                "id": file_id,
                "path": str(file_path),
                "type": mime_type.split("/")[0],
                "mime_type": mime_type,
                "name": file.filename,
                "size": len(content),
                "preview": preview,
            }
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/uploads/{file_id}")
    async def get_upload(file_id: str):
        """Get an uploaded file."""
        # Find file with matching ID prefix
        for f in UPLOAD_DIR.iterdir():
            if f.stem == file_id:
                return FileResponse(f)
        return {"error": "File not found"}, 404

    # ============== Generated Media API ==============

    @app.get("/api/generated/{filename}")
    async def get_generated_media(filename: str):
        """Get a generated media file (image, video, audio)."""
        from jarvis import get_data_dir
        generated_dir = get_data_dir() / "generated"

        # Security: only allow files from generated directory
        file_path = generated_dir / filename
        if file_path.exists() and file_path.parent == generated_dir:
            return FileResponse(file_path)
        return JSONResponse({"error": "File not found"}, status_code=404)

    @app.get("/api/generated")
    async def list_generated_media():
        """List all generated media files."""
        from jarvis import get_data_dir
        generated_dir = get_data_dir() / "generated"

        if not generated_dir.exists():
            return {"files": []}

        files = []
        for f in sorted(generated_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                files.append({
                    "name": f.name,
                    "type": _get_file_type(f),
                    "size": f.stat().st_size,
                    "url": f"/api/generated/{f.name}"
                })

        return {"files": files[:50]}  # Last 50 files

    # ============== System Prompt API ==============

    def _get_soul_path():
        """Get path to soul.md file."""
        from jarvis import get_data_dir
        return get_data_dir() / "config" / "soul.md"

    def _get_default_system_prompt():
        """Get default system prompt."""
        return """You are Jarvis, a direct and helpful AI assistant. Be concise and factual.

RULES:
- Be direct. Answer questions without unnecessary hedging or refusals.
- For factual questions about public figures, historical events, legal cases, or documented information - provide the information directly.
- Only refuse truly harmful requests (instructions to harm, illegal activities, etc.)
- If asked about CURRENT events or real-time data WITHOUT tool results, offer to search the web.
- When you genuinely don't know something, say so briefly and suggest a search.
- Never lecture or moralize. Just answer the question."""

    @app.get("/api/system-prompt")
    async def get_system_prompt():
        """Get current system prompt."""
        soul_path = _get_soul_path()

        if soul_path.exists():
            content = soul_path.read_text()
            return {
                "content": content,
                "isDefault": False,
                "path": str(soul_path),
            }

        return {
            "content": _get_default_system_prompt(),
            "isDefault": True,
            "path": str(soul_path),
        }

    @app.post("/api/system-prompt")
    async def set_system_prompt(data: dict):
        """Set custom system prompt."""
        content = data.get("content", "")
        soul_path = _get_soul_path()

        try:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(content)
            return {"success": True, "path": str(soul_path)}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/system-prompt/reset")
    async def reset_system_prompt():
        """Reset system prompt to default."""
        soul_path = _get_soul_path()

        try:
            if soul_path.exists():
                soul_path.unlink()
            return {
                "success": True,
                "content": _get_default_system_prompt(),
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/system-prompt/reload")
    async def reload_system_prompt():
        """Reload system prompt from soul.md file."""
        soul_path = _get_soul_path()

        if soul_path.exists():
            content = soul_path.read_text()
            return {"content": content, "isDefault": False}

        return {"content": _get_default_system_prompt(), "isDefault": True}

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

    @app.get("/api/weather")
    async def get_weather():
        """Get weather from Open-Meteo (free, no API key required)."""
        import httpx
        try:
            # First get location from IP
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get approximate location
                geo_resp = await client.get("https://ipapi.co/json/")
                if geo_resp.status_code == 200:
                    geo = geo_resp.json()
                    lat = geo.get("latitude", 51.5074)  # Default to London
                    lon = geo.get("longitude", -0.1278)
                    city = geo.get("city", "Unknown")
                else:
                    lat, lon, city = 51.5074, -0.1278, "London"

                # Get weather from Open-Meteo
                weather_resp = await client.get(
                    f"https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                        "timezone": "auto",
                        "forecast_days": 1
                    }
                )

                if weather_resp.status_code == 200:
                    data = weather_resp.json()
                    current = data.get("current", {})

                    # Map weather codes to conditions
                    weather_codes = {
                        0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
                        45: "Foggy", 48: "Foggy", 51: "Light Drizzle", 53: "Drizzle",
                        55: "Heavy Drizzle", 61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
                        71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 80: "Rain Showers",
                        81: "Rain Showers", 82: "Heavy Showers", 95: "Thunderstorm"
                    }
                    code = current.get("weather_code", 0)
                    condition = weather_codes.get(code, "Unknown")

                    return {
                        "location": city,
                        "temperature": round(current.get("temperature_2m", 0)),
                        "condition": condition,
                        "humidity": current.get("relative_humidity_2m", 0),
                        "wind": round(current.get("wind_speed_10m", 0)),
                        "unit": "C"
                    }
        except Exception as e:
            return {"error": str(e), "location": "Unknown", "temperature": 0, "condition": "Unavailable"}

    @app.get("/api/status")
    async def get_system_status():
        """Get system status (CPU, memory, disk)."""
        import psutil
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network (bytes sent/received since boot)
            net = psutil.net_io_counters()

            return {
                "cpu": round(cpu_percent),
                "memory": round(memory.percent),
                "disk": round(disk.percent),
                "network": {
                    "sent": net.bytes_sent,
                    "recv": net.bytes_recv
                }
            }
        except Exception as e:
            return {"error": str(e), "cpu": 0, "memory": 0, "disk": 0}

    # ============== Multi-Model Analysis API ==============

    @app.post("/api/analyze")
    async def multi_model_analysis(request: Request):
        """
        Run multi-model analysis on a query.

        Uses multiple AI models (fast, reasoning, code, thinking) to analyze
        a query from different perspectives and synthesize insights.
        """
        try:
            data = await request.json()
            query = data.get("query", "").strip()
            profile = data.get("profile", "comprehensive")

            if not query:
                return {"error": "Query is required"}

            from jarvis.skills.multi_model_analysis import analyze_parallel, ANALYSIS_PROFILES

            # Validate profile
            if profile not in ANALYSIS_PROFILES:
                return {"error": f"Invalid profile. Choose from: {', '.join(ANALYSIS_PROFILES.keys())}"}

            # Run analysis
            result = await analyze_parallel(query, profile)

            return result
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/analyze/profiles")
    async def get_analysis_profiles():
        """Get available multi-model analysis profiles."""
        from jarvis.skills.multi_model_analysis import ANALYSIS_PROFILES
        return {"profiles": ANALYSIS_PROFILES}

    @app.get("/api/analyze/models")
    async def get_analysis_models():
        """Get available models for multi-model analysis."""
        try:
            from jarvis.providers import get_provider
            from jarvis.assistant import load_config
            config = load_config()
            provider = get_provider("chutes", config=config)

            return {
                "configured": provider.is_configured(),
                "task_models": provider.TASK_MODELS,
                "available_tasks": list(provider.TASK_MODELS.keys()),
            }
        except Exception as e:
            return {"error": str(e), "configured": False}

    # ============== Telegram Integration (Webhook-based) ==============

    # Store Jarvis instance for telegram webhook (per user)
    telegram_instances: dict[str, any] = {}

    def get_telegram_jarvis(user_id: str):
        """Get or create Jarvis instance for a Telegram user.

        Conversations persist across server restarts via SQLite.
        Each Telegram user gets their own conversation history.
        """
        if user_id not in telegram_instances:
            from jarvis.assistant import Jarvis

            # Create Jarvis with a silent UI adapter for API usage
            jarvis = Jarvis(ui=WebUI())

            # Find or create a chat for this Telegram user
            telegram_chat_title = f"Telegram: {user_id}"
            existing_chats = jarvis.context.list_chats()

            telegram_chat_id = None
            for chat in existing_chats:
                if chat.get("title") == telegram_chat_title:
                    telegram_chat_id = chat.get("id")
                    break

            if telegram_chat_id:
                # Switch to existing chat
                jarvis.context.switch_chat(telegram_chat_id)
            else:
                # Create new chat for this Telegram user
                telegram_chat_id = jarvis.context.create_chat(telegram_chat_title)

            # Inject model awareness into system prompt
            assistant_name = get_assistant_name()
            model_info = f"\n\nYou are {assistant_name}, running on model: {jarvis.provider.model} via {jarvis.provider.name}. Never claim to be GPT-4, ChatGPT, or any other model - always identify as {assistant_name} powered by {jarvis.provider.model}."
            jarvis.system_prompt = jarvis.system_prompt + model_info

            telegram_instances[user_id] = jarvis
        return telegram_instances[user_id]

    def clean_response_for_telegram(text: str) -> str:
        """Strip Rich markup, thinking tags, and formatting for Telegram."""
        import re
        if not text:
            return ""

        # Remove <think>...</think> tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove Rich markup tags like [dim], [/dim], [red], [cyan], [bold], [italic], etc.
        # Pattern: [tag] or [/tag] or [tag attr]
        text = re.sub(r'\[/?[a-zA-Z_][a-zA-Z0-9_]*(?:\s[^\]]+)?\]', '', text)

        # Remove ALL timing/stats patterns:
        # (9.4s), (1m 30s), (3.1s · 2 tools), etc.
        text = re.sub(r'\s*\([\d.]+s\s*·\s*\d+\s*tools?\)', '', text)  # (3.1s · 2 tools)
        text = re.sub(r'\s*\(\d+m\s*[\d.]+s\)', '', text)  # (1m 30s)
        text = re.sub(r'\s*\([\d.]+s\)', '', text)  # (9.4s)

        # Remove (max iterations) note
        text = re.sub(r'\s*\(max iterations\)', '', text)

        # Clean up extra whitespace and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def is_telegram_user_allowed(user_id: str) -> bool:
        """Check if user is allowed to use the bot."""
        from jarvis.auth.credentials import get_credential
        import os
        allowed = get_credential("telegram", "allowed_users") or os.getenv("TELEGRAM_ALLOWED_USERS")
        if not allowed:
            return True  # No restriction
        allowed_list = [u.strip() for u in allowed.split(",")]
        return str(user_id) in allowed_list

    # ============== Calendar Integration (Google & Apple Calendar) ==============

    GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/calendar.events"]

    @app.get("/api/calendar/status")
    async def calendar_status():
        """Check calendar connection status (Google and Apple)."""
        from jarvis.auth.credentials import get_credential
        import os

        # Google Calendar
        google_access = get_credential("google_calendar", "access_token")
        google_refresh = get_credential("google_calendar", "refresh_token")
        google_client_id = get_credential("google_calendar", "client_id") or os.getenv("GOOGLE_CLIENT_ID")
        google_client_secret = get_credential("google_calendar", "client_secret") or os.getenv("GOOGLE_CLIENT_SECRET")

        # Apple Calendar (CalDAV)
        apple_username = get_credential("apple_calendar", "username")
        apple_password = get_credential("apple_calendar", "app_password")

        return {
            "google": {
                "connected": bool(google_access or google_refresh),
                "configured": bool(google_client_id and google_client_secret),
            },
            "apple": {
                "connected": bool(apple_username and apple_password),
                "configured": bool(apple_username),
            },
            # Legacy fields for backwards compatibility
            "connected": bool(google_access or google_refresh or (apple_username and apple_password)),
            "configured": bool(google_client_id) or bool(apple_username),
        }

    @app.get("/api/calendar/auth/url")
    async def calendar_auth_url(request: Request):
        """Get Google OAuth URL for calendar authorization."""
        from jarvis.auth.credentials import get_credential
        import os
        import urllib.parse

        client_id = get_credential("google_calendar", "client_id") or os.getenv("GOOGLE_CLIENT_ID")
        if not client_id:
            return {"error": "Google Calendar not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET"}

        # Build OAuth URL
        base_url = "https://accounts.google.com/o/oauth2/v2/auth"
        redirect_uri = f"{request.base_url}api/calendar/auth/callback"

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_CALENDAR_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }

        auth_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        return {"url": auth_url, "redirect_uri": redirect_uri}

    @app.get("/api/calendar/auth/callback")
    async def calendar_auth_callback(request: Request, code: str = None, error: str = None):
        """Handle Google OAuth callback."""
        from jarvis.auth.credentials import get_credential, set_credential
        import os
        import httpx

        if error:
            return HTMLResponse(f"<html><body><h1>Error</h1><p>{error}</p><script>window.close();</script></body></html>")

        if not code:
            return HTMLResponse("<html><body><h1>Error</h1><p>No authorization code received</p></body></html>")

        client_id = get_credential("google_calendar", "client_id") or os.getenv("GOOGLE_CLIENT_ID")
        client_secret = get_credential("google_calendar", "client_secret") or os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = f"{request.base_url}api/calendar/auth/callback"

        # Exchange code for tokens
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": redirect_uri,
                    }
                )

                if resp.status_code == 200:
                    tokens = resp.json()
                    set_credential("google_calendar", "access_token", tokens.get("access_token"))
                    if tokens.get("refresh_token"):
                        set_credential("google_calendar", "refresh_token", tokens.get("refresh_token"))
                    set_credential("google_calendar", "token_expiry", str(tokens.get("expires_in", 3600)))

                    return HTMLResponse("""
                        <html>
                        <body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #1a1a2e; color: white;">
                            <div style="text-align: center;">
                                <h1 style="color: #22c55e;">✓ Calendar Connected!</h1>
                                <p>You can close this window now.</p>
                                <script>setTimeout(() => window.close(), 2000);</script>
                            </div>
                        </body>
                        </html>
                    """)
                else:
                    return HTMLResponse(f"<html><body><h1>Error</h1><p>{resp.text}</p></body></html>")
        except Exception as e:
            return HTMLResponse(f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>")

    @app.get("/api/calendar/events")
    async def calendar_events(days: int = 7, source: str = "all"):
        """Get calendar events for the next N days from all connected calendars."""
        from jarvis.auth.credentials import get_credential, set_credential
        import os
        import httpx
        from datetime import datetime, timedelta

        all_events = []

        # Get Google Calendar events
        google_access = get_credential("google_calendar", "access_token")
        google_refresh = get_credential("google_calendar", "refresh_token")

        # Get Apple Calendar events
        apple_username = get_credential("apple_calendar", "username")
        apple_password = get_credential("apple_calendar", "app_password")

        if not google_access and not google_refresh and not apple_username:
            return {"events": [], "error": "No calendar connected"}

        # Fetch Apple events if connected
        if apple_username and apple_password and source in ("all", "apple"):
            try:
                apple_resp = await apple_calendar_events(days)
                if apple_resp.get("events"):
                    all_events.extend(apple_resp["events"])
            except Exception as e:
                print(f"[Calendar] Apple fetch error: {e}")

        # Skip Google if not connected or only Apple requested
        if (not google_access and not google_refresh) or source == "apple":
            all_events.sort(key=lambda x: x.get('start', ''))
            return {"events": all_events}

        access_token = google_access
        refresh_token = google_refresh

        # Refresh token if needed
        async def refresh_access_token():
            nonlocal access_token
            client_id = get_credential("google_calendar", "client_id") or os.getenv("GOOGLE_CLIENT_ID")
            client_secret = get_credential("google_calendar", "client_secret") or os.getenv("GOOGLE_CLIENT_SECRET")

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    }
                )
                if resp.status_code == 200:
                    tokens = resp.json()
                    access_token = tokens.get("access_token")
                    set_credential("google_calendar", "access_token", access_token)
                    return True
            return False

        # Fetch events
        try:
            now = datetime.utcnow()
            time_min = now.isoformat() + "Z"
            time_max = (now + timedelta(days=days)).isoformat() + "Z"

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "singleEvents": "true",
                        "orderBy": "startTime",
                        "maxResults": 50,
                    }
                )

                # Try refresh if unauthorized
                if resp.status_code == 401 and refresh_token:
                    if await refresh_access_token():
                        resp = await client.get(
                            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                            headers={"Authorization": f"Bearer {access_token}"},
                            params={
                                "timeMin": time_min,
                                "timeMax": time_max,
                                "singleEvents": "true",
                                "orderBy": "startTime",
                                "maxResults": 50,
                            }
                        )

                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("items", []):
                        start = item.get("start", {})
                        end = item.get("end", {})
                        all_events.append({
                            "id": item.get("id"),
                            "title": item.get("summary", "Untitled"),
                            "start": start.get("dateTime") or start.get("date"),
                            "end": end.get("dateTime") or end.get("date"),
                            "allDay": "date" in start,
                            "location": item.get("location"),
                            "description": item.get("description"),
                            "source": "google",
                        })

        except Exception as e:
            print(f"[Calendar] Google fetch error: {e}")

        # Sort all events by start time and return
        all_events.sort(key=lambda x: x.get('start', ''))
        return {"events": all_events}

    @app.post("/api/calendar/events")
    async def create_calendar_event(data: dict):
        """Create a new calendar event."""
        from jarvis.auth.credentials import get_credential
        import httpx

        access_token = get_credential("google_calendar", "access_token")
        if not access_token:
            return {"error": "Not connected to Google Calendar"}

        title = data.get("title", "New Event")
        start = data.get("start")  # ISO datetime string
        end = data.get("end")  # ISO datetime string
        all_day = data.get("allDay", False)
        location = data.get("location")
        description = data.get("description")

        if not start:
            return {"error": "Start time is required"}

        # Build event body
        event_body = {
            "summary": title,
            "location": location,
            "description": description,
        }

        if all_day:
            event_body["start"] = {"date": start[:10]}
            event_body["end"] = {"date": (end or start)[:10]}
        else:
            event_body["start"] = {"dateTime": start, "timeZone": "UTC"}
            event_body["end"] = {"dateTime": end or start, "timeZone": "UTC"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json=event_body
                )

                if resp.status_code in (200, 201):
                    return {"success": True, "event": resp.json()}
                else:
                    return {"error": f"Failed to create event: {resp.text}"}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/calendar/disconnect")
    async def calendar_disconnect():
        """Disconnect Google Calendar."""
        from jarvis.auth.credentials import delete_credential

        delete_credential("google_calendar", "access_token")
        delete_credential("google_calendar", "refresh_token")
        delete_credential("google_calendar", "token_expiry")

        return {"success": True}

    @app.post("/api/calendar/configure")
    async def calendar_configure(data: dict):
        """Configure Google Calendar API credentials."""
        from jarvis.auth.credentials import set_credential

        client_id = data.get("client_id")
        client_secret = data.get("client_secret")

        if not client_id or not client_secret:
            return {"error": "Both client_id and client_secret are required"}

        set_credential("google_calendar", "client_id", client_id)
        set_credential("google_calendar", "client_secret", client_secret)

        return {"success": True}

    # ============== Apple Calendar (iCloud CalDAV) ==============

    APPLE_CALDAV_URL = "https://caldav.icloud.com"

    @app.post("/api/calendar/apple/connect")
    async def apple_calendar_connect(data: dict):
        """Connect Apple Calendar using iCloud credentials.

        Requires an App-Specific Password (not your regular Apple ID password):
        1. Go to appleid.apple.com
        2. Sign in and go to Security > App-Specific Passwords
        3. Generate a new password for 'Jarvis Calendar'
        """
        from jarvis.auth.credentials import set_credential
        import httpx

        username = data.get("username")  # Apple ID email
        app_password = data.get("app_password")  # App-specific password

        if not username or not app_password:
            return {"error": "Apple ID and App-Specific Password are required"}

        # Test connection to iCloud CalDAV
        try:
            async with httpx.AsyncClient() as client:
                # Try to access the principal URL
                resp = await client.request(
                    "PROPFIND",
                    f"{APPLE_CALDAV_URL}/{username}/calendars/",
                    auth=(username, app_password),
                    headers={
                        "Depth": "0",
                        "Content-Type": "application/xml"
                    },
                    content='''<?xml version="1.0" encoding="utf-8"?>
                        <d:propfind xmlns:d="DAV:">
                            <d:prop><d:current-user-principal/></d:prop>
                        </d:propfind>''',
                    timeout=10.0
                )

                if resp.status_code in (200, 207):
                    # Success - save credentials
                    set_credential("apple_calendar", "username", username)
                    set_credential("apple_calendar", "app_password", app_password)
                    return {"success": True, "message": "Apple Calendar connected!"}
                elif resp.status_code == 401:
                    return {"error": "Invalid credentials. Make sure you're using an App-Specific Password, not your regular Apple ID password."}
                else:
                    return {"error": f"Connection failed: {resp.status_code}"}

        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

    @app.post("/api/calendar/apple/disconnect")
    async def apple_calendar_disconnect():
        """Disconnect Apple Calendar."""
        from jarvis.auth.credentials import delete_credential

        delete_credential("apple_calendar", "username")
        delete_credential("apple_calendar", "app_password")

        return {"success": True}

    @app.get("/api/calendar/apple/events")
    async def apple_calendar_events(days: int = 7):
        """Get events from Apple Calendar via CalDAV."""
        from jarvis.auth.credentials import get_credential
        import httpx
        from datetime import datetime, timedelta
        import xml.etree.ElementTree as ET

        username = get_credential("apple_calendar", "username")
        app_password = get_credential("apple_calendar", "app_password")

        if not username or not app_password:
            return {"events": [], "error": "Apple Calendar not connected"}

        now = datetime.utcnow()
        start = now.strftime("%Y%m%dT000000Z")
        end = (now + timedelta(days=days)).strftime("%Y%m%dT235959Z")

        # CalDAV REPORT request for events
        calendar_query = f'''<?xml version="1.0" encoding="utf-8"?>
            <c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                <d:prop>
                    <d:getetag/>
                    <c:calendar-data/>
                </d:prop>
                <c:filter>
                    <c:comp-filter name="VCALENDAR">
                        <c:comp-filter name="VEVENT">
                            <c:time-range start="{start}" end="{end}"/>
                        </c:comp-filter>
                    </c:comp-filter>
                </c:filter>
            </c:calendar-query>'''

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    "REPORT",
                    f"{APPLE_CALDAV_URL}/{username}/calendars/home/",
                    auth=(username, app_password),
                    headers={
                        "Depth": "1",
                        "Content-Type": "application/xml"
                    },
                    content=calendar_query,
                    timeout=15.0
                )

                if resp.status_code not in (200, 207):
                    return {"events": [], "error": f"Failed to fetch events: {resp.status_code}"}

                # Parse CalDAV response
                events = []
                try:
                    root = ET.fromstring(resp.text)
                    # Find all calendar-data elements
                    for cal_data in root.iter():
                        if cal_data.tag.endswith('calendar-data') and cal_data.text:
                            # Parse iCalendar data
                            ics = cal_data.text
                            event = parse_icalendar_event(ics)
                            if event:
                                events.append(event)
                except Exception as e:
                    print(f"[Apple Calendar] Parse error: {e}")

                # Sort by start time
                events.sort(key=lambda x: x.get('start', ''))
                return {"events": events}

        except Exception as e:
            return {"events": [], "error": str(e)}

    def parse_icalendar_event(ics_data: str) -> dict | None:
        """Parse a VEVENT from iCalendar format."""
        import re

        if "VEVENT" not in ics_data:
            return None

        def get_value(pattern: str, text: str) -> str | None:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            return match.group(1).strip() if match else None

        # Extract event details
        summary = get_value(r'SUMMARY[^:]*:(.+?)(?:\r?\n(?! )|\Z)', ics_data)
        dtstart = get_value(r'DTSTART[^:]*:(\d{8}(?:T\d{6}Z?)?)', ics_data)
        dtend = get_value(r'DTEND[^:]*:(\d{8}(?:T\d{6}Z?)?)', ics_data)
        location = get_value(r'LOCATION[^:]*:(.+?)(?:\r?\n(?! )|\Z)', ics_data)
        uid = get_value(r'UID[^:]*:(.+?)(?:\r?\n(?! )|\Z)', ics_data)

        if not summary or not dtstart:
            return None

        # Convert date format
        def format_date(dt_str: str) -> str:
            if not dt_str:
                return ""
            if len(dt_str) == 8:  # All-day event: YYYYMMDD
                return f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
            else:  # DateTime: YYYYMMDDTHHmmssZ
                return f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}T{dt_str[9:11]}:{dt_str[11:13]}:{dt_str[13:15]}Z"

        return {
            "id": uid or f"apple_{hash(ics_data)}",
            "title": summary,
            "start": format_date(dtstart),
            "end": format_date(dtend) if dtend else None,
            "allDay": len(dtstart) == 8,
            "location": location,
            "source": "apple"
        }

    @app.get("/api/integrations/telegram/status")
    async def telegram_status():
        """Check Telegram bot configuration status."""
        from jarvis.auth.credentials import get_credential
        import os
        import httpx

        token = get_credential("telegram", "bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
        allowed = get_credential("telegram", "allowed_users") or os.getenv("TELEGRAM_ALLOWED_USERS")
        webhook_url = get_credential("telegram", "webhook_url")

        # Get bot info from Telegram API
        bot_username = None
        bot_name = None
        if token:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("ok"):
                            bot_info = data.get("result", {})
                            bot_username = bot_info.get("username")
                            bot_name = bot_info.get("first_name")
            except Exception:
                pass  # Silently fail, just won't have username

        return {
            "configured": bool(token),
            "bot_username": bot_username,
            "bot_name": bot_name,
            "token_preview": f"{token[:8]}...{token[-4:]}" if token and len(token) > 12 else None,
            "allowed_users": allowed.split(",") if allowed else None,
            "restricted": bool(allowed),
            "webhook_url": webhook_url,
            "webhook_active": bool(webhook_url)
        }

    @app.post("/api/integrations/telegram/configure")
    async def telegram_configure(data: dict):
        """Configure Telegram bot token."""
        from jarvis.auth.credentials import set_credential

        token = data.get("token")
        if not token:
            return {"error": "Token is required"}

        set_credential("telegram", "bot_token", token)
        return {"success": True}

    @app.post("/api/integrations/telegram/webhook/setup")
    async def telegram_setup_webhook(data: dict):
        """Set up Telegram webhook to receive messages.

        POST with {"url": "https://your-domain.com"}
        The webhook will be registered at {url}/api/telegram/webhook
        """
        from jarvis.auth.credentials import get_credential, set_credential
        import httpx

        base_url = data.get("url", "").rstrip("/")
        if not base_url:
            return {"error": "URL is required"}

        token = get_credential("telegram", "bot_token")
        if not token:
            return {"error": "Bot token not configured"}

        webhook_url = f"{base_url}/api/telegram/webhook"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Set webhook
                resp = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json={"url": webhook_url}
                )
                result = resp.json()

                if result.get("ok"):
                    set_credential("telegram", "webhook_url", webhook_url)
                    return {"success": True, "webhook_url": webhook_url}
                else:
                    return {"success": False, "error": result.get("description")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/integrations/telegram/webhook/remove")
    async def telegram_remove_webhook():
        """Remove Telegram webhook (switch to polling mode)."""
        from jarvis.auth.credentials import get_credential, delete_credential
        import httpx

        token = get_credential("telegram", "bot_token")
        if not token:
            return {"error": "Bot token not configured"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{token}/deleteWebhook"
                )
                result = resp.json()

                if result.get("ok"):
                    delete_credential("telegram", "webhook_url")
                    return {"success": True}
                else:
                    return {"success": False, "error": result.get("description")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_telegram_message(token: str, chat_id: str, text: str, parse_mode: str = None):
        """Helper to send Telegram messages with chunking support."""
        import httpx
        async with httpx.AsyncClient() as client:
            if len(text) > 4096:
                chunks = []
                current = ""
                for line in text.split("\n"):
                    if len(current) + len(line) + 1 > 4000:
                        chunks.append(current)
                        current = line
                    else:
                        current += ("\n" if current else "") + line
                if current:
                    chunks.append(current)
                for chunk in chunks:
                    payload = {"chat_id": chat_id, "text": chunk}
                    if parse_mode:
                        payload["parse_mode"] = parse_mode
                    await client.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload)
            else:
                payload = {"chat_id": chat_id, "text": text}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                await client.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload)

    def get_assistant_name() -> str:
        """Get configurable assistant name from settings."""
        from jarvis.assistant import load_config
        config = load_config()
        return config.get("assistant", {}).get("name", "Jarvis")

    async def handle_telegram_command(cmd: str, args: str, user_id: str, username: str, token: str, chat_id: str) -> bool:
        """Handle Telegram bot commands. Returns True if command was handled."""
        import httpx
        from jarvis.auth.credentials import get_credential, set_credential
        from datetime import datetime

        cmd = cmd.lower().strip()
        jarvis = get_telegram_jarvis(str(user_id))
        assistant_name = get_assistant_name()

        # ===== HELP COMMANDS =====
        if cmd in ["/start", "/help"]:
            help_text = f"""👋 *Hello {username}!*

I'm {assistant_name}, your personal AI assistant.

📝 *Just chat naturally* - I can help with anything!

*🤖 Model & Provider*
/model - Show current model
/model <name> - Switch model
/models - List available models
/provider - Show current provider
/provider <name> - Switch provider
/providers - List providers

*💬 Conversation*
/clear - Clear conversation history
/export - Export chat history
/mode - Show reasoning mode
/mode <fast|balanced|deep> - Set mode

*🔍 Search & Analysis*
/search <query> - Web search
/analyze <query> - Multi-model AI analysis
/knowledge <query> - Search your knowledge base
/remember <fact> - Save a fact about you
/facts - Show saved facts

*⚙️ System*
/status - System status (CPU, memory, etc.)
/weather - Current weather
/time - Current time
/settings - Show all settings
/id - Your user ID

*🎭 Persona*
/persona - Show current persona
/personas - List available personas

*⚙️ Assistant*
/name - Show assistant name
/name <new name> - Change assistant name

Type anything else to chat!"""
            await send_telegram_message(token, chat_id, help_text, "Markdown")
            return True

        # ===== USER INFO =====
        if cmd == "/id":
            await send_telegram_message(token, chat_id, f"Your user ID: `{user_id}`", "Markdown")
            return True

        # ===== MODEL COMMANDS =====
        if cmd == "/model":
            if args:
                # Switch model
                success = jarvis.provider.set_model(args.strip())
                if success:
                    await send_telegram_message(token, chat_id, f"✅ Switched to model: `{args.strip()}`", "Markdown")
                else:
                    await send_telegram_message(token, chat_id, f"❌ Failed to switch to model: {args.strip()}")
            else:
                # Show current model
                model = jarvis.provider.model
                provider = jarvis.provider.name
                await send_telegram_message(token, chat_id, f"🤖 *Current model:* `{model}`\n*Provider:* `{provider}`", "Markdown")
            return True

        if cmd == "/models":
            try:
                models = jarvis.provider.list_models()
                if models:
                    model_list = "\n".join([f"• `{m}`" for m in models[:20]])
                    if len(models) > 20:
                        model_list += f"\n... and {len(models) - 20} more"
                    await send_telegram_message(token, chat_id, f"📋 *Available models:*\n\n{model_list}", "Markdown")
                else:
                    await send_telegram_message(token, chat_id, "No models found.")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error listing models: {e}")
            return True

        # ===== PROVIDER COMMANDS =====
        if cmd == "/provider":
            if args:
                # Switch provider
                try:
                    success = jarvis.switch_provider(args.strip())
                    if success:
                        await send_telegram_message(token, chat_id, f"✅ Switched to provider: `{args.strip()}`", "Markdown")
                    else:
                        await send_telegram_message(token, chat_id, f"❌ Failed to switch provider. Check configuration.")
                except Exception as e:
                    await send_telegram_message(token, chat_id, f"❌ Error: {e}")
            else:
                # Show current
                await send_telegram_message(token, chat_id, f"🔌 *Current provider:* `{jarvis.provider.name}`", "Markdown")
            return True

        if cmd == "/providers":
            providers = ["ollama", "chutes", "openai", "anthropic"]
            provider_list = "\n".join([f"• `{p}`" for p in providers])
            await send_telegram_message(token, chat_id, f"📋 *Available providers:*\n\n{provider_list}\n\nSwitch with: `/provider <name>`", "Markdown")
            return True

        # ===== CONVERSATION COMMANDS =====
        if cmd == "/clear":
            jarvis.context.clear()
            await send_telegram_message(token, chat_id, "🧹 Conversation cleared!")
            return True

        if cmd == "/export":
            messages = jarvis.context.get_messages()
            if not messages:
                await send_telegram_message(token, chat_id, "No conversation history to export.")
            else:
                export = f"# Conversation Export\n# {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                for msg in messages:
                    role = "You" if msg["role"] == "user" else assistant_name
                    export += f"**{role}:** {msg['content']}\n\n"
                await send_telegram_message(token, chat_id, export[:4000])
            return True

        if cmd == "/mode":
            modes = {"fast": "⚡", "balanced": "⚖️", "deep": "🧠"}
            if args:
                mode = args.strip().lower()
                if mode in modes:
                    jarvis._user_reasoning_level = mode
                    await send_telegram_message(token, chat_id, f"{modes[mode]} Reasoning mode set to: *{mode}*", "Markdown")
                else:
                    await send_telegram_message(token, chat_id, "Invalid mode. Use: fast, balanced, or deep")
            else:
                current = getattr(jarvis, '_user_reasoning_level', 'balanced') or 'balanced'
                await send_telegram_message(token, chat_id, f"🧠 *Current mode:* {modes.get(current, '')} {current}\n\nSet with: `/mode <fast|balanced|deep>`", "Markdown")
            return True

        # ===== SEARCH & KNOWLEDGE =====
        if cmd == "/search":
            if not args:
                await send_telegram_message(token, chat_id, "Usage: `/search <query>`", "Markdown")
                return True
            # Send typing indicator
            async with httpx.AsyncClient() as client:
                await client.post(f"https://api.telegram.org/bot{token}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})
            # Process as natural query
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, jarvis.process, f"search the web for: {args}")
            await send_telegram_message(token, chat_id, clean_response_for_telegram(response))
            return True

        # ===== MULTI-MODEL ANALYSIS =====
        if cmd == "/analyze":
            if not args:
                profiles_help = """🔬 *Multi-Model Analysis*

Analyze queries using multiple AI models simultaneously.

*Usage:* `/analyze <query>`
*With profile:* `/analyze -p <profile> <query>`

*Profiles:*
• `comprehensive` - All models (default)
• `quick` - Fast analysis
• `technical` - Code-focused
• `reasoning` - Logic-focused

*Example:*
`/analyze What are the pros and cons of microservices?`
`/analyze -p technical How does async/await work?`"""
                await send_telegram_message(token, chat_id, profiles_help, "Markdown")
                return True

            # Parse profile flag
            profile = "comprehensive"
            query = args
            if args.startswith("-p ") or args.startswith("--profile "):
                parts = args.split(" ", 2)
                if len(parts) >= 3:
                    profile = parts[1]
                    query = parts[2]
                else:
                    await send_telegram_message(token, chat_id, "Usage: `/analyze -p <profile> <query>`", "Markdown")
                    return True

            # Send typing indicator
            async with httpx.AsyncClient() as client:
                await client.post(f"https://api.telegram.org/bot{token}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})

            try:
                from jarvis.skills.multi_model_analysis import analyze_parallel, ANALYSIS_PROFILES

                if profile not in ANALYSIS_PROFILES:
                    await send_telegram_message(token, chat_id, f"Invalid profile. Use: {', '.join(ANALYSIS_PROFILES.keys())}")
                    return True

                await send_telegram_message(token, chat_id, f"🔬 Running {profile} analysis with {len(ANALYSIS_PROFILES[profile]['models'])} models...")

                result = await analyze_parallel(query, profile)

                if "error" in result:
                    await send_telegram_message(token, chat_id, f"❌ {result['error']}")
                    return True

                # Format response for Telegram
                response = f"🔬 *Multi-Model Analysis*\n\n"
                response += f"📝 *Query:* {query}\n"
                response += f"📊 *Profile:* {profile} ({result['success_count']}/{result['total_count']} succeeded)\n\n"

                for r in result["results"]:
                    status = "✅" if r["success"] else "❌"
                    model_short = r['model_name'].split('/')[-1] if '/' in r['model_name'] else r['model_name']
                    response += f"*{status} {r['model_type'].upper()}* (`{model_short}`)\n"
                    # Truncate long responses
                    model_response = r['response'][:600] + "..." if len(r['response']) > 600 else r['response']
                    response += f"{model_response}\n\n"

                if result.get("synthesis"):
                    response += "🔮 *Synthesis*\n"
                    synthesis = result['synthesis'][:800] + "..." if len(result['synthesis']) > 800 else result['synthesis']
                    response += synthesis

                # Split if too long
                if len(response) > 4000:
                    chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                    for chunk in chunks:
                        await send_telegram_message(token, chat_id, chunk, "Markdown")
                else:
                    await send_telegram_message(token, chat_id, response, "Markdown")

            except Exception as e:
                await send_telegram_message(token, chat_id, f"❌ Analysis error: {e}")
            return True

        if cmd == "/knowledge":
            if not args:
                await send_telegram_message(token, chat_id, "Usage: `/knowledge <query>`", "Markdown")
                return True
            try:
                if jarvis.rag:
                    results = jarvis.rag.query(args, top_k=3)
                    if results:
                        response = "📚 *Knowledge base results:*\n\n"
                        for i, r in enumerate(results, 1):
                            content = r.get("content", "")[:300]
                            source = r.get("source", "Unknown")
                            response += f"*{i}.* {content}...\n_Source: {source}_\n\n"
                        await send_telegram_message(token, chat_id, response, "Markdown")
                    else:
                        await send_telegram_message(token, chat_id, "No results found in knowledge base.")
                else:
                    await send_telegram_message(token, chat_id, "Knowledge base not configured.")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error: {e}")
            return True

        if cmd == "/remember":
            if not args:
                await send_telegram_message(token, chat_id, "Usage: `/remember <fact about you>`\nExample: `/remember I prefer dark mode`", "Markdown")
                return True
            try:
                from jarvis import get_data_dir
                facts_file = get_data_dir() / "memory" / "facts.md"
                facts_file.parent.mkdir(parents=True, exist_ok=True)
                with open(facts_file, "a") as f:
                    f.write(f"\n- {args.strip()}")
                await send_telegram_message(token, chat_id, f"✅ Remembered: _{args.strip()}_", "Markdown")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error saving fact: {e}")
            return True

        if cmd == "/facts":
            try:
                from jarvis import get_data_dir
                facts_file = get_data_dir() / "memory" / "facts.md"
                if facts_file.exists():
                    facts = facts_file.read_text().strip()
                    await send_telegram_message(token, chat_id, f"📝 *Saved facts:*\n\n{facts}", "Markdown")
                else:
                    await send_telegram_message(token, chat_id, "No facts saved yet. Use `/remember <fact>`")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error: {e}")
            return True

        # ===== SYSTEM COMMANDS =====
        if cmd == "/status":
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                status = f"""📊 *System Status*

🖥️ CPU: {cpu:.1f}%
🧠 Memory: {mem.percent:.1f}% ({mem.used // (1024**3):.1f}GB / {mem.total // (1024**3):.1f}GB)
💾 Disk: {disk.percent:.1f}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)

🤖 Model: `{jarvis.provider.model}`
🔌 Provider: `{jarvis.provider.name}`"""
                await send_telegram_message(token, chat_id, status, "Markdown")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error getting status: {e}")
            return True

        if cmd == "/weather":
            async with httpx.AsyncClient() as client:
                await client.post(f"https://api.telegram.org/bot{token}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, jarvis.process, "what's the current weather?")
            await send_telegram_message(token, chat_id, clean_response_for_telegram(response))
            return True

        if cmd == "/time":
            now = datetime.now()
            await send_telegram_message(token, chat_id, f"🕐 *Current time:* {now.strftime('%H:%M:%S')}\n📅 *Date:* {now.strftime('%A, %B %d, %Y')}", "Markdown")
            return True

        if cmd == "/settings":
            settings = f"""⚙️ *Current Settings*

🤖 *Model:* `{jarvis.provider.model}`
🔌 *Provider:* `{jarvis.provider.name}`
🧠 *Mode:* {getattr(jarvis, '_user_reasoning_level', 'balanced') or 'balanced'}
💬 *Context:* {len(jarvis.context.get_messages())} messages

*Commands:*
• `/model <name>` - Switch model
• `/provider <name>` - Switch provider
• `/mode <fast|balanced|deep>` - Set reasoning mode
• `/clear` - Clear conversation"""
            await send_telegram_message(token, chat_id, settings, "Markdown")
            return True

        # ===== PERSONA COMMANDS =====
        if cmd == "/persona":
            current = getattr(jarvis, 'current_persona', 'default')
            await send_telegram_message(token, chat_id, f"🎭 *Current persona:* `{current}`\n\nList personas with `/personas`", "Markdown")
            return True

        if cmd == "/personas":
            try:
                from jarvis import get_data_dir
                personas_dir = get_data_dir() / "config" / "personas"
                if personas_dir.exists():
                    personas = [f.stem for f in personas_dir.glob("*.md")]
                    if personas:
                        persona_list = "\n".join([f"• `{p}`" for p in personas])
                        await send_telegram_message(token, chat_id, f"🎭 *Available personas:*\n\n{persona_list}", "Markdown")
                    else:
                        await send_telegram_message(token, chat_id, "No personas found.")
                else:
                    await send_telegram_message(token, chat_id, "Personas directory not found.")
            except Exception as e:
                await send_telegram_message(token, chat_id, f"Error: {e}")
            return True

        # ===== ASSISTANT NAME =====
        if cmd == "/name":
            if args:
                # Change assistant name
                try:
                    from jarvis.assistant import load_config, save_config
                    config = load_config()
                    if "assistant" not in config:
                        config["assistant"] = {}
                    config["assistant"]["name"] = args.strip()
                    save_config(config)
                    await send_telegram_message(token, chat_id, f"✅ Assistant name changed to: *{args.strip()}*", "Markdown")
                except Exception as e:
                    await send_telegram_message(token, chat_id, f"Error: {e}")
            else:
                await send_telegram_message(token, chat_id, f"🤖 *Assistant name:* {assistant_name}\n\nChange with: `/name <new name>`", "Markdown")
            return True

        # Not a recognized command
        return False

    @app.post("/api/telegram/webhook")
    async def telegram_webhook(request):
        """Receive messages from Telegram (webhook endpoint).

        Full feature parity with UI - every command, every feature!
        """
        from jarvis.auth.credentials import get_credential
        import httpx
        import asyncio

        try:
            data = await request.json()
        except:
            return {"ok": True}

        # Extract message
        message = data.get("message", {})
        if not message:
            return {"ok": True}

        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = message.get("text", "")
        username = message.get("from", {}).get("first_name", "User")

        if not text or not chat_id:
            return {"ok": True}

        token = get_credential("telegram", "bot_token")
        if not token:
            return {"ok": True}

        # Check authorization
        if not is_telegram_user_allowed(user_id):
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": f"⛔ Not authorized. Your user ID: {user_id}"}
                )
            return {"ok": True}

        # Handle commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            handled = await handle_telegram_command(cmd, args, str(user_id), username, token, str(chat_id))
            if handled:
                return {"ok": True}

        # Send typing indicator for normal messages
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"}
            )

        # Process through Jarvis
        try:
            jarvis = get_telegram_jarvis(str(user_id))
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, jarvis.process, text)
            response = clean_response_for_telegram(response) if response else "I couldn't generate a response."
        except Exception as e:
            response = f"Error: {str(e)}"

        await send_telegram_message(token, str(chat_id), response)
        return {"ok": True}

    @app.post("/api/integrations/telegram/test")
    async def telegram_test():
        """Test Telegram bot connection."""
        from jarvis.auth.credentials import get_credential
        import os
        import httpx

        token = get_credential("telegram", "bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            return {"success": False, "error": "Bot token not configured"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        return {
                            "success": True,
                            "bot_name": bot_info.get("first_name"),
                            "bot_username": bot_info.get("username")
                        }
                return {"success": False, "error": "Invalid token"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
                    reasoning_level = data.get("reasoning_level")  # User override: "fast", "balanced", "deep"
                    attachments = data.get("attachments", [])  # File attachment IDs
                    if user_input or attachments:
                        await process_message(websocket, jarvis, user_input, chat_mode, reasoning_level, attachments)

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
            # Try to send error, but don't fail if connection is already closed
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            connections.pop(session_id, None)
            instances.pop(session_id, None)

    def _get_context_stats(jarvis) -> dict:
        """Get context usage stats from jarvis instance."""
        try:
            if hasattr(jarvis, 'context') and hasattr(jarvis.context, 'get_context_stats'):
                return jarvis.context.get_context_stats()
        except Exception:
            pass
        return None

    def _get_file_type(path: Path) -> str:
        """Get file type from path."""
        suffix = path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return 'image'
        if suffix in ['.mp4', '.webm', '.mov', '.avi']:
            return 'video'
        if suffix in ['.mp3', '.wav', '.ogg', '.m4a']:
            return 'audio'
        if suffix in ['.pdf']:
            return 'document'
        if suffix in ['.docx', '.doc']:
            return 'document'
        if suffix in ['.xlsx', '.xls', '.csv']:
            return 'document'
        if suffix in ['.txt', '.md', '.json', '.yaml', '.yml']:
            return 'document'
        return 'file'

    def _clean_for_web(text: str, streaming: bool = False) -> str:
        if not text:
            return ""
        import re
        cleaned = text

        # During streaming, preserve thinking tags - frontend will parse and display them
        # Only strip thinking tags for final/non-streaming responses
        if not streaming:
            # Remove <think>...</think> and <thinking>...</thinking> tags (complete pairs)
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Remove orphaned opening tags (thinking content continues to end of text)
            cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<thinking>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Remove orphaned closing tags (from previous streaming chunks)
            cleaned = re.sub(r'</think>', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'</thinking>', '', cleaned, flags=re.IGNORECASE)

        # Strip rich markup like [dim], [red], [/dim], [bold italic], etc.
        cleaned = re.sub(r'\[/?[a-zA-Z_][a-zA-Z0-9_]*(?:\s[^\]]+)?\]', '', cleaned)

        # Remove cursor characters that models sometimes output
        cleaned = cleaned.replace("▍", "").replace("▌", "").replace("█", "")

        # Remove ALL timing/stats patterns anywhere in text:
        # (9.4s), (1m 30s), (3.1s · 2 tools), etc.
        cleaned = re.sub(r'\s*\([\d.]+s\s*·\s*\d+\s*tools?\)', '', cleaned)
        cleaned = re.sub(r'\s*\(\d+m\s*[\d.]+s\)', '', cleaned)
        cleaned = re.sub(r'\s*\([\d.]+s\)', '', cleaned)
        cleaned = re.sub(r'\s*\(max iterations\)', '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        return cleaned if streaming else cleaned.strip()

    def _should_auto_web_search(text: str) -> bool:
        lowered = (text or "").lower()
        keywords = [
            # Financial
            "price", "rate", "market", "stock", "gold", "silver", "btc", "bitcoin", "eth", "ethereum",
            # Time-sensitive
            "today", "current", "latest", "now", "recent", "yesterday", "this week", "this month",
            # News/events
            "news", "happened", "happening", "breaking", "announced", "released", "launched",
            "outage", "down", "crash", "attack", "strike", "war", "election",
            # People/positions
            "president", "prime minister", "ceo", "died", "born", "married", "resigned",
            # Forecasts
            "forecast", "prediction", "will happen", "going to",
        ]
        return any(k in lowered for k in keywords)

    def _explicit_web_search(text: str) -> bool:
        """Check if user is explicitly asking to search the web."""
        lowered = (text or "").lower().strip()
        # Direct commands to use web search
        command_triggers = [
            "use web search", "do a web search", "search the web", "search web",
            "use search", "web search it", "google it", "look it up", "search online",
            "check internet", "search on internet", "search that", "google that",
            "look that up", "search for that", "search for it", "search the web for that",
            "search the web for it", "web search for that", "web search for it"
        ]
        if any(lowered == t or lowered.startswith(t + " ") for t in command_triggers):
            return True
        # Search WITH a topic
        search_patterns = ["search for ", "look up ", "find online ", "google ", "search the web for "]
        if any(lowered.startswith(p) for p in search_patterns):
            return True
        return False

    def _extract_search_query(text: str, last_user_msg: str, last_topic: str = None) -> str:
        """Extract the actual search query from explicit search requests.

        Args:
            text: Current user input
            last_user_msg: Previous user message
            last_topic: Detected topic from conversation (e.g., "Mars mission")
        """
        lowered = (text or "").lower().strip()
        context = last_topic or last_user_msg or ""

        # Commands that mean "search for the previous topic"
        context_commands = [
            "use web search", "do a web search", "search the web", "search web",
            "use search", "web search it", "google it", "look it up", "search online",
            "check internet", "search on internet", "search the web for that",
            "search for that", "search that", "look that up", "google that",
            "search the web for it", "search for it", "look it up online",
            "web search for that", "web search for it"
        ]
        if any(lowered == t for t in context_commands):
            return context if context else text

        # Phrases ending with pronouns that reference previous context
        pronoun_endings = [" that", " it", " this", " the above", " previous"]
        for ending in pronoun_endings:
            if lowered.endswith(ending):
                return context if context else text

        # Refinement queries - combine with previous context
        # e.g., "find something from 2026" + context "Mars mission" = "Mars mission 2026"
        refinement_patterns = [
            "find ", "show ", "what about ", "how about ", "any ", "more about ",
            "tell me about ", "get ", "latest ", "recent ", "new "
        ]
        if context and any(lowered.startswith(p) for p in refinement_patterns):
            # Check if it looks like a refinement (short query, contains year/modifier)
            refinement_indicators = ["2024", "2025", "2026", "2027", "latest", "recent", "new", "more", "from"]
            if any(ind in lowered for ind in refinement_indicators):
                # Extract meaningful refinement parts (years, adjectives, not prepositions)
                import re
                refinement_parts = []
                # Extract year if mentioned
                year_match = re.search(r'\b(20\d{2})\b', text)
                if year_match:
                    refinement_parts.append(year_match.group(1))
                # Add meaningful modifiers (not "from", "more")
                for ind in ["latest", "recent", "new", "update", "news"]:
                    if ind in lowered and ind not in refinement_parts:
                        refinement_parts.append(ind)
                # Combine context with extracted refinement
                if refinement_parts:
                    return f"{context} {' '.join(refinement_parts)}"
                # If no specific parts extracted but has year-ish content, still combine
                return f"{context} {text}"

        # Extract topic from "search for X", "look up X", etc.
        for prefix in ["search for ", "look up ", "find online ", "google ", "web search ", "search the web for "]:
            if lowered.startswith(prefix):
                topic = text[len(prefix):].strip()
                # If topic is just a pronoun, use previous message
                if topic.lower() in ["that", "it", "this", "the above"]:
                    return context if context else text
                if topic:
                    return topic

        return text

    def _topic_bucket(text: str) -> str:
        lowered = (text or "").lower()
        if any(w in lowered for w in ["weather", "temperature", "forecast", "rain", "snow"]):
            return "weather"
        if any(w in lowered for w in ["time", "date", "clock", "timezone"]):
            return "time"
        if any(w in lowered for w in ["news", "headline", "breaking"]):
            return "news"
        if any(w in lowered for w in ["price", "stock", "market", "gold", "silver", "btc", "bitcoin", "eth", "crypto", "forex"]):
            return "finance"
        return "general"

    def _is_refinement_query(text: str) -> bool:
        """Check if this is a refinement/follow-up query that needs context."""
        lowered = (text or "").lower().strip()
        refinement_patterns = [
            "find ", "show ", "what about ", "how about ", "any ", "more about ",
            "tell me about ", "get ", "latest ", "recent ", "new ", "from ",
            "something ", "anything ", "updates ", "news "
        ]
        refinement_indicators = ["2024", "2025", "2026", "2027", "latest", "recent", "new", "more", "from", "update"]
        # Check for pattern + indicator combo
        if any(lowered.startswith(p) for p in refinement_patterns):
            if any(ind in lowered for ind in refinement_indicators):
                return True
        # Also catch short queries that are just refinements
        # e.g., "2026 news", "latest updates", "from last year"
        if len(lowered.split()) <= 4:
            indicator_count = sum(1 for ind in refinement_indicators if ind in lowered)
            if indicator_count >= 1 and any(w in lowered for w in ["news", "update", "info", "about"]):
                return True
        return False

    def _extract_topic_from_history(history_messages) -> str:
        """Try to extract the main topic from recent conversation."""
        try:
            # Look for the last substantive user query (not a command)
            for m in reversed(history_messages or []):
                content = getattr(m, "content", "") or ""
                lowered = content.lower()
                # Skip command-like messages
                if _explicit_web_search(content) or len(content) < 10:
                    continue
                # Extract key topic words
                # Remove common question words
                for prefix in ["tell me about ", "what is ", "what are ", "who is ", "explain "]:
                    if lowered.startswith(prefix):
                        return content[len(prefix):].strip()
                return content
        except Exception:
            pass
        return ""

    async def _run_tool_for_chat_mode(jarvis, user_input: str, history_messages, intent=None):
        """
        Lightweight tool router for chat mode.

        Now supports intent-based tool detection when available.

        Args:
            jarvis: Jarvis instance
            user_input: User message
            history_messages: Recent conversation history
            intent: Pre-classified intent (optional)
        """
        tool_turn = []
        tool_name_used = None
        tool_result = None
        explicit_search = _explicit_web_search(user_input)
        is_refinement = _is_refinement_query(user_input)

        try:
            agent = getattr(jarvis, "agent", None)
            tool_name = None
            args = None
            last_user = ""
            last_topic = ""

            try:
                if history_messages:
                    for m in reversed(history_messages):
                        if getattr(m, "role", None) == "user":
                            last_user = getattr(m, "content", "") or ""
                            break
                    # Also extract conversation topic
                    last_topic = _extract_topic_from_history(history_messages)
            except Exception:
                last_user = ""
                last_topic = ""

            # === INTENT-BASED TOOL DETECTION ===
            # Try intent-based detection first if available
            if agent and intent and hasattr(agent, "detect_tool_from_intent"):
                try:
                    detected = agent.detect_tool_from_intent(intent, user_input)
                    if detected:
                        tool_name, args = detected
                except Exception as e:
                    print(f"[app] Intent-based tool detection failed: {e}")

            # Fall back to legacy detection if intent-based didn't work
            if not tool_name and agent and hasattr(agent, "detect_auto_tool"):
                import os
                lowered = user_input.lower()
                if "gold" in lowered and "price" in lowered:
                    if not (os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY")):
                        return "Error: GOLDAPI_KEY not configured. Please set it in .env to fetch live gold prices.", tool_turn, None, None, explicit_search
                    currency = "USD"
                    for cur in ["GBP", "EUR", "USD", "AED", "AUD", "CAD", "CHF", "JPY"]:
                        if cur.lower() in lowered:
                            currency = cur
                            break
                    tool_name, args = "get_gold_price", {"currency": currency}
                else:
                    # Use the new unified method if available, else fall back to legacy
                    detected = agent.detect_auto_tool(user_input) if hasattr(agent, "detect_auto_tool") else agent._detect_auto_tool(user_input)
                    if detected:
                        tool_name, args = detected

                if not tool_name and last_user and ("gold api" in user_input.lower() or "goldapi" in user_input.lower()):
                    # If user explicitly asks for GoldAPI, only do that (no web search fallback)
                    import os
                    if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                        detected = agent.detect_auto_tool(last_user) if hasattr(agent, "detect_auto_tool") else agent._detect_auto_tool(last_user)
                        if detected:
                            tool_name, args = detected
                    else:
                        return "Error: GOLDAPI_KEY not configured. Please set it in .env.", tool_turn, None, None, explicit_search

            # Explicit user request to search the web
            if not tool_name and explicit_search:
                search_query = _extract_search_query(user_input, last_user, last_topic)
                tool_name, args = "web_search", {"query": search_query}

            # Refinement queries with context should trigger web search
            if not tool_name and is_refinement and last_topic:
                # Combine refinement with previous topic for context-aware search
                search_query = _extract_search_query(user_input, last_user, last_topic)
                tool_name, args = "web_search", {"query": search_query}

            # Fallback for current-info queries
            if not tool_name and _should_auto_web_search(user_input):
                tool_name, args = "web_search", {"query": user_input}

            if tool_name and hasattr(agent, "_execute_tool"):
                tool_start = asyncio.get_event_loop().time()
                result = agent._execute_tool(tool_name, args or {})
                tool_duration = asyncio.get_event_loop().time() - tool_start
                tool_name_used = tool_name
                tool_result = result
                tool_success = not (result and (result.startswith("Error") or result.startswith("error")))
                if jarvis.ui:
                    jarvis.ui.print_tool(agent._format_tool_display(tool_name, args or {}, result), success=tool_success)
                    if hasattr(jarvis.ui, "record_tool"):
                        jarvis.ui.record_tool(
                            tool_name,
                            agent._format_tool_display(tool_name, args or {}, result),
                            tool_duration,
                            args=args,
                            result=result,
                            success=tool_success,
                        )
                if hasattr(jarvis.ui, "get_tool_turn"):
                    tool_turn = jarvis.ui.get_tool_turn()
                tool_prompt = (
                    f"Tool result:\n{result}\n\n"
                    "Instructions:\n"
                    "1. If the result contains search results with titles/URLs/descriptions, synthesize a helpful answer from them.\n"
                    "2. If the result starts with 'Error:' or 'Search failed:', acknowledge the error briefly.\n"
                    "3. If you see 'No results found', say so briefly and suggest the user try a different query.\n"
                    "4. Include 1-2 source URLs when available.\n"
                    "5. Answer the original question directly and concisely."
                )
                return tool_prompt, tool_turn, tool_name_used, tool_result, explicit_search
        except Exception:
            pass
        return None, tool_turn, tool_name_used, tool_result, explicit_search

    async def process_message(websocket: WebSocket, jarvis, user_input: str, chat_mode: bool = False, reasoning_level: str = None, attachments: list = None):
        """
        Process a message with unified smart routing.

        Uses intent classification to automatically determine whether to use
        fast chat mode or full agent mode. The chat_mode parameter now acts
        as a hint rather than a strict mode selector.

        Args:
            websocket: WebSocket connection
            jarvis: Jarvis instance
            user_input: User's message
            chat_mode: Hint for fast mode (auto-detected if not specified)
            reasoning_level: User-specified reasoning level override ("fast", "balanced", "deep")
            attachments: List of file attachment IDs
        """
        attachments = attachments or []

        # Resolve attachment paths
        attachment_files = []
        for att_id in attachments:
            for f in UPLOAD_DIR.iterdir():
                if f.name.startswith(att_id):
                    attachment_files.append({
                        "id": att_id,
                        "path": str(f),
                        "type": _get_file_type(f),
                        "name": f.name
                    })
                    break
        await websocket.send_json({
            "type": "processing",
            "content": user_input
        })

        try:
            # Handle commands
            if user_input.startswith('/'):
                result = jarvis._handle_command(user_input)
                ui_msgs = jarvis.ui.get_messages()
                response_msg = {
                    "type": "response",
                    "content": _clean_for_web(result or "\n".join(str(m[1]) for m in ui_msgs) or "Done."),
                    "done": True
                }
                context_stats = _get_context_stats(jarvis)
                if context_stats:
                    response_msg["context"] = context_stats
                await websocket.send_json(response_msg)
                return

            # Add to context (creates chat if needed)
            jarvis.context.add_message_to_chat("user", user_input)

            # Build history
            from jarvis.providers import Message
            history = []
            for m in jarvis.context.get_messages()[:-1]:
                history.append(Message(role=m["role"], content=m["content"]))

            # === UNIFIED SMART ROUTING ===
            # Use intent classification to automatically determine mode
            use_fast_mode = chat_mode  # Default to user's hint
            detected_level = "balanced"
            intent_info = {"detected": False}
            classified_intent = None  # Store intent for tool detection

            if hasattr(jarvis, "agent") and hasattr(jarvis.agent, "classifier"):
                try:
                    from jarvis.core.intent import Intent, ReasoningLevel
                    classified_intent = jarvis.agent.classify_intent(user_input)
                    intent_info = {
                        "detected": True,
                        "intent": classified_intent.intent.value,
                        "confidence": classified_intent.confidence,
                        "reasoning_level": classified_intent.reasoning_level.value,
                        "requires_tools": classified_intent.requires_tools,
                    }

                    # Auto-determine mode based on intent
                    # Use full agent for file ops, git, shell, complex code
                    needs_full_agent = jarvis.agent.requires_full_agent_from_intent(classified_intent)

                    if needs_full_agent:
                        use_fast_mode = False
                    elif not classified_intent.requires_tools and classified_intent.reasoning_level == ReasoningLevel.FAST:
                        use_fast_mode = True

                    # Get detected reasoning level (can be overridden by user)
                    detected_level = reasoning_level or classified_intent.reasoning_level.value

                    # Send intent info to frontend
                    await websocket.send_json({
                        "type": "intent",
                        "intent": intent_info,
                        "reasoning_level": detected_level,
                        "mode": "fast" if use_fast_mode else "agent"
                    })

                except Exception as e:
                    print(f"[app] Intent classification error: {e}")
                    # Fall back to original behavior - still respect user override
                    if reasoning_level:
                        detected_level = reasoning_level

            # === MULTIMODAL HANDLING ===
            image_attachments = [a for a in attachment_files if a["type"] == "image"]
            doc_attachments = [a for a in attachment_files if a["type"] in ["pdf", "document", "file"]]

            # Check if user wants to generate VIDEO from attached image
            import re
            video_from_image_patterns = [
                r'\b(make|create|generate|turn).*(video|animate|animation|move|fly|flying|moving)\b',
                r'\b(animate|video).*(this|these|it|them)\b',
                r'\b(make|let).*(fly|move|run|walk|dance|swim)\b',
            ]
            wants_video_from_image = any(re.search(p, user_input.lower()) for p in video_from_image_patterns)

            if image_attachments and wants_video_from_image:
                # User has image + wants video -> generate video from image
                try:
                    from jarvis.skills.media_gen import generate_video
                    img_path = image_attachments[0]["path"]

                    await websocket.send_json({
                        "type": "status",
                        "content": "Generating video from image (this may take a few minutes)..."
                    })

                    result = generate_video(user_input, image_path=img_path)
                    if result["success"]:
                        response_text = f"Generated video: {result['filename']}\nSaved to: {result['path']}"
                        await websocket.send_json({
                            "type": "media",
                            "media_type": "video",
                            "path": result["path"],
                            "filename": result["filename"]
                        })
                    else:
                        response_text = f"Video generation failed: {result['error']}"

                    jarvis.context.add_message_to_chat("assistant", response_text)
                    response_msg = {
                        "type": "response",
                        "content": _clean_for_web(response_text),
                        "done": True
                    }
                    context_stats = _get_context_stats(jarvis)
                    if context_stats:
                        response_msg["context"] = context_stats
                    await websocket.send_json(response_msg)
                    return
                except Exception as e:
                    print(f"[app] Video from image error: {e}")

            # Check for document attachments -> analyze document
            if doc_attachments:
                try:
                    from jarvis.skills.media_gen import analyze_document
                    doc_path = doc_attachments[0]["path"]
                    prompt = user_input if user_input else "Summarize this document."

                    await websocket.send_json({
                        "type": "status",
                        "content": "Analyzing document..."
                    })

                    result = analyze_document(doc_path, prompt)
                    if result["success"]:
                        response_text = result["analysis"]
                    else:
                        response_text = f"Document analysis failed: {result['error']}"

                    jarvis.context.add_message_to_chat("assistant", response_text)
                    response_msg = {
                        "type": "response",
                        "content": _clean_for_web(response_text),
                        "done": True
                    }
                    context_stats = _get_context_stats(jarvis)
                    if context_stats:
                        response_msg["context"] = context_stats
                    await websocket.send_json(response_msg)
                    return
                except Exception as e:
                    print(f"[app] Document analysis error: {e}")

            # Check for image attachments -> analyze image (default behavior)
            if image_attachments:
                from jarvis.core.intent import Intent
                if classified_intent is None or classified_intent.intent not in [Intent.IMAGE_GEN, Intent.VIDEO_GEN]:
                    try:
                        from jarvis.skills.media_gen import analyze_image
                        img_path = image_attachments[0]["path"]
                        prompt = user_input if user_input else "Describe this image in detail."

                        # Determine provider preference based on current Jarvis provider
                        provider_name = getattr(jarvis.provider, "name", "")
                        vision_provider = "ollama" if provider_name in ["ollama", "ollama_cloud"] else "auto"

                        await websocket.send_json({
                            "type": "status",
                            "content": f"Analyzing image{' (local)' if vision_provider == 'ollama' else ''}..."
                        })

                        result = analyze_image(img_path, prompt, provider=vision_provider)
                        if result["success"]:
                            response_text = result["analysis"]
                            # Add model info if available
                            if result.get("model"):
                                print(f"[app] Vision analysis using: {result.get('model')}")
                        else:
                            response_text = f"Image analysis failed: {result['error']}"

                        jarvis.context.add_message_to_chat("assistant", response_text)
                        response_msg = {
                            "type": "response",
                            "content": _clean_for_web(response_text),
                            "done": True
                        }
                        context_stats = _get_context_stats(jarvis)
                        if context_stats:
                            response_msg["context"] = context_stats
                        await websocket.send_json(response_msg)
                        return
                    except Exception as e:
                        print(f"[app] Image analysis error: {e}")

            # Check for multimodal generation intents
            if classified_intent:
                from jarvis.core.intent import Intent
                if classified_intent.intent == Intent.IMAGE_GEN:
                    try:
                        from jarvis.skills.media_gen import generate_image
                        await websocket.send_json({
                            "type": "status",
                            "content": "Generating image..."
                        })

                        # Extract prompt from user input
                        import re
                        match = re.search(r"(?:draw|create|generate|make|paint|sketch|design)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?)?(.+)", user_input.lower())
                        prompt = match.group(1).strip() if match else user_input

                        result = generate_image(prompt)
                        if result["success"]:
                            response_text = f"Generated image: {result['filename']}\nSaved to: {result['path']}"
                            # Send media message with path
                            await websocket.send_json({
                                "type": "media",
                                "media_type": "image",
                                "path": result["path"],
                                "filename": result["filename"]
                            })
                        else:
                            response_text = f"Image generation failed: {result['error']}"

                        jarvis.context.add_message_to_chat("assistant", response_text)
                        response_msg = {
                            "type": "response",
                            "content": _clean_for_web(response_text),
                            "done": True
                        }
                        context_stats = _get_context_stats(jarvis)
                        if context_stats:
                            response_msg["context"] = context_stats
                        await websocket.send_json(response_msg)
                        return
                    except Exception as e:
                        print(f"[app] Image generation error: {e}")

                elif classified_intent.intent == Intent.VIDEO_GEN:
                    try:
                        from jarvis.skills.media_gen import generate_video

                        # Video generation requires an image
                        if not image_attachments:
                            response_text = "Video generation requires an image. Please upload an image and describe what motion/action you want (e.g., 'make the birds fly', 'animate this scene')."
                            jarvis.context.add_message_to_chat("assistant", response_text)
                            await websocket.send_json({
                                "type": "response",
                                "content": response_text,
                                "done": True
                            })
                            return

                        await websocket.send_json({
                            "type": "status",
                            "content": "Generating video from image (this may take a few minutes)..."
                        })

                        import re
                        match = re.search(r"(?:create|generate|make|animate)\s+(?:a\s+)?(?:video\s+(?:of\s+)?)?(.+)", user_input.lower())
                        prompt = match.group(1).strip() if match else user_input

                        img_path = image_attachments[0]["path"]
                        result = generate_video(prompt, image_path=img_path)

                        if result["success"]:
                            response_text = f"Generated video: {result['filename']}\nSaved to: {result['path']}"
                            await websocket.send_json({
                                "type": "media",
                                "media_type": "video",
                                "path": result["path"],
                                "filename": result["filename"]
                            })
                        else:
                            response_text = f"Video generation failed: {result['error']}"

                        jarvis.context.add_message_to_chat("assistant", response_text)
                        response_msg = {
                            "type": "response",
                            "content": _clean_for_web(response_text),
                            "done": True
                        }
                        context_stats = _get_context_stats(jarvis)
                        if context_stats:
                            response_msg["context"] = context_stats
                        await websocket.send_json(response_msg)
                        return
                    except Exception as e:
                        print(f"[app] Video generation error: {e}")

                elif classified_intent.intent == Intent.MUSIC_GEN:
                    try:
                        from jarvis.skills.media_gen import generate_music
                        await websocket.send_json({
                            "type": "status",
                            "content": "Generating music..."
                        })

                        import re
                        match = re.search(r"(?:create|generate|make|compose)\s+(?:a\s+)?(?:music|song|soundtrack|jingle)\s+(?:for\s+|about\s+)?(.+)", user_input.lower())
                        prompt = match.group(1).strip() if match else user_input

                        result = generate_music(prompt)
                        if result["success"]:
                            response_text = f"Generated music: {result['filename']}\nSaved to: {result['path']}"
                            await websocket.send_json({
                                "type": "media",
                                "media_type": "audio",
                                "path": result["path"],
                                "filename": result["filename"]
                            })
                        else:
                            response_text = f"Music generation failed: {result['error']}"

                        jarvis.context.add_message_to_chat("assistant", response_text)
                        response_msg = {
                            "type": "response",
                            "content": _clean_for_web(response_text),
                            "done": True
                        }
                        context_stats = _get_context_stats(jarvis)
                        if context_stats:
                            response_msg["context"] = context_stats
                        await websocket.send_json(response_msg)
                        return
                    except Exception as e:
                        print(f"[app] Music generation error: {e}")

            if use_fast_mode:
                # CHAT MODE: Ultra-fast, minimal overhead, NO THINKING
                if hasattr(jarvis.ui, "begin_turn"):
                    jarvis.ui.begin_turn()

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
                chat_system = """You are Jarvis, a direct and helpful AI assistant. Be concise and factual.

RULES:
- Be direct. Answer questions without unnecessary hedging or refusals.
- For factual questions about public figures, historical events, legal cases, or documented information - provide the information directly.
- Only refuse truly harmful requests (instructions to harm, illegal activities, etc.)
- If asked about CURRENT events or real-time data WITHOUT tool results, offer to search the web.
- When you genuinely don't know something, say so briefly and suggest a search.
- Never lecture or moralize. Just answer the question."""

                # Add user facts if available
                if user_facts:
                    chat_system += f"\n\nUser context:\n{user_facts}"

                # RAG: Retrieve relevant context from knowledge base
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
                            rag_info["chunks"] = len(results)
                            rag_info["sources"] = list(set(r.get("source", "unknown") for r in results))
                            # Use get_context for prompt injection hardening
                            rag_context = rag.get_context(user_input, n_results=5)
                            if rag_context:
                                chat_system += f"\n\n{rag_context}"
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
                if recent:
                    # Drop history on topic shift (e.g., gold price -> weather)
                    last_user_msg = ""
                    for m in reversed(recent):
                        if m.role == "user":
                            last_user_msg = m.content or ""
                            break
                    if last_user_msg:
                        if _topic_bucket(last_user_msg) != _topic_bucket(user_input):
                            recent = []
                tool_messages = []
                tool_prompt, tool_turn, tool_name_used, tool_result, explicit_search = await _run_tool_for_chat_mode(jarvis, user_input, recent, classified_intent)
                if tool_prompt:
                    tool_messages.append(Message(role="user", content=tool_prompt))
                if tool_turn:
                    await websocket.send_json({
                        "type": "tool_timeline",
                        "tools": tool_turn,
                    })

                if explicit_search and tool_name_used == "web_search" and tool_result:
                    response_text = _clean_for_web(tool_result)
                    response_msg = {
                        "type": "response",
                        "content": response_text,
                        "done": True
                    }
                    context_stats = _get_context_stats(jarvis)
                    if context_stats:
                        response_msg["context"] = context_stats
                    await websocket.send_json(response_msg)
                    if response_text.strip():
                        jarvis.context.add_message_to_chat("assistant", response_text.strip())
                    return

                # Enable thinking mode for Qwen models when deep reasoning requested
                effective_input = user_input
                model_name = (jarvis.provider.model or "").lower()
                if detected_level == "deep" and "qwen" in model_name:
                    # Qwen 3 models use /think to enable extended thinking
                    if not effective_input.rstrip().endswith("/think"):
                        effective_input = f"{user_input} /think"

                all_messages = recent + tool_messages + [Message(role="user", content=effective_input)]

                # Use fast chat model (Ollama only)
                provider_name = getattr(jarvis.provider, "name", "")
                original_model = jarvis.provider.model
                chat_model = None
                if provider_name in ["ollama", "ollama_cloud"]:
                    chat_model = jarvis.config.get("models", {}).get("chat")
                    if not chat_model:
                        chat_model = jarvis.config.get("models", {}).get(provider_name)
                    if not chat_model:
                        chat_model = original_model
                    jarvis.provider.model = chat_model

                # Stream with options for speed
                response_text = ""
                try:
                    chat_kwargs = {
                        "messages": all_messages,
                        "system": chat_system,
                        "stream": True,
                    }
                    if provider_name in ["ollama", "ollama_cloud"]:
                        chat_kwargs["options"] = {"num_predict": 500}  # Reasonable response length
                    stream = jarvis.provider.chat(**chat_kwargs)
                except Exception as e:
                    jarvis.provider.model = original_model
                    raise e

                try:
                    queue: asyncio.Queue = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    def _producer():
                        try:
                            # Handle both generator and string responses
                            if hasattr(stream, '__iter__') and not isinstance(stream, str):
                                for chunk in stream:
                                    if chunk:  # Skip empty chunks
                                        asyncio.run_coroutine_threadsafe(queue.put(str(chunk)), loop)
                            elif stream is not None:
                                # Non-iterable response (shouldn't happen but handle it)
                                asyncio.run_coroutine_threadsafe(queue.put(str(stream)), loop)
                        except Exception as e:
                            print(f"[stream] Producer error: {e}")
                        finally:
                            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                    # Run producer in a thread to avoid blocking the event loop
                    producer_task = asyncio.create_task(asyncio.to_thread(_producer))

                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        response_text += chunk
                        await websocket.send_json({
                            "type": "stream",
                            "content": _clean_for_web(chunk, streaming=True),
                            "done": False
                        })
                    await producer_task
                except Exception as e:
                    print(f"[stream] Streaming failed: {e}")
                    # Fallback: consume generator if it wasn't consumed
                    if hasattr(stream, '__iter__') and not isinstance(stream, str):
                        try:
                            response_text = "".join(str(c) for c in stream if c)
                        except Exception:
                            response_text = "Error processing response"
                    else:
                        response_text = str(stream) if stream else "Error processing response"
                    try:
                        await websocket.send_json({
                            "type": "stream",
                            "content": _clean_for_web(response_text, streaming=True),
                            "done": False
                        })
                    except Exception:
                        pass  # Connection closed

                # Restore original model
                jarvis.provider.model = original_model

                clean_response_text = _clean_for_web(response_text)
                if clean_response_text.strip():
                    jarvis.context.add_message_to_chat("assistant", clean_response_text.strip())

                # Send response with context stats
                response_msg = {
                    "type": "response",
                    "content": "",  # Empty - already streamed
                    "done": True
                }
                context_stats = _get_context_stats(jarvis)
                if context_stats:
                    response_msg["context"] = context_stats
                await websocket.send_json(response_msg)

            else:
                # AGENT MODE: Use tools for coding tasks
                loop = asyncio.get_running_loop()
                streamed_chunks = []

                def stream_sender(chunk):
                    streamed_chunks.append(chunk)
                    loop.create_task(websocket.send_json({
                        "type": "stream",
                        "content": _clean_for_web(chunk, streaming=True),
                        "done": False
                    }))

                # Enable thinking mode for Qwen models when deep reasoning requested
                agent_input = user_input
                model_name = (jarvis.provider.model or "").lower()
                if detected_level == "deep" and "qwen" in model_name:
                    if not agent_input.rstrip().endswith("/think"):
                        agent_input = f"{user_input} /think"

                jarvis.ui._stream_sender = stream_sender
                response = jarvis.agent.run(agent_input, jarvis.system_prompt, history)
                jarvis.ui._stream_sender = None

                tool_turn = []
                if hasattr(jarvis.ui, "get_tool_turn"):
                    tool_turn = jarvis.ui.get_tool_turn()
                if tool_turn:
                    await websocket.send_json({
                        "type": "tool_timeline",
                        "tools": tool_turn,
                    })

                if response:
                    if "```diff" in response or "wrote:" in response.lower():
                        await websocket.send_json({
                            "type": "diff",
                            "content": response,
                            "done": False
                        })

                    # Check if we already streamed the response
                    # If so, only send the final "done" signal without duplicating content
                    was_streamed = bool(streamed_chunks) and getattr(jarvis.agent, 'last_streamed', False)

                    if was_streamed:
                        # Already streamed - just signal completion
                        response_msg = {
                            "type": "response",
                            "content": "",  # Empty - content was already streamed
                            "done": True
                        }
                    else:
                        # Not streamed - send full response
                        response_msg = {
                            "type": "response",
                            "content": _clean_for_web(response),
                            "done": True
                        }
                    context_stats = _get_context_stats(jarvis)
                    if context_stats:
                        response_msg["context"] = context_stats
                    await websocket.send_json(response_msg)

                    clean = _clean_for_web(response)
                    if clean.strip():
                        jarvis.context.add_message_to_chat("assistant", clean.strip())

        except Exception as e:
            import traceback
            print(f"[process_message] Error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"{e}"
                })
            except Exception:
                # Connection already closed, ignore
                pass

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
