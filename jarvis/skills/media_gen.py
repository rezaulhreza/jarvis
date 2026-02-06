"""Media generation skills - image, video, music via Chutes AI."""

import os
import base64
import httpx
from pathlib import Path
from typing import Optional
from datetime import datetime


def _get_chutes_key() -> Optional[str]:
    """Get Chutes API key from credentials or environment."""
    try:
        from jarvis.auth.credentials import get_credential
        key = get_credential("chutes", "api_key")
        if key:
            return key
    except ImportError:
        pass
    return os.environ.get("CHUTES_API_KEY")


def _get_output_dir() -> Path:
    """Get output directory for generated media."""
    from jarvis import get_data_dir
    output_dir = get_data_dir() / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _auto_cleanup(max_age_hours: int = 48, max_files: int = 100):
    """Auto-cleanup old generated files (runs in background, non-blocking)."""
    import time
    try:
        output_dir = _get_output_dir()
        files = list(output_dir.iterdir())

        # Cleanup by count (keep only last N files)
        if len(files) > max_files:
            files_sorted = sorted(files, key=lambda f: f.stat().st_mtime)
            for f in files_sorted[:-max_files]:
                try:
                    f.unlink()
                except Exception:
                    pass

        # Cleanup by age
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        for f in output_dir.iterdir():
            if f.is_file():
                age = now - f.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        f.unlink()
                    except Exception:
                        pass
    except Exception:
        pass  # Silently ignore cleanup errors


def generate_image(
    prompt: str,
    model: str = "FLUX.1-schnell",
    width: int = 1024,
    height: int = 1024,
    steps: int = 10
) -> dict:
    """
    Generate an image from a text prompt using Chutes AI.

    Args:
        prompt: Text description of the image to generate
        model: Model to use (FLUX.1-schnell, FLUX.1-dev, etc.)
        width: Image width (default 1024, max 2048)
        height: Image height (default 1024, max 2048)
        steps: Number of inference steps (default 10, max 30)

    Returns:
        Dict with success status, file path, and preview URL
    """
    # Auto-cleanup old files
    _auto_cleanup()

    api_key = _get_chutes_key()
    if not api_key:
        return {"success": False, "error": "Chutes API key not configured. Set CHUTES_API_KEY."}

    # Normalize model name
    model_mapping = {
        "flux-schnell": "FLUX.1-schnell",
        "flux-dev": "FLUX.1-dev",
        "schnell": "FLUX.1-schnell",
        "dev": "FLUX.1-dev",
    }
    model_id = model_mapping.get(model.lower(), model)

    try:
        with httpx.Client(timeout=120.0) as client:
            # Chutes image API returns raw JPEG bytes
            response = client.post(
                "https://image.chutes.ai/generate",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "width": min(width, 2048),
                    "height": min(height, 2048),
                    "num_inference_steps": min(steps, 30),
                    "guidance_scale": 7.5
                }
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                content_length = len(response.content)
                print(f"[ImageGen] Response: {response.status_code}, Content-Type: {content_type}, Size: {content_length} bytes")

                # If we got substantial data, save it as an image
                if content_length > 1000:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Detect format from content-type or default to jpg
                    if "png" in content_type:
                        ext = "png"
                    elif "webp" in content_type:
                        ext = "webp"
                    else:
                        ext = "jpg"
                    filename = f"image_{timestamp}.{ext}"
                    output_path = _get_output_dir() / filename

                    output_path.write_bytes(response.content)
                    print(f"[ImageGen] Saved to: {output_path}")

                    return {
                        "success": True,
                        "path": str(output_path),
                        "filename": filename,
                        "prompt": prompt,
                        "model": model_id,
                        "size": f"{width}x{height}"
                    }

                return {"success": False, "error": f"Response too small ({content_length} bytes). Content-Type: {content_type}"}
            else:
                error_msg = response.text[:300] if response.text else f"HTTP {response.status_code}"
                return {"success": False, "error": f"API error ({response.status_code}): {error_msg}"}

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out (image generation can take up to 2 minutes)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_video(
    prompt: str,
    image_path: Optional[str] = None,
    duration: Optional[float] = None,
    frames: Optional[int] = None,
    fps: int = 16,
    resolution: str = "480p",
    fast: bool = True
) -> dict:
    """
    Generate a video from text or image using Chutes AI (Wan 2.2 I2V).

    If no image is provided, generates one first using FLUX.1, then converts to video.

    Args:
        prompt: Text description of what should happen in the video
        image_path: Path to source image (optional - will generate if not provided)
        duration: Duration in seconds (default ~5s). Max ~5.8s at 24fps or ~8.7s at 16fps.
        frames: Number of frames (21-140). If not set, calculated from duration.
        fps: Frames per second (16-24, default 24 for smooth playback)
        resolution: "720p" or "480p" (default "720p")
        fast: Use fast mode (default True)

    Returns:
        Dict with success status and file path
    """
    import re as regex

    # Auto-cleanup old files
    _auto_cleanup()

    api_key = _get_chutes_key()
    if not api_key:
        return {"success": False, "error": "Chutes API key not configured. Set CHUTES_API_KEY."}

    # Parse duration from prompt if not explicitly provided (e.g., "8 seconds", "10 sec", "5s")
    prompt_lower = prompt.lower()
    if duration is None:
        duration_match = regex.search(r'(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b', prompt_lower)
        if duration_match:
            duration = float(duration_match.group(1))
            print(f"[VideoGen] Detected duration from prompt: {duration}s")

    # Detect resolution from prompt - default 480p, upgrade to 720p if user asks for higher
    if resolution == "480p":  # Only override if still at default
        high_res_patterns = [r'\b720p\b', r'\bhd\b', r'\bhigh\s*(quality|res|resolution)\b', r'\bbetter\s*quality\b']
        for pattern in high_res_patterns:
            if regex.search(pattern, prompt_lower):
                resolution = "720p"
                print(f"[VideoGen] Detected high resolution request: 720p")
                break

    # Detect fps from prompt - default 16fps, upgrade to 24fps if user asks for smoother
    if fps == 16:  # Only override if still at default
        smooth_patterns = [r'\b24\s*fps\b', r'\bsmooth\b', r'\bcinematic\b', r'\bhigh\s*fps\b']
        for pattern in smooth_patterns:
            if regex.search(pattern, prompt_lower):
                fps = 24
                print(f"[VideoGen] Detected smooth fps request: 24fps")
                break

    # Clamp fps to valid range
    fps = min(max(fps, 16), 24)

    # Calculate frames from duration, or use default
    # API limit: 21-140 frames
    # At 24fps: max ~5.8s, at 16fps: max ~8.7s
    if frames is None:
        if duration is not None:
            frames = int(duration * fps)
            # If requested duration exceeds max at current fps, try lower fps
            if frames > 140 and fps > 16:
                fps = 16
                frames = int(duration * fps)
                print(f"[VideoGen] Lowered fps to 16 to accommodate {duration}s duration")
        else:
            frames = 81  # Default ~5s at 16fps or ~3.4s at 24fps

    # Clamp frames to valid range
    frames = min(max(frames, 21), 140)
    actual_duration = frames / fps
    print(f"[VideoGen] Final: {frames} frames at {fps}fps = {actual_duration:.1f}s")

    # If no image provided, generate one first (two-step T2V workaround)
    generated_image = False
    if not image_path:
        print(f"[VideoGen] No image provided. Generating base image from prompt...")

        # Create an image prompt optimized for video generation
        image_prompt = f"cinematic still frame, {prompt}, high quality, detailed, suitable for animation"

        # Determine image dimensions based on resolution
        if resolution == "720p":
            img_width, img_height = 1280, 720
        else:
            img_width, img_height = 854, 480

        image_result = generate_image(
            prompt=image_prompt,
            model="FLUX.1-schnell",
            width=img_width,
            height=img_height,
            steps=10
        )

        if not image_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to generate base image: {image_result.get('error', 'Unknown error')}"
            }

        image_path = image_result["path"]
        generated_image = True
        print(f"[VideoGen] Base image generated: {image_path}")

    img_path = Path(image_path)
    if not img_path.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    try:
        # Encode image to base64
        img_data = base64.b64encode(img_path.read_bytes()).decode()

        # Build payload for Wan 2.2 I2V
        payload = {
            "prompt": prompt,
            "image": img_data,
            "frames": frames,
            "fps": fps,
            "resolution": resolution,
            "fast": fast,
        }

        print(f"[VideoGen] Payload: frames={frames}, fps={fps}, resolution={resolution}, fast={fast}")
        print(f"[VideoGen] Generating video from image: {img_path.name}, {frames} frames @ {fps}fps, {resolution}")

        with httpx.Client(timeout=600.0) as client:  # 10 min timeout for video
            response = client.post(
                "https://chutes-wan-2-2-i2v-14b-fast.chutes.ai/generate",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            content_type = response.headers.get("content-type", "")
            content_length = len(response.content)
            print(f"[VideoGen] Response: {response.status_code}, Content-Type: {content_type}, Size: {content_length} bytes")

            if response.status_code == 200:
                # Check if response is video data (raw bytes)
                if content_length > 10000:  # Video should be at least 10KB
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"video_{timestamp}.mp4"
                    output_path = _get_output_dir() / filename
                    output_path.write_bytes(response.content)
                    print(f"[VideoGen] Saved to: {output_path}")

                    return {
                        "success": True,
                        "path": str(output_path),
                        "filename": filename,
                        "prompt": prompt,
                        "frames": frames,
                        "fps": fps,
                        "duration": round(frames / fps, 1),
                        "resolution": resolution,
                        "source_image": image_path,
                        "image_generated": generated_image
                    }

                # Try to parse as JSON (might contain URL or error)
                try:
                    data = response.json()
                    if "url" in data:
                        video_response = client.get(data["url"], timeout=120.0)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"video_{timestamp}.mp4"
                        output_path = _get_output_dir() / filename
                        output_path.write_bytes(video_response.content)
                        return {
                            "success": True,
                            "path": str(output_path),
                            "filename": filename,
                            "prompt": prompt,
                            "frames": frames,
                            "fps": fps,
                            "duration": round(frames / fps, 1),
                            "resolution": resolution,
                            "source_image": image_path,
                            "image_generated": generated_image
                        }
                    elif "error" in data:
                        return {"success": False, "error": data["error"]}
                except Exception:
                    pass

                return {"success": False, "error": f"Unexpected response format. Size: {content_length} bytes"}
            else:
                error_msg = response.text[:300] if response.text else f"HTTP {response.status_code}"
                return {"success": False, "error": f"API error ({response.status_code}): {error_msg}"}

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out (video generation can take several minutes)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_music(
    prompt: str,
    duration: float = 30.0,
    model: str = "diffrhythm"
) -> dict:
    """
    Generate music from a text prompt using Chutes AI.

    Args:
        prompt: Text description of the music to generate
        duration: Duration in seconds (default 30)
        model: Model to use (diffrhythm)

    Returns:
        Dict with success status and file path
    """
    # Auto-cleanup old files
    _auto_cleanup()

    api_key = _get_chutes_key()
    if not api_key:
        return {"success": False, "error": "Chutes API key not configured. Set CHUTES_API_KEY."}

    models = {
        "diffrhythm": "ASLP-lab/DiffRhythm",
    }
    model_id = models.get(model, model)

    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                "https://api.chutes.ai/v1/audio/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "duration": duration,
                }
            )

            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    audio_data = data["data"][0]

                    if "url" in audio_data:
                        audio_response = client.get(audio_data["url"])
                        audio_bytes = audio_response.content
                    elif "b64_json" in audio_data:
                        audio_bytes = base64.b64decode(audio_data["b64_json"])
                    else:
                        return {"success": False, "error": "No audio data in response"}

                    # Save to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"music_{timestamp}.mp3"
                    output_path = _get_output_dir() / filename
                    output_path.write_bytes(audio_bytes)

                    return {
                        "success": True,
                        "path": str(output_path),
                        "filename": filename,
                        "prompt": prompt,
                        "duration": duration,
                        "model": model
                    }

                return {"success": False, "error": "No audio data in response"}
            else:
                error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                return {"success": False, "error": f"API error: {error_msg}"}

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out (music generation can take a few minutes)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_image_ollama(image_path: str, prompt: str = "Describe this image in detail.") -> dict:
    """
    Analyze an image using local Ollama vision model.

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image

    Returns:
        Dict with success status and analysis result
    """
    import time

    img_path = Path(image_path)
    if not img_path.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    print(f"[Vision] Starting image analysis: {image_path}")

    try:
        from jarvis.providers.ollama import OllamaProvider
        provider = OllamaProvider()

        # Check if Ollama is running and has a vision model
        vision_model = provider.get_vision_model()
        if not vision_model:
            print("[Vision] No Ollama vision model found")
            return {"success": False, "error": "No Ollama vision model found. Install with: ollama pull llava"}

        print(f"[Vision] Using model: {vision_model}")
        start_time = time.time()

        # Analyze the image
        result = provider.vision(str(img_path), prompt, model=vision_model)

        elapsed = time.time() - start_time
        print(f"[Vision] Analysis complete in {elapsed:.1f}s")

        return {
            "success": True,
            "analysis": result,
            "image": image_path,
            "prompt": prompt,
            "model": vision_model,
            "provider": "ollama"
        }

    except Exception as e:
        print(f"[Vision] Error: {e}")
        return {"success": False, "error": f"Ollama vision error: {e}"}


def analyze_image(image_path: str, prompt: str = "Describe this image in detail.", provider: str = "auto") -> dict:
    """
    Analyze an image using vision model (Chutes cloud or Ollama local).

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image
        provider: "chutes", "ollama", or "auto" (tries Ollama first if available)

    Returns:
        Dict with success status and analysis result
    """
    # Auto-detect: try Ollama first (local, free), fall back to Chutes
    if provider == "auto":
        ollama_result = analyze_image_ollama(image_path, prompt)
        if ollama_result["success"]:
            return ollama_result
        # Fall through to Chutes if Ollama failed
        provider = "chutes"

    if provider == "ollama":
        return analyze_image_ollama(image_path, prompt)

    # Chutes provider
    api_key = _get_chutes_key()
    if not api_key:
        return {"success": False, "error": "Chutes API key not configured. Set CHUTES_API_KEY."}

    img_path = Path(image_path)
    if not img_path.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    try:
        # Read and encode image
        img_data = base64.b64encode(img_path.read_bytes()).decode()

        # Detect mime type
        suffix = img_path.suffix.lower()
        mime_types = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp"}
        mime = mime_types.get(suffix, "jpeg")

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://llm.chutes.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{mime};base64,{img_data}"}
                            }
                        ]
                    }],
                    "max_tokens": 1024
                }
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "success": True,
                    "analysis": content,
                    "image": image_path,
                    "prompt": prompt
                }
            else:
                error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                return {"success": False, "error": f"API error: {error_msg}"}

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_document(doc_path: str, prompt: str = "Summarize this document.") -> dict:
    """
    Analyze a document (PDF, DOCX, XLSX, TXT) using Chutes vision/LLM.

    Args:
        doc_path: Path to the document file
        prompt: Question or instruction about the document

    Returns:
        Dict with success status and analysis result
    """
    api_key = _get_chutes_key()
    if not api_key:
        return {"success": False, "error": "Chutes API key not configured. Set CHUTES_API_KEY."}

    doc = Path(doc_path)
    if not doc.exists():
        return {"success": False, "error": f"Document not found: {doc_path}"}

    suffix = doc.suffix.lower()

    try:
        # Extract text based on file type
        text_content = ""

        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF
                pdf = fitz.open(doc_path)
                for page in pdf:
                    text_content += page.get_text() + "\n"
                pdf.close()
            except ImportError:
                # Fallback: try pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(doc_path) as pdf:
                        for page in pdf.pages:
                            text_content += (page.extract_text() or "") + "\n"
                except ImportError:
                    return {"success": False, "error": "PDF support requires PyMuPDF or pdfplumber. Install with: pip install pymupdf"}

        elif suffix in [".docx", ".doc"]:
            try:
                from docx import Document
                document = Document(doc_path)
                text_content = "\n".join([para.text for para in document.paragraphs])
            except ImportError:
                return {"success": False, "error": "DOCX support requires python-docx. Install with: pip install python-docx"}

        elif suffix in [".xlsx", ".xls"]:
            try:
                import pandas as pd
                df = pd.read_excel(doc_path)
                text_content = f"Excel file with {len(df)} rows and {len(df.columns)} columns.\n\n"
                text_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                text_content += "First 20 rows:\n"
                text_content += df.head(20).to_string()
            except ImportError:
                return {"success": False, "error": "Excel support requires pandas and openpyxl. Install with: pip install pandas openpyxl"}

        elif suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(doc_path)
                text_content = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n\n"
                text_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                text_content += "First 20 rows:\n"
                text_content += df.head(20).to_string()
            except ImportError:
                return {"success": False, "error": "CSV support requires pandas. Install with: pip install pandas"}

        elif suffix in [".txt", ".md", ".json", ".yaml", ".yml"]:
            text_content = doc.read_text(errors="ignore")

        else:
            return {"success": False, "error": f"Unsupported file type: {suffix}"}

        # Truncate if too long
        if len(text_content) > 15000:
            text_content = text_content[:15000] + "\n\n[... content truncated ...]"

        if not text_content.strip():
            return {"success": False, "error": "Could not extract text from document"}

        # Send to LLM for analysis
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://llm.chutes.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen3-32B",
                    "messages": [{
                        "role": "user",
                        "content": f"Document content:\n\n{text_content}\n\n---\n\nUser question: {prompt}"
                    }],
                    "max_tokens": 2048
                }
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "success": True,
                    "analysis": content,
                    "document": doc_path,
                    "prompt": prompt,
                    "file_type": suffix
                }
            else:
                error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                return {"success": False, "error": f"API error: {error_msg}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def cleanup_generated_files(max_age_hours: int = 24) -> dict:
    """
    Clean up old generated media files.

    Args:
        max_age_hours: Delete files older than this many hours

    Returns:
        Dict with count of deleted files
    """
    import time

    output_dir = _get_output_dir()
    if not output_dir.exists():
        return {"success": True, "deleted": 0}

    deleted = 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for f in output_dir.iterdir():
        if f.is_file():
            age = now - f.stat().st_mtime
            if age > max_age_seconds:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass

    return {"success": True, "deleted": deleted, "max_age_hours": max_age_hours}


def list_media_models() -> dict:
    """List available media generation models."""
    models = {
        "image_generation": {
            "provider": "chutes",
            "models": {
                "flux-schnell": "Fast, high quality (recommended)",
                "flux-dev": "Higher quality, slower",
                "sdxl": "Stable Diffusion XL",
                "hidream": "Artistic style",
                "juggernaut": "Photorealistic",
                "dreamshaper": "Versatile"
            }
        },
        "video_generation": {
            "provider": "chutes",
            "models": {
                "wan-i2v": "Wan2.2 Image-to-Video (direct)",
                "wan-t2v": "Text-to-Video (generates image first, then animates)"
            },
            "note": "T2V uses FLUX.1 to generate base image, then Wan I2V to animate"
        },
        "music_generation": {
            "provider": "chutes",
            "models": {
                "diffrhythm": "Text to music"
            }
        },
        "vision": {
            "chutes": {
                "qwen-vl": "Qwen2.5-VL-72B (cloud, high quality)"
            },
            "ollama": {
                "llama3.2-vision": "Best quality local",
                "llava-llama3": "Good quality local",
                "llava": "Classic vision model",
                "moondream": "Fast/small local"
            }
        },
        "document": {
            "supported": ["PDF", "DOCX", "XLSX", "CSV", "TXT", "MD", "JSON", "YAML"]
        }
    }

    # Check for available Ollama vision models
    try:
        from jarvis.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        vision_model = provider.get_vision_model()
        if vision_model:
            models["vision"]["ollama_available"] = vision_model
    except Exception:
        pass

    return models
