"""
Voice input mode for Jarvis

Uses Whisper for speech recognition and system TTS for output.
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_voice_mode():
    """Run Jarvis in voice input mode."""
    # Check dependencies
    try:
        import whisper
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install jarvis-ai-assistant[voice]")
        sys.exit(1)

    from ..assistant import Jarvis

    console.print(Panel.fit(
        "[bold blue]JARVIS[/bold blue] - Voice Mode\n"
        "[dim]Press Enter to speak, Ctrl+C to exit[/dim]",
        border_style="blue"
    ))

    # Load Whisper model
    console.print("[dim]Loading Whisper model...[/dim]")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    console.print("[green]✓ Whisper loaded[/green]")

    # Initialize Jarvis
    try:
        jarvis = Jarvis()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)

    console.print(f"[dim]Persona: {jarvis.current_persona} | Model: {jarvis.ollama.default_model}[/dim]\n")

    sample_rate = 16000
    duration = 5  # seconds to record

    while True:
        try:
            input("\n[Press Enter to speak, then talk for 5 seconds...]")
            console.print("[cyan]🎤 Listening...[/cyan]")

            # Record audio
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()

            console.print("[dim]Processing speech...[/dim]")

            # Transcribe with Whisper
            audio_data = audio.flatten()
            result = model.transcribe(audio_data, fp16=False)
            text = result["text"].strip()

            if not text:
                console.print("[yellow]No speech detected. Try again.[/yellow]")
                continue

            console.print(f"\n[bold green]You said:[/bold green] {text}")

            # Check for exit commands
            if text.lower() in ["exit", "quit", "goodbye", "bye"]:
                speak("Goodbye!")
                console.print("[yellow]Goodbye![/yellow]")
                break

            # Process with Jarvis
            console.print("\n[bold blue]Jarvis:[/bold blue]", end=" ")
            response = jarvis.process(text)

            # Speak the response (if not too long)
            if len(response) < 500:
                speak(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def speak(text: str):
    """Use system TTS to speak text."""
    if sys.platform == "darwin":
        # macOS: use 'say' command
        subprocess.run(["say", text], capture_output=True)
    elif sys.platform == "linux":
        # Linux: try espeak
        try:
            subprocess.run(["espeak", text], capture_output=True)
        except FileNotFoundError:
            pass  # espeak not installed
    # Windows: could use pyttsx3 or powershell


def record_audio_continuous(sample_rate: int = 16000, silence_threshold: float = 0.01,
                            silence_duration: float = 1.5, max_duration: float = 30.0):
    """
    Record audio until silence is detected.

    Args:
        sample_rate: Audio sample rate
        silence_threshold: RMS threshold for silence
        silence_duration: Seconds of silence to stop recording
        max_duration: Maximum recording duration

    Returns:
        numpy array of audio data
    """
    import numpy as np
    import sounddevice as sd

    console.print("[cyan]🎤 Listening... (speak now, will stop on silence)[/cyan]")

    chunk_duration = 0.1  # seconds per chunk
    chunk_samples = int(sample_rate * chunk_duration)
    silence_chunks = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)

    audio_chunks = []
    silent_count = 0
    has_speech = False

    for _ in range(max_chunks):
        chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()

        rms = np.sqrt(np.mean(chunk**2))
        audio_chunks.append(chunk)

        if rms > silence_threshold:
            has_speech = True
            silent_count = 0
        else:
            silent_count += 1

        # Stop if we've had speech and then silence
        if has_speech and silent_count >= silence_chunks:
            break

    return np.concatenate(audio_chunks).flatten() if audio_chunks else np.array([])


if __name__ == "__main__":
    run_voice_mode()
