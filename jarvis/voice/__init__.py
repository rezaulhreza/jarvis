"""
Jarvis Voice Input Module

Uses Whisper for speech-to-text and system TTS for responses.
"""

from .voice_mode import run_voice_mode

__all__ = ["run_voice_mode"]
