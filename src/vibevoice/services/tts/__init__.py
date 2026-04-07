"""
TTS backend abstraction and implementations.

Provides a unified interface for text-to-speech generation across
Qwen3-TTS, and optionally XTTS and Bark.
"""
from .base import SpeakerRef, TTSBackend
from .segments import TranscriptSegment
from .qwen3_backend import Qwen3Backend
from .segments import parse_transcript_into_segments

__all__ = [
    "SpeakerRef",
    "TranscriptSegment",
    "TTSBackend",
    "Qwen3Backend",
    "parse_transcript_into_segments",
]
