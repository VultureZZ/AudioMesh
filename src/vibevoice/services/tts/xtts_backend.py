"""
XTTS (Coqui TTS) backend stub.

Implement using Coqui TTS API when needed; same segment-and-concatenate
pattern as Qwen3Backend. Voice refs: speaker_wav path or cached speaker ID.
"""
from pathlib import Path
from typing import List

from .base import SpeakerRef, TTSBackend
from .segments import TranscriptSegment


class XTTSBackend(TTSBackend):
    """TTS backend using Coqui XTTS. Not yet implemented."""

    def generate(
        self,
        segments: List[TranscriptSegment],
        speaker_refs: List[SpeakerRef],
        language: str,
        output_path: Path,
    ) -> Path:
        raise NotImplementedError(
            "XTTS backend is not implemented. Set TTS_BACKEND=qwen3 or vibevoice."
        )
