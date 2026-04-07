"""
Bark (Suno) backend stub.

Implement using Bark/suno API when needed; map voices to Bark presets
or cloned voices. More generative (speech, effects); different API.
"""
from pathlib import Path
from typing import List

from .base import SpeakerRef, TTSBackend
from .segments import TranscriptSegment


class BarkBackend(TTSBackend):
    """TTS backend using Suno Bark. Not yet implemented."""

    def generate(
        self,
        segments: List[TranscriptSegment],
        speaker_refs: List[SpeakerRef],
        language: str,
        output_path: Path,
    ) -> Path:
        raise NotImplementedError(
            "Bark backend is not implemented. Set TTS_BACKEND=qwen3 or vibevoice."
        )
