"""
Audio transcription service (speech-to-text) for voice profiling.

Uses `faster-whisper` to transcribe an audio file into text. This enables
style-oriented voice profiling (cadence/tone/vocabulary) based on what was said.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str] = None


class AudioTranscriber:
    """
    Wrap `faster-whisper` in a small, lazy-loading service.

    Notes:
    - Requires `faster-whisper` dependency.
    - Requires ffmpeg available on PATH for decoding most formats.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Audio transcription requires the 'faster-whisper' package. "
                "Install it with: pip install faster-whisper"
            ) from e

        logger.info(
            "Loading faster-whisper model",
            extra={
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
            },
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> TranscriptionResult:
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        # We keep options conservative to avoid long transcriptions.
        # The caller can trim/choose different model sizes if needed later.
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
        )

        parts: list[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                parts.append(text)

        transcript = " ".join(parts).strip()
        detected_lang = getattr(info, "language", None)
        return TranscriptionResult(text=transcript, language=detected_lang)


# Global instance
audio_transcriber = AudioTranscriber()

