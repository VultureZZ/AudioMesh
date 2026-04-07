"""
WhisperX transcription/alignment service for transcript processing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ...config import config

logger = logging.getLogger(__name__)


class TranscriptTranscriber:
    """Wrapper around WhisperX with lazy model loading."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._align_model: Optional[Any] = None
        self._align_metadata: Optional[dict[str, Any]] = None
        self._device = "cuda"
        self._compute_type = "float16"

    def _load_whisperx(self):
        try:
            import whisperx  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Transcript service requires whisperx. Install dependencies from requirements.txt."
            ) from exc
        return whisperx

    def _load_model(self):
        if self._model is not None:
            return self._model
        whisperx = self._load_whisperx()
        self._model = whisperx.load_model(
            config.TRANSCRIPT_WHISPER_MODEL,
            self._device,
            compute_type=self._compute_type,
        )
        return self._model

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict[str, Any]:
        model = self._load_model()
        logger.info("Transcribing audio with WhisperX: %s", audio_path)
        result = await asyncio.to_thread(
            model.transcribe,
            audio_path,
            language=language,
        )
        if "segments" not in result:
            result["segments"] = []
        return result

    async def align(self, transcript: dict[str, Any], audio_path: str) -> dict[str, Any]:
        whisperx = self._load_whisperx()
        language = transcript.get("language") or "en"
        if self._align_model is None or self._align_metadata is None:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self._device,
            )
            self._align_model = align_model
            self._align_metadata = align_metadata

        # WhisperX versions differ in defaults around interpolation.
        # Explicitly set interpolate_method to avoid pandas ValueError
        # ("method should be a string, not None") seen in some envs.
        try:
            return await asyncio.to_thread(
                whisperx.align,
                transcript.get("segments", []),
                self._align_model,
                self._align_metadata,
                audio_path,
                self._device,
                return_char_alignments=False,
                interpolate_method="nearest",
            )
        except TypeError:
            # Older whisperx versions may not support interpolate_method kwarg.
            # Retry with a minimal argument set.
            try:
                return await asyncio.to_thread(
                    whisperx.align,
                    transcript.get("segments", []),
                    self._align_model,
                    self._align_metadata,
                    audio_path,
                    self._device,
                    False,
                )
            except ValueError as exc:
                if "method" not in str(exc):
                    raise
                logger.warning("WhisperX align fallback hit pandas method=None bug; returning raw transcript.")
                return transcript
        except ValueError as exc:
            if "method" not in str(exc):
                raise
            # Final defensive retry for version-mismatch edge cases.
            try:
                return await asyncio.to_thread(
                    whisperx.align,
                    transcript.get("segments", []),
                    self._align_model,
                    self._align_metadata,
                    audio_path,
                    self._device,
                    return_char_alignments=False,
                    interpolate_method="linear",
                )
            except Exception:
                logger.warning("WhisperX align failed after retries; returning raw transcript.")
                return transcript


transcript_transcriber = TranscriptTranscriber()

