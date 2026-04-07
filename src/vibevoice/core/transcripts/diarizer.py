"""
PyAnnote diarization service and transcript speaker assignment.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config import config

logger = logging.getLogger(__name__)


class TranscriptDiarizer:
    """Speaker diarization + assignment helper."""

    def __init__(self) -> None:
        self._pipeline: Any = None

    def _load_pyannote(self):
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Transcript service requires pyannote.audio. Install dependencies from requirements.txt."
            ) from exc
        return Pipeline

    def _load_whisperx(self):
        try:
            import whisperx  # type: ignore
        except Exception as exc:
            raise RuntimeError("WhisperX is required for speaker assignment.") from exc
        return whisperx

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not config.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is required for speaker diarization. "
                "Set HF_TOKEN and accept pyannote model terms on HuggingFace."
            )
        Pipeline = self._load_pyannote()
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.HF_TOKEN,
        )
        return self._pipeline

    async def run(self, audio_path: str):
        pipeline = self._load_pipeline()
        logger.info("Running diarization: %s", audio_path)
        return await asyncio.to_thread(pipeline, audio_path)

    async def assign_speakers(self, aligned_transcript: dict[str, Any], diarization: Any) -> list[dict[str, Any]]:
        whisperx = self._load_whisperx()
        enriched = await asyncio.to_thread(whisperx.assign_word_speakers, diarization, aligned_transcript)

        segments_out: list[dict[str, Any]] = []
        for segment in enriched.get("segments", []):
            speaker_id = segment.get("speaker") or "SPEAKER_00"
            start_ms = int(float(segment.get("start", 0.0)) * 1000)
            end_ms = int(float(segment.get("end", 0.0)) * 1000)
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            confidence = 0.0
            words = segment.get("words") or []
            confidences = [float(w.get("score", 0.0)) for w in words if isinstance(w, dict) and "score" in w]
            if confidences:
                confidence = sum(confidences) / len(confidences)
            segments_out.append(
                {
                    "speaker_id": speaker_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "confidence": confidence,
                }
            )
        return segments_out


transcript_diarizer = TranscriptDiarizer()

