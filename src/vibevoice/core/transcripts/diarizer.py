"""
PyAnnote diarization service and transcript speaker assignment.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config import config

logger = logging.getLogger(__name__)


def _audio_file_to_pyannote_input(audio_path: str) -> dict[str, Any]:
    """
    Build the in-memory input pyannote accepts when file-based decoding is broken
    (e.g. TorchCodec/AudioDecoder missing — NameError in pyannote.audio.core.io).
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # data: (frames, channels) -> waveform: (channel, time)
    if data.shape[1] == 1:
        waveform = torch.from_numpy(data[:, 0]).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T.copy())
    return {"waveform": waveform, "sample_rate": int(sr)}


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
        hf_token = (config.HF_TOKEN or "").strip()
        # huggingface_hub renamed use_auth_token -> token; newer pyannote rejects the old name.
        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        return self._pipeline

    def _run_pipeline_on_file(self, pipeline: Any, audio_path: str) -> Any:
        """Run diarization using preloaded waveform to avoid pyannote file I/O / torchcodec issues."""
        audio_in = _audio_file_to_pyannote_input(audio_path)
        return pipeline(audio_in)

    async def run(self, audio_path: str):
        pipeline = self._load_pipeline()
        logger.info("Running diarization (in-memory waveform): %s", audio_path)
        return await asyncio.to_thread(self._run_pipeline_on_file, pipeline, audio_path)

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

