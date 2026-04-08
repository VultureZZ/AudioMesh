"""
PyAnnote diarization service and transcript speaker assignment.
"""
from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from ...config import config

logger = logging.getLogger(__name__)

# Reduce log noise from optional stack pieces we do not rely on:
# - torchcodec: pyannote warns on import even when inputs are preloaded waveform dicts (see _audio_file_to_pyannote_input).
# - pooling std: benign for very short internal segments during diarization.
# - speechbrain: torchaudio backend deprecation inside their utils.
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings(
    "ignore",
    message=".*degrees of freedom is <= 0.*",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"speechbrain\.utils\.torch_audio_backend")


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


def as_pyannote_annotation(diarization: Any) -> Any:
    """
    Normalize pipeline output to an object with itertracks (pyannote.core.Annotation).
    pyannote.audio 4.x returns DiarizeOutput; 3.x returns Annotation directly.
    """
    if diarization is None:
        raise ValueError("diarization is None")
    if hasattr(diarization, "itertracks"):
        return diarization
    ann = getattr(diarization, "speaker_diarization", None)
    if ann is not None and hasattr(ann, "itertracks"):
        return ann
    for name in ("diarization", "annotation"):
        ann = getattr(diarization, name, None)
        if ann is not None and hasattr(ann, "itertracks"):
            return ann
    raise TypeError(
        f"Unsupported diarization result {type(diarization)!r}: "
        "expected Annotation or DiarizeOutput with speaker_diarization"
    )


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
        raw = pipeline(audio_in)
        return as_pyannote_annotation(raw)

    async def run(self, audio_path: str):
        pipeline = self._load_pipeline()
        logger.info("Running diarization (in-memory waveform): %s", audio_path)
        return await asyncio.to_thread(self._run_pipeline_on_file, pipeline, audio_path)

    async def assign_speakers(self, aligned_transcript: dict[str, Any], diarization: Any) -> list[dict[str, Any]]:
        diarization = as_pyannote_annotation(diarization)
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

