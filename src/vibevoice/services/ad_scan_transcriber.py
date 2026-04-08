"""
Transcription for podcast ad scanning.

Uses faster-whisper by default so we do not import WhisperX's Pyannote VAD stack
(which pulls torchcodec and triggers FFmpeg/libavutil warnings on many servers).
Model name comes from TRANSCRIPT_WHISPER_MODEL (same as transcript settings).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import config

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required for AD_SCAN_TRANSCRIBE_BACKEND=faster_whisper. "
            "Install faster-whisper or set AD_SCAN_TRANSCRIBE_BACKEND=whisperx."
        ) from exc

    logger.info(
        "Loading faster-whisper for ad scan",
        extra={
            "model": config.TRANSCRIPT_WHISPER_MODEL,
            "device": config.AD_SCAN_WHISPER_DEVICE,
            "compute_type": config.AD_SCAN_WHISPER_COMPUTE_TYPE,
        },
    )
    _model = WhisperModel(
        config.TRANSCRIPT_WHISPER_MODEL,
        device=config.AD_SCAN_WHISPER_DEVICE,
        compute_type=config.AD_SCAN_WHISPER_COMPUTE_TYPE,
    )
    return _model


def transcribe_for_ad_scan(audio_path: str, language: Optional[str] = None) -> dict[str, Any]:
    """
    Return a Whisper-like dict: { "segments": [ {start, end, text}, ... ], "language": ... }.
    """
    model = _get_model()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
    )
    out_segments: list[dict[str, Any]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        out_segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )
    detected = getattr(info, "language", None)
    return {"segments": out_segments, "language": detected}
