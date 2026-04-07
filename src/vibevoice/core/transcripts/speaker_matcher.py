"""
Speaker embedding extraction and voice-library matching.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydub import AudioSegment

from ...config import config
from ...models.voice_storage import voice_storage

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class TranscriptSpeakerMatcher:
    """SpeechBrain ECAPA embedding matcher."""

    def __init__(self) -> None:
        self._model: Any = None

    def _load_encoder(self):
        if self._model is not None:
            return self._model
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "speechbrain is required for speaker matching. Install dependencies from requirements.txt."
            ) from exc
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.PROJECT_ROOT / "models" / "speechbrain-ecapa"),
        )
        return self._model

    def _embedding_from_file(self, audio_path: Path) -> np.ndarray:
        model = self._load_encoder()
        emb = model.encode_batch(str(audio_path))
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        emb_arr = np.asarray(emb).reshape(-1)
        return emb_arr.astype(np.float32)

    async def extract_embedding(self, audio_path: str, speaker_id: str, diarization: Any) -> np.ndarray:
        return await asyncio.to_thread(
            self._extract_embedding_sync,
            audio_path,
            speaker_id,
            diarization,
        )

    def _extract_embedding_sync(self, audio_path: str, speaker_id: str, diarization: Any) -> np.ndarray:
        source = AudioSegment.from_file(audio_path)
        collected = AudioSegment.silent(duration=0)
        min_seg_ms = int(config.TRANSCRIPT_MIN_SEGMENT_DURATION_SECONDS * 1000)
        for segment, _, label in diarization.itertracks(yield_label=True):
            if str(label) != speaker_id:
                continue
            start_ms = int(float(segment.start) * 1000)
            end_ms = int(float(segment.end) * 1000)
            if end_ms - start_ms < min_seg_ms:
                continue
            clip = source[start_ms:end_ms]
            collected += clip + AudioSegment.silent(duration=120)

        if len(collected) == 0:
            return np.zeros(0, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            collected = collected.set_frame_rate(16000).set_channels(1)
            collected.export(str(tmp_path), format="wav")
            return self._embedding_from_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def match_against_library(self, embedding: np.ndarray) -> Optional[tuple[str, float]]:
        if embedding.size == 0:
            return None
        threshold = config.TRANSCRIPT_SPEAKER_MATCH_THRESHOLD
        best_voice_id: Optional[str] = None
        best_score = -1.0
        for voice in voice_storage.list_voices():
            emb = voice.get("speaker_embedding")
            if not emb:
                continue
            try:
                candidate = np.asarray(emb, dtype=np.float32)
                score = _cosine_similarity(embedding, candidate)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_voice_id = voice.get("id")

        if best_voice_id is None or best_score < threshold:
            return None
        return best_voice_id, best_score

    async def match_all(self, speaker_ids: list[str], audio_path: str, diarization: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for speaker_id in speaker_ids:
            try:
                embedding = await self.extract_embedding(audio_path, speaker_id, diarization)
                matched = await self.match_against_library(embedding)
                if matched is None:
                    out.append({"speaker_id": speaker_id, "voice_id": None, "confidence": None})
                else:
                    voice_id, confidence = matched
                    out.append({"speaker_id": speaker_id, "voice_id": voice_id, "confidence": confidence})
            except Exception as exc:
                logger.warning("Speaker matching failed for %s: %s", speaker_id, exc)
                out.append({"speaker_id": speaker_id, "voice_id": None, "confidence": None})
        return out


def compute_file_embedding(audio_path: Path) -> Optional[list[float]]:
    """
    Helper used by voice creation to store reference embeddings.
    Returns None when extraction is unavailable/fails.
    """
    matcher = TranscriptSpeakerMatcher()
    try:
        emb = matcher._embedding_from_file(audio_path)
        if emb.size == 0:
            return None
        return emb.tolist()
    except Exception as exc:
        logger.warning("Could not compute speaker embedding for %s: %s", audio_path, exc)
        return None


transcript_speaker_matcher = TranscriptSpeakerMatcher()

