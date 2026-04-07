"""
Per-speaker audio extraction service.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from ...config import config


class TranscriptAudioExtractor:
    async def extract_all(
        self,
        audio_path: str,
        speaker_ids: list[str],
        diarization: Any,
        transcript_id: str,
    ) -> dict[str, str]:
        return await asyncio.to_thread(
            self._extract_all_sync,
            audio_path,
            speaker_ids,
            diarization,
            transcript_id,
        )

    def _extract_all_sync(
        self,
        audio_path: str,
        speaker_ids: list[str],
        diarization: Any,
        transcript_id: str,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        if not config.TRANSCRIPT_EXTRACT_SPEAKER_AUDIO:
            return out

        src = AudioSegment.from_file(audio_path)
        segments_dir = config.TRANSCRIPTS_DIR / "segments" / transcript_id
        segments_dir.mkdir(parents=True, exist_ok=True)
        min_ms = int(config.TRANSCRIPT_MIN_SEGMENT_DURATION_SECONDS * 1000)

        for speaker_id in speaker_ids:
            combined = AudioSegment.silent(duration=0)
            for segment, _, label in diarization.itertracks(yield_label=True):
                if str(label) != speaker_id:
                    continue
                start_ms = int(float(segment.start) * 1000)
                end_ms = int(float(segment.end) * 1000)
                if end_ms - start_ms < min_ms:
                    continue
                combined += src[start_ms:end_ms] + AudioSegment.silent(duration=120)

            if len(combined) == 0:
                continue

            combined = combined.set_frame_rate(24000).set_channels(1)
            out_path = segments_dir / f"{speaker_id}.wav"
            combined.export(str(out_path), format="wav", parameters=["-ar", "24000"])
            out[speaker_id] = str(out_path)

        return out


transcript_audio_extractor = TranscriptAudioExtractor()

