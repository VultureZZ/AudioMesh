"""
Timing alignment helpers for podcast production mode.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from ..core.transcripts.transcriber import transcript_transcriber

logger = logging.getLogger(__name__)


class PodcastTimingService:
    """Derives dialogue timing hints from rendered voice audio using WhisperX alignment."""

    _speaker_line_pattern = re.compile(r"^(Speaker\s+\d+):\s*(.+)$", re.IGNORECASE)

    def parse_dialogue_lines(self, script: str) -> List[Dict]:
        dialogue: List[Dict] = []
        for raw in script.split("\n"):
            line = raw.strip()
            if not line:
                continue
            match = self._speaker_line_pattern.match(line)
            if not match:
                continue
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            if not text:
                continue
            dialogue.append({"speaker": speaker, "text": text})
        return dialogue

    async def build_dialogue_timing(self, script: str, voice_path: str) -> List[Dict]:
        dialogue = self.parse_dialogue_lines(script)
        if not dialogue:
            return []

        try:
            transcript = await transcript_transcriber.transcribe(voice_path, language="en")
            aligned = await transcript_transcriber.align(transcript, voice_path)
            aligned_segments = aligned.get("segments", []) if isinstance(aligned, dict) else []
            text_segments = [seg for seg in aligned_segments if isinstance(seg, dict) and str(seg.get("text", "")).strip()]

            if not text_segments:
                raise RuntimeError("No aligned segments available for timing")

            timing: List[Dict] = []
            pointer = 0
            for line in dialogue:
                seg = text_segments[min(pointer, len(text_segments) - 1)]
                start = max(float(seg.get("start", 0.0)), 0.0)
                end = max(float(seg.get("end", start + 1.0)), start + 0.1)
                timing.append(
                    {
                        "speaker": line["speaker"],
                        "text": line["text"],
                        "start_time_hint": round(start, 2),
                        "duration_ms": int((end - start) * 1000),
                    }
                )
                pointer += 1
            return timing
        except Exception as exc:
            logger.warning("WhisperX timing alignment unavailable, using fallback timing: %s", exc)
            return self._fallback_timing(dialogue)

    def _fallback_timing(self, dialogue: List[Dict]) -> List[Dict]:
        current = 2.0
        out: List[Dict] = []
        for line in dialogue:
            text = line.get("text", "")
            words = max(len(text.split()), 1)
            duration_seconds = max(words / 2.6, 1.0)
            out.append(
                {
                    "speaker": line.get("speaker", "Speaker 1"),
                    "text": text,
                    "start_time_hint": round(current, 2),
                    "duration_ms": int(duration_seconds * 1000),
                }
            )
            current += duration_seconds
        return out


podcast_timing_service = PodcastTimingService()

