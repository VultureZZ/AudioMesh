"""
Music cue generation service for podcast production mode.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Literal, Optional

from .music_generator import music_generator

logger = logging.getLogger(__name__)

CueType = Literal["intro", "outro", "transition", "bed"]
StyleType = Literal["tech_talk", "casual", "news", "storytelling"]


class PodcastMusicService:
    """Generates podcast music cues through the existing ACE-Step pipeline."""

    _PROMPT_PRESETS: Dict[tuple[CueType, str], str] = {
        ("intro", "tech_talk"): "upbeat modern podcast intro, energetic synth, 12 seconds, broadcast quality",
        ("intro", "casual"): "friendly podcast intro music, warm acoustic groove, 12 seconds, clean mix",
        ("intro", "news"): "concise authoritative news podcast intro, modern pulse, 10-12 seconds, broadcast quality",
        ("intro", "storytelling"): "cinematic storytelling podcast intro, emotional but subtle, 12 seconds",
        ("bed", "tech_talk"): "modern ambient podcast bed, subtle rhythmic synth texture, low intensity, loop-friendly",
        ("bed", "casual"): "warm ambient background music, subtle, looping, conversational podcast",
        ("bed", "news"): "neutral newsroom background bed, steady pulse, subtle, loop-friendly",
        ("bed", "storytelling"): "soft cinematic underscore for spoken storytelling, unobtrusive, loop-friendly",
        ("transition", "any"): "short punchy podcast transition hit, 2-3 seconds",
        ("outro", "tech_talk"): "closing podcast music, warm fade out, 10 seconds",
        ("outro", "casual"): "friendly outro music, gentle fade out, 10 seconds, conversational tone",
        ("outro", "news"): "clean news podcast outro sting, authoritative close, 8-10 seconds",
        ("outro", "storytelling"): "cinematic closing podcast music, warm fade out, 10 seconds",
    }

    def resolve_prompt(self, cue_type: CueType, style: StyleType) -> str:
        if cue_type == "transition":
            return self._PROMPT_PRESETS[("transition", "any")]
        return self._PROMPT_PRESETS.get((cue_type, style)) or self._PROMPT_PRESETS[(cue_type, "casual")]

    def generate_cue(
        self,
        cue_type: CueType,
        style: StyleType,
        *,
        timeout_seconds: int = 300,
        poll_interval_seconds: float = 2.5,
    ) -> str:
        """
        Submit and wait for a cue generation task.

        Returns:
            Absolute file path to generated cue audio.
        """
        prompt = self.resolve_prompt(cue_type, style)
        logger.info("Generating podcast cue: cue_type=%s, style=%s", cue_type, style)
        task_id = music_generator.generate_music(
            {
                "caption": prompt,
                "lyrics": "",
                "duration": 12 if cue_type == "intro" else 10 if cue_type == "outro" else 3 if cue_type == "transition" else 45,
                "instrumental": True,
                "thinking": False,
                "batch_size": 1,
                "audio_format": "wav",
            }
        )
        return self.wait_for_task(task_id, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds)

    def wait_for_task(
        self,
        task_id: str,
        *,
        timeout_seconds: int = 300,
        poll_interval_seconds: float = 2.5,
    ) -> str:
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for cue task {task_id}")

            status = music_generator.get_status(task_id)
            state = status.get("status")
            if state == "succeeded":
                metadata = status.get("metadata") or []
                if not metadata:
                    raise RuntimeError(f"Cue task {task_id} succeeded but returned no metadata")
                file_path = metadata[0].get("file_path")
                if not file_path:
                    raise RuntimeError(f"Cue task {task_id} succeeded but returned no file path")
                path = Path(file_path)
                if not path.exists():
                    raise RuntimeError(f"Cue output file missing for task {task_id}: {path}")
                return str(path)
            if state == "failed":
                raise RuntimeError(status.get("error") or f"Cue task {task_id} failed")

            time.sleep(poll_interval_seconds)

    def health_check(self) -> Dict:
        """Proxy music system health for production orchestration."""
        return music_generator.health()


podcast_music_service = PodcastMusicService()

