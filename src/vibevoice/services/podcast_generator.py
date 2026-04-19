"""
Podcast generation service that orchestrates article scraping, script generation, and audio creation.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from .article_scraper import article_scraper
from .ollama_client import (
    estimated_duration_seconds_for_segmentation,
    infer_num_voices_from_script,
    normalize_podcast_speaker_labels,
    ollama_client,
)
from .voice_generator import voice_generator

logger = logging.getLogger(__name__)


def _resolve_voice_profile_for_script(
    voice_name: str,
    voice_storage: Any,
    voice_manager: Any,
) -> Optional[Dict]:
    """
    Load stored profile for a voice: by voice id (all types), embedded metadata, then name fallbacks.
    Keys in the caller's voice_profiles dict must stay as the request's voice_name strings.
    """
    voice_data = voice_manager.get_voice_by_name(voice_name)
    if not voice_data:
        voice_data = voice_manager.get_voice_by_name(voice_manager.normalize_voice_name(voice_name))

    profile: Optional[Dict] = None
    if voice_data:
        voice_id = voice_data.get("id")
        if voice_id:
            profile = voice_storage.get_voice_profile(voice_id)
        embedded = voice_data.get("profile")
        if not profile and isinstance(embedded, dict) and embedded:
            profile = dict(embedded)

    if not profile:
        canonical = voice_manager.normalize_voice_name(voice_name)
        profile = voice_storage.get_voice_profile(canonical) or voice_storage.get_voice_profile(voice_name)

    return profile


def production_style_to_genre_style(style: Optional[str]) -> Optional[str]:
    """Map production music ``style`` to a human-readable tone line for segmentation prompts."""
    if not style:
        return None
    return {
        "tech_talk": "Technical, clear, structured",
        "casual": "Conversational, relaxed",
        "news": "Journalistic, neutral",
        "storytelling": "Narrative, engaging",
    }.get(style, style)


_PRODUCTION_CUE_LINE_ONLY = re.compile(r"^\s*\[CUE:\s*[^\]]+\]\s*$", re.IGNORECASE)
_INLINE_PRODUCTION_CUE = re.compile(r"\s*\[CUE:\s*[^\]]+\]\s*", re.IGNORECASE)


def strip_production_cue_markers(script: str) -> str:
    """
    Remove ``[CUE: ...]`` production tokens before TTS.

    Standalone cue lines are dropped. Inline cues are stripped from speaker lines.
    Segmentation should run on the original script (with cues); voice rendering uses this output.
    """
    if not script or not script.strip():
        return script
    out: List[str] = []
    for raw in script.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _PRODUCTION_CUE_LINE_ONLY.match(line):
            continue
        cleaned = _INLINE_PRODUCTION_CUE.sub(" ", line).strip()
        cleaned = re.sub(r" +", " ", cleaned)
        if cleaned:
            out.append(cleaned)
    return "\n".join(out).strip()


def _validate_script_duration_inputs(duration: Optional[str], approximate_duration_minutes: Optional[float]) -> None:
    has_d = duration is not None and str(duration).strip() != ""
    has_m = approximate_duration_minutes is not None
    if not has_d and not has_m:
        raise ValueError("Provide either duration (e.g. '10 min') or approximate_duration_minutes")


class PodcastGenerator:
    """Service for generating podcasts from articles."""

    def __init__(self):
        """Initialize podcast generator."""
        self.scraper = article_scraper
        self.ollama = ollama_client
        self.voice_gen = voice_generator

    def generate_script(
        self,
        url: str,
        genre: str,
        duration: Optional[str],
        voices: List[str],
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        approximate_duration_minutes: Optional[float] = None,
        include_production_cues: bool = False,
    ) -> str:
        """
        Generate podcast script from article URL.

        Args:
            url: Article URL to scrape
            genre: Podcast genre (Comedy, Serious, News, etc.)
            duration: Target duration label (e.g. 5 min, 10 min) when not using approximate_duration_minutes
            voices: List of voice names (1-4 voices)
            ollama_url: Optional custom Ollama server URL
            ollama_model: Optional custom Ollama model name
            approximate_duration_minutes: Optional target length in minutes for word-count targeting

        Returns:
            Generated podcast script with speaker labels

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If scraping or script generation fails
        """
        _validate_script_duration_inputs(duration, approximate_duration_minutes)
        if not voices or len(voices) == 0:
            raise ValueError("At least one voice is required")
        if len(voices) > 4:
            raise ValueError("Maximum 4 voices allowed")

        num_voices = len(voices)

        logger.info(
            "Generating podcast script: url=%s, genre=%s, duration=%s, approx_min=%s, voices=%d",
            url,
            genre,
            duration,
            approximate_duration_minutes,
            num_voices,
        )

        # Step 1: Scrape article
        logger.info("Step 1: Scraping article...")
        article_text = self.scraper.scrape_article(url)
        logger.info(f"Scraped {len(article_text)} characters from article")

        # Step 2: Load voice profiles
        logger.info("Step 2: Loading voice profiles...")
        voice_profiles = {}
        from ..models.voice_storage import voice_storage
        from .voice_manager import voice_manager

        for voice_name in voices:
            profile = _resolve_voice_profile_for_script(voice_name, voice_storage, voice_manager)
            if profile:
                voice_profiles[voice_name] = profile
                logger.info(f"Loaded profile for voice: {voice_name}")

        # Step 3: Generate script with Ollama
        logger.info("Step 3: Generating script with Ollama...")
        if ollama_url or ollama_model:
            # Create temporary client with custom settings
            from .ollama_client import OllamaClient

            custom_client = OllamaClient(base_url=ollama_url, model=ollama_model)
            script = custom_client.generate_script(
                article_text,
                genre,
                duration,
                num_voices,
                voice_profiles=voice_profiles if voice_profiles else None,
                voice_names=voices,
                approximate_duration_minutes=approximate_duration_minutes,
                include_production_cues=include_production_cues,
            )
        else:
            script = self.ollama.generate_script(
                article_text,
                genre,
                duration,
                num_voices,
                voice_profiles=voice_profiles if voice_profiles else None,
                voice_names=voices,
                approximate_duration_minutes=approximate_duration_minutes,
                include_production_cues=include_production_cues,
            )

        logger.info(f"Generated script: {len(script)} characters")
        return script

    def generate_script_from_article(
        self,
        article_text: str,
        genre: str,
        duration: Optional[str],
        voices: List[str],
        narrator_speaker_index: int,
        article_title: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        approximate_duration_minutes: Optional[float] = None,
        include_production_cues: bool = False,
    ) -> str:
        """
        Generate podcast script from raw article text (same pipeline as URL-based generation, without scraping).

        Voice profiles are loaded by name so the LLM can match dialogue style to each registered voice.
        ``narrator_speaker_index`` is 1-based and aligns with ``voices`` order (Speaker 1 = voices[0], etc.).
        """
        _validate_script_duration_inputs(duration, approximate_duration_minutes)
        if not voices or len(voices) == 0:
            raise ValueError("At least one voice is required")
        if len(voices) > 4:
            raise ValueError("Maximum 4 voices allowed")
        body = (article_text or "").strip()
        if not body:
            raise ValueError("article_text cannot be empty")
        if narrator_speaker_index < 1 or narrator_speaker_index > len(voices):
            raise ValueError(
                f"narrator_speaker_index must be between 1 and {len(voices)} (number of voices)"
            )

        num_voices = len(voices)
        combined = body
        if article_title and str(article_title).strip():
            combined = f"**HEADLINE:** {str(article_title).strip()}\n\n{body}"

        logger.info(
            "Generating podcast script from article body: genre=%s, duration=%s, approx_min=%s, voices=%d, narrator=Speaker %d",
            genre,
            duration,
            approximate_duration_minutes,
            num_voices,
            narrator_speaker_index,
        )

        from ..models.voice_storage import voice_storage
        from .voice_manager import voice_manager

        voice_profiles: Dict[str, Dict] = {}
        for voice_name in voices:
            profile = _resolve_voice_profile_for_script(voice_name, voice_storage, voice_manager)
            if profile:
                voice_profiles[voice_name] = profile
                logger.info("Loaded profile for voice: %s", voice_name)

        if ollama_url or ollama_model:
            from .ollama_client import OllamaClient

            custom_client = OllamaClient(base_url=ollama_url, model=ollama_model)
            script = custom_client.generate_script(
                combined,
                genre,
                duration,
                num_voices,
                voice_profiles=voice_profiles if voice_profiles else None,
                voice_names=voices,
                narrator_speaker_index=narrator_speaker_index,
                approximate_duration_minutes=approximate_duration_minutes,
                include_production_cues=include_production_cues,
            )
        else:
            script = self.ollama.generate_script(
                combined,
                genre,
                duration,
                num_voices,
                voice_profiles=voice_profiles if voice_profiles else None,
                voice_names=voices,
                narrator_speaker_index=narrator_speaker_index,
                approximate_duration_minutes=approximate_duration_minutes,
                include_production_cues=include_production_cues,
            )

        logger.info("Generated script from article: %d characters", len(script))
        return script

    def generate_audio(
        self,
        script: str,
        voices: List[str],
        voice_direction: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate audio from podcast script.

        Args:
            script: Podcast script with speaker labels
            voices: List of voice names (mapped to speakers in order)
            voice_direction: Optional per-line prosody (ProductionPlan voice_direction rows)

        Returns:
            Path to generated audio file

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If audio generation fails
        """
        if not script or not script.strip():
            raise ValueError("Script cannot be empty")
        if not voices or len(voices) == 0:
            raise ValueError("At least one voice is required")
        if len(voices) > 4:
            raise ValueError("Maximum 4 voices allowed")

        script = strip_production_cue_markers(script)
        if not script or not script.strip():
            raise ValueError("Script cannot be empty after removing production cue markers")

        script = normalize_podcast_speaker_labels(script, len(voices), include_production_cues=False)

        logger.info(f"Generating audio from script: {len(script)} characters, {len(voices)} voices")

        # Format script to ensure proper speaker mapping
        formatted_script = self._format_script_for_voices(script, voices)

        # Generate audio using existing voice generator
        output_path = self.voice_gen.generate_speech(
            transcript=formatted_script,
            speakers=voices,
            voice_direction=voice_direction,
        )

        logger.info(f"Audio generated: {output_path}")
        return str(output_path)

    def generate_script_segments(
        self,
        script: str,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        *,
        estimated_duration_seconds: Optional[float] = None,
        num_voices: Optional[int] = None,
        genre: Optional[str] = None,
        genre_style: Optional[str] = None,
        duration: Optional[str] = None,
        approximate_duration_minutes: Optional[float] = None,
    ) -> List[Dict]:
        """
        Build production cue segment structure with Ollama and fallback parsing.

        When ``estimated_duration_seconds`` is omitted, it is derived from ``duration`` /
        ``approximate_duration_minutes`` (same targeting as script generation) or from the script text.
        """
        if not script or not script.strip():
            return []

        est = estimated_duration_seconds
        if est is None:
            est = estimated_duration_seconds_for_segmentation(script, duration, approximate_duration_minutes)
        nv = num_voices if num_voices is not None else infer_num_voices_from_script(script)
        g = (genre or "").strip() or None
        gs = (genre_style or "").strip() or None
        if not gs and g:
            gs = g

        seg_kw = dict(
            estimated_duration_seconds=est,
            num_voices=nv,
            genre=g or "General",
            genre_style=gs or "General",
        )

        try:
            if ollama_url or ollama_model:
                from .ollama_client import OllamaClient

                custom_client = OllamaClient(base_url=ollama_url, model=ollama_model)
                return custom_client.generate_script_segments(script, ollama_model, **seg_kw)
            return self.ollama.generate_script_segments(script, None, **seg_kw)
        except Exception as exc:
            logger.warning("Falling back to deterministic script segmentation: %s", exc)
            return self._fallback_segments_from_script(script)

    def _format_script_for_voices(self, script: str, voices: List[str]) -> str:
        """
        Format script to ensure proper speaker-to-voice mapping.

        Args:
            script: Script with speaker labels
            voices: List of voice names

        Returns:
            Formatted script
        """
        lines = script.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line has a speaker label
            speaker_num = None
            for i in range(1, len(voices) + 1):
                if line.startswith(f"Speaker {i}:"):
                    speaker_num = i
                    break

            if speaker_num:
                # Keep the speaker label as-is (voice_generator will map it)
                formatted_lines.append(line)
            else:
                # If no speaker label, assign to first speaker
                if formatted_lines:
                    # Append to previous line if it was a speaker line
                    if formatted_lines and any(
                        formatted_lines[-1].startswith(f"Speaker {i}:") for i in range(1, len(voices) + 1)
                    ):
                        formatted_lines[-1] += " " + line
                        continue
                # Otherwise, assign to Speaker 1
                formatted_lines.append(f"Speaker 1: {line}")

        return "\n".join(formatted_lines)

    def _fallback_segments_from_script(self, script: str) -> List[Dict]:
        """
        Fallback segmentation from speaker lines when Ollama JSON segmentation fails.
        """
        segments: List[Dict] = [
            {
                "segment_type": "intro_music",
                "speaker": None,
                "text": None,
                "start_time_hint": 0.0,
                "duration_hint": 5.0,
                "energy_level": "high",
                "notes": None,
            }
        ]
        current_time = 2.0
        speaker_pattern = re.compile(r"^(Speaker\s+\d+):\s*(.+)$", re.IGNORECASE)

        for raw_line in script.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = speaker_pattern.match(line)
            if not match:
                continue
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            if not text:
                continue
            word_count = max(len(text.split()), 1)
            dur_hint = round((word_count / 140.0) * 60.0, 2)
            segments.append(
                {
                    "segment_type": "dialogue",
                    "speaker": speaker,
                    "text": text,
                    "start_time_hint": round(current_time, 2),
                    "duration_hint": dur_hint,
                    "energy_level": "medium",
                    "notes": None,
                }
            )
            current_time += max((word_count / 2.6), 1.0)

        if len(segments) > 3:
            midpoint = round(current_time / 2.0, 2)
            segments.insert(
                2,
                {
                    "segment_type": "transition_sting",
                    "speaker": None,
                    "text": None,
                    "start_time_hint": midpoint,
                    "duration_hint": 2.0,
                    "energy_level": "medium",
                    "notes": None,
                },
            )
        segments.append(
            {
                "segment_type": "outro_music",
                "speaker": None,
                "text": None,
                "start_time_hint": round(max(current_time - 2.0, 0.0), 2),
                "duration_hint": 8.0,
                "energy_level": "low",
                "notes": None,
            }
        )
        for idx, seg in enumerate(segments, start=1):
            seg["segment_id"] = idx
        return segments


# Global podcast generator instance
podcast_generator = PodcastGenerator()
