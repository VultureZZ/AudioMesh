"""
Podcast generation service that orchestrates article scraping, script generation, and audio creation.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from .article_scraper import article_scraper
from .ollama_client import ollama_client
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
        duration: str,
        voices: List[str],
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> str:
        """
        Generate podcast script from article URL.

        Args:
            url: Article URL to scrape
            genre: Podcast genre (Comedy, Serious, News, etc.)
            duration: Target duration (5 min, 10 min, 15 min, 30 min)
            voices: List of voice names (1-4 voices)
            ollama_url: Optional custom Ollama server URL
            ollama_model: Optional custom Ollama model name

        Returns:
            Generated podcast script with speaker labels

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If scraping or script generation fails
        """
        if not voices or len(voices) == 0:
            raise ValueError("At least one voice is required")
        if len(voices) > 4:
            raise ValueError("Maximum 4 voices allowed")

        num_voices = len(voices)

        logger.info(f"Generating podcast script: url={url}, genre={genre}, duration={duration}, voices={num_voices}")

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
            )
        else:
            script = self.ollama.generate_script(
                article_text,
                genre,
                duration,
                num_voices,
                voice_profiles=voice_profiles if voice_profiles else None,
                voice_names=voices,
            )

        logger.info(f"Generated script: {len(script)} characters")
        return script

    def generate_audio(
        self,
        script: str,
        voices: List[str],
    ) -> str:
        """
        Generate audio from podcast script.

        Args:
            script: Podcast script with speaker labels
            voices: List of voice names (mapped to speakers in order)

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

        logger.info(f"Generating audio from script: {len(script)} characters, {len(voices)} voices")

        # Format script to ensure proper speaker mapping
        formatted_script = self._format_script_for_voices(script, voices)

        # Generate audio using existing voice generator
        output_path = self.voice_gen.generate_speech(
            transcript=formatted_script,
            speakers=voices,
        )

        logger.info(f"Audio generated: {output_path}")
        return str(output_path)

    def generate_script_segments(
        self,
        script: str,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build production cue segment structure with Ollama and fallback parsing.
        """
        if not script or not script.strip():
            return []

        try:
            if ollama_url or ollama_model:
                from .ollama_client import OllamaClient

                custom_client = OllamaClient(base_url=ollama_url, model=ollama_model)
                return custom_client.generate_script_segments(script)
            return self.ollama.generate_script_segments(script)
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
        segments: List[Dict] = [{"segment_type": "intro_music", "start_time_hint": 0.0}]
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
            segments.append(
                {
                    "segment_type": "dialogue",
                    "speaker": speaker,
                    "text": text,
                    "start_time_hint": round(current_time, 2),
                }
            )
            word_count = max(len(text.split()), 1)
            current_time += max((word_count / 2.6), 1.0)

        if len(segments) > 3:
            midpoint = round(current_time / 2.0, 2)
            segments.insert(2, {"segment_type": "transition_sting", "start_time_hint": midpoint})
        segments.append({"segment_type": "outro_music", "start_time_hint": round(max(current_time - 2.0, 0.0), 2)})
        return segments


# Global podcast generator instance
podcast_generator = PodcastGenerator()
