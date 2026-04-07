"""
Podcast generation service that orchestrates article scraping, script generation, and audio creation.
"""
import logging
from typing import List, Optional

from .article_scraper import article_scraper
from .ollama_client import ollama_client
from .voice_generator import voice_generator

logger = logging.getLogger(__name__)


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
            # Get voice data to find voice_id
            voice_data = voice_manager.get_voice_by_name(voice_name)
            if voice_data and voice_data.get("type") == "custom":
                voice_id = voice_data.get("id")
                if voice_id:
                    profile = voice_storage.get_voice_profile(voice_id)
                    if profile:
                        voice_profiles[voice_name] = profile
                        logger.info(f"Loaded profile for voice: {voice_name}")
            else:
                # Try direct lookup by name as ID
                profile = voice_storage.get_voice_profile(voice_name)
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


# Global podcast generator instance
podcast_generator = PodcastGenerator()
