"""
Ollama client service for LLM script generation.
"""
import logging
import re
from typing import Dict, List, Optional

import httpx

from ..config import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL (defaults to config)
            model: Ollama model name (defaults to config)
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_MODEL
        self.timeout = 300  # 5 minutes for long-running requests
        self.client = httpx.Client(timeout=self.timeout)

        logger.info(f"OllamaClient initialized: {self.base_url}, model: {self.model}")

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    def generate_script(
        self,
        article_text: str,
        genre: str,
        duration: str,
        num_voices: int,
        custom_model: Optional[str] = None,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate podcast script from article using Ollama.

        Args:
            article_text: Scraped article content
            genre: Podcast genre (Comedy, Serious, News, etc.)
            duration: Target duration (e.g., "5 min", "30 min")
            num_voices: Number of voices/speakers (1-4)
            custom_model: Optional custom model name (overrides default)
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order (maps to Speaker 1, 2, etc.)

        Returns:
            Generated podcast script with speaker labels

        Raises:
            RuntimeError: If Ollama request fails
        """
        model = custom_model or self.model

        # Truncate article if too long (keep first ~8000 chars for context)
        max_article_length = 8000
        if len(article_text) > max_article_length:
            logger.info(f"Truncating article from {len(article_text)} to {max_article_length} characters")
            article_text = article_text[:max_article_length] + "\n[... article truncated ...]"

        # Build prompt for script generation
        prompt = self._build_prompt(article_text, genre, duration, num_voices, voice_profiles, voice_names)

        logger.info(f"Generating script with Ollama (model: {model}, genre: {genre}, duration: {duration})")

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            script = result.get("response", "").strip()

            if not script:
                raise RuntimeError("Ollama returned empty script")

            # Clean up the script
            script = self._clean_script(script, num_voices)

            logger.info(f"Generated script: {len(script)} characters")
            return script

        except httpx.HTTPError as e:
            error_msg = f"Ollama API request failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during script generation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _build_prompt(
        self,
        article_text: str,
        genre: str,
        duration: str,
        num_voices: int,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for Ollama script generation.

        Args:
            article_text: Article content
            genre: Podcast genre
            duration: Target duration
            num_voices: Number of voices
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order

        Returns:
            Formatted prompt string
        """
        # Map duration to approximate word count (without "words" suffix)
        duration_map = {
            "5 min": "500-700",
            "10 min": "1000-1400",
            "15 min": "1500-2100",
            "30 min": "3000-4200",
        }
        word_target = duration_map.get(duration, "1000-1400")

        # Genre-specific instructions
        genre_instructions = {
            "Comedy": "Make it light-hearted, humorous, and engaging. Include jokes and witty commentary.",
            "Serious": "Keep it professional, informative, and thoughtful. Maintain a serious tone throughout.",
            "News": "Present information clearly and objectively. Use a journalistic style with facts and analysis.",
            "Educational": "Focus on teaching and explaining concepts clearly. Use examples and analogies.",
            "Storytelling": "Narrate the content as a story with engaging descriptions and narrative flow.",
            "Interview": "Format as a conversation with questions and answers. Make it feel like a natural interview.",
            "Documentary": "Present information in a documentary style with detailed explanations and context.",
        }
        genre_style = genre_instructions.get(genre, "Keep it informative and engaging.")

        # Build speaker examples based on num_voices (1-4)
        speaker_examples = []
        for i in range(1, num_voices + 1):
            speaker_examples.append(f"Speaker {i}: [dialogue]")

        # Build voice profiles section if available
        voice_profiles_section = ""
        if voice_profiles and voice_names:
            voice_profiles_section = "\n\n**VOICE PROFILES:**\n"
            for i, voice_name in enumerate(voice_names[:num_voices], 1):
                profile = voice_profiles.get(voice_name)
                if profile:
                    voice_profiles_section += f"\nSpeaker {i} ({voice_name}):\n"
                    if profile.get("cadence"):
                        voice_profiles_section += f"- Cadence: {profile['cadence']}\n"
                    if profile.get("tone"):
                        voice_profiles_section += f"- Tone: {profile['tone']}\n"
                    if profile.get("vocabulary_style"):
                        voice_profiles_section += f"- Vocabulary: {profile['vocabulary_style']}\n"
                    if profile.get("sentence_structure"):
                        voice_profiles_section += f"- Sentence Structure: {profile['sentence_structure']}\n"
                    if profile.get("unique_phrases"):
                        phrases = ", ".join(profile['unique_phrases'][:5])  # Limit to 5 phrases
                        voice_profiles_section += f"- Unique Phrases: {phrases}\n"
                    voice_profiles_section += f"\nWhen writing dialogue for Speaker {i}, ensure the script reflects these characteristics. "
                    voice_profiles_section += "Match the cadence, tone, vocabulary style, and sentence structure described above.\n"

        # Build conditional speaker differentiation section
        speaker_differentiation = ""
        if num_voices == 1:
            speaker_differentiation = "   - Speaker 1: Solo host, presents all information and guides the narrative"
        elif num_voices == 2:
            speaker_differentiation = """   - Speaker 1: Primary host, guides the conversation, asks questions
   - Speaker 2: Co-host/expert, provides insights and elaboration"""
        else:
            # num_voices > 2
            speaker_differentiation = """   - Speaker 1: Primary host, guides the conversation, asks questions
   - Speaker 2: Co-host/expert, provides insights and elaboration
   - Additional speakers: Contribute unique perspectives or counterpoints"""

        prompt = f"""You are an expert podcast script writer specializing in creating natural, engaging dialogue for text-to-speech synthesis.
{voice_profiles_section}

**ARTICLE TO ADAPT:**
{article_text}

**PODCAST SPECIFICATIONS:**
- Genre: {genre}
- Tone/Style: {genre_style}
- Target word count: {word_target} words (~{duration})
- Speakers: {num_voices}

**OUTPUT FORMAT:**
Use exactly this format with no additional markup:

{chr(10).join(speaker_examples)}

**SCRIPT REQUIREMENTS:**

1. **Natural Speech Patterns**
   - Use contractions (don't, we're, that's)
   - Include verbal fillers sparingly (well, you know, I mean)
   - Add brief reactions (Exactly, Right, Interesting)
   - Vary sentence length—mix short punchy lines with longer explanations

2. **Speaker Differentiation**
{speaker_differentiation}

3. **Structure**
   - Opening hook (grab attention in first 2-3 exchanges)
   - Brief intro of topic
   - Main content covering key article points
   - Smooth transitions between subtopics
   - Memorable conclusion with takeaway

4. **TTS Optimization**
   - Avoid abbreviations (write "versus" not "vs.")
   - Spell out numbers under 10
   - Use em-dashes for natural pauses
   - Avoid complex punctuation that may confuse TTS

5. **Pacing**
   - Balance information density with breathing room
   - Include moments of agreement/reaction between substantive points
   - Aim for 130-150 words per minute of target duration

**DO NOT:**
- Include stage directions or notes in parentheses or brackets
- Use placeholders like [Host Name], [Co-host Name], [Name], etc. - write actual dialogue only
- Use asterisks or other formatting
- Add sound effect cues
- Add or imply background music, theme music, intro/outro music, ambiance, jingles, or any non-speech audio
- Include timestamps
- Include any bracketed placeholders or notes
- Start with greeting-style introductions (e.g., "Welcome to...", "Hello and welcome...", "Thanks for joining us...") or any "show intro" language

Generate the complete podcast script now:"""

        return prompt

    def _clean_script(self, script: str, num_voices: int) -> str:
        """
        Clean and format generated script.

        Args:
            script: Raw script from Ollama
            num_voices: Expected number of voices

        Returns:
            Cleaned script
        """
        # Remove markdown code blocks if present
        if script.startswith("```"):
            lines = script.split("\n")
            # Remove first line if it's ``` or ```markdown
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            script = "\n".join(lines)

        # Remove placeholders in brackets (e.g., [Host Name], [Co-host Name], [Name], etc.)
        # Pattern matches brackets with text inside, but preserves speaker labels like "Speaker 1:"
        # Remove placeholder and any space before it if followed by punctuation
        placeholder_pattern = r'\s*\[[^\]]+\]'
        script = re.sub(placeholder_pattern, '', script)

        # Ensure proper speaker labels
        lines = script.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with a speaker label
            has_speaker = any(line.startswith(f"Speaker {i}:") for i in range(1, num_voices + 1))

            if not has_speaker and cleaned_lines:
                # If previous line was a speaker line, append to it
                if cleaned_lines and any(cleaned_lines[-1].startswith(f"Speaker {i}:") for i in range(1, num_voices + 1)):
                    cleaned_lines[-1] += " " + line
                    continue

            cleaned_lines.append(line)

        # Clean up any extra whitespace that might have been left after removing placeholders
        cleaned_script = "\n".join(cleaned_lines)
        # Remove multiple spaces and clean up spacing around punctuation
        cleaned_script = re.sub(r' +', ' ', cleaned_script)
        cleaned_script = re.sub(r' \.', '.', cleaned_script)
        cleaned_script = re.sub(r' ,', ',', cleaned_script)
        cleaned_script = re.sub(r' \?', '?', cleaned_script)
        cleaned_script = re.sub(r' !', '!', cleaned_script)

        return cleaned_script

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


# Global Ollama client instance
ollama_client = OllamaClient()
