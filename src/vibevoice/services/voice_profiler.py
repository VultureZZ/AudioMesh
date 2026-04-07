"""
Voice profiling service using LLM to analyze speech patterns.
"""
import json
import logging
from typing import Dict, List, Optional

import httpx

from ..config import config
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class VoiceProfiler:
    """Service for profiling voices using LLM analysis."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize voice profiler.
        
        Args:
            base_url: Optional Ollama server base URL (defaults to config)
            model: Optional Ollama model name (defaults to config)
        """
        self.ollama = OllamaClient(base_url=base_url, model=model)

    def profile_voice_from_audio(
        self,
        voice_name: str,
        voice_description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> Dict:
        """
        Profile a voice using LLM analysis.

        Args:
            voice_name: Name of the voice
            voice_description: Optional description of the voice
            keywords: Optional keywords for context (e.g., person names)

        Returns:
            Structured profile dictionary
        """
        logger.info(f"Profiling voice: {voice_name}")
        if keywords:
            logger.info(f"Using keywords: {keywords}")

        # Use custom Ollama settings if provided
        ollama_client = self.ollama
        if ollama_url or ollama_model:
            ollama_client = OllamaClient(base_url=ollama_url, model=ollama_model)
            logger.info(f"Using custom Ollama settings: URL={ollama_url or 'default'}, Model={ollama_model or 'default'}")

        # Check Ollama connection first
        if not ollama_client.check_connection():
            error_msg = f"Ollama server not available at {ollama_client.base_url}. Please ensure Ollama is running."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Build prompt for profiling
        prompt = self.generate_profile_prompt(voice_name, voice_description, keywords)

        try:
            # Call Ollama to generate profile
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    f"{ollama_client.base_url}/api/generate",
                    json={
                        "model": ollama_client.model,
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
            profile_text = result.get("response", "").strip()

            if not profile_text:
                logger.warning("Ollama returned empty profile")
                return self._create_empty_profile()

            # Parse profile response
            profile = self.parse_profile_response(profile_text, keywords)
            
            # Ensure keywords are always included in the profile
            if keywords:
                profile["keywords"] = keywords
            elif "keywords" not in profile:
                profile["keywords"] = []

            logger.info(f"Profile generated successfully for {voice_name}")
            logger.debug(f"Profile data: {profile}")
            return profile

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = f"Ollama model '{ollama_client.model}' not found. Please ensure the model is installed: ollama pull {ollama_client.model}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                error_msg = f"Ollama API error (HTTP {e.response.status_code}): {e.response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to Ollama at {ollama_client.base_url}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to profile voice {voice_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def enhance_profile_with_keywords(
        self,
        voice_name: str,
        existing_profile: Optional[Dict],
        keywords: List[str],
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> Dict:
        """
        Enhance existing profile using keywords.

        Args:
            voice_name: Name of the voice
            existing_profile: Existing profile data (optional)
            keywords: Keywords for context

        Returns:
            Enhanced profile dictionary
        """
        logger.info(f"Enhancing profile for {voice_name} with keywords: {keywords}")

        # Use custom Ollama settings if provided
        ollama_client = self.ollama
        if ollama_url or ollama_model:
            ollama_client = OllamaClient(base_url=ollama_url, model=ollama_model)
            logger.info(f"Using custom Ollama settings: URL={ollama_url or 'default'}, Model={ollama_model or 'default'}")

        # Check Ollama connection first
        if not ollama_client.check_connection():
            error_msg = f"Ollama server not available at {ollama_client.base_url}. Please ensure Ollama is running."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Build enhancement prompt
        prompt = self.generate_enhancement_prompt(voice_name, existing_profile, keywords)

        try:
            # Call Ollama to enhance profile
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    f"{ollama_client.base_url}/api/generate",
                    json={
                        "model": ollama_client.model,
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
            profile_text = result.get("response", "").strip()

            if not profile_text:
                logger.warning("Ollama returned empty enhanced profile")
                return existing_profile or self._create_empty_profile()

            # Parse enhanced profile
            enhanced_profile = self.parse_profile_response(profile_text, keywords)
            
            # Ensure keywords are always included
            if keywords:
                enhanced_profile["keywords"] = keywords
            elif "keywords" not in enhanced_profile:
                enhanced_profile["keywords"] = []

            # Merge with existing profile if present
            if existing_profile:
                enhanced_profile = self._merge_profiles(existing_profile, enhanced_profile)
                # Ensure merged profile has keywords
                if keywords:
                    enhanced_profile["keywords"] = list(set(existing_profile.get("keywords", []) + keywords))

            logger.info(f"Profile enhanced successfully for {voice_name}")
            logger.debug(f"Enhanced profile data: {enhanced_profile}")
            return enhanced_profile

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = f"Ollama model '{ollama_client.model}' not found. Please ensure the model is installed: ollama pull {ollama_client.model}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                error_msg = f"Ollama API error (HTTP {e.response.status_code}): {e.response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to Ollama at {ollama_client.base_url}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to enhance profile for {voice_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def generate_profile_prompt(
        self,
        voice_name: str,
        voice_description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for LLM profiling.

        Args:
            voice_name: Name of the voice
            voice_description: Optional description
            keywords: Optional keywords for context

        Returns:
            Formatted prompt string
        """
        keyword_context = ""
        if keywords:
            keyword_context = f"\n\n**KEYWORDS/CONTEXT:**\n"
            keyword_context += ", ".join(keywords)
            keyword_context += "\n\nUse these keywords to help identify the unique characteristics of this voice. "
            keyword_context += "If you recognize these keywords (e.g., a famous person's name), incorporate known speech patterns."

        description_text = ""
        if voice_description:
            description_text = f"\n\n**VOICE DESCRIPTION:**\n{voice_description}"

        prompt = f"""You are an expert at analyzing speech patterns and voice characteristics. Analyze the following voice and provide a structured profile.

**VOICE NAME:** {voice_name}
{description_text}
{keyword_context}

**TASK:**
Provide a detailed analysis of this voice's speech patterns. Structure your response as JSON with the following fields:

{{
  "cadence": "Description of speech rhythm, pace, and timing patterns",
  "tone": "Emotional tone, delivery style, and vocal characteristics",
  "vocabulary_style": "Word choice patterns (formal, casual, technical, simple, complex, etc.)",
  "sentence_structure": "Typical sentence patterns (short, long, complex, simple, etc.)",
  "unique_phrases": ["List of common phrases or expressions"],
  "profile_text": "Full comprehensive description of the voice's speech characteristics"
}}

**INSTRUCTIONS:**
- Be specific and detailed in your analysis
- If keywords are provided and you recognize them, incorporate known characteristics
- Focus on speech patterns that would affect how text should be written for this voice
- Include specific examples of vocabulary, phrases, or speech patterns when possible
- The profile_text should be a comprehensive paragraph describing all aspects of the voice

Provide ONLY the JSON response, no additional text:"""

        return prompt

    def generate_enhancement_prompt(
        self,
        voice_name: str,
        existing_profile: Optional[Dict],
        keywords: List[str],
    ) -> str:
        """
        Build prompt for profile enhancement.

        Args:
            voice_name: Name of the voice
            existing_profile: Existing profile (optional)
            keywords: Keywords for enhancement

        Returns:
            Formatted prompt string
        """
        existing_text = ""
        if existing_profile:
            existing_text = f"\n\n**EXISTING PROFILE:**\n{json.dumps(existing_profile, indent=2)}"

        prompt = f"""You are an expert at analyzing speech patterns. Enhance the voice profile using the provided keywords.

**VOICE NAME:** {voice_name}
**KEYWORDS:** {', '.join(keywords)}
{existing_text}

**TASK:**
Using the keywords provided, enhance or create a detailed voice profile. If you recognize these keywords (e.g., a famous person's name), incorporate their known speech patterns and characteristics.

Provide a JSON response with the following structure:

{{
  "cadence": "Description of speech rhythm, pace, and timing patterns",
  "tone": "Emotional tone, delivery style, and vocal characteristics",
  "vocabulary_style": "Word choice patterns (formal, casual, technical, simple, complex, etc.)",
  "sentence_structure": "Typical sentence patterns (short, long, complex, simple, etc.)",
  "unique_phrases": ["List of common phrases or expressions"],
  "profile_text": "Full comprehensive description of the voice's speech characteristics"
}}

**INSTRUCTIONS:**
- If an existing profile is provided, enhance it with keyword-based insights
- If no existing profile, create a new one based on the keywords
- Be specific about speech patterns, vocabulary, and delivery style
- Include known phrases or expressions if applicable
- The profile_text should be comprehensive

Provide ONLY the JSON response, no additional text:"""

        return prompt

    def parse_profile_response(self, response_text: str, keywords: Optional[List[str]] = None) -> Dict:
        """
        Parse LLM response into structured profile.

        Args:
            response_text: Raw response from LLM
            keywords: Optional keywords to include in profile

        Returns:
            Structured profile dictionary
        """
        # Try to extract JSON from response
        profile = {
            "cadence": None,
            "tone": None,
            "vocabulary_style": None,
            "sentence_structure": None,
            "unique_phrases": [],
            "keywords": keywords or [],
            "profile_text": None,
        }

        # Try to find JSON in the response
        try:
            # Look for JSON block
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                parsed = json.loads(json_text)

                # Map parsed fields to profile
                profile["cadence"] = parsed.get("cadence")
                profile["tone"] = parsed.get("tone")
                profile["vocabulary_style"] = parsed.get("vocabulary_style")
                profile["sentence_structure"] = parsed.get("sentence_structure")
                profile["unique_phrases"] = parsed.get("unique_phrases", [])
                profile["profile_text"] = parsed.get("profile_text")
                profile["keywords"] = keywords or []

            else:
                # If no JSON found, use the text as profile_text
                profile["profile_text"] = response_text.strip()

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON from profile response: {e}")
            # Use the raw text as profile_text
            profile["profile_text"] = response_text.strip()

        return profile

    def _create_empty_profile(self) -> Dict:
        """Create an empty profile structure."""
        return {
            "cadence": None,
            "tone": None,
            "vocabulary_style": None,
            "sentence_structure": None,
            "unique_phrases": [],
            "keywords": [],
            "profile_text": None,
        }

    def _merge_profiles(self, existing: Dict, enhanced: Dict) -> Dict:
        """
        Merge existing profile with enhanced profile.

        Args:
            existing: Existing profile data
            enhanced: Enhanced profile data

        Returns:
            Merged profile dictionary
        """
        merged = existing.copy()

        # Update fields from enhanced profile (only if they have values)
        for key in ["cadence", "tone", "vocabulary_style", "sentence_structure", "profile_text"]:
            if enhanced.get(key):
                merged[key] = enhanced[key]

        # Merge unique phrases
        existing_phrases = set(existing.get("unique_phrases", []))
        enhanced_phrases = set(enhanced.get("unique_phrases", []))
        merged["unique_phrases"] = list(existing_phrases | enhanced_phrases)

        # Merge keywords
        existing_keywords = set(existing.get("keywords", []))
        enhanced_keywords = set(enhanced.get("keywords", []))
        merged["keywords"] = list(existing_keywords | enhanced_keywords)

        return merged


# Global voice profiler instance
voice_profiler = VoiceProfiler()
