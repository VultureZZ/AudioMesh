"""
Voice profile creation from uploaded audio.

This is style-oriented profiling: we transcribe speech to text, then ask an LLM
(Ollama) to summarize cadence/tone/vocabulary/sentence structure and common phrases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from .audio_transcriber import audio_transcriber
from .audio_validator import AudioValidator
from .ollama_client import OllamaClient
from .voice_profiler import voice_profiler

logger = logging.getLogger(__name__)


class VoiceProfileFromAudioService:
    def __init__(self) -> None:
        self.audio_validator = AudioValidator()

    def analyze(
        self,
        audio_path: Path,
        keywords: Optional[List[str]] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> Tuple[Dict, Dict, str]:
        """
        Analyze audio and produce a voice profile.

        Returns:
            (profile_dict, validation_feedback, transcript)
        """
        validation = self.audio_validator.validate_audio_files([audio_path])

        transcript = ""
        transcript_language = None
        try:
            transcription = audio_transcriber.transcribe(audio_path)
            transcript = transcription.text
            transcript_language = transcription.language
        except Exception as e:
            # Don't fail profiling outright if ASR isn't available;
            # return a minimal profile that explains the limitation.
            logger.warning(f"Transcription failed: {e}")
            profile = voice_profiler.parse_profile_response(
                response_text="",
                keywords=keywords,
            )
            profile["profile_text"] = (
                "Voice profile could not be derived from audio transcript because transcription failed. "
                "Ensure ffmpeg is installed and the 'faster-whisper' dependency is available, then try again."
            )
            if keywords:
                profile["keywords"] = keywords
            return profile, validation, transcript

        if not transcript:
            profile = voice_profiler.parse_profile_response(response_text="", keywords=keywords)
            profile["profile_text"] = (
                "Transcription completed but produced no text. The audio may be silent, too noisy, or too short."
            )
            if transcript_language:
                profile["profile_text"] += f" Detected language: {transcript_language}."
            if keywords:
                profile["keywords"] = keywords
            return profile, validation, transcript

        # Synthesize a structured profile via Ollama from transcript text.
        ollama = OllamaClient(base_url=ollama_url, model=ollama_model)
        if not ollama.check_connection():
            raise RuntimeError(
                f"Ollama server not available at {ollama.base_url}. Please ensure Ollama is running."
            )

        prompt = self._build_prompt(transcript=transcript, keywords=keywords, language=transcript_language)
        try:
            response = ollama.client.post(
                f"{ollama.base_url}/api/generate",
                json={
                    "model": ollama.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "top_p": 0.9},
                },
            )
            response.raise_for_status()
            result = response.json()
            profile_text = (result.get("response") or "").strip()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during profile generation: {e}") from e

        profile = voice_profiler.parse_profile_response(profile_text, keywords=keywords)
        if keywords:
            profile["keywords"] = keywords
        if transcript_language and profile.get("profile_text"):
            profile["profile_text"] = f"[Transcript language: {transcript_language}] {profile['profile_text']}"

        return profile, validation, transcript

    def _build_prompt(self, transcript: str, keywords: Optional[List[str]], language: Optional[str]) -> str:
        keyword_context = ""
        if keywords:
            keyword_context = (
                "\n\nKEYWORDS/CONTEXT:\n"
                + ", ".join(keywords)
                + "\nUse these keywords as context only; base the style primarily on the transcript."
            )

        lang_line = ""
        if language:
            lang_line = f"\n\nTRANSCRIPT_LANGUAGE: {language}"

        return f"""You are an expert at analyzing speech patterns and writing style based on a transcript.\n\nTASK:\nGiven the transcript below, infer the speaker's conversational style and produce a JSON object with these fields:\n\n{{\n  \"cadence\": \"Description of rhythm/pace/timing\",\n  \"tone\": \"Emotional tone and delivery style\",\n  \"vocabulary_style\": \"Word choice patterns (formal/casual/technical/etc.)\",\n  \"sentence_structure\": \"Typical sentence patterns\",\n  \"unique_phrases\": [\"Common phrases or expressions\"],\n  \"profile_text\": \"A comprehensive paragraph describing how this speaker tends to talk\"\n}}\n\nRULES:\n- Output ONLY valid JSON.\n- Derive characteristics from the transcript; do not invent biographical facts.\n- Keep unique_phrases grounded in phrases present in, or strongly suggested by, the transcript.\n\nTRANSCRIPT:\n{transcript}\n{lang_line}{keyword_context}\n"""


voice_profile_from_audio = VoiceProfileFromAudioService()

