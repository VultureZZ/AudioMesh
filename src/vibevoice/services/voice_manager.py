"""
Voice management service.

Handles custom voice creation, listing, and deletion.
"""
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import config
from ..models.voice_storage import voice_storage
from ..core.transcripts.speaker_matcher import compute_file_embedding
from .audio_quality_analyzer import audio_quality_analyzer
from .audio_transcriber import audio_transcriber
from .audio_validator import AudioValidator

# Default voices that cannot be deleted
# Mapping of short names to full voice file names
VOICE_NAME_MAPPING = {
    "Alice": "en-Alice_woman",
    "Frank": "en-Frank_man",
    "Mary": "en-Mary_woman_bgm",
    "Carter": "en-Carter_man",
    "Maya": "en-Maya_woman",
}

# Default voices list (both short and full names)
DEFAULT_VOICES = list(VOICE_NAME_MAPPING.keys()) + list(VOICE_NAME_MAPPING.values())

LANGUAGE_LABELS = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    # Non-standard code used by some bundled voices
    "in": "Indian",
}


def _get_language_label(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    normalized = code.strip().lower()
    if not normalized:
        return None
    return LANGUAGE_LABELS.get(normalized, normalized.upper())


def _parse_default_voice_stem(stem: str) -> dict:
    """
    Parse default voice filename stem like:
      - en-Alice_woman_bgm
      - in-Samuel_man
      - zh-Anchen_man_bgm

    Returns:
      { display_name, language_code, language_label, gender }
    """
    display_name = stem
    language_code: Optional[str] = None
    gender: str = "unknown"

    if "-" in stem:
        language_code, remainder = stem.split("-", 1)
        language_code = language_code.lower()
        parts = remainder.split("_")
        if parts:
            display_name = parts[0]
            tags = {p.lower() for p in parts[1:] if p}
            if "woman" in tags or "female" in tags:
                gender = "female"
            elif "man" in tags or "male" in tags:
                gender = "male"
            elif "neutral" in tags or "nonbinary" in tags or "gender_neutral" in tags:
                gender = "neutral"
    return {
        "display_name": display_name,
        "language_code": language_code,
        "language_label": _get_language_label(language_code),
        "gender": gender,
    }


def _normalize_gender(value: Optional[str]) -> Optional[str]:
    """
    Normalize gender inputs to one of: male, female, neutral, unknown.

    Accepts common variants from UIs/integrations (e.g. "gender_neutral", "nonbinary").
    Returns None when the input is empty/whitespace so callers can treat it as "unset".
    """
    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None

    mapping = {
        # canonical
        "male": "male",
        "female": "female",
        "neutral": "neutral",
        "unknown": "unknown",
        # common variants
        "man": "male",
        "woman": "female",
        "nonbinary": "neutral",
        "non-binary": "neutral",
        "nb": "neutral",
        "gender_neutral": "neutral",
        "gender-neutral": "neutral",
        "gender neutral": "neutral",
    }
    return mapping.get(v, v)


class VoiceManager:
    """Service for managing voices."""

    def __init__(self):
        """Initialize voice manager."""
        self.custom_voices_dir = config.CUSTOM_VOICES_DIR
        self.vibevoice_repo_dir = config.VIBEVOICE_REPO_DIR
        self.default_voices_dir = self.vibevoice_repo_dir / "demo" / "voices"
        self.audio_validator = AudioValidator()

    def is_default_voice(self, voice_name: str) -> bool:
        """
        Check if a voice is a default voice.

        Args:
            voice_name: Voice name to check

        Returns:
            True if default voice, False otherwise
        """
        if voice_name in DEFAULT_VOICES:
            return True
        # Also treat any on-disk default voice name as reserved.
        #
        # IMPORTANT: Custom voices may be symlinked into the default voices directory
        # so the underlying inference code can find them. Those symlinks must NOT be
        # treated as default/reserved voices, otherwise custom voices become impossible
        # to update (e.g., language/gender) after they've been used for inference.
        try:
            candidate = self.default_voices_dir / f"{voice_name}.wav"
            if not candidate.exists():
                return False

            if candidate.is_symlink():
                try:
                    target_path = candidate.resolve()
                    if self.custom_voices_dir in target_path.parents:
                        return False
                except (OSError, RuntimeError):
                    # Broken symlink: not a real default voice.
                    return False

            return True
        except Exception:
            return False

    def get_voice_id_from_name(self, name: str) -> str:
        """
        Convert voice name to a valid voice ID (sanitized).

        Args:
            name: Voice name

        Returns:
            Sanitized voice ID
        """
        # Replace spaces and special characters with underscores
        voice_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        return voice_id.lower()

    ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    def create_custom_voice(
        self,
        name: str,
        description: Optional[str],
        audio_files: List[Path],
        keywords: Optional[List[str]] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        language_code: Optional[str] = None,
        gender: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> dict:
        """
        Create a custom voice from uploaded audio files.

        Args:
            name: Voice name
            description: Voice description
            audio_files: List of paths to uploaded audio files
            image_path: Optional path to avatar image (jpg, png, webp)

        Returns:
            Dict with voice metadata

        Raises:
            ValueError: If name is invalid or files cannot be processed
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Voice name cannot be empty")

        # Check if name is a default voice
        if self.is_default_voice(name):
            raise ValueError(f"Voice name '{name}' is reserved for default voices")

        # Generate voice ID
        voice_id = self.get_voice_id_from_name(name)

        # Check if voice ID already exists
        if voice_storage.voice_exists(voice_id):
            raise ValueError(f"Voice with name '{name}' already exists")

        # Check if name already exists (case-insensitive)
        if voice_storage.name_exists(name):
            raise ValueError(f"Voice with name '{name}' already exists")

        # Create voice directory structure
        voice_dir = self.custom_voices_dir / voice_id
        original_dir = voice_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Save original files and combine them
        saved_files = []
        audio_segments = []

        try:
            normalized_language_code = None
            if language_code is not None:
                normalized_language_code = language_code.strip().lower() or None

            normalized_gender = None
            if gender is not None:
                normalized_gender = _normalize_gender(gender)
            allowed_genders = {"male", "female", "neutral", "unknown"}
            if normalized_gender is not None and normalized_gender not in allowed_genders:
                raise ValueError("gender must be one of: male, female, neutral, unknown")

            for i, audio_file in enumerate(audio_files):
                # Validate file exists
                if not audio_file.exists():
                    raise ValueError(f"Audio file not found: {audio_file}")

                # Determine file extension
                ext = audio_file.suffix.lower()
                if ext not in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
                    raise ValueError(f"Unsupported audio format: {ext}")

                # Save original file
                saved_filename = f"file_{i+1}{ext}"
                saved_path = original_dir / saved_filename
                shutil.copy2(audio_file, saved_path)
                saved_files.append(saved_filename)

                # Load audio segment
                try:
                    from pydub import AudioSegment

                    if ext == ".wav":
                        segment = AudioSegment.from_wav(str(saved_path))
                    elif ext == ".mp3":
                        segment = AudioSegment.from_mp3(str(saved_path))
                    elif ext == ".flac":
                        segment = AudioSegment.from_flac(str(saved_path))
                    elif ext == ".m4a":
                        segment = AudioSegment.from_file(str(saved_path), format="m4a")
                    elif ext == ".ogg":
                        segment = AudioSegment.from_ogg(str(saved_path))
                    else:
                        raise ValueError(f"Unsupported format: {ext}")

                    # Normalize sample rate to 24000 Hz (VibeVoice standard)
                    if segment.frame_rate != 24000:
                        segment = segment.set_frame_rate(24000)

                    # Convert to mono if stereo
                    if segment.channels > 1:
                        segment = segment.set_channels(1)

                    audio_segments.append(segment)
                except Exception as e:
                    raise ValueError(f"Failed to process audio file {audio_file}: {str(e)}")

            # Combine all audio segments
            if not audio_segments:
                raise ValueError("No valid audio files provided")

            combined_audio = sum(audio_segments)
            combined_path = voice_dir / "combined.wav"

            # Calculate combined duration
            combined_duration_seconds = len(combined_audio) / 1000.0
            truncated_for_qwen3 = False
            if (config.TTS_BACKEND or "qwen3").strip().lower() == "qwen3" and combined_duration_seconds > 60:
                combined_audio = combined_audio[:60000]
                combined_duration_seconds = 60.0
                truncated_for_qwen3 = True

            # Validate audio files (analyze individual files and combined result)
            # Build list of saved file paths for validation
            saved_file_paths = [original_dir / filename for filename in saved_files]
            validation_feedback = self.audio_validator.validate_audio_files(
                audio_files=saved_file_paths,
                combined_duration_seconds=combined_duration_seconds,
            )
            if truncated_for_qwen3:
                validation_feedback["warnings"].append(
                    "Combined reference was over 60s and was truncated to 60s for Qwen3-TTS."
                )

            # Export combined audio as WAV
            combined_audio.export(str(combined_path), format="wav", parameters=["-ar", "24000"])

            # Analyze audio quality (background music, noise, recording quality, clone quality)
            quality_analysis = None
            try:
                quality_analysis = audio_quality_analyzer.analyze_quality(
                    audio_files=saved_file_paths,
                    combined_path=combined_path,
                    total_duration_seconds=combined_duration_seconds,
                )
                validation_feedback["quality_metrics"].update(quality_analysis)
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.warning("Audio quality analysis failed: %s", e, exc_info=True)

            # Save optional avatar image
            image_filename = None
            if image_path and image_path.exists():
                ext = image_path.suffix.lower()
                if ext in self.ALLOWED_IMAGE_EXTENSIONS:
                    stored_name = "image" + ext
                    dest = voice_dir / stored_name
                    shutil.copy2(image_path, dest)
                    image_filename = stored_name

            # Save metadata
            speaker_embedding = compute_file_embedding(combined_path)
            voice_storage.add_voice(
                voice_id=voice_id,
                name=name,
                description=description,
                audio_files=saved_files,
                language_code=normalized_language_code,
                gender=normalized_gender,
                image_filename=image_filename,
                quality_analysis=quality_analysis,
                speaker_embedding=speaker_embedding,
            )

            # Transcribe combined reference audio for Qwen3-TTS (ref_text improves clone quality)
            reference_transcript = None
            try:
                transcription = audio_transcriber.transcribe(combined_path)
                if transcription and transcription.text.strip():
                    reference_transcript = transcription.text.strip()
                    import logging
                    logging.getLogger(__name__).info(
                        "Transcribed reference audio for voice %s (%d chars)", name, len(reference_transcript)
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Reference transcription skipped for %s: %s", name, e)

            # Automatically profile the voice (non-blocking - don't fail if profiling fails)
            import logging
            logger = logging.getLogger(__name__)
            try:
                from .voice_profiler import voice_profiler

                logger.info(f"Starting automatic profiling for voice: {name} (ID: {voice_id})")
                if keywords:
                    logger.info(f"Using keywords for profiling: {keywords}")
                else:
                    logger.info("No keywords provided for profiling")
                
                profile = voice_profiler.profile_voice_from_audio(
                    voice_name=name,
                    voice_description=description,
                    keywords=keywords,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                )
                if profile:
                    # Ensure keywords are included in profile
                    if keywords:
                        profile["keywords"] = keywords
                    # Store reference transcript for Qwen3-TTS (ref_text) when available
                    if reference_transcript:
                        profile["transcript"] = reference_transcript
                    # Store profile in voice metadata
                    voice_storage.update_voice_profile(voice_id, profile)
                    logger.info(f"Successfully created and saved profile for voice: {name}")
                else:
                    logger.warning(f"Profile creation returned empty profile for voice: {name}")
                    # Still save transcript if we have it and profiling produced nothing
                    if reference_transcript:
                        voice_storage.update_voice_profile(voice_id, {"transcript": reference_transcript})
            except RuntimeError as e:
                # RuntimeError means Ollama is not available or model missing - log as warning
                logger.warning(f"Could not profile voice {name}: {e}")
                if reference_transcript:
                    voice_storage.update_voice_profile(voice_id, {"transcript": reference_transcript})
            except Exception as e:
                # Log error but don't fail voice creation
                logger.error(f"Failed to automatically profile voice {name}: {e}", exc_info=True)
                if reference_transcript:
                    voice_storage.update_voice_profile(voice_id, {"transcript": reference_transcript})

            # Return voice metadata with validation feedback
            voice_data = voice_storage.get_voice(voice_id)
            if voice_data:
                # Ensure display fields exist for the frontend.
                voice_data.setdefault("display_name", voice_data.get("name"))
                lc = voice_data.get("language_code")
                if lc:
                    voice_data["language_label"] = _get_language_label(lc)
                g = voice_data.get("gender")
                if not g:
                    voice_data["gender"] = "unknown"
            voice_data["validation_feedback"] = validation_feedback
            return voice_data

        except Exception as e:
            # Clean up on error
            if voice_dir.exists():
                shutil.rmtree(voice_dir)
            raise

    def create_voice_from_prompt(
        self,
        name: str,
        description: Optional[str],
        voice_design_prompt: str,
        language_code: Optional[str] = None,
        gender: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> dict:
        """
        Create a VoiceDesign voice from a natural-language description (no audio).

        Args:
            name: Voice name (must be unique)
            description: Optional voice description
            voice_design_prompt: Text description of the voice (e.g. "young female, calm tone")
            language_code: Optional language code
            gender: Optional gender
            image_path: Optional path to avatar image

        Returns:
            Dict with voice metadata (type=voice_design)
        """
        if not name or not name.strip():
            raise ValueError("Voice name cannot be empty")
        if not voice_design_prompt or not voice_design_prompt.strip():
            raise ValueError("Voice design prompt cannot be empty")
        if self.is_default_voice(name):
            raise ValueError(f"Voice name '{name}' is reserved for default voices")

        voice_id = self.get_voice_id_from_name(name)
        if voice_storage.voice_exists(voice_id):
            raise ValueError(f"Voice with name '{name}' already exists")
        if voice_storage.name_exists(name):
            raise ValueError(f"Voice with name '{name}' already exists")

        normalized_language_code = None
        if language_code:
            normalized_language_code = language_code.strip().lower() or None
        normalized_gender = None
        if gender:
            normalized_gender = _normalize_gender(gender)
        if normalized_gender is not None and normalized_gender not in {"male", "female", "neutral", "unknown"}:
            raise ValueError("gender must be one of: male, female, neutral, unknown")

        profile = {"voice_design_prompt": voice_design_prompt.strip()}
        image_filename = None
        if image_path and image_path.exists():
            voice_dir = self.custom_voices_dir / voice_id
            voice_dir.mkdir(parents=True, exist_ok=True)
            ext = image_path.suffix.lower()
            if ext in self.ALLOWED_IMAGE_EXTENSIONS:
                dest = voice_dir / ("image" + ext)
                shutil.copy2(image_path, dest)
                image_filename = "image" + ext

        voice_storage.add_voice(
            voice_id=voice_id,
            name=name,
            description=description or "",
            audio_files=None,
            profile=profile,
            language_code=normalized_language_code,
            gender=normalized_gender,
            image_filename=image_filename,
            voice_type="voice_design",
        )
        voice_data = voice_storage.get_voice(voice_id)
        if voice_data:
            voice_data.setdefault("display_name", voice_data.get("name"))
            lc = voice_data.get("language_code")
            if lc:
                voice_data["language_label"] = _get_language_label(lc)
            if not voice_data.get("gender"):
                voice_data["gender"] = "unknown"
        return voice_data

    def delete_custom_voice(self, voice_id: str) -> bool:
        """
        Delete a custom voice.

        Args:
            voice_id: Voice identifier

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If trying to delete a default voice
        """
        # Get voice metadata
        voice_data = voice_storage.get_voice(voice_id)
        if not voice_data:
            return False

        # Check if it's a default voice
        if voice_data.get("type") == "default" or self.is_default_voice(voice_data.get("name", "")):
            raise ValueError("Cannot delete default voices")

        # Get the canonical voice name to find the symlink
        voice_name = voice_data.get("name", "")
        if voice_name:
            # Normalize to get canonical name (in case it was stored with different casing)
            canonical_name = self.normalize_voice_name(voice_name)
            # Remove symlink from default voices directory if it exists
            symlink_path = self.default_voices_dir / f"{canonical_name}.wav"
            if symlink_path.exists() or symlink_path.is_symlink():
                try:
                    symlink_path.unlink()
                except OSError as e:
                    # Log but don't fail deletion if symlink removal fails
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to remove symlink {symlink_path}: {e}")

        # Delete voice directory
        voice_dir = self.custom_voices_dir / voice_id
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

        # Delete from storage
        return voice_storage.delete_voice(voice_id)

    def list_all_voices(self) -> List[dict]:
        """
        List all available voices (default + custom).

        Returns:
            List of voice metadata dicts
        """
        voices = []
        seen_voices = set()

        # Get list of custom voice IDs to exclude symlinks
        custom_voice_ids = {v["id"] for v in voice_storage.list_voices()}

        # Always expose built-in default voice names (Alice, Frank, Mary, Carter, Maya).
        # They are used by Qwen3-TTS CustomVoice; no on-disk files required.
        for short_name in VOICE_NAME_MAPPING:
            if short_name not in seen_voices and short_name not in custom_voice_ids:
                voices.append(
                    {
                        "id": short_name,
                        "name": short_name,
                        "display_name": short_name,
                        "language_code": "en",
                        "language_label": "English",
                        "gender": "unknown",
                        "description": f"Default voice: {short_name}",
                        "type": "default",
                        "created_at": None,
                        "audio_files": None,
                    }
                )
                seen_voices.add(short_name)

        # Add default voices from actual files in the directory (e.g. VibeVoice demo/voices).
        # If a voice has a short-name mapping (e.g. en-Alice_woman -> Alice),
        # expose only the short name to avoid duplicate entries in the UI.
        if self.default_voices_dir.exists():
            for voice_file in self.default_voices_dir.glob("*.wav"):
                full_name = voice_file.stem
                if full_name.startswith("."):
                    continue

                # Check if this is a symlink to a custom voice
                is_custom_symlink = False
                if voice_file.is_symlink():
                    try:
                        target_path = voice_file.resolve()
                        # Check if target is in a custom voice directory
                        if self.custom_voices_dir in target_path.parents:
                            is_custom_symlink = True
                    except (OSError, RuntimeError):
                        # If symlink is broken, skip it
                        continue

                # Skip if it's a symlink to a custom voice or if the name matches a custom voice ID
                if is_custom_symlink or full_name in custom_voice_ids:
                    continue

                # Prefer the short name for mapped voices.
                short_name = None
                for candidate_short, mapped_name in VOICE_NAME_MAPPING.items():
                    if mapped_name == full_name:
                        short_name = candidate_short
                        break

                if short_name:
                    if short_name not in seen_voices:
                        parsed = _parse_default_voice_stem(full_name)
                        voices.append(
                            {
                                "id": short_name,
                                "name": short_name,
                                "display_name": short_name,
                                "language_code": parsed.get("language_code"),
                                "language_label": parsed.get("language_label"),
                                "gender": parsed.get("gender", "unknown"),
                                "description": f"Default VibeVoice voice: {short_name} (maps to {full_name})",
                                "type": "default",
                                "created_at": None,
                                "audio_files": None,
                            }
                        )
                        seen_voices.add(short_name)
                else:
                    # No mapping: expose full voice id.
                    if full_name not in seen_voices:
                        parsed = _parse_default_voice_stem(full_name)
                        voices.append(
                            {
                                "id": full_name,
                                "name": full_name,
                                "display_name": parsed.get("display_name", full_name),
                                "language_code": parsed.get("language_code"),
                                "language_label": parsed.get("language_label"),
                                "gender": parsed.get("gender", "unknown"),
                                "description": f"Default VibeVoice voice: {full_name}",
                                "type": "default",
                                "created_at": None,
                                "audio_files": None,
                            }
                        )
                        seen_voices.add(full_name)

        # IMPORTANT:
        # Do not add hardcoded "default" voices that aren't present on disk.
        # This prevents the API/UX from listing voices that cannot be used on this host.

        # Add custom voices
        custom_voices = voice_storage.list_voices()
        for voice_data in custom_voices:
            # Parse created_at if it's a string
            if isinstance(voice_data.get("created_at"), str):
                try:
                    voice_data["created_at"] = datetime.fromisoformat(
                        voice_data["created_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            # Ensure optional display fields exist for the frontend.
            if isinstance(voice_data, dict):
                voice_data.setdefault("display_name", voice_data.get("name"))
                lc = voice_data.get("language_code")
                if lc and not voice_data.get("language_label"):
                    voice_data["language_label"] = _get_language_label(lc)
                g = voice_data.get("gender")
                if not g:
                    voice_data["gender"] = "unknown"
            voices.append(voice_data)

        return voices

    def get_voice(self, voice_id: str) -> Optional[dict]:
        """
        Get voice metadata by ID.

        Args:
            voice_id: Voice identifier

        Returns:
            Voice metadata dict or None if not found
        """
        # Check custom voices first
        voice_data = voice_storage.get_voice(voice_id)
        if voice_data:
            return voice_data

        # Check default voices based on files present in the demo voices directory.
        # This must align with list_all_voices(), which enumerates the directory contents.
        resolved = VOICE_NAME_MAPPING.get(voice_id, voice_id)
        voice_path = self.default_voices_dir / f"{resolved}.wav"
        if voice_path.exists():
            parsed = _parse_default_voice_stem(resolved)
            # If this voice_id is the short mapped name (e.g. Alice), keep display_name as the short name.
            display_name = voice_id if voice_id in VOICE_NAME_MAPPING else parsed.get("display_name", voice_id)
            return {
                "id": voice_id,
                "name": voice_id,
                "display_name": display_name,
                "language_code": parsed.get("language_code"),
                "language_label": parsed.get("language_label"),
                "gender": parsed.get("gender", "unknown"),
                "description": f"Default VibeVoice voice: {voice_id}",
                "type": "default",
                "created_at": None,
                "audio_files": None,
            }

        return None

    def enhance_voice_profile(
        self,
        voice_id: str,
        keywords: List[str],
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> dict:
        """
        Enhance voice profile with keywords.

        Args:
            voice_id: Voice identifier
            keywords: Keywords for enhancement

        Returns:
            Updated voice data with enhanced profile

        Raises:
            ValueError: If voice is not found
        """
        voice_data = voice_storage.get_voice(voice_id)
        if not voice_data:
            raise ValueError(f"Voice '{voice_id}' not found")

        # Get existing profile
        existing_profile = voice_storage.get_voice_profile(voice_id)

        # Enhance profile with keywords
        try:
            from .voice_profiler import voice_profiler

            enhanced_profile = voice_profiler.enhance_profile_with_keywords(
                voice_name=voice_data.get("name", voice_id),
                existing_profile=existing_profile,
                keywords=keywords,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
            if enhanced_profile:
                voice_storage.update_voice_profile(voice_id, enhanced_profile)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to enhance profile for voice {voice_id}: {e}")
            raise ValueError(f"Failed to enhance profile: {e}") from e

        # Return updated voice data
        return voice_storage.get_voice(voice_id)

    def update_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        language_code: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> dict:
        """
        Update voice metadata.

        Args:
            voice_id: Voice identifier
            name: New voice name (optional)
            description: New voice description (optional)

        Returns:
            Updated voice data

        Raises:
            ValueError: If voice is not found or is a default voice
        """
        voice_data = voice_storage.get_voice(voice_id)
        if not voice_data:
            raise ValueError(f"Voice '{voice_id}' not found")

        # Check if it's a default voice
        if voice_data.get("type") == "default" or self.is_default_voice(voice_data.get("name", "")):
            raise ValueError("Cannot update default voices")

        # Validate new name if provided
        if name is not None:
            if not name or not name.strip():
                raise ValueError("Voice name cannot be empty")
            if self.is_default_voice(name):
                raise ValueError(f"Voice name '{name}' is reserved for default voices")
            if voice_storage.name_exists(name, exclude_voice_id=voice_id):
                raise ValueError(f"Voice with name '{name}' already exists")

        normalized_language_code = None
        if language_code is not None:
            normalized_language_code = language_code.strip().lower()

        normalized_gender = None
        if gender is not None:
            normalized_gender = _normalize_gender(gender)
            allowed_genders = {"male", "female", "neutral", "unknown"}
            if normalized_gender is not None and normalized_gender not in allowed_genders:
                raise ValueError("gender must be one of: male, female, neutral, unknown")

        # Update via storage
        updated = voice_storage.update_voice(
            voice_id=voice_id,
            name=name,
            description=description,
            language_code=normalized_language_code,
            gender=normalized_gender,
        )

        if not updated:
            raise ValueError(f"Failed to update voice '{voice_id}'")

        # Return updated voice data with computed display fields
        updated_voice = voice_storage.get_voice(voice_id)
        if isinstance(updated_voice, dict):
            updated_voice.setdefault("display_name", updated_voice.get("name"))
            lc = updated_voice.get("language_code")
            if lc and not updated_voice.get("language_label"):
                updated_voice["language_label"] = _get_language_label(lc)
            g = updated_voice.get("gender")
            if not g:
                updated_voice["gender"] = "unknown"
        return updated_voice

    def update_voice_image(self, voice_id: str, image_path: Path) -> dict:
        """
        Set or replace the avatar image for a custom voice.

        Args:
            voice_id: Voice identifier
            image_path: Path to image file (jpg, png, webp)

        Returns:
            Updated voice data

        Raises:
            ValueError: If voice not found, not custom, or image invalid
        """
        voice_data = voice_storage.get_voice(voice_id)
        if not voice_data:
            raise ValueError(f"Voice '{voice_id}' not found")
        if voice_data.get("type") != "custom":
            raise ValueError("Cannot set image for default voices")

        ext = image_path.suffix.lower()
        if ext not in self.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {ext}. Use one of {self.ALLOWED_IMAGE_EXTENSIONS}"
            )
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        voice_dir = self.custom_voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        stored_name = "image" + ext
        dest = voice_dir / stored_name
        shutil.copy2(image_path, dest)

        voice_storage.update_voice(voice_id=voice_id, image_filename=stored_name)

        updated = voice_storage.get_voice(voice_id)
        if isinstance(updated, dict):
            updated.setdefault("display_name", updated.get("name"))
            lc = updated.get("language_code")
            if lc and not updated.get("language_label"):
                updated["language_label"] = _get_language_label(lc)
            g = updated.get("gender")
            if not g:
                updated["gender"] = "unknown"
        return updated

    def get_voice_image_path(self, voice_id: str) -> Optional[Path]:
        """
        Return the path to a custom voice's avatar image, or None if no image.

        Args:
            voice_id: Voice identifier

        Returns:
            Path to image file, or None
        """
        voice_data = voice_storage.get_voice(voice_id)
        if not voice_data or voice_data.get("type") != "custom":
            return None
        image_filename = voice_data.get("image_filename")
        if not image_filename:
            return None
        voice_dir = self.custom_voices_dir / voice_id
        path = voice_dir / image_filename
        return path if path.exists() else None

    def get_voice_by_name(self, name: str) -> Optional[dict]:
        """
        Get voice metadata by name (searches custom voices by name).

        Args:
            name: Voice name to search for

        Returns:
            Voice metadata dict or None if not found
        """
        # Check custom voices by name
        custom_voices = voice_storage.list_voices()
        for voice_data in custom_voices:
            if voice_data.get("name", "").lower() == name.lower():
                return voice_data

        # Check default voices
        if self.is_default_voice(name) or name in DEFAULT_VOICES:
            return {
                "id": name,
                "name": name,
                "description": f"Default VibeVoice voice: {name}",
                "type": "default",
                "created_at": None,
                "audio_files": None,
            }

        return None

    def ensure_voice_accessible(self, voice_name: str) -> str:
        """
        Ensure a voice file is accessible in the default voices directory.
        For custom voices, creates a symlink in the default voices directory.

        Args:
            voice_name: Voice name (canonical or mapped format)

        Returns:
            Resolved voice name (for default voices) or voice name (for custom voices)

        Raises:
            ValueError: If voice is not found
        """
        # Normalize to canonical form first to handle mapped names
        canonical_name = self.normalize_voice_name(voice_name)

        # Check if it's a default voice that needs mapping (canonical -> mapped)
        if canonical_name in VOICE_NAME_MAPPING:
            return VOICE_NAME_MAPPING[canonical_name]

        # Check if it's a default voice
        if self.is_default_voice(canonical_name):
            return canonical_name

        # Check if it's a custom voice (use canonical name for lookup)
        voice_data = self.get_voice_by_name(canonical_name)
        if voice_data and voice_data.get("type") == "custom":
            voice_id = voice_data.get("id")
            if not voice_id:
                raise ValueError(f"Custom voice '{canonical_name}' has no ID")

            # Ensure default voices directory exists
            self.default_voices_dir.mkdir(parents=True, exist_ok=True)

            # Get source file path
            voice_dir = self.custom_voices_dir / voice_id
            source_path = voice_dir / "combined.wav"

            if not source_path.exists():
                raise ValueError(f"Voice file not found for '{canonical_name}' at {source_path}")

            # Create symlink in default voices directory (use canonical name)
            target_path = self.default_voices_dir / f"{canonical_name}.wav"

            # Remove existing symlink/file if it exists
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()

            # Create symlink
            try:
                target_path.symlink_to(source_path)
            except OSError:
                # If symlink fails (e.g., on Windows), copy the file
                shutil.copy2(source_path, target_path)

            return canonical_name

        # Return canonical name if not found (will cause error in inference)
        return canonical_name

    def normalize_voice_name(self, voice_name: str) -> str:
        """
        Normalize a voice name to its canonical form.

        Accepts both canonical names (e.g., "Alice") and mapped aliases
        (e.g., "en-Alice_woman") and returns the canonical name.
        Matching is case-insensitive.

        Args:
            voice_name: Voice name in any format

        Returns:
            Canonical voice name with correct casing

        Examples:
            - "en-Alice_woman" -> "Alice"
            - "Alice" -> "Alice"
            - "alice" -> "Alice" (case-insensitive)
            - "en-Carter_man" -> "Carter"
            - "donaldtrump" -> "DonaldTrump" (if exists, case-insensitive)
            - "zh-Anchen_man_bgm" -> "zh-Anchen_man_bgm" (no mapping, returns as-is)
        """
        voice_name_lower = voice_name.lower()
        
        # Check if it's a mapped name (reverse lookup, case-insensitive)
        for canonical, mapped in VOICE_NAME_MAPPING.items():
            if voice_name_lower == mapped.lower():
                return canonical

        # Check if it matches a canonical name (case-insensitive)
        for canonical in VOICE_NAME_MAPPING.keys():
            if voice_name_lower == canonical.lower():
                return canonical

        # If not found in mapping, check against all available voices (case-insensitive)
        # This handles custom voices and default voices not in the mapping
        all_voices = self.list_all_voices()
        for voice in all_voices:
            voice_name_from_list = voice.get("name", "")
            if voice_name_lower == voice_name_from_list.lower():
                return voice_name_from_list

        # If not found, return as-is (preserve original casing)
        # This allows for future voices that might not be in the list yet
        return voice_name

    def resolve_voice_name(self, voice_name: str) -> str:
        """
        Resolve a voice name to its actual file name.

        Maps short names (e.g., "Alice") to full names (e.g., "en-Alice_woman").
        For custom voices, ensures they are accessible in the default voices directory.

        Args:
            voice_name: Voice name (short or full)

        Returns:
            Resolved voice name (full name if mapping exists, otherwise original)
        """
        return self.ensure_voice_accessible(voice_name)

    def get_voice_path(self, voice_id: str) -> Optional[Path]:
        """
        Get the path to a voice file.

        Args:
            voice_id: Voice identifier

        Returns:
            Path to voice file or None if not found
        """
        # Check custom voices
        voice_data = voice_storage.get_voice(voice_id)
        if voice_data:
            voice_dir = self.custom_voices_dir / voice_id
            combined_path = voice_dir / "combined.wav"
            if combined_path.exists():
                return combined_path

        # Resolve voice name (map short names to full names)
        resolved_name = self.resolve_voice_name(voice_id)

        # Check default voices
        if self.default_voices_dir.exists():
            # Try exact match first
            exact_path = self.default_voices_dir / f"{resolved_name}.wav"
            if exact_path.exists():
                return exact_path

            # Try different naming conventions as fallback
            for pattern in [f"{resolved_name}.wav", f"{voice_id}.wav", f"en-{voice_id}.wav", f"{voice_id}_*.wav"]:
                matches = list(self.default_voices_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def get_bgm_risk_warnings(self, voice_names: List[str]) -> List[str]:
        """
        Return best-effort warnings about background music (BGM) risk for selected voices.

        This is warn-only: it does not block generation.
        """
        warnings: List[str] = []
        seen: set[str] = set()

        def add(msg: str) -> None:
            if msg not in seen:
                warnings.append(msg)
                seen.add(msg)

        for voice_name in voice_names or []:
            vn = (voice_name or "").strip()
            if not vn:
                continue

            # Best practices: "Alice" has higher BGM probability.
            if vn.lower() == "alice":
                add(
                    "Selected voice 'Alice' has a higher probability of spontaneous background music. "
                    "If you hear music, try a different voice preset."
                )

            # Detect `_bgm` in the resolved on-disk voice name.
            try:
                resolved = self.resolve_voice_name(vn)
            except Exception:
                resolved = vn
            if "_bgm" in (resolved or "").lower():
                add(
                    f"Selected voice '{vn}' maps to '{resolved}', which is a BGM-tagged preset. "
                    "Choose a different voice to reduce background music risk."
                )

        # General best-practice warning (we can't reliably detect BGM in custom samples here).
        add(
            "If a custom voice sample contains background music, the model is more likely to generate BGM. "
            "Use clean voice-only samples for best results."
        )

        # Another best-practice: greeting-style intros can trigger BGM.
        add(
            "Greeting-style intros (e.g., 'Welcome to…', 'Hello and welcome…') can increase the likelihood of background music. "
            "If you hear music, try removing those phrases from the script."
        )

        return warnings


# Global voice manager instance
voice_manager = VoiceManager()