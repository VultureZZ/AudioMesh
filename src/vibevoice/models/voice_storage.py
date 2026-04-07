"""
Voice metadata storage using JSON files.

Thread-safe operations for managing voice metadata.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..config import config


class VoiceStorage:
    """Thread-safe voice metadata storage."""

    def __init__(self, storage_file: Optional[Path] = None):
        """
        Initialize voice storage.

        Args:
            storage_file: Path to JSON storage file (defaults to custom_voices_dir/voice_metadata.json)
        """
        if storage_file is None:
            storage_file = config.CUSTOM_VOICES_DIR / "voice_metadata.json"
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        """Ensure storage file exists with proper structure."""
        if not self.storage_file.exists():
            with self.lock:
                self.storage_file.parent.mkdir(parents=True, exist_ok=True)
                # `voices`: custom voice metadata keyed by voice_id
                # `profiles`: optional profiles for any voice_id (including default voices)
                initial_data = {"voices": {}, "profiles": {}}
                self.storage_file.write_text(json.dumps(initial_data, indent=2))

    def _load(self) -> Dict:
        """Load metadata from storage file."""
        try:
            if self.storage_file.exists():
                content = self.storage_file.read_text()
                data = json.loads(content)
                if not isinstance(data, dict):
                    return {"voices": {}, "profiles": {}}
                # Back-compat: older files may only have `voices`
                data.setdefault("voices", {})
                data.setdefault("profiles", {})
                return data
            return {"voices": {}, "profiles": {}}
        except (json.JSONDecodeError, IOError):
            return {"voices": {}, "profiles": {}}

    def _save(self, data: Dict) -> None:
        """Save metadata to storage file."""
        with self.lock:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            data.setdefault("voices", {})
            data.setdefault("profiles", {})
            self.storage_file.write_text(json.dumps(data, indent=2))

    def add_voice(
        self,
        voice_id: str,
        name: str,
        description: Optional[str] = None,
        audio_files: Optional[List[str]] = None,
        profile: Optional[Dict] = None,
        language_code: Optional[str] = None,
        gender: Optional[str] = None,
        image_filename: Optional[str] = None,
        quality_analysis: Optional[Dict] = None,
        speaker_embedding: Optional[List[float]] = None,
        voice_type: str = "custom",
    ) -> None:
        """
        Add a new voice to storage.

        Args:
            voice_id: Unique voice identifier
            name: Voice name
            description: Voice description
            audio_files: List of audio file names (for type custom)
            profile: Optional voice profile data (for custom: transcript etc.; for voice_design: voice_design_prompt)
            voice_type: "custom" (from audio) or "voice_design" (from text description)
            image_filename: Optional image filename (e.g. image.jpg) for voice avatar
            quality_analysis: Optional audio quality analysis (clone_quality, issues, etc.)
            speaker_embedding: Optional ECAPA embedding for matching in transcript service
        """
        data = self._load()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        voice_data = {
            "name": name,
            "description": description or "",
            "type": voice_type,
            "created_at": now,
            "audio_files": audio_files or [],
        }
        if language_code:
            voice_data["language_code"] = language_code
        if gender:
            voice_data["gender"] = gender
        if profile:
            voice_data["profile"] = profile
        if image_filename:
            voice_data["image_filename"] = image_filename
        if quality_analysis:
            voice_data["quality_analysis"] = quality_analysis
        if speaker_embedding:
            voice_data["speaker_embedding"] = speaker_embedding
        data["voices"][voice_id] = voice_data
        self._save(data)

    def get_voice(self, voice_id: str) -> Optional[Dict]:
        """
        Get voice metadata by ID.

        Args:
            voice_id: Voice identifier

        Returns:
            Voice metadata dict or None if not found
        """
        data = self._load()
        voice = data["voices"].get(voice_id)
        if voice:
            voice["id"] = voice_id
        return voice

    def list_voices(self) -> List[Dict]:
        """
        List all custom voices.

        Returns:
            List of voice metadata dicts
        """
        data = self._load()
        voices = []
        for voice_id, voice_data in data["voices"].items():
            voice_data["id"] = voice_id
            voices.append(voice_data)
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice from storage.

        Args:
            voice_id: Voice identifier

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        if voice_id in data["voices"]:
            del data["voices"][voice_id]
            self._save(data)
            return True
        return False

    def voice_exists(self, voice_id: str) -> bool:
        """
        Check if a voice exists.

        Args:
            voice_id: Voice identifier

        Returns:
            True if voice exists, False otherwise
        """
        data = self._load()
        return voice_id in data["voices"]

    def name_exists(self, name: str, exclude_voice_id: Optional[str] = None) -> bool:
        """
        Check if a voice name already exists.

        Args:
            name: Voice name to check
            exclude_voice_id: Voice ID to exclude from check (for updates)

        Returns:
            True if name exists, False otherwise
        """
        data = self._load()
        for voice_id, voice_data in data["voices"].items():
            if exclude_voice_id and voice_id == exclude_voice_id:
                continue
            if voice_data["name"].lower() == name.lower():
                return True
        return False

    def update_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        language_code: Optional[str] = None,
        gender: Optional[str] = None,
        image_filename: Optional[str] = None,
        speaker_embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update voice name, description, and/or image.

        Args:
            voice_id: Voice identifier
            name: New voice name (optional)
            description: New voice description (optional)
            image_filename: New image filename, or empty string to clear (optional)

        Returns:
            True if updated, False if not found
        """
        data = self._load()
        if voice_id not in data["voices"]:
            return False

        if name is not None:
            data["voices"][voice_id]["name"] = name
        if description is not None:
            data["voices"][voice_id]["description"] = description
        if language_code is not None:
            if language_code:
                data["voices"][voice_id]["language_code"] = language_code
            else:
                data["voices"][voice_id].pop("language_code", None)
        if gender is not None:
            if gender:
                data["voices"][voice_id]["gender"] = gender
            else:
                data["voices"][voice_id].pop("gender", None)
        if image_filename is not None:
            if image_filename:
                data["voices"][voice_id]["image_filename"] = image_filename
            else:
                data["voices"][voice_id].pop("image_filename", None)
        if speaker_embedding is not None:
            if speaker_embedding:
                data["voices"][voice_id]["speaker_embedding"] = speaker_embedding
            else:
                data["voices"][voice_id].pop("speaker_embedding", None)

        self._save(data)
        return True

    def update_voice_profile(self, voice_id: str, profile: Dict) -> bool:
        """
        Update voice profile data.

        Args:
            voice_id: Voice identifier
            profile: Profile data dictionary

        Returns:
            True if updated, False if not found
        """
        data = self._load()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # If this is a custom voice, store profile inside the voice record.
        if voice_id in data["voices"]:
            if "profile" not in data["voices"][voice_id]:
                data["voices"][voice_id]["profile"] = {}

            data["voices"][voice_id]["profile"].update(profile)
            data["voices"][voice_id]["profile"]["updated_at"] = now
            if "created_at" not in data["voices"][voice_id]["profile"]:
                data["voices"][voice_id]["profile"]["created_at"] = now
        else:
            # Otherwise store it in the global `profiles` mapping so we can
            # attach profiles to default voices without polluting the custom-voice list.
            if voice_id not in data["profiles"]:
                data["profiles"][voice_id] = {}
            data["profiles"][voice_id].update(profile)
            data["profiles"][voice_id]["updated_at"] = now
            if "created_at" not in data["profiles"][voice_id]:
                data["profiles"][voice_id]["created_at"] = now

        self._save(data)
        return True

    def get_voice_profile(self, voice_id: str) -> Optional[Dict]:
        """
        Get voice profile by ID.

        Args:
            voice_id: Voice identifier

        Returns:
            Voice profile dict or None if not found
        """
        data = self._load()
        voice = data["voices"].get(voice_id)
        if voice and isinstance(voice, dict):
            profile = voice.get("profile")
            if profile:
                return profile

        profile = data.get("profiles", {}).get(voice_id)
        if profile and isinstance(profile, dict):
            return profile

        return None


# Global storage instance
voice_storage = VoiceStorage()
