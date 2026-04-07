"""
Voice generation service.

Supports Qwen3-TTS (default) and legacy VibeVoice subprocess.
"""
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from ..config import config
from .voice_manager import voice_manager

logger = logging.getLogger(__name__)


def _get_tts_backend():
    """Return the configured TTS backend instance (lazy)."""
    backend_name = (config.TTS_BACKEND or "qwen3").strip().lower()
    if backend_name == "qwen3":
        from .tts import Qwen3Backend
        return Qwen3Backend()
    if backend_name == "vibevoice":
        return None
    logger.warning("Unknown TTS_BACKEND=%s; using qwen3", backend_name)
    from .tts import Qwen3Backend
    return Qwen3Backend()


class VoiceGenerator:
    """Service for generating speech from text."""

    def __init__(self):
        """Initialize voice generator."""
        self.output_dir = config.OUTPUT_DIR
        self._backend = None
        self._use_legacy = (config.TTS_BACKEND or "qwen3").strip().lower() == "vibevoice"
        if self._use_legacy:
            self.model_path = config.MODEL_PATH
            self.vibevoice_repo_dir = config.VIBEVOICE_REPO_DIR
            self.inference_script = self.vibevoice_repo_dir / "demo" / "inference_from_file.py"
            logger.info("VoiceGenerator initialized (legacy VibeVoice)")
            logger.info("  Model path: %s", self.model_path)
            logger.info("  VibeVoice repo: %s", self.vibevoice_repo_dir)
        else:
            logger.info("VoiceGenerator initialized (TTS backend: %s)", config.TTS_BACKEND or "qwen3")
        logger.info("  Output directory: %s", self.output_dir)

    def _get_backend(self):
        if self._backend is None and not self._use_legacy:
            self._backend = _get_tts_backend()
        return self._backend

    def validate_speakers(self, speakers: List[str]) -> None:
        """
        Validate that all speakers exist.

        Args:
            speakers: List of speaker names (canonical or mapped format)

        Raises:
            ValueError: If any speaker is invalid
        """
        logger.debug("Validating speakers: %s", speakers)

        normalized_speakers = [voice_manager.normalize_voice_name(s) for s in speakers]
        logger.debug("Normalized speakers: %s", normalized_speakers)

        all_voices = voice_manager.list_all_voices()
        available_names = {v["name"] for v in all_voices}
        logger.debug("Available voices: %s", sorted(available_names))

        available_names_lower = {name.lower(): name for name in available_names}
        invalid_speakers = []
        for normalized in normalized_speakers:
            if normalized.lower() not in available_names_lower:
                invalid_speakers.append(normalized)

        if invalid_speakers:
            logger.error("Invalid speakers detected: %s", invalid_speakers)
            raise ValueError(f"Invalid speakers: {', '.join(invalid_speakers)}")

        logger.info("Speaker validation passed: %s (normalized: %s)", speakers, normalized_speakers)

    def create_transcript_file(self, transcript: str) -> Path:
        """Create a temporary transcript file (used by legacy path)."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            prefix="transcript_",
        )
        temp_file.write(transcript)
        temp_file.close()
        return Path(temp_file.name)

    def generate_speech(
        self,
        transcript: str,
        speakers: List[str],
        output_filename: Optional[str] = None,
        language: Optional[str] = None,
        speaker_instructions: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """
        Generate speech from transcript.

        Uses Qwen3-TTS backend by default; set TTS_BACKEND=vibevoice for legacy subprocess.
        speaker_instructions: optional list of style/emotion instructions (one per speaker).
        """
        logger.info("Starting speech generation process...")
        logger.info("  Transcript length: %s characters", len(transcript))
        logger.info("  Speakers: %s", speakers)
        logger.info("  Output filename: %s", output_filename or "auto-generated")

        self.validate_speakers(speakers)
        if speaker_instructions is not None and len(speaker_instructions) != len(speakers):
            raise ValueError(
                f"speaker_instructions length ({len(speaker_instructions)}) must match speakers ({len(speakers)})"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_generated.wav"
        output_path = self.output_dir / output_filename

        if self._use_legacy:
            return self._generate_speech_legacy(transcript, speakers, output_path)
        return self._generate_speech_backend(
            transcript,
            speakers,
            output_path,
            language or "en",
            speaker_instructions,
            progress_callback,
        )

    def _generate_speech_backend(
        self,
        transcript: str,
        speakers: List[str],
        output_path: Path,
        language: str = "en",
        speaker_instructions: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Generate using TTS backend (Qwen3-TTS)."""
        from .tts import parse_transcript_into_segments
        from .tts.base import SpeakerRef
        from .tts.qwen3_backend import resolve_speaker_to_qwen3_ref

        backend = self._get_backend()
        if backend is None:
            raise RuntimeError("TTS backend not available")

        formatted = self.format_transcript(transcript, speakers)
        segments = parse_transcript_into_segments(formatted, len(speakers))
        if not segments:
            raise ValueError("No segments to generate (empty or unparseable transcript)")

        speaker_refs: List[SpeakerRef] = []
        for name in speakers:
            ref = resolve_speaker_to_qwen3_ref(name, voice_manager)
            speaker_refs.append(ref)
        if speaker_instructions:
            for i, instr in enumerate(speaker_instructions):
                if i < len(speaker_refs) and instr and isinstance(instr, str):
                    speaker_refs[i].instruct = instr.strip()

        backend.generate(
            segments, speaker_refs, language, output_path, progress_callback
        )
        logger.info("Speech generation completed: %s", output_path)
        return output_path

    def _generate_speech_legacy(
        self,
        transcript: str,
        speakers: List[str],
        output_path: Path,
    ) -> Path:
        """Generate using legacy VibeVoice subprocess."""
        if not self.model_path.exists():
            logger.error("Model not found at %s", self.model_path)
            raise RuntimeError(f"Model not found at {self.model_path}")
        if not self.inference_script.exists():
            logger.error("Inference script not found at %s", self.inference_script)
            raise RuntimeError(f"Inference script not found at {self.inference_script}")

        transcript_file = self.create_transcript_file(transcript)
        try:
            resolved_speakers = []
            for speaker in speakers:
                resolved = voice_manager.resolve_voice_name(speaker)
                resolved_speakers.append(resolved)

            cmd = [
                sys.executable,
                str(self.inference_script),
                "--model_path",
                str(self.model_path),
                "--txt_path",
                str(transcript_file),
                "--speaker_names",
            ] + resolved_speakers

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.vibevoice_repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

            logger.info("Executing VibeVoice inference: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=str(config.PROJECT_ROOT),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            generated_files = list(self.output_dir.glob("*generated.wav"))
            if generated_files:
                latest_file = max(generated_files, key=lambda p: p.stat().st_mtime)
                if latest_file != output_path:
                    latest_file.rename(output_path)
                return output_path
            if output_path.exists():
                return output_path
            raise RuntimeError("Generated audio file not found")
        except subprocess.CalledProcessError as e:
            logger.error("Inference failed: %s", e.stderr or e.stdout or "Unknown error")
            raise RuntimeError(f"Inference failed: {e.stderr or e.stdout or 'Unknown error'}") from e
        finally:
            if transcript_file.exists():
                transcript_file.unlink()

    def format_transcript(self, transcript: str, speakers: List[str]) -> str:
        """
        Format transcript to ensure proper speaker labels.

        If transcript already has speaker labels, return as-is.
        Otherwise format with "Speaker 1: ...", "Speaker 2: ..." in round-robin.
        """
        if "Speaker" in transcript or ":" in transcript:
            return transcript

        lines = transcript.strip().split("\n")
        formatted_lines = []
        speaker_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            speaker_name = speakers[speaker_index % len(speakers)]
            formatted_lines.append(f"Speaker {speaker_index + 1}: {line}")
            speaker_index += 1

        return "\n".join(formatted_lines)


# Global voice generator instance
voice_generator = VoiceGenerator()
