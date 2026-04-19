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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

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

    def validate_speakers(self, speakers: List[str]) -> List[str]:
        """
        Validate that all speakers exist.

        Args:
            speakers: List of speaker names (canonical or mapped format)

        Returns:
            Canonical voice names in request order (same length as ``speakers``).

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
        voice_map = ", ".join(
            f"Speaker {i + 1}→{normalized_speakers[i]}" for i in range(len(normalized_speakers))
        )
        logger.info("  Script Speaker N uses TTS voice: %s", voice_map)
        return normalized_speakers

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
        voice_direction: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
        breath_audio_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate speech from transcript.

        Uses Qwen3-TTS backend by default; set TTS_BACKEND=vibevoice for legacy subprocess.
        speaker_instructions: optional list of style/emotion instructions (one per speaker).

        When ``voice_direction`` is set (list of dicts or VoiceDirectionLine-like rows aligned
        to dialogue lines), Qwen3-TTS receives per-utterance style instructions and optional
        ``pause_after_ms`` silences between lines (before WhisperX alignment).
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
            voice_direction=voice_direction,
            breath_audio_path=breath_audio_path,
        )

    @staticmethod
    def build_qwen3_line_instruct(emotion: str, emphasis_words: Optional[Sequence[str]] = None) -> str:
        """Natural-language instruct line for one utterance (merged with SpeakerRef.instruct)."""
        from .tts.qwen3_backend import _expand_instruct

        em = (emotion or "neutral").strip().lower()
        base = _expand_instruct(em) or _expand_instruct("neutral") or "Speak clearly."
        words = [w.strip() for w in (emphasis_words or []) if w and str(w).strip()]
        if words:
            stress = ", ".join(words[:12])
            return f"{base} Emphasize these words slightly: {stress}."
        return str(base)

    def _apply_voice_direction_to_segments(
        self,
        segments: List[Any],
        voice_direction: Sequence[Union[Dict[str, Any], Any]],
    ) -> None:
        """Mutate ``TranscriptSegment`` rows with per-line instruct and pause_after_ms."""

        def _li(r: Union[Dict[str, Any], Any]) -> int:
            if isinstance(r, dict):
                return int(r.get("line_index", 0))
            return int(getattr(r, "line_index", 0))

        rows = sorted(voice_direction, key=_li)
        for i, seg in enumerate(segments):
            if i >= len(rows):
                break
            row = rows[i]
            if isinstance(row, dict):
                em = str(row.get("emotion") or "neutral")
                emph = row.get("emphasis_words") or []
                pause = int(row.get("pause_after_ms") or 0)
            else:
                em = str(getattr(row, "emotion", None) or "neutral")
                emph = list(getattr(row, "emphasis_words", None) or [])
                pause = int(getattr(row, "pause_after_ms", 0) or 0)
            seg.instruct = self.build_qwen3_line_instruct(em, emph)
            seg.pause_after_ms = max(0, pause)

    def _load_breath_mono(self, path: Path, target_sr: int) -> Any:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        mono = data[:, 0] if data.shape[1] else data.reshape(-1)
        if sr != target_sr and len(mono) > 0:
            n_dst = int(round(len(mono) * target_sr / float(sr)))
            t_src = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
            t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
            mono = np.interp(t_dst, t_src, mono).astype(np.float32)
        return mono.astype(np.float32)

    def _generate_speech_backend(
        self,
        transcript: str,
        speakers: List[str],
        output_path: Path,
        language: str = "en",
        speaker_instructions: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        voice_direction: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
        breath_audio_path: Optional[Path] = None,
    ) -> Path:
        """Generate using TTS backend (Qwen3-TTS)."""
        import random

        from .tts import parse_transcript_into_segments
        from .tts.base import SpeakerRef
        from .tts.qwen3_backend import Qwen3Backend, resolve_speaker_to_qwen3_ref

        backend = self._get_backend()
        if backend is None:
            raise RuntimeError("TTS backend not available")

        formatted = self.format_transcript(transcript, speakers)
        segments = parse_transcript_into_segments(formatted, len(speakers))
        if not segments:
            raise ValueError("No segments to generate (empty or unparseable transcript)")

        if voice_direction:
            self._apply_voice_direction_to_segments(list(segments), voice_direction)

        speaker_refs: List[SpeakerRef] = []
        for name in speakers:
            ref = resolve_speaker_to_qwen3_ref(name, voice_manager)
            speaker_refs.append(ref)
        if speaker_instructions:
            for i, instr in enumerate(speaker_instructions):
                if i < len(speaker_refs) and instr and isinstance(instr, str):
                    speaker_refs[i].instruct = instr.strip()

        breath_np = None
        breath_idx = None
        if isinstance(backend, Qwen3Backend):
            from app.services.voice_prosody import breath_after_indices, resolve_breath_audio_path

            env_breath = getattr(config, "BREATH_SFX_PATH", None) or os.getenv("BREATH_SFX_PATH")
            bpath = breath_audio_path or resolve_breath_audio_path(
                Path(env_breath) if env_breath else None
            )
            breath_idx, _stride = breath_after_indices(len(segments), random.Random())
            if bpath and bpath.is_file():
                try:
                    breath_np = self._load_breath_mono(bpath, 24000)
                except Exception as exc:
                    logger.warning("Could not load breath sample %s: %s", bpath, exc)

            backend.generate(
                segments,
                speaker_refs,
                language,
                output_path,
                progress_callback,
                breath_audio=breath_np,
                breath_after_segment_indices=breath_idx if breath_np is not None else None,
            )
        else:
            backend.generate(segments, speaker_refs, language, output_path, progress_callback)
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

    def tts_has_inflight_generation(self) -> bool:
        """True while Qwen3-TTS (or another backend) is inside generate()."""
        if self._use_legacy:
            return False
        backend = self._backend
        if backend is None:
            return False
        check = getattr(backend, "has_inflight_generation", None)
        return bool(callable(check) and check())

    def release_gpu_memory_after_speech(self) -> None:
        """
        Unload in-process TTS weights and release CUDA caches so another workload (e.g.
        ACE-Step in a subprocess) can use the GPU.
        """
        if self._use_legacy:
            return
        backend = self._backend
        if backend is None:
            return
        unload = getattr(backend, "unload_models_immediately", None)
        if callable(unload):
            unload()
        from ..gpu_memory import release_torch_cuda_memory

        release_torch_cuda_memory()


# Global voice generator instance
voice_generator = VoiceGenerator()
