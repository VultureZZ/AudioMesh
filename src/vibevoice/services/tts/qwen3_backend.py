"""
Qwen3-TTS backend using the official qwen-tts package.

Uses CustomVoice model for built-in speakers and Base model for voice cloning.
"""
import gc
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config import config
from .base import SpeakerRef, TTSBackend
from .segments import TranscriptSegment

logger = logging.getLogger(__name__)

# Map our default voice names (Alice, Frank, etc.) to Qwen3 CustomVoice speaker names.
# Qwen3 1.7B CustomVoice speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
QWEN3_DEFAULT_VOICE_MAPPING = {
    "Alice": "Vivian",
    "Frank": "Ryan",
    "Mary": "Serena",
    "Carter": "Aiden",
    "Maya": "Sohee",
}

# Map short language codes to Qwen3 language names
QWEN3_LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _get_qwen3_language(lang: str) -> str:
    """Map short code (e.g. en) to Qwen3 language name (e.g. English)."""
    normalized = (lang or "en").strip().lower()
    return QWEN3_LANGUAGE_MAP.get(normalized, "English")


# Expand short emotion keywords into more descriptive instructions for better model response.
# Qwen3-TTS responds better to explicit phrasing like "Speak in an X tone" than single words.
_EMOTION_EXPANSIONS = {
    "angry": "Speak in an angry, frustrated tone.",
    "excited": "Speak in an excited, enthusiastic tone.",
    "happy": "Speak in a happy, cheerful tone.",
    "sad": "Speak in a sad, somber tone.",
    "calm": "Speak in a calm, relaxed tone.",
    "nervous": "Speak in a nervous, anxious tone.",
    "serious": "Speak in a serious, solemn tone.",
    "playful": "Speak in a playful, lighthearted tone.",
    "whisper": "Speak in a soft whisper.",
    "shouting": "Speak loudly, as if shouting.",
}


def _expand_instruct(instruct: str) -> str:
    """Expand short emotion keywords for clearer model instruction."""
    if not instruct or not instruct.strip():
        return ""
    s = instruct.strip().lower()
    # Only expand if it looks like a single short keyword (no punctuation, few words)
    if len(s) <= 20 and " " not in s:
        expanded = _EMOTION_EXPANSIONS.get(s)
        if expanded:
            return expanded
    return instruct.strip()


class Qwen3Backend(TTSBackend):
    """
    TTS backend using Qwen3-TTS (CustomVoice for built-in, Base for cloning).

    Lazy-loads CustomVoice and Base models on first use; unloads after idle
    (configurable via TTS_MODEL_IDLE_UNLOAD_SECONDS, default 15s) to free memory.
    Caches voice_clone_prompt per ref_audio_path for custom voices.
    """

    def __init__(
        self,
        custom_voice_model: Optional[str] = None,
        base_model: Optional[str] = None,
        device_map: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        self._custom_voice_model = (
            custom_voice_model or getattr(config, "QWEN_TTS_CUSTOM_VOICE_MODEL", None)
            or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        self._base_model = (
            base_model or getattr(config, "QWEN_TTS_BASE_MODEL", None)
            or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        self._voice_design_model = (
            getattr(config, "QWEN_TTS_VOICE_DESIGN_MODEL", None)
            or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )
        self._device_map = device_map or getattr(config, "QWEN_TTS_DEVICE", "cuda:0")
        self._dtype_str = dtype or getattr(config, "QWEN_TTS_DTYPE", "bfloat16")
        self._custom_voice_model_instance: Any = None
        self._base_model_instance: Any = None
        self._voice_design_model_instance: Any = None
        self._clone_prompt_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._unload_timer: Optional[threading.Timer] = None
        self._idle_unload_seconds: int = getattr(
            config, "TTS_MODEL_IDLE_UNLOAD_SECONDS", 15
        )

    def _model_kwargs(self):
        """Build common kwargs for from_pretrained (device, dtype, attn). Only use flash_attention_2 if installed."""
        import torch
        dtype = getattr(torch, self._dtype_str, torch.bfloat16)
        kwargs = {"device_map": self._device_map, "dtype": dtype}
        if self._dtype_str in ("float16", "bfloat16"):
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.debug("flash_attn not installed; using default attention")
        return kwargs

    def _do_unload(self) -> None:
        """Unload all models and clear caches to free memory. Called by idle timer."""
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer = None
            if (
                self._custom_voice_model_instance is None
                and self._base_model_instance is None
                and self._voice_design_model_instance is None
            ):
                return
            logger.info("Unloading Qwen3-TTS models after idle timeout")
            models = [
                self._custom_voice_model_instance,
                self._base_model_instance,
                self._voice_design_model_instance,
            ]
            self._custom_voice_model_instance = None
            self._base_model_instance = None
            self._voice_design_model_instance = None
            self._clone_prompt_cache.clear()
        try:
            import torch

            # Some wrappers keep internal module refs; move any underlying module to CPU first.
            for model in models:
                if model is None:
                    continue
                try:
                    inner = getattr(model, "model", None)
                    if inner is not None and hasattr(inner, "to"):
                        inner.to("cpu")
                    elif hasattr(model, "to"):
                        model.to("cpu")
                except Exception:
                    logger.debug("Best-effort CPU offload failed during model unload", exc_info=True)

            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)
                torch.cuda.synchronize()
                del models
                gc.collect()
                torch.cuda.empty_cache()
                # Also release CUDA IPC memory held by allocator internals.
                torch.cuda.ipc_collect()
                allocated_after = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
                logger.info(
                    "VRAM cleanup complete: allocated %.0f->%.0f MiB, reserved %.0f->%.0f MiB",
                    allocated_before,
                    allocated_after,
                    reserved_before,
                    reserved_after,
                )
            else:
                del models
                gc.collect()
        except Exception:
            logger.warning("Qwen3 unload cleanup encountered an error", exc_info=True)

    def _schedule_unload(self) -> None:
        """Schedule model unload after idle timeout. Call with lock held."""
        if self._idle_unload_seconds <= 0:
            return
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None
        self._unload_timer = threading.Timer(
            self._idle_unload_seconds,
            self._do_unload,
        )
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _get_custom_voice_model(self):
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            if self._custom_voice_model_instance is None:
                from qwen_tts import Qwen3TTSModel

                logger.info("Loading Qwen3-TTS CustomVoice model: %s", self._custom_voice_model)
                self._custom_voice_model_instance = Qwen3TTSModel.from_pretrained(
                    self._custom_voice_model,
                    **self._model_kwargs(),
                )
            self._schedule_unload()
            return self._custom_voice_model_instance

    def _get_base_model(self):
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            if self._base_model_instance is None:
                from qwen_tts import Qwen3TTSModel

                logger.info("Loading Qwen3-TTS Base model: %s", self._base_model)
                self._base_model_instance = Qwen3TTSModel.from_pretrained(
                    self._base_model,
                    **self._model_kwargs(),
                )
            self._schedule_unload()
            return self._base_model_instance

    def _get_voice_design_model(self):
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            if self._voice_design_model_instance is None:
                from qwen_tts import Qwen3TTSModel

                logger.info("Loading Qwen3-TTS VoiceDesign model: %s", self._voice_design_model)
                self._voice_design_model_instance = Qwen3TTSModel.from_pretrained(
                    self._voice_design_model,
                    **self._model_kwargs(),
                )
            self._schedule_unload()
            return self._voice_design_model_instance

    def _get_or_create_clone_prompt(self, ref_audio_path: Path, ref_text: Optional[str]) -> Any:
        """Get cached voice_clone_prompt or create and cache it."""
        cache_key = str(ref_audio_path)
        if ref_text:
            cache_key += "|" + ref_text[:200]
        if cache_key in self._clone_prompt_cache:
            return self._clone_prompt_cache[cache_key]
        model = self._get_base_model()
        ref_audio = str(ref_audio_path) if ref_audio_path else None
        if not ref_audio or not Path(ref_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        kwargs = {"ref_audio": ref_audio, "x_vector_only_mode": not ref_text}
        if ref_text:
            kwargs["ref_text"] = ref_text
        prompt = model.create_voice_clone_prompt(**kwargs)
        self._clone_prompt_cache[cache_key] = prompt
        return prompt

    def _generate_segment(
        self,
        text: str,
        speaker_ref: SpeakerRef,
        language: str,
    ) -> tuple:
        """Generate audio for one segment. Returns (wav_array, sample_rate)."""
        import numpy as np

        qwen_lang = _get_qwen3_language(language)
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 24000

        style_instruct = (speaker_ref.instruct or "").strip() if getattr(speaker_ref, "instruct", None) else ""
        style_instruct = _expand_instruct(style_instruct)

        if getattr(speaker_ref, "use_voice_design", False) and getattr(
            speaker_ref, "voice_design_instructions", None
        ):
            model = self._get_voice_design_model()
            voice_desc = (speaker_ref.voice_design_instructions or "").strip()
            # Combine voice description with per-segment style instruction when provided
            instruct = voice_desc
            if style_instruct:
                instruct = f"{voice_desc}. {style_instruct}" if voice_desc else style_instruct
            # Ensure we use VoiceDesign model (has tts_model_type "voice_design")
            tts_type = getattr(model.model, "tts_model_type", None)
            if tts_type != "voice_design":
                raise ValueError(
                    f"VoiceDesign segment requires VoiceDesign model, got tts_model_type={tts_type}. "
                    "Check QWEN_TTS_VOICE_DESIGN_MODEL config."
                )
            # Tuned for adherence to description (inspired by qwen3-TTS-studio)
            gen_kwargs = {
                "temperature": 0.35,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "subtalker_temperature": 0.35,
                "subtalker_top_k": 50,
                "subtalker_top_p": 0.9,
            }
            wavs, sr = model.generate_voice_design(
                text=text.strip(),
                instruct=instruct,
                language=qwen_lang,
                **gen_kwargs,
            )
            return wavs[0], sr
        if speaker_ref.use_custom_voice:
            model = self._get_custom_voice_model()
            speaker_name = speaker_ref.speaker_id or "Ryan"
            wavs, sr = model.generate_custom_voice(
                text=text.strip(),
                language=qwen_lang,
                speaker=speaker_name,
                instruct=style_instruct or None,
            )
            return wavs[0], sr
        else:
            model = self._get_base_model()
            if speaker_ref.voice_clone_prompt is not None:
                prompt = speaker_ref.voice_clone_prompt
            elif speaker_ref.ref_audio_path and speaker_ref.ref_audio_path.exists():
                prompt = self._get_or_create_clone_prompt(
                    speaker_ref.ref_audio_path,
                    speaker_ref.ref_text,
                )
            else:
                raise ValueError(
                    "Custom voice segment requires ref_audio_path or voice_clone_prompt"
                )
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=qwen_lang,
                voice_clone_prompt=prompt,
            )
            return wavs[0], sr

    def generate(
        self,
        segments: List[TranscriptSegment],
        speaker_refs: List[SpeakerRef],
        language: str,
        output_path: Path,
        progress_callback: Optional[Any] = None,
    ) -> Path:
        """Generate speech from segments and concatenate into one WAV."""
        import numpy as np
        import soundfile as sf

        if not segments:
            raise ValueError("No segments to generate")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(segments)
        if progress_callback:
            progress_callback(0, total, "Starting generation...")

        all_wavs: List[np.ndarray] = []
        sample_rate = 24000

        for i, seg in enumerate(segments):
            if seg.speaker_index >= len(speaker_refs):
                raise ValueError(
                    f"Segment speaker_index {seg.speaker_index} out of range (have {len(speaker_refs)} speaker_refs)"
                )
            ref = speaker_refs[seg.speaker_index]
            wav, sr = self._generate_segment(seg.text, ref, language)
            if progress_callback:
                progress_callback(i + 1, total, f"Generated segment {i + 1} of {total}")
            if wav is not None and len(wav) > 0:
                if not all_wavs:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning("Segment sample rate %s != %s; using first segment sr", sr, sample_rate)
                all_wavs.append(wav.astype(np.float32))

        if not all_wavs:
            raise RuntimeError("No audio generated from segments")

        concatenated = np.concatenate(all_wavs)
        sf.write(str(output_path), concatenated, sample_rate)
        logger.info("Wrote Qwen3-TTS output: %s (%s samples, %s Hz)", output_path, len(concatenated), sample_rate)
        return output_path


def resolve_speaker_to_qwen3_ref(
    speaker_name: str,
    voice_manager: Any,
    clone_prompt_cache: Optional[Dict[str, Any]] = None,
) -> SpeakerRef:
    """
    Resolve a speaker name (from our API) to a SpeakerRef for Qwen3Backend.

    - If speaker is a default (Alice, Frank, etc.), map to Qwen3 CustomVoice speaker.
    - If speaker is a custom voice, return SpeakerRef with ref_audio_path and optional ref_text.
    """
    from .base import SpeakerRef

    normalized = voice_manager.normalize_voice_name(speaker_name)
    qwen_speaker = QWEN3_DEFAULT_VOICE_MAPPING.get(normalized)
    if qwen_speaker is not None:
        return SpeakerRef(use_custom_voice=True, speaker_id=qwen_speaker)

    voice_data = voice_manager.get_voice_by_name(normalized) or voice_manager.get_voice_by_name(speaker_name)
    if not voice_data:
        raise ValueError(f"Voice not found: {speaker_name}")

    voice_type = voice_data.get("type", "custom")

    if voice_type == "voice_design":
        profile = voice_data.get("profile") or {}
        instructions = profile.get("voice_design_prompt") or voice_data.get("voice_design_prompt") or ""
        if not (instructions and isinstance(instructions, str) and instructions.strip()):
            raise ValueError(f"VoiceDesign voice '{speaker_name}' has no voice_design_prompt")
        return SpeakerRef(
            use_custom_voice=False,
            use_voice_design=True,
            voice_design_instructions=instructions.strip(),
        )

    if voice_type != "custom":
        raise ValueError(f"Voice not found or not custom: {speaker_name}")

    voice_id = voice_data.get("id")
    if not voice_id:
        raise ValueError(f"Custom voice has no id: {speaker_name}")

    path = voice_manager.get_voice_path(voice_id)
    if path is None or not path.exists():
        raise ValueError(f"Voice file not found for: {speaker_name}")

    ref_text = None
    if voice_data and isinstance(voice_data.get("profile"), dict):
        profile = voice_data["profile"]
        if isinstance(profile.get("transcript"), str):
            ref_text = profile["transcript"]

    return SpeakerRef(
        use_custom_voice=False,
        ref_audio_path=path,
        ref_text=ref_text or None,
    )
