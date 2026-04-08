"""
Infer human-readable speaker labels for Voice Isolator from short intro audio:
transcribe early speech per diarization speaker, then regex (and optionally Ollama) for self-introductions.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydub import AudioSegment

from ..config import config
from ..core.transcripts.transcriber import transcript_transcriber

logger = logging.getLogger(__name__)

_NAME_BLOCKLIST = frozenset(
    {
        "sure",
        "here",
        "going",
        "not",
        "very",
        "sorry",
        "back",
        "fine",
        "okay",
        "ok",
        "ready",
        "happy",
        "glad",
        "excited",
        "speaking",
        "today",
        "recording",
        "host",
        "the",
        "a",
        "an",
    }
)


def _safe_speaker_file_id(pyannote_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", pyannote_id)[:48] or "spk"


def slice_intro_audio(
    full: AudioSegment,
    intervals: list[tuple[float, float]],
    max_sec: float,
) -> Optional[AudioSegment]:
    """Concatenate this speaker's speech from the start of the file up to max_sec seconds."""
    if not intervals:
        return None
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    combined = AudioSegment.silent(duration=0)
    max_ms = int(max_sec * 1000)
    for a, b in sorted_iv:
        if len(combined) >= max_ms:
            break
        start_ms = int(max(0.0, a) * 1000)
        end_ms = int(max(start_ms + 1, b) * 1000)
        chunk = full[start_ms:end_ms]
        combined += chunk
        if len(combined) > max_ms:
            combined = combined[:max_ms]
            break
    if len(combined) < 800:
        return None
    return combined


def _whisper_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for seg in result.get("segments") or []:
        t = (seg.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def sanitize_display_name(name: str) -> Optional[str]:
    name = re.sub(r"\s+", " ", (name or "").strip())
    if len(name) < 2 or len(name) > 48:
        return None
    words = name.split()
    if len(words) > 4:
        return None
    if any(ch.isdigit() for ch in name):
        return None
    if not re.match(r"^[\w\s'\-\.]+$", name, re.UNICODE):
        return None
    # Title-case words conservatively
    titled = " ".join(w[:1].upper() + w[1:].lower() if w else w for w in words)
    return titled


def extract_name_regex(text: str) -> Optional[str]:
    """Match common English self-introduction phrases."""
    t = " ".join(text.split())
    if len(t) < 4:
        return None
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,3})\b",
        r"\b(?:i am|i'm|i m)\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,3})\b",
        r"\bcall me\s+([A-Za-z][A-Za-z'\-]+)\b",
        r"\bthis is\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})\b",
        r"\bits\s+([A-Za-z][A-Za-z'\-]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).strip()
        cleaned = sanitize_display_name(raw)
        if not cleaned:
            continue
        first = cleaned.split()[0].lower()
        if first in _NAME_BLOCKLIST:
            continue
        return cleaned
    return None


def dedupe_speaker_labels(speakers: list[dict[str, Any]]) -> None:
    """If two speakers get the same inferred label, suffix the second as 'Name (2)'."""
    seen: dict[str, int] = {}
    for sp in speakers:
        label = (sp.get("label") or "").strip()
        key = label.lower()
        c = seen.get(key, 0)
        seen[key] = c + 1
        if c > 0:
            sp["label"] = f"{label} ({c + 1})"


async def infer_speaker_labels_for_isolation(
    *,
    full_audio: AudioSegment,
    by_speaker: dict[str, list[tuple[float, float]]],
    speakers_payload: list[dict[str, Any]],
    job_dir: Path,
) -> None:
    mode = (config.SPEAKER_NAME_INFERENCE or "regex").strip().lower()
    if mode in ("", "off", "false", "0", "no"):
        return
    max_sec = float(getattr(config, "SPEAKER_NAME_INTRO_MAX_SECONDS", 45.0) or 45.0)
    max_sec = max(15.0, min(120.0, max_sec))

    use_regex = mode in ("regex", "both")
    use_ollama = mode in ("ollama", "both")

    for sp in speakers_payload:
        pid = sp.get("speaker_id")
        if not pid:
            continue
        intervals = by_speaker.get(str(pid), [])
        intro = slice_intro_audio(full_audio, intervals, max_sec)
        if intro is None:
            continue
        sid = _safe_speaker_file_id(str(pid))
        intro_path = job_dir / f"intro_{sid}.wav"
        try:
            await asyncio.to_thread(intro.export, str(intro_path), format="wav")
            result = await transcript_transcriber.transcribe(str(intro_path), language="en")
            text = _whisper_text(result)
        except Exception:
            logger.debug("Intro transcription failed for %s", pid, exc_info=True)
            continue
        finally:
            try:
                intro_path.unlink(missing_ok=True)
            except OSError:
                pass

        if not text:
            continue

        label: Optional[str] = None
        if use_regex:
            label = extract_name_regex(text)
        if label is None and use_ollama:
            try:
                from .ollama_client import ollama_client

                raw = await asyncio.to_thread(ollama_client.infer_speaker_display_name_from_transcript, text)
                if raw:
                    label = sanitize_display_name(raw)
            except Exception:
                logger.debug("Ollama name inference failed for %s", pid, exc_info=True)

        if label:
            sp["label"] = label
            sp["label_source"] = "inferred"

    dedupe_speaker_labels(speakers_payload)
