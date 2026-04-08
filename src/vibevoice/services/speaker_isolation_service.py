"""
Async job pipeline: speaker diarization + representative clip extraction for voice cloning.
Uses the same pyannote diarization entrypoint as the transcript service (transcript_diarizer).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import UploadFile
from pydub import AudioSegment

from ..config import config
from ..core.transcripts.diarizer import transcript_diarizer
from ..core.transcripts.transcriber import transcript_transcriber
from ..gpu_memory import release_torch_cuda_memory
from .speaker_name_inference import infer_speaker_labels_for_isolation

logger = logging.getLogger(__name__)

_ISOLATE_FORMATS = frozenset({"mp3", "wav", "m4a", "mp4"})
_MAX_SPEAKERS = 6
_CLIP_MIN_S = 10.0
_CLIP_MAX_S = 15.0
_MERGE_GAP_S = 0.35
_EDGE_AVOID_S = 5.0

_JOB_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CLIP_FILE_RE = re.compile(r"^speaker_([1-6])_clip_([123])\.mp3$")

# Same requirement as transcript diarization (pyannote); checked before upload to avoid wasted transfers.
_HF_TOKEN_MISSING_MSG = (
    "Speaker isolation requires a Hugging Face token for pyannote diarization. "
    "Set HF_TOKEN in the API server environment and accept the model agreements for "
    "pyannote/speaker-diarization-3.1 on huggingface.co."
)


def _ext(name: str) -> str:
    return (Path(name).suffix or "").lower().lstrip(".")


def _merge_close_intervals(intervals: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_iv[0]]
    for a, b in sorted_iv[1:]:
        la, lb = merged[-1]
        if a - lb <= gap_s:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _overlap(a: tuple[float, float], b: tuple[float, float], eps: float = 0.05) -> bool:
    return not (a[1] <= b[0] + eps or b[1] <= a[0] + eps)


def _length_score(duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    if _CLIP_MIN_S <= duration_s <= _CLIP_MAX_S:
        return 1.0
    if duration_s < _CLIP_MIN_S:
        return max(0.0, duration_s / _CLIP_MIN_S) * 0.85
    return max(0.35, (_CLIP_MAX_S / duration_s) * 0.9)


def _position_score(center_s: float, file_duration_s: float) -> float:
    if file_duration_s <= 2 * _EDGE_AVOID_S:
        return 0.5
    if center_s < _EDGE_AVOID_S or center_s > file_duration_s - _EDGE_AVOID_S:
        return 0.25
    return 1.0


def _energy_score_dbfs(segment: AudioSegment) -> float:
    try:
        db = segment.dBFS
    except Exception:
        return 0.35
    if db == float("-inf"):
        return 0.0
    # Typical speech roughly -45 .. -10 dBFS
    t = (db + 50.0) / 40.0
    return max(0.0, min(1.0, t))


def _normalize_to_clip_window(
    full: AudioSegment,
    start_s: float,
    end_s: float,
) -> AudioSegment:
    """Trim/pad segment to 10–15 seconds from the requested window."""
    start_ms = int(max(0.0, start_s) * 1000)
    end_ms = int(max(start_ms + 1, end_s * 1000))
    raw = full[start_ms:end_ms]
    dur_ms = len(raw)
    lo, hi = int(_CLIP_MIN_S * 1000), int(_CLIP_MAX_S * 1000)
    if dur_ms > hi:
        excess = dur_ms - hi
        trim_start = start_ms + excess // 2
        raw = full[trim_start : trim_start + hi]
    elif dur_ms < lo:
        pad_total = lo - dur_ms
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        raw = AudioSegment.silent(duration=pad_l) + raw + AudioSegment.silent(duration=pad_r)
    if raw.frame_rate != 24000:
        raw = raw.set_frame_rate(24000)
    if raw.channels > 1:
        raw = raw.set_channels(1)
    return raw


def _segment_score(
    full: AudioSegment,
    start_s: float,
    end_s: float,
    file_duration_s: float,
) -> float:
    dur = max(0.0, end_s - start_s)
    center = (start_s + end_s) / 2.0
    ls = _length_score(dur)
    ps = _position_score(center, file_duration_s)
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    chunk = full[start_ms:end_ms]
    es = _energy_score_dbfs(chunk)
    return 0.35 * ls + 0.45 * es + 0.20 * ps


def _pick_top_non_overlapping(
    full: AudioSegment,
    intervals: list[tuple[float, float]],
    file_duration_s: float,
    k: int = 3,
) -> list[tuple[float, float, float]]:
    """Returns up to k (start, end, score) tuples, best non-overlapping by score."""
    scored: list[tuple[float, float, float]] = []
    for a, b in intervals:
        if b - a < 0.2:
            continue
        sc = _segment_score(full, a, b, file_duration_s)
        scored.append((a, b, sc))
    scored.sort(key=lambda x: -x[2])
    picked: list[tuple[float, float, float]] = []
    for a, b, sc in scored:
        if any(_overlap((a, b), (p[0], p[1])) for p in picked):
            continue
        picked.append((a, b, sc))
        if len(picked) >= k:
            break
    return picked


def _mfcc_profile_for_interval(full: AudioSegment, start_s: float, end_s: float) -> np.ndarray:
    """Short timbre vector from the same 10–15s window we export (librosa MFCC means)."""
    import librosa

    seg = _normalize_to_clip_window(full, start_s, end_s)
    seg = seg.set_channels(1).set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if samples.size == 0:
        return np.zeros(13, dtype=np.float32)
    denom = float(2 ** (8 * seg.sample_width - 1))
    y = samples / denom
    y = y / (np.max(np.abs(y)) + 1e-9)
    if len(y) < 800:
        return np.zeros(13, dtype=np.float32)
    n_fft = min(2048, len(y))
    hop_length = max(1, min(512, n_fft // 4))
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfcc, axis=1).astype(np.float32)


def _cosine_sim_unit(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def _pick_coherent_non_overlapping(
    full: AudioSegment,
    intervals: list[tuple[float, float]],
    file_duration_s: float,
    k: int = 3,
    *,
    strict_same_voice: bool = False,
) -> list[tuple[float, float, float]]:
    """
    Pick non-overlapping clips ranked by score, but prefer segments whose exported audio
    matches the best-scoring (anchor) clip in MFCC space. Reduces cases where pyannote
    assigns one label to acoustically different regions (e.g. same person mis-clustered
    with another voice or noise).
    """
    scored: list[tuple[float, float, float]] = []
    for a, b in intervals:
        if b - a < 0.2:
            continue
        sc = _segment_score(full, a, b, file_duration_s)
        scored.append((a, b, sc))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[2])
    pool = scored[:30]

    try:
        anchor = pool[0]
        ref = _mfcc_profile_for_interval(full, anchor[0], anchor[1])
    except Exception:
        logger.debug("MFCC coherence skipped; using score-only selection", exc_info=True)
        return _pick_top_non_overlapping(full, intervals, file_duration_s, k)

    # Stricter when diarization only found one speaker (often one human + label noise).
    sim_high = 0.80 if strict_same_voice else 0.74
    sim_mid = 0.64 if strict_same_voice else 0.60

    picked: list[tuple[float, float, float]] = [anchor]

    def overlaps_picked(a: float, b: float) -> bool:
        return any(_overlap((a, b), (p[0], p[1])) for p in picked)

    for a, b, sc in pool[1:]:
        if len(picked) >= k:
            break
        if overlaps_picked(a, b):
            continue
        if _cosine_sim_unit(ref, _mfcc_profile_for_interval(full, a, b)) >= sim_high:
            picked.append((a, b, sc))

    if len(picked) < k:
        for a, b, sc in pool[1:]:
            if len(picked) >= k:
                break
            if overlaps_picked(a, b):
                continue
            if (a, b, sc) in picked:
                continue
            if _cosine_sim_unit(ref, _mfcc_profile_for_interval(full, a, b)) >= sim_mid:
                picked.append((a, b, sc))

    if len(picked) < k:
        for a, b, sc in pool[1:]:
            if len(picked) >= k:
                break
            if overlaps_picked(a, b):
                continue
            if (a, b, sc) in picked:
                continue
            picked.append((a, b, sc))

    return picked[:k]


def _collect_diarization_intervals(diarization: Any) -> dict[str, list[tuple[float, float]]]:
    by_sp: dict[str, list[tuple[float, float]]] = {}
    for segment, _, label in diarization.itertracks(yield_label=True):
        sid = str(label)
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        by_sp.setdefault(sid, []).append((start, end))
    for sid in list(by_sp.keys()):
        by_sp[sid] = _merge_close_intervals(by_sp[sid], _MERGE_GAP_S)
    return by_sp


class SpeakerIsolationService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _require_hf_token() -> None:
        if not (config.HF_TOKEN or "").strip():
            raise ValueError(_HF_TOKEN_MISSING_MSG)

    def _job_output_dir(self, job_id: str) -> Path:
        return (config.OUTPUT_DIR / "isolate_speakers" / job_id).resolve()

    def _validate_upload(self, filename: str, size_bytes: int) -> None:
        ext = _ext(filename)
        if ext not in _ISOLATE_FORMATS:
            raise ValueError(f"Unsupported format '{ext}'. Use mp3, wav, m4a, or mp4.")
        max_bytes = config.AUDIO_TOOLS_MAX_UPLOAD_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"File too large. Maximum size is {config.AUDIO_TOOLS_MAX_UPLOAD_MB}MB."
            )

    async def _save_upload(self, upload: UploadFile, job_id: str) -> tuple[str, int]:
        out_dir = self._job_output_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(upload.filename or "audio.mp3").suffix.lower() or ".mp3"
        if _ext(upload.filename or "") not in _ISOLATE_FORMATS:
            ext = ".mp3"
        path = out_dir / f"source{ext}"
        content = await upload.read()
        if not content:
            raise ValueError("Uploaded file is empty.")
        self._validate_upload(upload.filename or path.name, len(content))
        path.write_bytes(content)
        return str(path), len(content)

    def _to_diarize_wav(self, source_path: str, job_id: str) -> str:
        out_dir = self._job_output_dir(job_id)
        wav_path = out_dir / "diarize.wav"
        seg = AudioSegment.from_file(source_path)
        seg = seg.set_frame_rate(16000).set_channels(1)
        seg.export(str(wav_path), format="wav")
        return str(wav_path)

    async def _run_job(self, job_id: str, source_audio_path: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            job["status"] = "diarizing"
            job["progress_pct"] = 15
            job["current_stage"] = "Running diarization..."

            wav_path = await asyncio.to_thread(self._to_diarize_wav, source_audio_path, job_id)
            full_audio = await asyncio.to_thread(AudioSegment.from_file, wav_path)
            duration_sec = len(full_audio) / 1000.0
            job["duration_seconds"] = duration_sec

            diarization = await transcript_diarizer.run(wav_path)
            by_speaker = _collect_diarization_intervals(diarization)

            totals: list[tuple[str, float]] = []
            for sid, intervals in by_speaker.items():
                total = sum(max(0.0, b - a) for a, b in intervals)
                totals.append((sid, total))
            totals.sort(key=lambda x: -x[1])
            ordered_ids = [x[0] for x in totals]

            job["status"] = "extracting"
            job["progress_pct"] = 45
            job["current_stage"] = "Extracting clips..."

            speakers_payload: list[dict[str, Any]] = []
            processed = 0
            id_count = max(1, len(ordered_ids))
            for pyannote_id in ordered_ids:
                if len(speakers_payload) >= _MAX_SPEAKERS:
                    break
                intervals = by_speaker.get(pyannote_id, [])
                total_s = sum(max(0.0, b - a) for a, b in intervals)
                picks = _pick_coherent_non_overlapping(
                    full_audio,
                    intervals,
                    duration_sec,
                    k=3,
                    strict_same_voice=(len(ordered_ids) == 1),
                )
                processed += 1
                job["progress_pct"] = min(90, 45 + int(40 * processed / id_count))

                if not picks:
                    continue

                display_num = len(speakers_payload) + 1
                label = f"Speaker {display_num}"
                clips_out: list[dict[str, Any]] = []
                for clip_i, (start_s, end_s, _) in enumerate(picks, start=1):
                    fname = f"speaker_{display_num}_clip_{clip_i}.mp3"
                    out_path = self._job_output_dir(job_id) / fname
                    clip_audio = _normalize_to_clip_window(full_audio, start_s, end_s)
                    await asyncio.to_thread(clip_audio.export, str(out_path), format="mp3", bitrate="192k")
                    dur_out = len(clip_audio) / 1000.0
                    clip_id = f"speaker_{display_num}_clip_{clip_i}"
                    clips_out.append(
                        {
                            "clip_id": clip_id,
                            "filename": fname,
                            "start_seconds": round(start_s, 3),
                            "end_seconds": round(end_s, 3),
                            "duration_seconds": round(dur_out, 3),
                            "download_url": f"/api/v1/audio-tools/isolate-speakers/clip/{job_id}/{fname}",
                        }
                    )

                speakers_payload.append(
                    {
                        "speaker_id": pyannote_id,
                        "label": label,
                        "total_speaking_seconds": round(total_s, 3),
                        "clips": clips_out,
                    }
                )

            mode = (config.SPEAKER_NAME_INFERENCE or "regex").strip().lower()
            if mode not in ("", "off", "false", "0", "no") and speakers_payload:
                job["status"] = "inferring_names"
                job["current_stage"] = "Inferring speaker names..."
                job["progress_pct"] = min(94, job.get("progress_pct", 90))
                await infer_speaker_labels_for_isolation(
                    full_audio=full_audio,
                    by_speaker=by_speaker,
                    speakers_payload=speakers_payload,
                    job_dir=self._job_output_dir(job_id),
                )
                for sp in speakers_payload:
                    if sp.get("label_source") is None:
                        sp["label_source"] = "default"

            result_path = self._job_output_dir(job_id) / "result.json"
            result_path.write_text(
                json.dumps({"job_id": job_id, "speakers": speakers_payload}, indent=2),
                encoding="utf-8",
            )
            job["speakers"] = speakers_payload
            job["result_path"] = str(result_path)
            job["status"] = "complete"
            job["progress_pct"] = 100
            job["current_stage"] = "Complete"
        except RuntimeError as exc:
            err = str(exc)
            if "HF_TOKEN" in err:
                logger.warning("Speaker isolation failed for %s: %s", job_id, err)
                job["status"] = "failed"
                job["progress_pct"] = 100
                job["current_stage"] = "Failed"
                job["error"] = _HF_TOKEN_MISSING_MSG
            else:
                logger.exception("Speaker isolation failed for %s", job_id)
                job["status"] = "failed"
                job["progress_pct"] = 100
                job["current_stage"] = "Failed"
                job["error"] = err
        except Exception as exc:
            logger.exception("Speaker isolation failed for %s", job_id)
            job["status"] = "failed"
            job["progress_pct"] = 100
            job["current_stage"] = "Failed"
            job["error"] = str(exc)
        finally:
            self._tasks.pop(job_id, None)
            if getattr(config, "ISOLATION_UNLOAD_MODELS_AFTER_JOB", True):
                try:

                    def _unload_isolation_models() -> None:
                        transcript_diarizer.unload_pipeline()
                        transcript_transcriber.unload_models()
                        release_torch_cuda_memory()

                    await asyncio.to_thread(_unload_isolation_models)
                    logger.info("Voice Isolator: released diarization/transcription models and CUDA cache")
                except Exception:
                    logger.debug("Voice Isolator VRAM cleanup failed", exc_info=True)

    async def upload_and_queue(self, audio_file: UploadFile) -> dict[str, Any]:
        self._require_hf_token()
        job_id = str(uuid.uuid4())
        audio_path, _ = await self._save_upload(audio_file, job_id)
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress_pct": 0,
            "current_stage": "Queued",
            "audio_path": audio_path,
            "duration_seconds": None,
            "speakers": None,
            "error": None,
            "result_path": None,
        }
        task = asyncio.create_task(self._run_job(job_id, audio_path))
        self._tasks[job_id] = task
        return {"job_id": job_id, "status": "queued", "progress_pct": 0}

    def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._jobs.get(job_id)

    def clip_path(self, job_id: str, filename: str) -> Path:
        if not _JOB_ID_RE.match(job_id) or not _CLIP_FILE_RE.match(filename):
            raise ValueError("Invalid job or clip filename")
        p = (self._job_output_dir(job_id) / filename).resolve()
        base = self._job_output_dir(job_id).resolve()
        if base not in p.parents and p != base:
            raise ValueError("Invalid path")
        return p

    def resolve_clip_audio_path(self, job_id: str, clip_id: str) -> Path:
        """Return path to an extracted clip file; clip_id is e.g. speaker_1_clip_2 (with or without .mp3)."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError("Job not found")
        if job.get("status") != "complete":
            raise ValueError("Job is not complete")
        raw = (clip_id or "").strip()
        if raw.endswith(".mp3"):
            filename = raw
        else:
            filename = f"{raw}.mp3"
        p = self.clip_path(job_id, filename)
        if not p.is_file():
            raise ValueError("Clip file not found")
        return p


speaker_isolation_service = SpeakerIsolationService()
