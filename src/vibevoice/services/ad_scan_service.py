"""
Async job pipeline for podcast advertisement scanning (Whisper + Ollama).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile
from pydub import AudioSegment

from ..config import config
from ..models.schemas import AdSegmentItem
from ..services.ad_scan_transcriber import transcribe_for_ad_scan
from ..services.ollama_client import ollama_client

logger = logging.getLogger(__name__)

_AD_SCAN_FORMATS = frozenset({"mp3", "wav", "m4a"})


def _ext(name: str) -> str:
    return (Path(name).suffix or "").lower().lstrip(".")


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_iv[0]]
    for a, b in sorted_iv[1:]:
        la, lb = merged[-1]
        if a <= lb + 0.001:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _non_ad_parts(duration_sec: float, ads: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = _merge_intervals(ads)
    if not merged:
        return [(0.0, duration_sec)] if duration_sec > 0 else []
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in merged:
        if a > cursor:
            out.append((cursor, min(a, duration_sec)))
        cursor = max(cursor, b)
    if cursor < duration_sec:
        out.append((cursor, duration_sec))
    return [(x, y) for x, y in out if y - x > 0.05]


class AdScanService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def _job_dir(self, job_id: str) -> Path:
        return config.AUDIO_TOOLS_DIR / "jobs" / job_id

    def _validate_upload(self, filename: str, size_bytes: int) -> None:
        ext = _ext(filename)
        if ext not in _AD_SCAN_FORMATS:
            raise ValueError(f"Unsupported format '{ext}'. Use mp3, wav, or m4a.")
        max_bytes = config.AUDIO_TOOLS_MAX_UPLOAD_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"File too large. Maximum size is {config.AUDIO_TOOLS_MAX_UPLOAD_MB}MB."
            )

    async def _save_upload(self, upload: UploadFile, job_id: str) -> tuple[str, int]:
        self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
        ext = Path(upload.filename or "audio.mp3").suffix.lower() or ".mp3"
        if _ext(upload.filename or "") not in _AD_SCAN_FORMATS:
            ext = ".mp3"
        path = self._job_dir(job_id) / f"source{ext}"
        content = await upload.read()
        if not content:
            raise ValueError("Uploaded file is empty.")
        self._validate_upload(upload.filename or path.name, len(content))
        path.write_bytes(content)
        return str(path), len(content)

    def _normalize_whisper_segments(self, raw_segments: list[Any], duration_sec: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            try:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", start))
            except (TypeError, ValueError):
                continue
            start = max(0.0, min(start, duration_sec))
            end = max(0.0, min(end, duration_sec))
            if end < start:
                start, end = end, start
            if not text and end - start < 0.01:
                continue
            out.append(
                {
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "text": text,
                }
            )
        return out

    async def _run_job(self, job_id: str, audio_path: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            job["status"] = "transcribing"
            job["progress_pct"] = 15
            job["current_stage"] = "Transcribing..."
            seg = AudioSegment.from_file(audio_path)
            duration_ms = len(seg)
            duration_sec = duration_ms / 1000.0
            job["duration_seconds"] = duration_sec

            backend = (config.AD_SCAN_TRANSCRIBE_BACKEND or "faster_whisper").strip().lower()
            logger.info("Ad scan job %s: transcription backend=%s", job_id, backend)
            if backend in ("whisperx", "whisper_x"):
                from ..core.transcripts.transcriber import transcript_transcriber

                transcript = await transcript_transcriber.transcribe(audio_path, language=None)
            elif backend in ("faster_whisper", "faster-whisper", "fasterwhisper"):
                transcript = await asyncio.to_thread(transcribe_for_ad_scan, audio_path, None)
            else:
                raise ValueError(
                    f"Invalid AD_SCAN_TRANSCRIBE_BACKEND={backend!r}. "
                    "Use 'faster_whisper' or 'whisperx'."
                )
            raw_segments = transcript.get("segments") or []
            segments_payload = self._normalize_whisper_segments(raw_segments, duration_sec)

            job["status"] = "analyzing"
            job["progress_pct"] = 55
            job["current_stage"] = "Analyzing for ads..."

            ad_raw = await asyncio.to_thread(
                ollama_client.identify_podcast_ad_segments,
                segments_payload,
                duration_sec,
            )
            ad_items: list[AdSegmentItem] = []
            for item in ad_raw:
                try:
                    ad_items.append(AdSegmentItem.model_validate(item))
                except Exception:
                    continue

            job["ad_segments"] = [a.model_dump() for a in ad_items]
            result_path = self._job_dir(job_id) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "duration_seconds": duration_sec,
                        "ad_segments": job["ad_segments"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            job["result_path"] = str(result_path)
            job["status"] = "complete"
            job["progress_pct"] = 100
            job["current_stage"] = "Complete"
        except Exception as exc:
            logger.exception("Ad scan job failed for %s", job_id)
            job["status"] = "failed"
            job["progress_pct"] = 100
            job["current_stage"] = "Failed"
            job["error"] = str(exc)
        finally:
            self._tasks.pop(job_id, None)

    async def upload_and_queue(self, audio_file: UploadFile) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        audio_path, _ = await self._save_upload(audio_file, job_id)
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress_pct": 0,
            "current_stage": "Queued",
            "audio_path": audio_path,
            "duration_seconds": None,
            "ad_segments": None,
            "error": None,
            "result_path": None,
        }
        task = asyncio.create_task(self._run_job(job_id, audio_path))
        self._tasks[job_id] = task
        return {"job_id": job_id, "status": "queued", "progress_pct": 0}

    def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._jobs.get(job_id)

    def _safe_export_filename(self, job_id: str, mode: str) -> str:
        short = job_id.replace("-", "")[:12]
        return f"adscan_{short}_{mode}_{uuid.uuid4().hex[:10]}.mp3"

    def export_audio(self, job_id: str, export_mode: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError("Job not found")
        if job.get("status") != "complete":
            raise ValueError("Job is not complete")
        audio_path = job.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            raise ValueError("Source audio not found")

        ads = [
            (float(a["start_seconds"]), float(a["end_seconds"]))
            for a in (job.get("ad_segments") or [])
            if isinstance(a, dict)
        ]
        if export_mode == "ads_only" and not ads:
            raise ValueError("No ad segments detected; ads-only export is not available.")

        duration_sec = float(job.get("duration_seconds") or 0)
        if duration_sec <= 0:
            seg = AudioSegment.from_file(audio_path)
            duration_sec = len(seg) / 1000.0

        audio = AudioSegment.from_file(audio_path)

        if export_mode == "clean":
            parts = _non_ad_parts(duration_sec, ads)
            if not parts:
                raise ValueError("Nothing left after removing ads")
            combined = AudioSegment.empty()
            for start, end in parts:
                combined += audio[int(start * 1000) : int(end * 1000)]
        else:
            merged = _merge_intervals(ads)
            if not merged:
                raise ValueError("No ad segments to export")
            combined = AudioSegment.empty()
            for start, end in merged:
                combined += audio[int(start * 1000) : int(end * 1000)]

        out_name = self._safe_export_filename(job_id, export_mode)
        out_path = (config.AUDIO_TOOLS_DIR / "exports" / out_name).resolve()
        export_base = (config.AUDIO_TOOLS_DIR / "exports").resolve()
        if export_base not in out_path.parents and out_path != export_base:
            raise ValueError("Invalid export path")
        combined.export(str(out_path), format="mp3", bitrate="192k")
        size_bytes = out_path.stat().st_size
        out_duration = len(combined) / 1000.0
        return {
            "download_url": f"/api/v1/audio-tools/podcast/download/{out_name}",
            "filename": out_name,
            "duration_seconds": out_duration,
            "file_size_bytes": size_bytes,
        }


ad_scan_service = AdScanService()
