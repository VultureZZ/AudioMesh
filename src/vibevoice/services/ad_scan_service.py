"""
Async job pipeline for podcast advertisement scanning (Whisper + Ollama).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile
from pydub import AudioSegment

from ..config import config
from ..models.schemas import AdSegmentItem
from ..services.ad_scan_segment_utils import filter_dominant_show_segments, is_commercial_ad_segment
from ..services.ad_scan_transcriber import transcribe_for_ad_scan
from ..services.ollama_client import ollama_client

logger = logging.getLogger(__name__)

_AD_SCAN_FORMATS = frozenset({"mp3", "wav", "m4a"})


def _log_ad_scan_blob(prefix: str, text: str) -> None:
    """Log a possibly large string; truncate using AD_SCAN_LOG_MAX_CHARS when > 0."""
    max_c = getattr(config, "AD_SCAN_LOG_MAX_CHARS", 50000)
    if max_c <= 0 or len(text) <= max_c:
        logger.info("%s\n%s", prefix, text)
        return
    omitted = len(text) - max_c
    logger.info(
        "%s (truncated: %d chars total, %d omitted)\n%s",
        prefix,
        len(text),
        omitted,
        text[:max_c],
    )


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

    @staticmethod
    def _merge_into_blocks(
        segments: list[dict[str, Any]],
        gap_threshold: float = 1.5,
        max_block_seconds: float = 45.0,
    ) -> list[dict[str, Any]]:
        """
        Merge consecutive Whisper segments into blocks when the gap between them is small.
        Also caps each block at max_block_seconds so ads + main episode never merge into one LLM
        input (Whisper often reports 0s gap between an ad end and the first news line).
        """
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda s: s["start_seconds"])
        blocks: list[dict[str, Any]] = []
        cur_start = float(sorted_segs[0]["start_seconds"])
        cur_end = float(sorted_segs[0]["end_seconds"])
        cur_texts: list[str] = [sorted_segs[0].get("text") or ""]

        def flush() -> None:
            joined = " ".join(t for t in cur_texts if t).strip()
            if joined:
                blocks.append(
                    {
                        "start_seconds": round(cur_start, 2),
                        "end_seconds": round(cur_end, 2),
                        "text": joined,
                    }
                )

        for seg in sorted_segs[1:]:
            seg_start = float(seg["start_seconds"])
            seg_end = float(seg["end_seconds"])
            seg_text = (seg.get("text") or "").strip()
            gap = seg_start - cur_end
            merged_end = max(cur_end, seg_end)
            merged_span = merged_end - cur_start

            if gap > gap_threshold:
                flush()
                cur_start = seg_start
                cur_end = seg_end
                cur_texts = [seg_text]
                continue

            if merged_span > max_block_seconds and (cur_end - cur_start) >= 0.05:
                flush()
                cur_start = seg_start
                cur_end = seg_end
                cur_texts = [seg_text]
                continue

            cur_end = merged_end
            if seg_text:
                cur_texts.append(seg_text)

        flush()
        return blocks

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
            t0 = time.perf_counter()
            logger.info(
                "[ad-scan] job=%s begin_transcription backend=%s whisper_model=%s audio=%s",
                job_id,
                backend,
                config.TRANSCRIPT_WHISPER_MODEL,
                audio_path,
            )
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
            transcribe_s = time.perf_counter() - t0
            raw_segments = transcript.get("segments") or []
            _log_ad_scan_blob(
                f"[ad-scan] job={job_id} whisper_raw_segments_json (pre-normalize, count={len(raw_segments)})",
                json.dumps(raw_segments, ensure_ascii=False, indent=2),
            )
            fine_segments = self._normalize_whisper_segments(raw_segments, duration_sec)
            blocks = self._merge_into_blocks(
                fine_segments,
                gap_threshold=float(getattr(config, "AD_SCAN_MERGE_GAP_SECONDS", 1.5)),
                max_block_seconds=float(getattr(config, "AD_SCAN_MAX_BLOCK_SECONDS", 45.0)),
            )
            lang = transcript.get("language")
            plain_text = " ".join(
                (s.get("text") or "").strip() for s in fine_segments if (s.get("text") or "").strip()
            )

            logger.info(
                "[ad-scan] job=%s transcription_done duration_sec=%.3f transcribe_wall_s=%.2f "
                "fine_segments=%d merged_blocks=%d language=%s plain_text_chars=%d",
                job_id,
                duration_sec,
                transcribe_s,
                len(fine_segments),
                len(blocks),
                lang,
                len(plain_text),
            )
            _log_ad_scan_blob(f"[ad-scan] job={job_id} transcription_plain_text", plain_text)
            blocks_json = json.dumps(blocks, ensure_ascii=False, indent=2)
            _log_ad_scan_blob(
                f"[ad-scan] job={job_id} merged_blocks_json (sent to LLM, count={len(blocks)})",
                blocks_json,
            )

            job["status"] = "analyzing"
            job["progress_pct"] = 55
            job["current_stage"] = "Analyzing for ads..."

            t_llm = time.perf_counter()
            ad_raw = await asyncio.to_thread(
                ollama_client.identify_podcast_ad_segments,
                blocks,
                duration_sec,
                None,
                job_id,
            )
            llm_s = time.perf_counter() - t_llm
            logger.info(
                "[ad-scan] job=%s ollama_returned raw_segments=%d wall_s=%.2f",
                job_id,
                len(ad_raw),
                llm_s,
            )
            ad_raw = filter_dominant_show_segments(ad_raw, duration_sec, job_id=job_id)
            logger.info(
                "[ad-scan] job=%s after_dominant_show_filter segments=%d",
                job_id,
                len(ad_raw),
            )
            ad_items: list[AdSegmentItem] = []
            pydantic_rejects = 0
            for item in ad_raw:
                try:
                    ad_items.append(AdSegmentItem.model_validate(item))
                except Exception as ex:
                    pydantic_rejects += 1
                    logger.debug(
                        "[ad-scan] job=%s pydantic_reject item=%r err=%s",
                        job_id,
                        item,
                        ex,
                    )
            if pydantic_rejects:
                logger.warning(
                    "[ad-scan] job=%s pydantic_rejected_segments=%d (see DEBUG for rows)",
                    job_id,
                    pydantic_rejects,
                )

            job["ad_segments"] = [a.model_dump() for a in ad_items]
            _log_ad_scan_blob(
                f"[ad-scan] job={job_id} final_ad_segments (stored on job)",
                json.dumps(job["ad_segments"], ensure_ascii=False, indent=2),
            )
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
            if isinstance(a, dict) and is_commercial_ad_segment(a)
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
