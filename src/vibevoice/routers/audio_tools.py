"""
Audio tools API: podcast ad scanning and export, speaker isolation clips.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..config import config
from ..models.schemas import (
    CreateVoiceFromIsolationClipRequest,
    PodcastAdExportRequest,
    PodcastAdExportResponse,
    PodcastAdScanStatusResponse,
    PodcastAdScanSubmitResponse,
    SpeakerIsolationClipItem,
    SpeakerIsolationSpeakerItem,
    SpeakerIsolationStatusResponse,
    SpeakerIsolationSubmitResponse,
    VoiceCreateResponse,
)
from ..routes.voices import _build_voice_create_response
from ..services.ad_scan_service import ad_scan_service
from ..services.speaker_isolation_service import speaker_isolation_service
from ..services.voice_manager import voice_manager

router = APIRouter(prefix="/api/v1/audio-tools", tags=["audio-tools"])

_SAFE_DOWNLOAD_NAME = re.compile(r"^[A-Za-z0-9._-]+\.mp3$")


def _isolation_status_response(row: dict) -> SpeakerIsolationStatusResponse:
    speakers = None
    raw_sp = row.get("speakers")
    if raw_sp:
        speakers = []
        for s in raw_sp:
            clips = [SpeakerIsolationClipItem.model_validate(c) for c in (s.get("clips") or [])]
            speakers.append(
                SpeakerIsolationSpeakerItem(
                    speaker_id=s["speaker_id"],
                    label=s["label"],
                    total_speaking_seconds=float(s["total_speaking_seconds"]),
                    clips=clips,
                )
            )
    return SpeakerIsolationStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        progress_pct=row.get("progress_pct", 0),
        current_stage=row.get("current_stage"),
        speakers=speakers,
        duration_seconds=row.get("duration_seconds"),
        error=row.get("error"),
    )


@router.post(
    "/podcast/scan-ads",
    response_model=PodcastAdScanSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def scan_podcast_ads(audio_file: UploadFile = File(...)):
    """
    Upload podcast audio and queue Whisper + LLM ad detection.
    """
    try:
        return await ad_scan_service.upload_and_queue(audio_file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ad scan: {exc}",
        ) from exc


@router.get(
    "/podcast/scan-ads/{job_id}/status",
    response_model=PodcastAdScanStatusResponse,
)
async def podcast_ad_scan_status(job_id: str):
    row = ad_scan_service.get_status(job_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return PodcastAdScanStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        progress_pct=row.get("progress_pct", 0),
        current_stage=row.get("current_stage"),
        ad_segments=row.get("ad_segments"),
        duration_seconds=row.get("duration_seconds"),
        error=row.get("error"),
    )


@router.post(
    "/podcast/export",
    response_model=PodcastAdExportResponse,
)
async def export_podcast_audio(payload: PodcastAdExportRequest):
    try:
        data = ad_scan_service.export_audio(payload.job_id, payload.export_mode)
        return PodcastAdExportResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {exc}",
        ) from exc


@router.get("/podcast/download/{filename}")
async def download_export(filename: str):
    if not _SAFE_DOWNLOAD_NAME.match(filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    base = (config.AUDIO_TOOLS_DIR / "exports").resolve()
    path = (base / filename).resolve()
    if base not in path.parents and path != base:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(
        path=str(path),
        media_type="audio/mpeg",
        filename=filename,
    )


@router.post(
    "/isolate-speakers",
    response_model=SpeakerIsolationSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def isolate_speakers(audio_file: UploadFile = File(...)):
    """
    Upload audio; run diarization and extract up to three 10–15s clips per speaker (max 6 speakers).
    """
    try:
        data = await speaker_isolation_service.upload_and_queue(audio_file)
        return SpeakerIsolationSubmitResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start speaker isolation: {exc}",
        ) from exc


@router.get(
    "/isolate-speakers/{job_id}/status",
    response_model=SpeakerIsolationStatusResponse,
)
async def speaker_isolation_status(job_id: str):
    row = speaker_isolation_service.get_status(job_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _isolation_status_response(row)


@router.get("/isolate-speakers/clip/{job_id}/{filename}")
async def speaker_isolation_clip_audio(job_id: str, filename: str):
    try:
        path = speaker_isolation_service.clip_path(job_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found")
    return FileResponse(
        path=str(path),
        media_type="audio/mpeg",
        content_disposition_type="inline",
    )


@router.post(
    "/isolate-speakers/create-voice",
    response_model=VoiceCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_voice_from_isolation_clip(payload: CreateVoiceFromIsolationClipRequest):
    try:
        clip_path = speaker_isolation_service.resolve_clip_audio_path(payload.job_id, payload.clip_id)
        voice_data = voice_manager.create_custom_voice(
            name=payload.voice_name.strip(),
            description=(payload.voice_description or "").strip() or None,
            audio_files=[clip_path],
            keywords=None,
            ollama_url=None,
            ollama_model=None,
            language_code=None,
            gender=None,
            image_path=None,
        )
        return _build_voice_create_response(voice_data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice creation failed: {exc}",
        ) from exc
