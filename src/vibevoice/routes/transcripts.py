"""
Transcript service endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from ..config import config
from ..services.transcript_service import transcript_service
from ..services.voice_manager import voice_manager

router = APIRouter(prefix="/api/v1/transcripts", tags=["transcripts"])


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_transcript(
    audio_file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: str = Form("en"),
    recording_type: str = Form("meeting"),
):
    try:
        return await transcript_service.upload_and_queue(
            audio_file,
            title=title,
            language=language,
            recording_type=recording_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload transcript audio: {exc}",
        ) from exc


@router.get("/{transcript_id}/status")
async def transcript_status(transcript_id: str):
    item = transcript_service.get(transcript_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")
    return {
        "transcript_id": item["id"],
        "status": item.get("status"),
        "progress_pct": item.get("progress_pct", 0),
        "current_stage": item.get("current_stage"),
        "duration_seconds": item.get("duration_seconds"),
        "speakers_detected": len(item.get("speakers") or []) if item.get("speakers") else None,
        "error": item.get("error"),
    }


@router.get("/{transcript_id}")
async def get_transcript(transcript_id: str):
    item = transcript_service.get(transcript_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")
    return item


@router.patch("/{transcript_id}/speakers")
async def update_transcript_speakers(
    transcript_id: str,
    payload: dict,
):
    item = transcript_service.get(transcript_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")

    speakers_update = payload.get("speakers") or []
    proceed_to_analysis = bool(payload.get("proceed_to_analysis"))

    current_speakers = item.get("speakers") or []
    by_id = {s.get("id"): s for s in current_speakers}
    for upd in speakers_update:
        sid = upd.get("id")
        if sid in by_id:
            by_id[sid]["label"] = upd.get("label")
    item["speakers"] = list(by_id.values())
    from ..models.transcript_storage import transcript_storage

    transcript_storage.update_transcript(transcript_id, speakers=item["speakers"])

    if proceed_to_analysis:
        await transcript_service.trigger_analysis(transcript_id)
        return {
            "transcript_id": transcript_id,
            "status": "analyzing",
            "message": "Speaker labels saved. Analysis started.",
        }

    return {
        "transcript_id": transcript_id,
        "status": item.get("status"),
        "message": "Speaker labels saved.",
    }


@router.get("/{transcript_id}/speakers/{speaker_id}/audio")
async def get_speaker_audio(transcript_id: str, speaker_id: str):
    item = transcript_service.get(transcript_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")

    speakers = item.get("speakers") or []
    speaker = next((s for s in speakers if s.get("id") == speaker_id), None)
    if not speaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker not found")
    audio_path = speaker.get("audio_segment_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker audio not found")

    return FileResponse(audio_path, media_type="audio/wav")


@router.post("/{transcript_id}/speakers/{speaker_id}/add-to-library", status_code=status.HTTP_201_CREATED)
async def add_speaker_to_library(transcript_id: str, speaker_id: str, payload: dict):
    item = transcript_service.get(transcript_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")

    speakers = item.get("speakers") or []
    speaker = next((s for s in speakers if s.get("id") == speaker_id), None)
    if not speaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker not found")

    audio_path = speaker.get("audio_segment_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker audio not found")

    voice_name = payload.get("voice_name") or speaker.get("label") or speaker_id
    description = payload.get("description") or f"Auto-extracted from transcript {transcript_id}"
    try:
        created = voice_manager.create_custom_voice(
            name=voice_name,
            description=description,
            audio_files=[Path(audio_path)],
            keywords=None,
            ollama_url=None,
            ollama_model=None,
        )
        return {
            "voice_id": created["id"],
            "message": f"Voice '{voice_name}' added to library successfully.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{transcript_id}/report")
async def get_transcript_report(
    transcript_id: str,
    format: str = Query("pdf", pattern="^(pdf|json|markdown)$"),
):
    reports_dir = config.TRANSCRIPTS_DIR / "reports"
    ext = "md" if format == "markdown" else format
    path = reports_dir / f"{transcript_id}.{ext}"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    media_type = {
        "pdf": "application/pdf",
        "json": "application/json",
        "md": "text/markdown",
    }[ext]
    filename = f"{transcript_id}.{ext}"
    return FileResponse(str(path), media_type=media_type, filename=filename)


@router.get("")
async def list_transcripts(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    recording_type: Optional[str] = Query(None),
):
    items, total = transcript_service.list(limit=limit, offset=offset, status=status, recording_type=recording_type)
    return {
        "transcripts": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/{transcript_id}")
async def delete_transcript(transcript_id: str):
    deleted = transcript_service.delete(transcript_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": f"Transcript '{transcript_id}' deleted."},
    )

