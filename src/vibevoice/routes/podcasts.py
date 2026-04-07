"""
Podcast library endpoints (list/search/download/delete).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse

from ..config import config
from ..models.podcast_storage import podcast_storage
from ..models.schemas import ErrorResponse, PodcastItem, PodcastListResponse

router = APIRouter(prefix="/api/v1/podcasts", tags=["podcasts"])


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


@router.get(
    "",
    response_model=PodcastListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_podcasts(query: str = Query(default="", description="Optional search query")) -> PodcastListResponse:
    """
    List and search saved podcasts.
    """
    try:
        items = podcast_storage.list_podcasts()
        q = (query or "").strip().lower()
        if q:
            def matches(item: dict) -> bool:
                title = str(item.get("title", "")).lower()
                source_url = str(item.get("source_url", "")).lower()
                voices = " ".join(item.get("voices", []) or []).lower()
                return q in title or q in source_url or q in voices

            items = [i for i in items if matches(i)]

        podcasts = []
        for item in items:
            pid = item.get("id")
            if not pid:
                continue
            podcasts.append(
                PodcastItem(
                    id=pid,
                    title=item.get("title", pid),
                    voices=item.get("voices", []) or [],
                    source_url=item.get("source_url"),
                    genre=item.get("genre"),
                    duration=item.get("duration"),
                    created_at=_parse_dt(item.get("created_at")),
                    audio_url=f"/api/v1/podcasts/{pid}/download",
                )
            )

        return PodcastListResponse(podcasts=podcasts, total=len(podcasts))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list podcasts: {str(e)}",
        ) from e


@router.get(
    "/{podcast_id}",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_podcast(podcast_id: str) -> JSONResponse:
    """
    Get podcast metadata (and script text if available).
    """
    try:
        item = podcast_storage.get_podcast(podcast_id)
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Podcast not found")

        script_text = None
        script_path = item.get("script_path")
        if script_path:
            p = Path(script_path)
            if p.exists():
                try:
                    script_text = p.read_text()
                except Exception:
                    script_text = None

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "podcast": item,
                "script": script_text,
                "audio_url": f"/api/v1/podcasts/{podcast_id}/download",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get podcast: {str(e)}",
        ) from e


@router.get(
    "/{podcast_id}/download",
    responses={404: {"model": ErrorResponse}},
)
async def download_podcast_by_id(podcast_id: str) -> FileResponse:
    item = podcast_storage.get_podcast(podcast_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Podcast not found")

    audio_path = Path(item.get("audio_path", ""))
    if not audio_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")

    # Safety: restrict downloads to configured podcasts dir
    resolved = audio_path.resolve()
    base_dir = config.PODCASTS_DIR.resolve()
    if base_dir not in resolved.parents and resolved != base_dir:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not accessible")

    return FileResponse(path=str(audio_path), media_type="audio/wav", filename=audio_path.name)


@router.delete(
    "/{podcast_id}",
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_podcast(podcast_id: str) -> JSONResponse:
    """
    Delete a saved podcast (metadata + audio + script file).
    """
    item = podcast_storage.delete_podcast(podcast_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Podcast not found")

    # Best-effort file cleanup
    for key in ("audio_path", "script_path"):
        p_str = item.get(key)
        if not p_str:
            continue
        p = Path(p_str)
        try:
            resolved = p.resolve()
            base_dir = config.PODCASTS_DIR.resolve()
            if base_dir in resolved.parents and p.exists():
                p.unlink()
        except Exception:
            # ignore file delete errors; metadata is already removed
            pass

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"success": True, "message": f"Podcast '{podcast_id}' deleted successfully"},
    )

