"""
Podcast generation endpoints.
"""
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from ..models.schemas import (
    ErrorResponse,
    PodcastGenerateRequest,
    PodcastGenerateResponse,
    PodcastScriptRequest,
    PodcastScriptResponse,
)
from ..config import config
from ..models.podcast_storage import podcast_storage
from ..services.podcast_generator import podcast_generator
from ..services.voice_manager import voice_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/podcast", tags=["podcast"])


@router.post(
    "/generate-script",
    response_model=PodcastScriptResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_podcast_script(
    request: PodcastScriptRequest, http_request: Request
) -> PodcastScriptResponse:
    """
    Generate podcast script from article URL using Ollama.

    Args:
        request: Podcast script generation request with URL, voices, genre, and duration
        http_request: HTTP request object for logging client info

    Returns:
        Podcast script response with generated script
    """
    client_ip = http_request.client.host if http_request.client else "unknown"

    logger.info("=" * 80)
    logger.info("Podcast Script Generation Request Received")
    logger.info("=" * 80)
    logger.info(f"Client IP: {client_ip}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Genre: {request.genre}")
    logger.info(f"Duration: {request.duration}")
    logger.info(f"Voices: {request.voices}")
    logger.info(f"Number of voices: {len(request.voices)}")
    if request.ollama_url:
        logger.info(f"Custom Ollama URL: {request.ollama_url}")
    if request.ollama_model:
        logger.info(f"Custom Ollama Model: {request.ollama_model}")
    logger.info("")

    try:
        warnings = voice_manager.get_bgm_risk_warnings(request.voices)

        # Generate script
        logger.info("Generating podcast script...")
        script = podcast_generator.generate_script(
            url=request.url,
            genre=request.genre,
            duration=request.duration,
            voices=request.voices,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
        )

        logger.info("")
        logger.info("Script Generation Completed Successfully")
        logger.info(f"  Script length: {len(script)} characters")
        logger.info("=" * 80)

        return PodcastScriptResponse(
            success=True,
            message="Podcast script generated successfully",
            script=script,
            warnings=warnings,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error(f"Runtime error during script generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script generation failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error during script generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/generate",
    response_model=PodcastGenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_podcast_audio(
    request: PodcastGenerateRequest, http_request: Request
) -> PodcastGenerateResponse:
    """
    Generate podcast audio from script using AudioMesh.

    Args:
        request: Podcast audio generation request with script and voices
        http_request: HTTP request object for logging client info

    Returns:
        Podcast generation response with audio file path
    """
    client_ip = http_request.client.host if http_request.client else "unknown"

    logger.info("=" * 80)
    logger.info("Podcast Audio Generation Request Received")
    logger.info("=" * 80)
    logger.info(f"Client IP: {client_ip}")
    logger.info(f"Script length: {len(request.script)} characters")
    logger.info(f"Voices: {request.voices}")
    logger.info(f"Number of voices: {len(request.voices)}")
    logger.info("")

    try:
        warnings = voice_manager.get_bgm_risk_warnings(request.voices)

        # Generate audio
        logger.info("Generating podcast audio...")
        output_path = podcast_generator.generate_audio(
            script=request.script,
            voices=request.voices,
        )

        output_file = Path(output_path)
        logger.info("")
        logger.info("Podcast Audio Generation Completed Successfully")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("=" * 80)

        podcast_id = None
        audio_url = f"/api/v1/podcast/download/{output_file.name}"
        saved_path = output_file

        if request.save_to_library:
            podcast_id = str(uuid4())
            # Derive title if missing
            title = (request.title or "").strip()
            if not title:
                title = f"Podcast {podcast_id[:8]}"

            # Save audio into podcasts dir (so delete is safe and contained)
            config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
            saved_audio_path = config.PODCASTS_DIR / f"{podcast_id}.wav"
            shutil.copy2(output_file, saved_audio_path)

            # Save script to file
            saved_script_path = config.PODCASTS_DIR / f"{podcast_id}.txt"
            saved_script_path.write_text(request.script)

            podcast_storage.add_podcast(
                podcast_id=podcast_id,
                title=title,
                voices=request.voices,
                audio_path=saved_audio_path,
                script_path=saved_script_path,
                source_url=request.source_url,
                genre=request.genre,
                duration=request.duration,
                extra={
                    "file_size_bytes": saved_audio_path.stat().st_size,
                },
            )

            audio_url = f"/api/v1/podcasts/{podcast_id}/download"
            saved_path = saved_audio_path

        return PodcastGenerateResponse(
            success=True,
            message="Podcast audio generated successfully",
            audio_url=audio_url,
            file_path=str(saved_path),
            script=request.script,
            podcast_id=podcast_id,
            warnings=warnings,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error(f"Runtime error during audio generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error during audio generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/download/{filename}",
    responses={
        404: {"model": ErrorResponse},
    },
)
async def download_podcast(filename: str) -> FileResponse:
    """
    Download generated podcast audio file.

    Args:
        filename: Name of the generated audio file

    Returns:
        Audio file as binary response
    """
    from ..config import config

    file_path = config.OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: {filename}",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )
