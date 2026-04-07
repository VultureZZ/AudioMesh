"""
Speech generation endpoints.
"""
import asyncio
import json
import logging
import queue
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from ..models.schemas import ErrorResponse, SpeechGenerateRequest, SpeechGenerateResponse
from ..services.voice_generator import voice_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/speech", tags=["speech"])


@router.post(
    "/generate",
    response_model=SpeechGenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_speech(
    request: SpeechGenerateRequest, http_request: Request
) -> SpeechGenerateResponse:
    """
    Generate speech from text transcript.

    Args:
        request: Speech generation request with transcript, speakers, and settings
        http_request: HTTP request object for logging client info

    Returns:
        Speech generation response with audio file path
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    logger.info("=" * 80)
    logger.info("Speech Generation Request Received")
    logger.info("=" * 80)
    logger.info(f"Client IP: {client_ip}")
    logger.info(f"Request ID: {id(request)}")
    logger.info("")
    logger.info("Request Parameters:")
    logger.info(f"  Transcript length: {len(request.transcript)} characters")
    logger.info(f"  Transcript preview: {request.transcript[:100]}...")
    logger.info(f"  Speakers: {request.speakers}")
    logger.info(f"  Number of speakers: {len(request.speakers)}")
    logger.info("")
    logger.info("Settings:")
    if request.settings:
        logger.info(f"  Language: {request.settings.language}")
        logger.info(f"  Output format: {request.settings.output_format}")
        logger.info(f"  Sample rate: {request.settings.sample_rate} Hz")
    else:
        logger.info("  Using default settings")
    logger.info("")

    try:
        # Format transcript if needed
        logger.info("Formatting transcript...")
        formatted_transcript = voice_generator.format_transcript(
            request.transcript, request.speakers
        )
        logger.info(f"Formatted transcript length: {len(formatted_transcript)} characters")
        logger.info("")

        # Generate speech
        logger.info("Starting speech generation...")
        language = request.settings.language if request.settings else "en"
        output_path = voice_generator.generate_speech(
            transcript=formatted_transcript,
            speakers=request.speakers,
            language=language,
            speaker_instructions=request.speaker_instructions,
        )

        logger.info("")
        logger.info("Speech Generation Completed Successfully")
        logger.info(f"  Output file: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("=" * 80)

        # Return response with file path
        return SpeechGenerateResponse(
            success=True,
            message="Speech generated successfully",
            audio_url=f"/api/v1/speech/download/{output_path.name}",
            file_path=str(output_path),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error(f"Runtime error during speech generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech generation failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error during speech generation: {e}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@router.post(
    "/generate-stream",
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_speech_stream(
    request: SpeechGenerateRequest, http_request: Request
) -> StreamingResponse:
    """
    Generate speech with Server-Sent Events for progress updates.

    Streams progress events (type: progress) and a final result event (type: complete or error).
    """
    progress_queue: queue.Queue = queue.Queue()

    def on_progress(current: int, total: int, message: str) -> None:
        progress_queue.put(("progress", current, total, message))

    def run_generation() -> None:
        try:
            formatted = voice_generator.format_transcript(
                request.transcript, request.speakers
            )
            language = request.settings.language if request.settings else "en"
            output_path = voice_generator.generate_speech(
                transcript=formatted,
                speakers=request.speakers,
                language=language,
                speaker_instructions=request.speaker_instructions,
                progress_callback=on_progress,
            )
            progress_queue.put(
                (
                    "complete",
                    str(output_path.name),
                    f"/api/v1/speech/download/{output_path.name}",
                )
            )
        except Exception as e:
            progress_queue.put(("error", str(e), None))

    async def event_generator():
        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, run_generation)

        while True:
            try:
                ev = await asyncio.to_thread(progress_queue.get, timeout=0.25)
            except queue.Empty:
                if gen_task.done():
                    break
                continue

            if ev[0] == "progress":
                _, current, total, message = ev
                yield _sse_event(
                    {"type": "progress", "current": current, "total": total, "message": message}
                )
            elif ev[0] == "complete":
                _, filename, audio_url = ev
                yield _sse_event(
                    {
                        "type": "complete",
                        "success": True,
                        "message": "Speech generated successfully",
                        "audio_url": audio_url,
                        "filename": filename,
                    }
                )
                break
            elif ev[0] == "error":
                _, detail, _ = ev
                yield _sse_event(
                    {"type": "error", "success": False, "detail": detail}
                )
                break

        await gen_task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/download/{filename}",
    responses={
        404: {"model": ErrorResponse},
    },
)
async def download_speech(filename: str) -> FileResponse:
    """
    Download generated speech file.

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
