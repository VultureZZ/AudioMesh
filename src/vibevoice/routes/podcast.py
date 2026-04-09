"""
Podcast generation endpoints.
"""
import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from ..models.schemas import (
    ErrorResponse,
    PodcastArticleScriptRequest,
    PodcastGenerateRequest,
    PodcastGenerateResponse,
    PodcastProductionRequest,
    PodcastProductionStatusResponse,
    PodcastProductionSubmitResponse,
    PodcastScriptRequest,
    PodcastScriptResponse,
)
from ..config import config
from ..core.transcripts.transcriber import transcript_transcriber
from ..gpu_memory import (
    cuda_device_index_from_string,
    release_torch_cuda_memory,
    wait_for_cuda_memory,
)
from ..models.podcast_storage import podcast_storage
from ..services.audio_compositor import CuePlacement, audio_compositor
from ..services.podcast_generator import podcast_generator, production_style_to_genre_style, strip_production_cue_markers
from ..services.podcast_music_service import podcast_music_service
from ..services.podcast_timing_service import podcast_timing_service
from ..services.voice_generator import voice_generator
from ..services.voice_manager import voice_manager

logger = logging.getLogger(__name__)


def _release_tts_and_wait_for_acestep_vram(music_cues_enabled: bool) -> None:
    """
    Free GPU memory used by production steps before ACE-Step: WhisperX (dialogue timing),
    then Qwen3-TTS, then optionally wait until enough global VRAM is free for the music worker.
    """
    logger.info("Releasing WhisperX and TTS GPU memory before music cues")
    transcript_transcriber.unload_models()
    voice_generator.release_gpu_memory_after_speech()
    release_torch_cuda_memory()
    if not music_cues_enabled:
        return
    idx = cuda_device_index_from_string(config.ACESTEP_DEVICE)
    if idx is None:
        return
    mib = config.ACESTEP_MIN_FREE_VRAM_MIB
    min_bytes = mib * 1024 * 1024
    ok = wait_for_cuda_memory(
        min_bytes,
        device_index=idx,
        timeout_seconds=config.GPU_VRAM_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds=config.GPU_VRAM_POLL_INTERVAL_SECONDS,
    )
    if not ok:
        raise RuntimeError(
            f"GPU did not reach ~{mib} MiB free within "
            f"{config.GPU_VRAM_WAIT_TIMEOUT_SECONDS}s; free other CUDA workloads or set "
            "GPU_VRAM_WAIT_TIMEOUT_SECONDS=0 to wait indefinitely."
        )

router = APIRouter(prefix="/api/v1/podcast", tags=["podcast"])

_PRODUCTION_TASKS: Dict[str, Dict] = {}
_PRODUCTION_TASKS_LOCK = Lock()


def _audio_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    return "audio/wav"


def _set_production_task(task_id: str, **updates) -> None:
    with _PRODUCTION_TASKS_LOCK:
        if task_id not in _PRODUCTION_TASKS:
            _PRODUCTION_TASKS[task_id] = {}
        _PRODUCTION_TASKS[task_id].update(updates)


def _get_production_task(task_id: str) -> Dict | None:
    with _PRODUCTION_TASKS_LOCK:
        data = _PRODUCTION_TASKS.get(task_id)
        return dict(data) if data else None


def _initial_stage_progress() -> Dict[str, str]:
    return {
        "generating_script": "pending",
        "generating_voice_track": "pending",
        "generating_music_cues": "pending",
        "mixing_production_audio": "pending",
        "ready_to_download": "pending",
    }


def _merge_dialogue_timing(segments: List[Dict], dialogue_timing: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    out: List[Dict] = []
    dialogue_index = 0
    for segment in segments:
        if segment.get("segment_type") == "dialogue":
            if dialogue_index < len(dialogue_timing):
                timing = dialogue_timing[dialogue_index]
                merged = dict(segment)
                merged["start_time_hint"] = timing.get("start_time_hint", segment.get("start_time_hint", 0.0))
                merged["duration_ms"] = timing.get("duration_ms")
                merged["speaker"] = timing.get("speaker", segment.get("speaker"))
                merged["text"] = timing.get("text", segment.get("text"))
                out.append(merged)
            else:
                out.append(dict(segment))
            dialogue_index += 1
        else:
            out.append(dict(segment))
    return out


def _save_podcast_to_library(
    *,
    audio_source_path: Path,
    script_text: str,
    title: str | None,
    voices: List[str],
    source_url: str | None,
    genre: str | None,
    duration: str | None,
) -> tuple[str, Path]:
    podcast_id = str(uuid4())
    resolved_title = (title or "").strip() or f"Podcast {podcast_id[:8]}"
    config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    target_audio_path = config.PODCASTS_DIR / f"{podcast_id}{audio_source_path.suffix.lower() or '.wav'}"
    shutil.copy2(audio_source_path, target_audio_path)

    script_path = config.PODCASTS_DIR / f"{podcast_id}.txt"
    script_path.write_text(script_text)

    podcast_storage.add_podcast(
        podcast_id=podcast_id,
        title=resolved_title,
        voices=voices,
        audio_path=target_audio_path,
        script_path=script_path,
        source_url=source_url,
        genre=genre,
        duration=duration,
        extra={
            "file_size_bytes": target_audio_path.stat().st_size,
        },
    )
    return podcast_id, target_audio_path


async def _run_production_task(task_id: str, request: PodcastProductionRequest) -> None:
    stage_progress = _initial_stage_progress()
    cue_status: Dict[str, str] = {}
    warnings: List[str] = []

    try:
        stage_progress["generating_script"] = "running"
        _set_production_task(
            task_id,
            status="running",
            current_stage="Generating Script",
            progress_pct=8,
            stage_progress=stage_progress,
            warnings=warnings,
        )

        def _segment_script() -> List[Dict]:
            return podcast_generator.generate_script_segments(
                request.script,
                request.ollama_url,
                request.ollama_model,
                num_voices=len(request.voices),
                genre=request.genre,
                genre_style=production_style_to_genre_style(request.style) or request.genre or "General",
                duration=request.duration,
            )

        script_segments = await asyncio.to_thread(_segment_script)
        stage_progress["generating_script"] = "completed"
        _set_production_task(
            task_id,
            script_segments=script_segments,
            current_stage="Generating Voice Track",
            progress_pct=25,
            stage_progress=stage_progress,
        )

        stage_progress["generating_voice_track"] = "running"
        tts_script = strip_production_cue_markers(request.script)
        voice_path = await asyncio.to_thread(
            podcast_generator.generate_audio,
            tts_script,
            request.voices,
        )
        stage_progress["generating_voice_track"] = "completed"

        dialogue_timing = await podcast_timing_service.build_dialogue_timing(tts_script, voice_path)
        script_segments = _merge_dialogue_timing(script_segments, dialogue_timing)

        health = podcast_music_service.health_check()
        music_available = bool(health.get("available"))
        enabled = set(request.enabled_cues or [])
        will_run_music_cues = music_available and bool(
            enabled.intersection({"intro", "outro", "transitions", "bed"})
        )
        await asyncio.to_thread(_release_tts_and_wait_for_acestep_vram, will_run_music_cues)

        _set_production_task(
            task_id,
            script_segments=script_segments,
            current_stage="Generating Music Cues",
            progress_pct=45,
            stage_progress=stage_progress,
        )

        stage_progress["generating_music_cues"] = "running"

        cue_paths: Dict[str, str] = {}
        if not music_available:
            warnings.append("ACE-Step not configured. Continuing with voice-only output.")
            stage_progress["generating_music_cues"] = "skipped"
            _set_production_task(task_id, stage_progress=stage_progress, warnings=warnings)
        else:
            async def _generate_named_cue(cue_name: str) -> None:
                cue_status[cue_name] = "running"
                _set_production_task(task_id, cue_status=cue_status, stage_progress=stage_progress)
                try:
                    cue_paths[cue_name] = await podcast_music_service.generate_cue(
                        cue_name,
                        request.style,
                    )
                    cue_status[cue_name] = "succeeded"
                except Exception as exc:
                    cue_status[cue_name] = "failed"
                    warnings.append(f"Failed generating {cue_name} cue: {exc}")
                _set_production_task(task_id, cue_status=cue_status, warnings=warnings)

            # Run cues one at a time so ACE-Step and the GPU are not overloaded in parallel.
            cue_order = [
                ("intro", "intro"),
                ("bed", "bed"),
                ("transitions", "transition"),
                ("outro", "outro"),
            ]
            ran_any = False
            for flag, cue_name in cue_order:
                if flag not in enabled:
                    continue
                await _generate_named_cue(cue_name)
                ran_any = True
            if ran_any:
                stage_progress["generating_music_cues"] = "completed"
            else:
                stage_progress["generating_music_cues"] = "skipped"

        _set_production_task(
            task_id,
            current_stage="Mixing Production Audio",
            progress_pct=75,
            stage_progress=stage_progress,
            cue_status=cue_status,
        )

        stage_progress["mixing_production_audio"] = "running"
        cue_placements: List[CuePlacement] = []
        for segment in script_segments:
            if segment.get("segment_type") == "dialogue":
                cue_placements.append(
                    CuePlacement(
                        cue_type="dialogue",
                        file_path=voice_path,
                        position_ms=int(float(segment.get("start_time_hint", 0.0)) * 1000),
                        duration_ms=int(segment.get("duration_ms") or 0),
                    )
                )

        if "intro" in cue_paths:
            cue_placements.append(CuePlacement(cue_type="intro", file_path=cue_paths["intro"], position_ms=0, volume_db=-1.5))
        if "bed" in cue_paths:
            cue_placements.append(CuePlacement(cue_type="bed", file_path=cue_paths["bed"], position_ms=0, volume_db=0.0))
        if "outro" in cue_paths:
            cue_placements.append(CuePlacement(cue_type="outro", file_path=cue_paths["outro"], position_ms=0, volume_db=-2.0))
        if "transition" in cue_paths:
            for segment in script_segments:
                if segment.get("segment_type") == "transition_sting":
                    cue_placements.append(
                        CuePlacement(
                            cue_type="transition",
                            file_path=cue_paths["transition"],
                            position_ms=int(float(segment.get("start_time_hint", 0.0)) * 1000),
                            volume_db=-1.0,
                        )
                    )

        final_path = voice_path
        if cue_paths:
            try:
                final_path = await asyncio.to_thread(audio_compositor.mix_podcast, voice_path, cue_placements)
            except Exception as exc:
                warnings.append(f"Audio mixing failed; returning voice-only output: {exc}")

        stage_progress["mixing_production_audio"] = "completed"
        stage_progress["ready_to_download"] = "completed"

        output_file = Path(final_path)
        podcast_id = None
        audio_url = f"/api/v1/podcast/download/{output_file.name}"
        saved_path = output_file
        if request.save_to_library:
            podcast_id, saved_path = _save_podcast_to_library(
                audio_source_path=output_file,
                script_text=request.script,
                title=request.title,
                voices=request.voices,
                source_url=request.source_url,
                genre=request.genre,
                duration=request.duration,
            )
            audio_url = f"/api/v1/podcasts/{podcast_id}/download"

        _set_production_task(
            task_id,
            success=True,
            message="Production podcast generated successfully",
            status="succeeded",
            current_stage="Ready to Download",
            progress_pct=100,
            stage_progress=stage_progress,
            cue_status=cue_status,
            audio_url=audio_url,
            file_path=str(saved_path),
            podcast_id=podcast_id,
            warnings=warnings,
        )
    except Exception as exc:
        stage_progress["ready_to_download"] = "failed"
        _set_production_task(
            task_id,
            status="failed",
            message="Production podcast generation failed",
            current_stage="Failed",
            progress_pct=100,
            stage_progress=stage_progress,
            cue_status=cue_status,
            warnings=warnings,
            error=str(exc),
        )


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
    logger.info(f"Approximate duration (minutes): {request.approximate_duration_minutes}")
    logger.info(f"Voices: {request.voices}")
    logger.info(f"Number of voices: {len(request.voices)}")
    if request.ollama_url:
        logger.info(f"Custom Ollama URL: {request.ollama_url}")
    if request.ollama_model:
        logger.info(f"Custom Ollama Model: {request.ollama_model}")
    logger.info("")

    try:
        warnings = voice_manager.get_bgm_risk_warnings(request.voices)

        # Generate script (CPU/network-heavy; must not block the asyncio event loop)
        logger.info("Generating podcast script...")
        script = await asyncio.to_thread(
            podcast_generator.generate_script,
            request.url,
            request.genre,
            request.duration,
            request.voices,
            request.ollama_url,
            request.ollama_model,
            request.approximate_duration_minutes,
            request.include_production_cues,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            script,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            num_voices=len(request.voices),
            genre=request.genre,
            genre_style=request.genre,
            duration=request.duration,
            approximate_duration_minutes=request.approximate_duration_minutes,
        )

        logger.info("")
        logger.info("Script Generation Completed Successfully")
        logger.info(f"  Script length: {len(script)} characters")
        logger.info("=" * 80)

        return PodcastScriptResponse(
            success=True,
            message="Podcast script generated successfully",
            script=script,
            script_segments=script_segments,
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
    "/generate-script-from-article",
    response_model=PodcastScriptResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_podcast_script_from_article(
    request: PodcastArticleScriptRequest, http_request: Request
) -> PodcastScriptResponse:
    """
    Generate a podcast script from raw article text, using the same voice-profile-aware pipeline as URL-based generation.

    ``narrator_speaker_index`` selects which logical speaker (Speaker 1..N) carries narration; it must not exceed
    the number of entries in ``voices``.
    """
    client_ip = http_request.client.host if http_request.client else "unknown"

    logger.info("=" * 80)
    logger.info("Podcast Script Generation (from article body) Request Received")
    logger.info("=" * 80)
    logger.info("Client IP: %s", client_ip)
    logger.info("Title: %s", request.title or "(none)")
    logger.info("Genre: %s", request.genre)
    logger.info("Duration: %s", request.duration)
    logger.info("Approximate duration (minutes): %s", request.approximate_duration_minutes)
    logger.info("Voices: %s", request.voices)
    logger.info("Narrator speaker index: %s", request.narrator_speaker_index)
    if request.ollama_url:
        logger.info("Custom Ollama URL: %s", request.ollama_url)
    if request.ollama_model:
        logger.info("Custom Ollama Model: %s", request.ollama_model)
    logger.info("")

    try:
        warnings = voice_manager.get_bgm_risk_warnings(request.voices)

        logger.info("Generating podcast script from article body...")
        script = await asyncio.to_thread(
            podcast_generator.generate_script_from_article,
            request.article_text,
            request.genre,
            request.duration,
            request.voices,
            request.narrator_speaker_index,
            request.title,
            request.ollama_url,
            request.ollama_model,
            request.approximate_duration_minutes,
            request.include_production_cues,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            script,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            num_voices=len(request.voices),
            genre=request.genre,
            genre_style=request.genre,
            duration=request.duration,
            approximate_duration_minutes=request.approximate_duration_minutes,
        )

        logger.info("")
        logger.info("Script Generation (from article) Completed Successfully")
        logger.info("  Script length: %d characters", len(script))
        logger.info("=" * 80)

        return PodcastScriptResponse(
            success=True,
            message="Podcast script generated successfully",
            script=script,
            script_segments=script_segments,
            warnings=warnings,
        )

    except ValueError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error("Runtime error during script generation: %s", e)
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script generation failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during script generation: %s", e)
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

        # Generate audio (TTS/GPU-heavy; must not block the asyncio event loop)
        logger.info("Generating podcast audio...")
        output_path = await asyncio.to_thread(
            podcast_generator.generate_audio,
            request.script,
            request.voices,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            request.script,
            genre=request.genre,
            genre_style=request.genre,
            duration=request.duration,
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
            podcast_id, saved_path = _save_podcast_to_library(
                audio_source_path=output_file,
                script_text=request.script,
                title=request.title,
                voices=request.voices,
                source_url=request.source_url,
                genre=request.genre,
                duration=request.duration,
            )
            audio_url = f"/api/v1/podcasts/{podcast_id}/download"

        return PodcastGenerateResponse(
            success=True,
            message="Podcast audio generated successfully",
            audio_url=audio_url,
            file_path=str(saved_path),
            script=request.script,
            script_segments=script_segments,
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


@router.post(
    "/generate-production",
    response_model=PodcastProductionSubmitResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_podcast_production(
    request: PodcastProductionRequest, http_request: Request
) -> PodcastProductionSubmitResponse:
    """
    Submit async production-mode podcast generation task.
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    logger.info("Production podcast task requested from %s", client_ip)

    if not request.script or not request.script.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Script cannot be empty")
    if not request.voices:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one voice is required")
    if len(request.voices) > 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 4 voices allowed")

    task_id = str(uuid4())
    _set_production_task(
        task_id,
        success=True,
        message="Production podcast task accepted",
        status="queued",
        current_stage="Queued",
        progress_pct=0,
        stage_progress=_initial_stage_progress(),
        cue_status={},
        audio_url=None,
        file_path=None,
        podcast_id=None,
        script_segments=[],
        warnings=[],
        error=None,
        created_at=datetime.utcnow().isoformat(),
    )
    asyncio.create_task(_run_production_task(task_id, request))
    return PodcastProductionSubmitResponse(
        success=True,
        message="Production podcast task accepted",
        task_id=task_id,
        status="queued",
    )


@router.get(
    "/status/{task_id}",
    response_model=PodcastProductionStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_podcast_production_status(task_id: str) -> PodcastProductionStatusResponse:
    task = _get_production_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Podcast production task not found")
    return PodcastProductionStatusResponse(
        success=bool(task.get("success", True)),
        message=task.get("message", "Task status"),
        task_id=task_id,
        status=task.get("status", "queued"),
        current_stage=task.get("current_stage"),
        progress_pct=int(task.get("progress_pct", 0)),
        stage_progress=task.get("stage_progress") or {},
        cue_status=task.get("cue_status") or {},
        audio_url=task.get("audio_url"),
        file_path=task.get("file_path"),
        podcast_id=task.get("podcast_id"),
        script_segments=task.get("script_segments") or [],
        warnings=task.get("warnings") or [],
        error=task.get("error"),
    )


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
        media_type=_audio_media_type(file_path),
        filename=filename,
    )
