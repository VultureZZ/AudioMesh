"""
Podcast generation endpoints.
"""
import asyncio
import copy
import json
import logging
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from ..models.schemas import (
    ErrorResponse,
    PodcastArticleScriptRequest,
    PodcastCompareRequest,
    PodcastCompareStatusResponse,
    PodcastCompareSubmitResponse,
    PodcastGenerateRequest,
    PodcastGenerateResponse,
    PodcastProductionRequest,
    PodcastProductionStatusResponse,
    PodcastProductionSubmitResponse,
    PodcastScriptRequest,
    PodcastScriptResponse,
    RegenerateEventRequest,
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

_PRODUCTION_RENDER_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=50)
_COMPARE_TASKS: Dict[str, Dict[str, Any]] = {}
_COMPARE_LOCK = Lock()


def _record_production_render(task_id: str, snapshot: Dict[str, Any]) -> None:
    row = {"task_id": task_id, **snapshot}
    _PRODUCTION_RENDER_HISTORY.appendleft(row)


def _get_production_render_history(limit: int = 50) -> List[Dict[str, Any]]:
    return list(_PRODUCTION_RENDER_HISTORY)[:limit]


def _set_compare_task(compare_id: str, **updates: Any) -> None:
    with _COMPARE_LOCK:
        if compare_id not in _COMPARE_TASKS:
            _COMPARE_TASKS[compare_id] = {}
        _COMPARE_TASKS[compare_id].update(updates)


def _get_compare_task(compare_id: str) -> Dict[str, Any] | None:
    with _COMPARE_LOCK:
        data = _COMPARE_TASKS.get(compare_id)
        return dict(data) if data else None


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


def has_running_production_tasks() -> bool:
    """True while an async production podcast task is queued or running."""
    with _PRODUCTION_TASKS_LOCK:
        for data in _PRODUCTION_TASKS.values():
            st = (data or {}).get("status")
            if st in ("queued", "running"):
                return True
    return False


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

    from app.services.genre_templates import resolve_genre_template

    genre_template = resolve_genre_template(
        template_id=request.production_genre,
        style=request.style,
        metadata_genre=request.genre,
    )

    try:
        from vibevoice.services.ollama_client import normalize_podcast_speaker_labels

        script_for_pipeline = normalize_podcast_speaker_labels(
            (request.script or "").strip(),
            len(request.voices),
            include_production_cues="[CUE:" in (request.script or "").upper(),
        )

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
                script_for_pipeline,
                request.ollama_url,
                request.ollama_model,
                llm_provider=request.llm_provider,
                openai_api_key=request.openai_api_key,
                openai_model=request.openai_model,
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
        tts_script = strip_production_cue_markers(script_for_pipeline)
        genre_label = (request.genre or "").strip() or production_style_to_genre_style(request.style) or "General"

        prosody_plan = None
        vd_for_tts: Optional[List[Any]] = None
        director_for_mix = None
        library = None

        if getattr(config, "VOICE_DIRECTION_ENABLED", True) and config.USE_PRODUCTION_DIRECTOR:
            from app.services.asset_library import AssetLibrary
            from app.services.generation_queue import GenerationQueue
            from app.services.production_director import ProductionDirector
            from app.services.voice_prosody import synthetic_timing_hints_from_segments

            _set_production_task(
                task_id,
                current_stage="Director prosody (pre-TTS)",
                progress_pct=28,
                stage_progress=stage_progress,
            )
            library = AssetLibrary()
            try:
                from app.services.backchannel_synth import ensure_backchannel_assets

                await asyncio.to_thread(ensure_backchannel_assets, request.voices, library)
            except Exception as exc:
                warnings.append(f"Backchannel asset cache skipped: {exc}")

            director_for_mix = ProductionDirector(
                base_url=request.ollama_url,
                model=request.ollama_model,
            )
            coarse_hints = synthetic_timing_hints_from_segments(script_segments)
            generation_queue = GenerationQueue(library, genre_template=genre_template)
            try:
                prosody_plan = await director_for_mix.plan(
                    script_for_pipeline,
                    script_segments,
                    genre_label,
                    [],
                    coarse_hints,
                    asset_library=library,
                    generation_queue=generation_queue,
                    use_tools=False,
                    word_index=None,
                    genre_template=genre_template,
                )
                if prosody_plan.voice_direction:
                    vd_for_tts = [ln.model_dump() for ln in prosody_plan.voice_direction]
                else:
                    from app.services.voice_prosody import fallback_voice_direction_for_script

                    vd_for_tts = fallback_voice_direction_for_script(tts_script, genre_label)
            except Exception as exc:
                warnings.append(f"Pre-TTS Director prosody failed; using fallback voice direction: {exc}")
                from app.services.voice_prosody import fallback_voice_direction_for_script

                vd_for_tts = fallback_voice_direction_for_script(tts_script, genre_label)
        elif getattr(config, "VOICE_DIRECTION_ENABLED", True):
            from app.services.voice_prosody import fallback_voice_direction_for_script

            vd_for_tts = fallback_voice_direction_for_script(tts_script, genre_label)

        t_tts = time.perf_counter()
        voice_path = await asyncio.to_thread(
            podcast_generator.generate_audio,
            tts_script,
            request.voices,
            voice_direction=vd_for_tts,
        )
        stage_progress["generating_voice_track"] = "completed"
        try:
            from app.services.pipeline_log import log_pipeline_event

            log_pipeline_event(
                "tts_voice",
                task_id=task_id,
                duration_ms=(time.perf_counter() - t_tts) * 1000.0,
            )
        except Exception:
            pass

        t_align = time.perf_counter()
        dialogue_timing, word_index = await podcast_timing_service.build_alignment_bundle(
            tts_script, voice_path
        )
        try:
            from app.services.pipeline_log import log_pipeline_event

            log_pipeline_event(
                "alignment_bundle",
                task_id=task_id,
                duration_ms=(time.perf_counter() - t_align) * 1000.0,
            )
        except Exception:
            pass
        script_segments = _merge_dialogue_timing(script_segments, dialogue_timing)

        timing_hints: List[Dict[str, Any]] = []
        for i, row in enumerate(dialogue_timing):
            st = float(row.get("start_time_hint", 0.0))
            dur = int(row.get("duration_ms") or 0)
            timing_hints.append(
                {
                    "line_index": i,
                    "start_ms": int(st * 1000),
                    "end_ms": int(st * 1000) + dur,
                    "speaker": row.get("speaker"),
                    "text": row.get("text"),
                }
            )

        voice_line_timing_ms: List[tuple] = [
            (int(h["line_index"]), float(h["start_ms"]), float(h["end_ms"])) for h in timing_hints
        ]

        _set_production_task(
            task_id,
            voice_track_path=str(voice_path),
            word_index=word_index,
            timing_hints=timing_hints,
            voice_line_timing_ms=voice_line_timing_ms,
            genre_template_id=genre_template.genre_id,
        )

        use_director = config.USE_PRODUCTION_DIRECTOR
        qa_pack: Optional[Dict[str, Any]] = None

        if use_director:
            from app.services.asset_library import AssetLibrary
            from app.services.generation_queue import GenerationQueue
            from app.services.production_director import ProductionDirector

            _set_production_task(
                task_id,
                script_segments=script_segments,
                current_stage="Director planning",
                progress_pct=35,
                stage_progress=stage_progress,
            )

            if library is None:
                library = AssetLibrary()
            try:
                from app.services.backchannel_synth import ensure_backchannel_assets

                await asyncio.to_thread(ensure_backchannel_assets, request.voices, library)
            except Exception as exc:
                warnings.append(f"Backchannel asset cache skipped: {exc}")

            generation_queue = GenerationQueue(library, genre_template=genre_template)
            if director_for_mix is None:
                director_for_mix = ProductionDirector(
                    base_url=request.ollama_url,
                    model=request.ollama_model,
                )

            production_plan = await director_for_mix.plan(
                script_for_pipeline,
                script_segments,
                genre_label,
                [],
                timing_hints,
                asset_library=library,
                generation_queue=generation_queue,
                use_tools=True,
                word_index=word_index,
                genre_template=genre_template,
            )
            if prosody_plan is not None and getattr(prosody_plan, "voice_direction", None):
                production_plan = production_plan.model_copy(update={"voice_direction": prosody_plan.voice_direction})

            try:
                from app.services.pipeline_log import log_pipeline_event

                log_pipeline_event("director_plan", task_id=task_id, extra={"tools": True})
            except Exception:
                pass

            _set_production_task(
                task_id,
                current_stage="Director planning",
                progress_pct=40,
                stage_progress=stage_progress,
                production_plan=production_plan.model_dump(mode="json"),
            )

            await asyncio.to_thread(_release_tts_and_wait_for_acestep_vram, True)

            stage_progress["generating_music_cues"] = "running"
            _set_production_task(
                task_id,
                current_stage="Generating production assets",
                progress_pct=48,
                stage_progress=stage_progress,
            )

            production_plan_with_prompts = production_plan.model_dump(mode="json")
            t_assets = time.perf_counter()
            try:
                production_plan = await generation_queue.run_queued_and_patch_plan(production_plan)
            except Exception as exc:
                warnings.append(f"Asset generation queue failed: {exc}")
            warnings.extend(generation_queue.warnings)
            try:
                from app.services.backchannel_resolve import patch_production_plan_voice_backchannels

                production_plan = patch_production_plan_voice_backchannels(
                    production_plan, library, request.voices
                )
            except Exception as exc:
                warnings.append(f"Voice backchannel patch failed: {exc}")
            stage_progress["generating_music_cues"] = "completed"
            try:
                from app.services.pipeline_log import log_pipeline_event

                log_pipeline_event(
                    "production_assets",
                    task_id=task_id,
                    duration_ms=(time.perf_counter() - t_assets) * 1000.0,
                )
            except Exception:
                pass

            _set_production_task(
                task_id,
                progress_pct=55,
                stage_progress=stage_progress,
                warnings=warnings,
                production_plan=production_plan.model_dump(mode="json"),
                production_plan_with_prompts=production_plan_with_prompts,
            )

            stage_progress["mixing_production_audio"] = "running"
            _set_production_task(
                task_id,
                current_stage="Mixing Production Audio",
                progress_pct=68,
                stage_progress=stage_progress,
            )

            final_path: str | Path = voice_path
            t_mix = time.perf_counter()
            try:
                final_path = await asyncio.to_thread(
                    audio_compositor.mix_production_plan,
                    str(voice_path),
                    production_plan,
                    library,
                    word_index=word_index,
                    timing_hints=timing_hints,
                    voice_line_timing_ms=voice_line_timing_ms,
                    genre_template=genre_template,
                )
            except Exception as exc:
                warnings.append(f"Production plan mix failed; returning voice-only output: {exc}")
            try:
                from app.services.pipeline_log import log_pipeline_event

                log_pipeline_event(
                    "mix_production",
                    task_id=task_id,
                    duration_ms=(time.perf_counter() - t_mix) * 1000.0,
                )
            except Exception:
                pass

            try:
                from app.services.production_artifacts import copy_production_cue_review_files

                review_dir = config.OUTPUT_DIR / f"production_{task_id}" / "cue_review"
                cue_review_files = copy_production_cue_review_files(
                    review_dir, production_plan, library
                )
                _set_production_task(
                    task_id,
                    cue_review_dir=str(review_dir.resolve()),
                    cue_review_files=cue_review_files,
                )
            except Exception as exc:
                warnings.append(f"Cue review copy failed: {exc}")

            try:
                from app.services.genre_templates import mastering_lufs
                from app.services.mix_qa import run_mix_qa

                dialogue_regions = [(int(h["start_ms"]), int(h["end_ms"])) for h in timing_hints]
                target_lufs = mastering_lufs(genre_template, production_plan.genre)
                qa_pack = run_mix_qa(
                    Path(final_path),
                    target_lufs=target_lufs,
                    plan_duration_seconds=float(production_plan.duration_target_seconds),
                    dialogue_regions_ms=dialogue_regions,
                    plan=production_plan,
                )
                _set_production_task(
                    task_id,
                    qa_results=qa_pack,
                    mix_metadata=qa_pack.get("summary"),
                    plan_duration_seconds=float(production_plan.duration_target_seconds),
                    episode_metadata_extra={"mix_qa": json.dumps(qa_pack, default=str)},
                )
            except Exception as exc:
                err_msg = f"Mix QA failed: {exc}"
                warnings.append(err_msg)
                logger.warning("%s", err_msg, exc_info=True)
                _set_production_task(task_id, mix_qa_error=str(exc))

            stage_progress["mixing_production_audio"] = "completed"
            stage_progress["ready_to_download"] = "completed"
            _set_production_task(
                task_id,
                progress_pct=80,
                stage_progress=stage_progress,
                warnings=warnings,
            )

        else:
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

            stage_progress["mixing_production_audio"] = "running"
            _set_production_task(
                task_id,
                current_stage="Mixing Production Audio",
                progress_pct=68,
                stage_progress=stage_progress,
                cue_status=cue_status,
            )

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
                cue_placements.append(
                    CuePlacement(cue_type="intro", file_path=cue_paths["intro"], position_ms=0, volume_db=-1.5)
                )
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

            final_path: str | Path = voice_path
            if cue_paths:
                try:
                    final_path = await asyncio.to_thread(audio_compositor.mix_podcast, voice_path, cue_placements)
                except Exception as exc:
                    warnings.append(f"Audio mixing failed; returning voice-only output: {exc}")

            if cue_paths:
                try:
                    from app.services.production_artifacts import copy_legacy_cue_paths_review

                    review_dir = config.OUTPUT_DIR / f"production_{task_id}" / "cue_review"
                    cue_review_files = copy_legacy_cue_paths_review(review_dir, cue_paths)
                    _set_production_task(
                        task_id,
                        cue_review_dir=str(review_dir.resolve()),
                        cue_review_files=cue_review_files,
                    )
                except Exception as exc:
                    warnings.append(f"Cue review copy failed: {exc}")

            stage_progress["mixing_production_audio"] = "completed"
            stage_progress["ready_to_download"] = "completed"
            _set_production_task(
                task_id,
                progress_pct=80,
                stage_progress=stage_progress,
                warnings=warnings,
            )

            if not use_director:
                try:
                    from app.services.mix_qa import run_mix_qa

                    est_dur = 0.0
                    for seg in script_segments or []:
                        du = seg.get("duration_ms")
                        if du is not None:
                            est_dur += float(du) / 1000.0
                    if est_dur <= 0:
                        est_dur = 120.0
                    qa_pack = run_mix_qa(
                        Path(final_path),
                        target_lufs=-16.0,
                        plan_duration_seconds=est_dur,
                        dialogue_regions_ms=[
                            (int(float(s.get("start_time_hint", 0)) * 1000), int(float(s.get("start_time_hint", 0)) * 1000) + int(s.get("duration_ms") or 0))
                            for s in (script_segments or [])
                            if str(s.get("segment_type") or "").lower() == "dialogue"
                        ],
                        plan=None,
                    )
                    _set_production_task(
                        task_id,
                        qa_results=qa_pack,
                        mix_metadata=qa_pack.get("summary"),
                        plan_duration_seconds=est_dur,
                        episode_metadata_extra={"mix_qa": json.dumps(qa_pack, default=str)},
                    )
                except Exception as exc:
                    err_msg = f"Mix QA failed: {exc}"
                    warnings.append(err_msg)
                    logger.warning("%s", err_msg, exc_info=True)
                    _set_production_task(task_id, mix_qa_error=str(exc))

        output_file = Path(final_path)
        podcast_id = None
        audio_url = f"/api/v1/podcast/download/{output_file.name}"
        saved_path = output_file
        if request.save_to_library:
            podcast_id, saved_path = _save_podcast_to_library(
                audio_source_path=output_file,
                script_text=script_for_pipeline,
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
        done = _get_production_task(task_id) or {}
        mm = done.get("mix_metadata") if isinstance(done.get("mix_metadata"), dict) else {}
        qa_summary = done.get("qa_results") if isinstance(done.get("qa_results"), dict) else {}
        if not mm and isinstance(qa_summary.get("summary"), dict):
            mm = qa_summary["summary"]
        plan_d = done.get("plan_duration_seconds")
        drift = None
        if plan_d is not None and mm.get("rendered_duration_sec") is not None:
            drift = abs(float(mm["rendered_duration_sec"]) - float(plan_d))
        qa_pass = mm.get("all_passed")
        if qa_pass is None and isinstance(qa_summary.get("summary"), dict):
            qa_pass = qa_summary["summary"].get("all_passed")
        _record_production_render(
            task_id,
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "qa_all_passed": qa_pass,
                "duration_drift_sec": drift,
                "audio_url": audio_url,
                "status": "succeeded",
            },
        )
        try:
            from app.services.pipeline_log import log_pipeline_event

            log_pipeline_event(
                "production_complete",
                task_id=task_id,
                extra={
                    "qa_all_passed": qa_pass,
                    "duration_drift_sec": drift,
                    "mix_qa_error": done.get("mix_qa_error"),
                },
            )
        except Exception:
            pass
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
    logger.info(f"LLM provider: {request.llm_provider}")
    if request.ollama_url:
        logger.info(f"Custom Ollama URL: {request.ollama_url}")
    if request.ollama_model:
        logger.info(f"Custom Ollama Model: {request.ollama_model}")
    if request.openai_model:
        logger.info(f"OpenAI model: {request.openai_model}")
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
            request.llm_provider,
            request.openai_api_key,
            request.openai_model,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            script,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            llm_provider=request.llm_provider,
            openai_api_key=request.openai_api_key,
            openai_model=request.openai_model,
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
    logger.info("LLM provider: %s", request.llm_provider)
    if request.ollama_url:
        logger.info("Custom Ollama URL: %s", request.ollama_url)
    if request.ollama_model:
        logger.info("Custom Ollama Model: %s", request.ollama_model)
    if request.openai_model:
        logger.info("OpenAI model: %s", request.openai_model)
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
            request.llm_provider,
            request.openai_api_key,
            request.openai_model,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            script,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            llm_provider=request.llm_provider,
            openai_api_key=request.openai_api_key,
            openai_model=request.openai_model,
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

        from vibevoice.services.ollama_client import normalize_podcast_speaker_labels

        script_norm = normalize_podcast_speaker_labels(
            (request.script or "").strip(),
            len(request.voices),
            include_production_cues="[CUE:" in (request.script or "").upper(),
        )

        # Generate audio (TTS/GPU-heavy; must not block the asyncio event loop)
        logger.info("Generating podcast audio...")
        output_path = await asyncio.to_thread(
            podcast_generator.generate_audio,
            script_norm,
            request.voices,
        )
        script_segments = await asyncio.to_thread(
            podcast_generator.generate_script_segments,
            script_norm,
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
                script_text=script_norm,
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
            script=script_norm,
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
        mix_qa_error=task.get("mix_qa_error"),
    )


@router.get("/production/{task_id}/audition")
async def get_production_audition_data(task_id: str) -> Dict[str, Any]:
    """JSON for operator audition UI: plan, QA, audio URL."""
    task = _get_production_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "production_plan": task.get("production_plan"),
        "production_plan_with_prompts": task.get("production_plan_with_prompts"),
        "qa_results": task.get("qa_results"),
        "mix_metadata": task.get("mix_metadata"),
        "audio_url": task.get("audio_url"),
        "file_path": task.get("file_path"),
        "voice_track_path": task.get("voice_track_path"),
        "genre_template_id": task.get("genre_template_id"),
        "cue_review_dir": task.get("cue_review_dir"),
        "cue_review_files": task.get("cue_review_files") or [],
        "warnings": task.get("warnings") or [],
    }


@router.post("/production/{task_id}/regenerate-event")
async def post_regenerate_track_event(task_id: str, body: RegenerateEventRequest) -> Dict[str, Any]:
    """Regenerate one stem via ACE-Step/SAO and remix."""
    task = _get_production_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task.get("status") != "succeeded":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task must have succeeded")
    vp = task.get("voice_track_path")
    if not vp or not Path(str(vp)).is_file():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Voice track missing for remix")
    plan_src = task.get("production_plan_with_prompts") or task.get("production_plan")
    if not plan_src:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No production plan stored")

    from app.services.asset_library import AssetLibrary
    from app.services.genre_templates import TEMPLATES, resolve_genre_template
    from app.services.generation_queue import GenerationQueue
    from app.services.mix_qa import run_mix_qa
    from app.services.production_director import ProductionPlan

    plan_pre = ProductionPlan.model_validate(plan_src)
    prompt = None
    duration_ms = 3000
    tr_role = None
    for tr in plan_pre.tracks:
        if tr.track_id != body.track_id:
            continue
        tr_role = str(tr.track_role)
        for ev in tr.events:
            if ev.event_id != body.event_id:
                continue
            if not ev.asset_ref or not ev.asset_ref.generation_prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Event has no generation_prompt; cannot regenerate",
                )
            prompt = ev.asset_ref.generation_prompt
            duration_ms = int(ev.duration_ms)
            break
    if not prompt or not tr_role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track/event not found")

    gid = task.get("genre_template_id")
    gt = TEMPLATES.get(str(gid)) if gid else None
    if gt is None:
        gt = resolve_genre_template(template_id=str(gid) if gid else None, style=None, metadata_genre=None)

    library = AssetLibrary()
    gq = GenerationQueue(library, genre_template=gt)
    aid = await gq.generate_for_track_event(
        track_role=tr_role,
        prompt=prompt,
        duration_ms=duration_ms,
        plan_genre=plan_pre.genre,
    )
    if not aid:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generation failed")

    final = task.get("production_plan")
    if not final:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No patched plan for remix")
    if isinstance(final, dict):
        raw = copy.deepcopy(final)
    else:
        raw = ProductionPlan.model_validate(final).model_dump(mode="json")
    for t in raw.get("tracks") or []:
        if t.get("track_id") != body.track_id:
            continue
        for e in t.get("events") or []:
            if e.get("event_id") == body.event_id:
                e["asset_ref"] = {"asset_id": aid}
                break
    new_plan = ProductionPlan.model_validate(raw)

    from app.services.genre_templates import mastering_lufs

    new_path = await asyncio.to_thread(
        audio_compositor.mix_production_plan,
        str(vp),
        new_plan,
        library,
        word_index=task.get("word_index"),
        timing_hints=task.get("timing_hints"),
        voice_line_timing_ms=task.get("voice_line_timing_ms"),
        genre_template=gt,
    )
    th = task.get("timing_hints") or []
    dialogue_regions = [(int(h["start_ms"]), int(h["end_ms"])) for h in th]
    qa = run_mix_qa(
        Path(new_path),
        target_lufs=mastering_lufs(gt, new_plan.genre),
        plan_duration_seconds=float(new_plan.duration_target_seconds),
        dialogue_regions_ms=dialogue_regions,
        plan=new_plan,
    )
    out_name = Path(new_path).name
    audio_url = f"/api/v1/podcast/download/{out_name}"
    _set_production_task(
        task_id,
        production_plan=raw,
        file_path=str(Path(new_path).resolve()),
        audio_url=audio_url,
        qa_results=qa,
        mix_metadata=qa.get("summary"),
    )
    return {"success": True, "audio_url": audio_url, "file_path": str(new_path), "qa_results": qa, "asset_id": aid}


@router.post("/compare", response_model=PodcastCompareSubmitResponse)
async def post_podcast_compare(request: PodcastCompareRequest) -> PodcastCompareSubmitResponse:
    """A/B two genre templates with the same voice track."""
    if not request.script.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Script cannot be empty")
    compare_id = str(uuid4())
    _set_compare_task(
        compare_id,
        status="queued",
        message="Compare queued",
        created_at=datetime.utcnow().isoformat(),
    )

    async def _run() -> None:
        from vibevoice.services.compare_ab_task import run_compare_ab_task

        genre_label = (request.genre or "").strip() or production_style_to_genre_style(request.style) or "General"
        await run_compare_ab_task(
            compare_id,
            script=request.script,
            voices=request.voices,
            genre_a=request.genres[0],
            genre_b=request.genres[1],
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            llm_provider=request.llm_provider,
            openai_api_key=request.openai_api_key,
            openai_model=request.openai_model,
            genre_label=genre_label,
            duration=request.duration,
            style=request.style,
            set_status=_set_compare_task,
        )

    asyncio.create_task(_run())
    return PodcastCompareSubmitResponse(compare_id=compare_id, status="queued", message="Compare task accepted")


@router.get("/compare/{compare_id}/status", response_model=PodcastCompareStatusResponse)
async def get_compare_status(compare_id: str) -> PodcastCompareStatusResponse:
    t = _get_compare_task(compare_id)
    if not t:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Compare task not found")
    return PodcastCompareStatusResponse(
        compare_id=compare_id,
        status=t.get("status", "queued"),
        message=t.get("message", ""),
        audio_url_a=t.get("audio_url_a"),
        audio_url_b=t.get("audio_url_b"),
        file_path_a=t.get("file_path_a"),
        file_path_b=t.get("file_path_b"),
        qa_a=t.get("qa_a"),
        qa_b=t.get("qa_b"),
        warnings=t.get("warnings") or [],
        error=t.get("error"),
    )


@router.get("/admin/production-renders")
async def get_admin_production_renders() -> Dict[str, Any]:
    """Last 50 production renders with QA summary for operator dashboard."""
    return {"renders": _get_production_render_history(50)}


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
