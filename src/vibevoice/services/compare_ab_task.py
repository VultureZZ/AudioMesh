"""
A/B production: one voice + alignment, two Director + mix passes with different genre templates.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_compare_ab_task(
    compare_id: str,
    *,
    script: str,
    voices: List[str],
    genre_a: str,
    genre_b: str,
    genre_label: str,
    duration: Optional[str],
    style: str,
    set_status: Callable[..., None],
    ollama_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    llm_provider: str = "ollama",
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
) -> None:
    """Run two full production branches (Director + assets + mix) sharing TTS and library."""
    from app.services.asset_library import AssetLibrary
    from app.services.backchannel_synth import ensure_backchannel_assets
    from app.services.genre_templates import mastering_lufs, resolve_genre_template
    from app.services.mix_qa import run_mix_qa
    from app.services.pipeline_log import log_pipeline_event
    from app.services.production_director import ProductionDirector
    from app.services.generation_queue import GenerationQueue
    from vibevoice.config import config
    from vibevoice.services.audio_compositor import audio_compositor
    from vibevoice.services.podcast_generator import podcast_generator, production_style_to_genre_style, strip_production_cue_markers
    from vibevoice.services.podcast_timing_service import podcast_timing_service

    warnings: List[str] = []
    try:
        if (llm_provider or "ollama").strip().lower() == "openai":
            set_status(
                compare_id,
                status="failed",
                error=(
                    "Genre compare requires the Ollama-based Production Director. "
                    "Set Primary LLM to Ollama in Settings for this workflow."
                ),
            )
            return

        set_status(compare_id, status="running", message="Segmenting script")
        t0 = time.perf_counter()

        def _segment() -> List[Dict[str, Any]]:
            return podcast_generator.generate_script_segments(
                script,
                ollama_url,
                ollama_model,
                llm_provider=llm_provider,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                num_voices=len(voices),
                genre=genre_label,
                genre_style=production_style_to_genre_style(style) or genre_label or "General",
                duration=duration,
            )

        script_segments = await asyncio.to_thread(_segment)
        log_pipeline_event(
            "compare_segment_script",
            task_id=compare_id,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"event": "compare_ab"},
        )

        tts_script = strip_production_cue_markers(script)
        t_tts = time.perf_counter()
        voice_path = await asyncio.to_thread(
            podcast_generator.generate_audio,
            tts_script,
            voices,
            voice_direction=None,
        )
        log_pipeline_event(
            "compare_tts",
            task_id=compare_id,
            duration_ms=(time.perf_counter() - t_tts) * 1000.0,
            extra={"event": "compare_ab"},
        )

        t_al = time.perf_counter()
        dialogue_timing, word_index = await podcast_timing_service.build_alignment_bundle(tts_script, voice_path)
        log_pipeline_event(
            "compare_alignment",
            task_id=compare_id,
            duration_ms=(time.perf_counter() - t_al) * 1000.0,
            extra={"event": "compare_ab"},
        )

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
        voice_line_timing_ms = [
            (int(h["line_index"]), float(h["start_ms"]), float(h["end_ms"])) for h in timing_hints
        ]

        if not config.USE_PRODUCTION_DIRECTOR:
            set_status(
                compare_id,
                status="failed",
                error="USE_PRODUCTION_DIRECTOR must be enabled for compare",
            )
            return

        library = AssetLibrary()
        await asyncio.to_thread(ensure_backchannel_assets, voices, library)

        tmpl_a = resolve_genre_template(template_id=genre_a, style=None, metadata_genre=None)
        tmpl_b = resolve_genre_template(template_id=genre_b, style=None, metadata_genre=None)

        out_dir = config.OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        async def _one_branch(suffix: str, gt: Any, label: str) -> tuple[str, Dict[str, Any]]:
            director = ProductionDirector(base_url=ollama_url, model=ollama_model)
            gq = GenerationQueue(library, genre_template=gt)
            t_plan = time.perf_counter()
            plan = await director.plan(
                script,
                script_segments,
                label,
                [],
                timing_hints,
                asset_library=library,
                generation_queue=gq,
                use_tools=True,
                word_index=word_index,
                genre_template=gt,
            )
            log_pipeline_event(
                f"compare_director_{suffix}",
                task_id=compare_id,
                duration_ms=(time.perf_counter() - t_plan) * 1000.0,
                extra={"event": "compare_ab", "genre": gt.genre_id},
            )
            plan = await gq.run_queued_and_patch_plan(plan)
            warnings.extend(gq.warnings)
            try:
                from app.services.backchannel_resolve import patch_production_plan_voice_backchannels

                plan = patch_production_plan_voice_backchannels(plan, library, voices)
            except Exception as exc:
                warnings.append(f"Voice backchannel patch failed: {exc}")
            t_mix = time.perf_counter()
            mp3 = await asyncio.to_thread(
                audio_compositor.mix_production_plan,
                str(voice_path),
                plan,
                library,
                word_index=word_index,
                timing_hints=timing_hints,
                voice_line_timing_ms=voice_line_timing_ms,
                genre_template=gt,
            )
            log_pipeline_event(
                f"compare_mix_{suffix}",
                task_id=compare_id,
                duration_ms=(time.perf_counter() - t_mix) * 1000.0,
                extra={"event": "compare_ab"},
            )
            dialogue_regions = [(int(h["start_ms"]), int(h["end_ms"])) for h in timing_hints]
            target = mastering_lufs(gt, plan.genre)
            qa = run_mix_qa(
                Path(mp3),
                target_lufs=target,
                plan_duration_seconds=float(plan.duration_target_seconds),
                dialogue_regions_ms=dialogue_regions,
                plan=plan,
            )
            return str(mp3), qa

        path_a, qa_a = await _one_branch("a", tmpl_a, tmpl_a.display_name)
        try:
            from vibevoice.core.transcripts.transcriber import transcript_transcriber
            from vibevoice.gpu_memory import cuda_device_index_from_string, release_torch_cuda_memory, wait_for_cuda_memory
            from vibevoice.services.voice_generator import voice_generator

            transcript_transcriber.unload_models()
            voice_generator.release_gpu_memory_after_speech()
            release_torch_cuda_memory()
            idx = cuda_device_index_from_string(config.ACESTEP_DEVICE)
            if idx is not None:
                mib = config.ACESTEP_MIN_FREE_VRAM_MIB
                wait_for_cuda_memory(
                    mib * 1024 * 1024,
                    device_index=idx,
                    timeout_seconds=config.GPU_VRAM_WAIT_TIMEOUT_SECONDS,
                    poll_interval_seconds=config.GPU_VRAM_POLL_INTERVAL_SECONDS,
                )
        except Exception:
            pass
        path_b, qa_b = await _one_branch("b", tmpl_b, tmpl_b.display_name)

        dest_a = out_dir / f"compare_{compare_id}_a.mp3"
        dest_b = out_dir / f"compare_{compare_id}_b.mp3"

        shutil.copy2(path_a, dest_a)
        shutil.copy2(path_b, dest_b)

        set_status(
            compare_id,
            status="succeeded",
            message="Compare renders ready",
            file_path_a=str(dest_a.resolve()),
            file_path_b=str(dest_b.resolve()),
            audio_url_a=f"/api/v1/podcast/download/{dest_a.name}",
            audio_url_b=f"/api/v1/podcast/download/{dest_b.name}",
            qa_a=qa_a,
            qa_b=qa_b,
            genre_ids=(genre_a, genre_b),
            warnings=warnings,
        )
        log_pipeline_event(
            "compare_complete",
            task_id=compare_id,
            extra={"event": "compare_ab", "genres": [genre_a, genre_b]},
        )
    except Exception as exc:
        logger.exception("compare_ab failed: %s", exc)
        set_status(compare_id, status="failed", error=str(exc), warnings=warnings)
