"""
Global idle memory purge for long-running API workers.

Unloads in-process ML models and releases CUDA caches after a period with no
meaningful HTTP traffic (see IdleActivityMiddleware). Skips while transcript,
ad-scan, isolation, or TTS work is in progress.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import sys
import threading
import time
from .config import config

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_last_activity: float = time.monotonic()
_purge_lock = threading.Lock()


def touch_activity() -> None:
    """Mark recent API activity (called from middleware)."""
    global _last_activity
    with _state_lock:
        _last_activity = time.monotonic()


def _maybe_trim_heap() -> None:
    if sys.platform != "linux":
        return
    if not getattr(config, "IDLE_MEMORY_TRIM_HEAP", True):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        logger.debug("malloc_trim skipped or failed", exc_info=True)


def _work_in_progress() -> bool:
    from .services.ad_scan_service import ad_scan_service
    from .services.speaker_isolation_service import speaker_isolation_service
    from .services.transcript_service import transcript_service
    from .services.voice_generator import voice_generator

    if transcript_service.has_active_jobs():
        return True
    if ad_scan_service.has_running_tasks():
        return True
    if speaker_isolation_service.has_running_tasks():
        return True
    if voice_generator.tts_has_inflight_generation():
        return True
    return False


def purge_idle_memory() -> None:
    """
    Unload heavyweight in-process models and release GPU / allocator memory.
    Safe to call when idle; no-ops if work is in progress.
    """
    if getattr(config, "IDLE_MEMORY_PURGE_SECONDS", 0) <= 0:
        return
    if not _purge_lock.acquire(blocking=False):
        return
    try:
        if _work_in_progress():
            return

        from .core.transcripts.diarizer import transcript_diarizer
        from .core.transcripts.transcriber import transcript_transcriber
        from .gpu_memory import release_torch_cuda_memory
        from .services import ad_scan_transcriber
        from .services.voice_generator import voice_generator

        logger.info("Idle memory purge: unloading models and releasing CUDA cache")

        transcript_transcriber.unload_models()
        transcript_diarizer.unload_pipeline()
        voice_generator.release_gpu_memory_after_speech()
        ad_scan_transcriber.unload_model()

        gc.collect()
        gc.collect()
        release_torch_cuda_memory()
        _maybe_trim_heap()

        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                logger.info(
                    "Idle memory purge done: CUDA allocated %.0f MiB, reserved %.0f MiB",
                    alloc,
                    reserved,
                )
        except Exception:
            logger.debug("CUDA memory stats after purge", exc_info=True)
    finally:
        _purge_lock.release()
        global _last_activity
        with _state_lock:
            _last_activity = time.monotonic()


async def idle_memory_watchdog() -> None:
    """Background loop: purge after IDLE_MEMORY_PURGE_SECONDS of no touched activity."""
    while True:
        poll = float(getattr(config, "IDLE_MEMORY_POLL_INTERVAL_SECONDS", 15.0))
        await asyncio.sleep(max(5.0, poll))
        threshold = int(getattr(config, "IDLE_MEMORY_PURGE_SECONDS", 0))
        if threshold <= 0:
            continue
        with _state_lock:
            idle_for = time.monotonic() - _last_activity
        if idle_for < threshold:
            continue
        if _work_in_progress():
            continue
        await asyncio.to_thread(purge_idle_memory)


def seconds_since_activity() -> float:
    with _state_lock:
        return time.monotonic() - _last_activity
