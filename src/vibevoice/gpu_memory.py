"""
Best-effort PyTorch CUDA cache release for long-running API workers.
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_torch_cuda_memory() -> None:
    """Synchronize, collect Python objects, and empty CUDA allocator caches."""
    try:
        import torch
    except ImportError:
        gc.collect()
        return
    if not torch.cuda.is_available():
        gc.collect()
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        logger.debug("cuda.synchronize during VRAM release", exc_info=True)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
