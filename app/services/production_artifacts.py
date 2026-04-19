"""
Copy production mix inputs (cue WAVs/MP3s) next to the rendered episode for review.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def copy_production_cue_review_files(
    review_dir: Path,
    plan: Any,
    library: Any,
    *,
    asset_overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Copy each distinct asset referenced by the plan into ``review_dir``.

    Filenames are prefixed with track role and event id for traceability.
    """
    review_dir.mkdir(parents=True, exist_ok=True)
    overrides = asset_overrides or {}
    copied: List[str] = []
    seen: set[str] = set()
    for tr in getattr(plan, "tracks", []) or []:
        role = (
            str(getattr(tr, "track_role", "") or "track")
            .replace("/", "_")
            .replace("\\", "_")
        )
        for ev in getattr(tr, "events", []) or []:
            ref = getattr(ev, "asset_ref", None)
            aid = getattr(ref, "asset_id", None) if ref else None
            if not aid:
                continue
            aid_s = str(aid)
            if aid_s in seen:
                continue
            seen.add(aid_s)
            try:
                src = Path(overrides[aid_s]) if aid_s in overrides else library.resolve_path(aid_s)
            except Exception as exc:
                logger.debug("Could not resolve path for review copy %s: %s", aid_s, exc)
                continue
            if not src.is_file():
                continue
            ev_id = str(getattr(ev, "event_id", "ev"))
            suffix = src.suffix.lower() or ".wav"
            dest = review_dir / f"{role}_{ev_id}_{aid_s}{suffix}"
            try:
                shutil.copy2(src, dest)
                copied.append(str(dest))
            except OSError as exc:
                logger.warning("Review copy failed %s -> %s: %s", src, dest, exc)
    return copied


def copy_legacy_cue_paths_review(review_dir: Path, cue_paths: Dict[str, str]) -> List[str]:
    """Copy ACE-Step (or other) cue files from the legacy non-director path."""
    review_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for key, path in sorted(cue_paths.items()):
        src = Path(path)
        if not src.is_file():
            continue
        suffix = src.suffix.lower() or ".wav"
        dest = review_dir / f"legacy_{key}{suffix}"
        try:
            shutil.copy2(src, dest)
            copied.append(str(dest))
        except OSError as exc:
            logger.warning("Legacy cue review copy failed %s: %s", src, exc)
    return copied
