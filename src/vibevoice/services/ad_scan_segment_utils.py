"""
Classify ad-scan segments: the LLM sometimes mislabels main episode content as ads.
Keep logic in sync with frontend/src/utils/adScanSegments.ts (editorial substrings only;
dominant-label filtering is server-side).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..config import config

logger = logging.getLogger(__name__)

# Substrings in `label` that indicate main episode / editorial content, not a sponsor spot.
_EDITORIAL_LABEL_SUBSTRINGS: tuple[str, ...] = (
    "news segment",
    "main content",
    "editorial",
    "episode content",
    "story segment",
    "discussion segment",
    "interview segment",
    "host segment",
    "cold open",
)

_EDITORIAL_LABELS_EXACT: frozenset[str] = frozenset({"news", "editorial"})


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_iv[0]]
    for a, b in sorted_iv[1:]:
        la, lb = merged[-1]
        if a <= lb + 0.001:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _merged_span_seconds(intervals: list[tuple[float, float]]) -> float:
    merged = _merge_intervals(intervals)
    return sum(max(0.0, b - a) for a, b in merged)


def filter_dominant_show_segments(
    segments: list[dict[str, Any]],
    total_duration_seconds: float,
    *,
    job_id: str | None = None,
    min_fraction: float | None = None,
) -> list[dict[str, Any]]:
    """
    If one label's merged coverage is >= min_fraction of the episode duration, treat that label as the
    show/network main content (not discrete ads) and remove all segments with that label.
    """
    if not segments or total_duration_seconds <= 0:
        return segments
    td = float(total_duration_seconds)
    frac = min_fraction if min_fraction is not None else float(
        getattr(config, "AD_SCAN_DOMINANT_LABEL_MIN_FRACTION", 0.45)
    )
    by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in segments:
        if not isinstance(s, dict):
            continue
        lab = (s.get("label") or "").strip().lower()
        if not lab:
            continue
        try:
            a = float(s["start_seconds"])
            b = float(s["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        a = max(0.0, min(a, td))
        b = max(0.0, min(b, td))
        if b <= a:
            continue
        by_label[lab].append((a, b))

    best_label: str | None = None
    best_span = 0.0
    for lab, ivs in by_label.items():
        span = _merged_span_seconds(ivs)
        if span > best_span:
            best_span = span
            best_label = lab

    if best_label is None or best_span <= 0:
        return segments

    ratio = best_span / td
    if ratio < frac:
        return segments

    jid = job_id or "-"
    logger.info(
        "[ad-scan] job=%s dominant_show_filter label=%r merged_span_s=%.1f episode_s=%.1f ratio=%.2f min=%.2f",
        jid,
        best_label,
        best_span,
        td,
        ratio,
        frac,
    )
    return [
        s
        for s in segments
        if isinstance(s, dict) and (s.get("label") or "").strip().lower() != best_label
    ]


def is_commercial_ad_segment(segment: dict[str, Any]) -> bool:
    """Return True if this row should be cut as a sponsor/ad; False for editorial mislabels."""
    label = (segment.get("label") or "").strip().lower()
    if not label:
        return True
    if label in _EDITORIAL_LABELS_EXACT:
        return False
    for snippet in _EDITORIAL_LABEL_SUBSTRINGS:
        if snippet in label:
            return False
    return True


def commercial_ad_segments_only(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in segments if isinstance(s, dict) and is_commercial_ad_segment(s)]


def _combine_ad_labels(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b or "ad"
    if not b or a == b:
        return a
    return f"{a} · {b}"


def merge_adjacent_ad_segments(
    segments: list[dict[str, Any]],
    total_duration_seconds: float,
    *,
    max_gap_seconds: float | None = None,
    pad_start_seconds: float | None = None,
    pad_end_seconds: float | None = None,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Cluster detected ads: bridge gaps (music/bumpers) between spots, pad start toward 0:00 when the
    first spot begins soon after the file start, and pad the last spot through short trailing silence.
    """
    if not segments or total_duration_seconds <= 0:
        return segments
    td = float(total_duration_seconds)
    gap_max = (
        max_gap_seconds
        if max_gap_seconds is not None
        else float(getattr(config, "AD_SCAN_AD_CLUSTER_GAP_SECONDS", 5.0))
    )
    pad_s = (
        pad_start_seconds
        if pad_start_seconds is not None
        else float(getattr(config, "AD_SCAN_AD_CLUSTER_PAD_START_SECONDS", 5.0))
    )
    pad_e = (
        pad_end_seconds
        if pad_end_seconds is not None
        else float(getattr(config, "AD_SCAN_AD_CLUSTER_PAD_END_SECONDS", 5.0))
    )

    rows: list[dict[str, Any]] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        try:
            a = float(s["start_seconds"])
            b = float(s["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        a = max(0.0, min(a, td))
        b = max(0.0, min(b, td))
        if b <= a:
            continue
        lab = str(s.get("label", "ad")).strip() or "ad"
        try:
            conf = float(s.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        rows.append({"start_seconds": a, "end_seconds": b, "label": lab, "confidence": conf})

    rows.sort(key=lambda x: x["start_seconds"])
    merged: list[dict[str, Any]] = []
    for r in rows:
        if not merged:
            merged.append(dict(r))
            continue
        prev = merged[-1]
        gap_between = r["start_seconds"] - prev["end_seconds"]
        if gap_between <= gap_max:
            prev["end_seconds"] = max(prev["end_seconds"], r["end_seconds"])
            prev["label"] = _combine_ad_labels(prev["label"], r["label"])
            prev["confidence"] = max(prev["confidence"], r["confidence"])
        else:
            merged.append(dict(r))

    if merged and merged[0]["start_seconds"] <= pad_s:
        merged[0]["start_seconds"] = 0.0

    if merged:
        tail = td - merged[-1]["end_seconds"]
        if 0 < tail <= pad_e:
            merged[-1]["end_seconds"] = td

    jid = job_id or "-"
    if len(merged) != len(rows):
        logger.info(
            "[ad-scan] job=%s ad_cluster_merge raw_rows=%d merged=%d gap_max=%.2fs pad_start=%.2fs pad_end=%.2fs",
            jid,
            len(rows),
            len(merged),
            gap_max,
            pad_s,
            pad_e,
        )
    elif merged and merged[0]["start_seconds"] == 0.0 and rows and rows[0]["start_seconds"] > 0:
        logger.info(
            "[ad-scan] job=%s ad_cluster_merge padded_first_start_to_zero (was %.3fs)",
            jid,
            rows[0]["start_seconds"],
        )
    elif merged and rows and merged[-1]["end_seconds"] > rows[-1]["end_seconds"]:
        logger.info(
            "[ad-scan] job=%s ad_cluster_merge padded_last_end_to_eof",
            jid,
        )

    return merged
