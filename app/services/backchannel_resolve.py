"""
Resolve voice_backchannel ProductionPlan events to pre-rendered library assets.

The mixer only overlays events with ``asset_ref.asset_id``; ``generation_prompt``-only
events are skipped. The generation queue also skips ``voice_backchannel`` tracks, so
we map trigger_word + anchor_speaker to ``bc_{voice}_{phrase}`` ids after planning.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _reactor_voice_name(voice_names: Sequence[str], anchor_speaker: Optional[str]) -> str:
    """Prefer a co-host reacting: pick a speaker that is not the anchor line."""
    if not voice_names:
        return ""
    m = re.match(r"Speaker\s+(\d+)", (anchor_speaker or "").strip(), re.I)
    if not m:
        return voice_names[min(1, len(voice_names) - 1)] if len(voice_names) > 1 else voice_names[0]
    idx = int(m.group(1)) - 1
    for i, name in enumerate(voice_names):
        if i != idx:
            return name
    return voice_names[0]


def resolve_voice_backchannel_asset_id(
    library: Any,
    voice_names: Sequence[str],
    *,
    trigger_word: Optional[str],
    anchor_speaker: Optional[str],
) -> Optional[str]:
    from app.services.backchannel_synth import PHRASES, _phrase_slug, _slug

    tw_raw = (trigger_word or "").strip().lower()
    tw = re.sub(r"[^a-z0-9\s\-]+", " ", tw_raw).strip()
    matched: List[str] = []
    seen_p: set[str] = set()
    for p in PHRASES:
        pl = p.lower()
        compact_tw = tw.replace(" ", "").replace("-", "")
        compact_pl = pl.replace(" ", "").replace("-", "")
        if (
            pl in tw
            or (tw and tw in pl)
            or tw.rstrip(".,!?") == pl
            or (compact_pl and compact_pl in compact_tw)
        ):
            if p not in seen_p:
                seen_p.add(p)
                matched.append(p)
    if not matched:
        for tok in re.split(r"\s+", tw):
            tok = tok.strip(".,!?")
            if not tok:
                continue
            for p in PHRASES:
                if _phrase_slug(p) == _phrase_slug(tok):
                    if p not in seen_p:
                        seen_p.add(p)
                        matched.append(p)
    if not matched:
        matched = list(PHRASES)

    reactor = _reactor_voice_name(voice_names, anchor_speaker)
    tried: List[str] = []

    def _try_voice(name: str) -> Optional[str]:
        if not name:
            return None
        for phrase in matched:
            aid = f"bc_{_slug(name)}_{_phrase_slug(phrase)}"
            tried.append(aid)
            try:
                library.get(aid)
                return aid
            except KeyError:
                continue
        return None

    if reactor:
        hit = _try_voice(reactor)
        if hit:
            return hit
    for name in voice_names:
        hit = _try_voice(str(name))
        if hit:
            return hit
    logger.warning(
        "No voice_backchannel asset matched (reactor=%r, trigger=%r); sample tried=%s",
        reactor,
        trigger_word,
        tried[:10],
    )
    return None


def patch_production_plan_voice_backchannels(
    plan: Any,
    library: Any,
    voice_names: Sequence[str],
) -> Any:
    from app.services.production_director import ProductionPlan

    if not voice_names:
        return plan
    raw = plan.model_dump(mode="json")
    changed = 0
    for t in raw.get("tracks") or []:
        if str(t.get("track_role")) != "voice_backchannel":
            continue
        for e in t.get("events") or []:
            ar = e.get("asset_ref")
            if not isinstance(ar, dict):
                continue
            aid = ar.get("asset_id")
            gp = ar.get("generation_prompt")
            valid = False
            if aid:
                try:
                    library.get(str(aid))
                    valid = True
                except KeyError:
                    valid = False
            if valid and not gp:
                continue
            resolved = resolve_voice_backchannel_asset_id(
                library,
                voice_names,
                trigger_word=e.get("trigger_word"),
                anchor_speaker=e.get("anchor_speaker"),
            )
            if not resolved:
                if gp or not valid:
                    logger.warning(
                        "voice_backchannel event %s could not be resolved to a library asset",
                        e.get("event_id"),
                    )
                continue
            e["asset_ref"] = {"asset_id": resolved}
            try:
                e["duration_ms"] = int(library.get(resolved).duration_ms)
            except Exception:
                pass
            changed += 1
    if changed:
        logger.info("Patched %s voice_backchannel event(s) with library asset_id", changed)
    return ProductionPlan.model_validate(raw)
