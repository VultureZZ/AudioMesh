"""
Routes generation requests to ACE-Step (music) or Stable Audio Open (SFX), or skip.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

BackendName = Literal["acestep", "stable_audio", "skip"]

_MUSIC_CATEGORIES = frozenset(
    {
        "music_bed",
        "music_transition",
        "music_intro",
        "music_outro",
    }
)
_SFX_CATEGORIES = frozenset(
    {
        "sfx_impact",
        "sfx_riser",
        "sfx_whoosh",
        "sfx_ambience",
        "sfx_laugh",
        "sfx_reveal",
    }
)


class PromptRouter:
    """Decides which generator handles a library ``category`` string."""

    def __init__(self) -> None:
        self._stable_enabled = False
        self._sfx_fallback_to_acestep = True
        try:
            from vibevoice.config import config  # type: ignore

            self._stable_enabled = bool(config.STABLE_AUDIO_OPEN_ENABLED)
            self._sfx_fallback_to_acestep = bool(getattr(config, "SFX_FALLBACK_TO_ACESTEP", True))
        except Exception:
            pass

    def route(self, category: str) -> BackendName:
        c = (category or "").strip()
        if c in _MUSIC_CATEGORIES:
            return "acestep"
        if c in _SFX_CATEGORIES:
            if self._stable_enabled:
                return "stable_audio"
            if self._sfx_fallback_to_acestep:
                logger.warning(
                    "Stable Audio Open disabled (STABLE_AUDIO_OPEN_ENABLED=false); "
                    "routing category=%s to ACE-Step fallback",
                    c,
                )
                return "acestep"
            logger.warning(
                "Stable Audio Open disabled (STABLE_AUDIO_OPEN_ENABLED=false); skipping category=%s",
                c,
            )
            return "skip"
        if c == "foley":
            if self._stable_enabled:
                return "stable_audio"
            if self._sfx_fallback_to_acestep:
                logger.warning("Stable Audio Open disabled; routing foley generation to ACE-Step fallback")
                return "acestep"
            logger.warning("Stable Audio Open disabled; skipping foley generation")
            return "skip"
        logger.warning("Unknown category for PromptRouter: %s", c)
        return "skip"

    def apply_genre_prompt_modifiers(
        self,
        prompt: str,
        category: str,
        genre_template: Optional[Any] = None,
    ) -> str:
        """Wrap prompts sent to ACE-Step / Stable Audio with :class:`GenreTemplate` modifiers."""
        from app.services.genre_templates import apply_generation_prompt_modifiers

        return apply_generation_prompt_modifiers(prompt, category, genre_template)
