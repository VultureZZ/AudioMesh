#!/usr/bin/env python3
"""PromptRouter backend selection tests."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


class TestPromptRouter(unittest.TestCase):
    def test_sfx_routes_to_acestep_fallback_when_stable_disabled(self) -> None:
        from app.services.prompt_router import PromptRouter

        mock_cfg = MagicMock()
        mock_cfg.STABLE_AUDIO_OPEN_ENABLED = False
        mock_cfg.SFX_FALLBACK_TO_ACESTEP = True
        with patch("vibevoice.config.config", mock_cfg):
            router = PromptRouter()
        self.assertEqual(router.route("sfx_whoosh"), "acestep")
        self.assertEqual(router.route("foley"), "acestep")

    def test_sfx_can_still_skip_when_fallback_disabled(self) -> None:
        from app.services.prompt_router import PromptRouter

        mock_cfg = MagicMock()
        mock_cfg.STABLE_AUDIO_OPEN_ENABLED = False
        mock_cfg.SFX_FALLBACK_TO_ACESTEP = False
        with patch("vibevoice.config.config", mock_cfg):
            router = PromptRouter()
        self.assertEqual(router.route("sfx_whoosh"), "skip")


if __name__ == "__main__":
    unittest.main()
