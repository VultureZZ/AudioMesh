#!/usr/bin/env python3
"""ACE-Step duration clamping for GenerationQueue."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


class TestEffectiveAcestepDuration(unittest.TestCase):
    def test_music_bed_raises_small_requests(self):
        from app.services.generation_queue import _effective_acestep_duration_seconds

        mock_cfg = MagicMock()
        mock_cfg.ACESTEP_MIN_MUSIC_DURATION_SECONDS = 30.0
        mock_cfg.ACESTEP_MAX_MUSIC_DURATION_SECONDS = 600.0
        mock_cfg.ACESTEP_MIN_TRANSITION_DURATION_SECONDS = 10.0
        with patch("vibevoice.config.config", mock_cfg):
            self.assertAlmostEqual(_effective_acestep_duration_seconds(5.0, "music_bed"), 30.0)
            self.assertAlmostEqual(_effective_acestep_duration_seconds(45.0, "music_intro"), 45.0)

    def test_transition_uses_lower_floor(self):
        from app.services.generation_queue import _effective_acestep_duration_seconds

        mock_cfg = MagicMock()
        mock_cfg.ACESTEP_MIN_MUSIC_DURATION_SECONDS = 30.0
        mock_cfg.ACESTEP_MAX_MUSIC_DURATION_SECONDS = 600.0
        mock_cfg.ACESTEP_MIN_TRANSITION_DURATION_SECONDS = 10.0
        with patch("vibevoice.config.config", mock_cfg):
            self.assertAlmostEqual(_effective_acestep_duration_seconds(3.0, "music_transition"), 10.0)


if __name__ == "__main__":
    unittest.main()
