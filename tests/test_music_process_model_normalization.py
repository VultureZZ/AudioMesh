#!/usr/bin/env python3
"""Tests for ACE-Step DiT model normalization before API startup."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


class TestMusicProcessModelNormalization(unittest.TestCase):
    def test_hf_style_model_id_is_normalized(self) -> None:
        from vibevoice.services.music_process import _normalize_dit_model_for_api_server

        self.assertEqual(
            _normalize_dit_model_for_api_server("ACE-Step/acestep-v15-xl-sft"),
            "acestep-v15-xl-sft",
        )

    def test_bare_model_id_is_preserved(self) -> None:
        from vibevoice.services.music_process import _normalize_dit_model_for_api_server

        self.assertEqual(
            _normalize_dit_model_for_api_server("acestep-v15-turbo"),
            "acestep-v15-turbo",
        )

    def test_unknown_model_falls_back_to_turbo(self) -> None:
        from vibevoice.services.music_process import _normalize_dit_model_for_api_server

        self.assertEqual(
            _normalize_dit_model_for_api_server("not-a-real-model"),
            "acestep-v15-turbo",
        )


if __name__ == "__main__":
    unittest.main()
