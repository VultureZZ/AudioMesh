#!/usr/bin/env python3
"""
Unit tests for simple music prompt mode payload building.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src to path for local test execution.
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from vibevoice.services.music_generator import music_generator


class TestMusicGeneratorModes(unittest.TestCase):
    def test_exact_mode_requires_caption_or_lyrics(self):
        with self.assertRaises(ValueError):
            music_generator.prepare_simple_payload(
                description="",
                input_mode="exact",
                instrumental=False,
                vocal_language="en",
                duration=45,
                batch_size=1,
                exact_caption="",
                exact_lyrics="",
            )

    def test_exact_mode_preserves_exact_fields(self):
        payload = music_generator.prepare_simple_payload(
            description="",
            input_mode="exact",
            instrumental=False,
            vocal_language="en",
            duration=45,
            batch_size=2,
            exact_caption="Boom bap rap with river imagery",
            exact_lyrics="[Verse 1]\nRowing with my homies",
            exact_bpm=92,
            exact_keyscale="C minor",
            exact_timesignature="4",
        )

        self.assertEqual(payload["prompt"], "Boom bap rap with river imagery")
        self.assertEqual(payload["lyrics"], "[Verse 1]\nRowing with my homies")
        self.assertEqual(payload["bpm"], 92)
        self.assertEqual(payload["keyscale"], "C minor")
        self.assertEqual(payload["timesignature"], "4")
        self.assertEqual(payload["duration"], 45)
        self.assertEqual(payload["vocal_language"], "en")
        self.assertEqual(payload["batch_size"], 2)
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["use_cot_caption"])
        self.assertFalse(payload["use_cot_language"])
        self.assertFalse(payload["use_cot_metas"])

    def test_refine_mode_applies_constraints_over_refined_values(self):
        refined = {
            "caption": "90s rap on a sunny river day",
            "lyrics": "[Verse 1]\nOn the river",
            "bpm": 88,
            "keyscale": "D minor",
            "timesignature": "4",
        }
        with patch.object(music_generator, "_refine_prompt_with_ollama", return_value=refined):
            payload = music_generator.prepare_simple_payload(
                description="a 90s rap about rowing with friends",
                input_mode="refine",
                instrumental=False,
                vocal_language="en",
                duration=45,
                batch_size=1,
            )

        self.assertEqual(payload["prompt"], "90s rap on a sunny river day")
        self.assertEqual(payload["lyrics"], "[Verse 1]\nOn the river")
        self.assertEqual(payload["duration"], 45)
        self.assertEqual(payload["vocal_language"], "en")
        self.assertEqual(payload["bpm"], 88)
        self.assertEqual(payload["keyscale"], "D minor")
        self.assertEqual(payload["timesignature"], "4")

    def test_refine_mode_raises_without_description(self):
        with self.assertRaises(ValueError):
            music_generator.prepare_simple_payload(
                description="",
                input_mode="refine",
                instrumental=False,
                vocal_language="en",
            )

    def test_refine_json_parser_extracts_embedded_object(self):
        parsed = music_generator._parse_ollama_json(
            "Preface text {\"caption\":\"x\",\"lyrics\":\"y\"} trailing"
        )
        self.assertEqual(parsed["caption"], "x")
        self.assertEqual(parsed["lyrics"], "y")


if __name__ == "__main__":
    unittest.main()
