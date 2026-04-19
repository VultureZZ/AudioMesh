#!/usr/bin/env python3
"""Tests for podcast Speaker 1..N label normalization."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from vibevoice.services.ollama_client import normalize_podcast_speaker_labels


class TestNormalizePodcastSpeakerLabels(unittest.TestCase):
    def test_maps_names_to_speakers_in_order(self):
        raw = """Marcus Parks: First line here.
Jad Abumrad: Second voice.
Henry Zebrowski: Third one."""
        out = normalize_podcast_speaker_labels(raw, 3)
        lines = [ln for ln in out.split("\n") if ln.strip()]
        self.assertTrue(lines[0].startswith("Speaker 1:"))
        self.assertTrue(lines[1].startswith("Speaker 2:"))
        self.assertTrue(lines[2].startswith("Speaker 3:"))
        self.assertIn("First line", lines[0])

    def test_preserves_canonical_speaker_lines(self):
        raw = "Speaker 2: Already labeled.\nSpeaker 1: Reply."
        out = normalize_podcast_speaker_labels(raw, 2)
        self.assertIn("Speaker 2: Already labeled.", out)
        self.assertIn("Speaker 1: Reply.", out)

    def test_clamps_speaker_index_to_num_voices(self):
        raw = "Speaker 9: Too high."
        out = normalize_podcast_speaker_labels(raw, 2)
        self.assertIn("Speaker 2: Too high.", out)

    def test_skips_cue_lines_when_flag_true(self):
        raw = "[CUE: INTRO_MUSIC]\nSomeone: Hello."
        out = normalize_podcast_speaker_labels(raw, 2, include_production_cues=True)
        self.assertIn("[CUE: INTRO_MUSIC]", out)
        self.assertIn("Speaker 1:", out)


if __name__ == "__main__":
    unittest.main()
