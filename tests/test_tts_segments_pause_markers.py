#!/usr/bin/env python3
"""Tests for inline [PAUSE_MS:N] markers in transcript parsing and script post-processing."""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from vibevoice.services.ollama_client import _inject_speaker_handoff_pauses, ollama_client
from vibevoice.services.podcast_pause_cues import PAUSE_MS_MAX, PAUSE_MS_MIN, redistribute_uniform_pause_markers
from vibevoice.services.tts.segments import parse_transcript_into_segments, strip_inline_pause_markers
from vibevoice.services.voice_generator import VoiceGenerator


class TestStripInlinePauseMarkers(unittest.TestCase):
    def test_strips_and_sums(self) -> None:
        t, ms = strip_inline_pause_markers("Hello [PAUSE_MS:100] there [PAUSE_MS:50]")
        self.assertEqual(t, "Hello there")
        self.assertEqual(ms, 150)

    def test_empty(self) -> None:
        t, ms = strip_inline_pause_markers("")
        self.assertEqual(ms, 0)


class TestParseTranscriptIntoSegments(unittest.TestCase):
    def test_two_speakers_with_pauses(self) -> None:
        transcript = (
            "Speaker 1: First line [PAUSE_MS:120]\n"
            "Speaker 2: Second line [PAUSE_MS:200]"
        )
        segs = parse_transcript_into_segments(transcript, 2)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "First line")
        self.assertEqual(segs[0].pause_after_ms, 120)
        self.assertEqual(segs[1].text, "Second line")
        self.assertEqual(segs[1].pause_after_ms, 200)

    def test_no_markers_regression(self) -> None:
        transcript = "Speaker 1: Only me.\nSpeaker 2: And me."
        segs = parse_transcript_into_segments(transcript, 2)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].pause_after_ms, 0)
        self.assertEqual(segs[1].pause_after_ms, 0)


def _pause_values(script: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[PAUSE_MS:\s*(\d+)\s*\]", script, flags=re.I)]


class TestInjectSpeakerHandoffPauses(unittest.TestCase):
    def test_adds_contextual_when_missing(self) -> None:
        script = "Speaker 1: Hello.\nSpeaker 2: Hi."
        out = _inject_speaker_handoff_pauses(script, include_production_cues=False)
        vals = _pause_values(out)
        self.assertEqual(len(vals), 1)
        self.assertGreaterEqual(vals[0], PAUSE_MS_MIN)
        self.assertLessEqual(vals[0], PAUSE_MS_MAX)
        self.assertIn("Hello.", out)

    def test_skips_when_present(self) -> None:
        script = "Speaker 1: Hello. [PAUSE_MS:300]\nSpeaker 2: Hi."
        out = _inject_speaker_handoff_pauses(script, include_production_cues=False)
        self.assertEqual(out.count("[PAUSE_MS:300]"), 1)


class TestRedistributeUniformPauses(unittest.TestCase):
    def test_replaces_flat_copy_paste(self) -> None:
        script = "\n".join(
            [
                "Speaker 1: Line one. [PAUSE_MS:220]",
                "Speaker 2: Line two. [PAUSE_MS:220]",
                "Speaker 1: Line three? [PAUSE_MS:220]",
                "Speaker 2: Line four. [PAUSE_MS:220]",
            ]
        )
        out = redistribute_uniform_pause_markers(script)
        vals = _pause_values(out)
        self.assertEqual(len(vals), 4)
        self.assertGreater(len(set(vals)), 1, "uniform 220 should diversify")
        for v in vals:
            self.assertGreaterEqual(v, PAUSE_MS_MIN)
            self.assertLessEqual(v, PAUSE_MS_MAX)

    def test_leaves_varied_scripts_alone(self) -> None:
        script = "Speaker 1: A [PAUSE_MS:140]\nSpeaker 2: B [PAUSE_MS:380]"
        self.assertEqual(redistribute_uniform_pause_markers(script), script)


class TestCleanScriptInjection(unittest.TestCase):
    def test_clean_script_injects_handoff(self) -> None:
        raw = "Speaker 1: Hello.\nSpeaker 2: There."
        cleaned = ollama_client._clean_script(raw, num_voices=2, include_production_cues=False)
        vals = _pause_values(cleaned)
        self.assertTrue(vals)
        self.assertTrue(all(PAUSE_MS_MIN <= v <= PAUSE_MS_MAX for v in vals))


class TestVoiceDirectionMergePause(unittest.TestCase):
    def test_max_merges_with_parsed_pause(self) -> None:
        from vibevoice.services.tts.segments import TranscriptSegment

        vg = VoiceGenerator.__new__(VoiceGenerator)  # type: ignore[misc]
        segs = [
            TranscriptSegment(speaker_index=0, text="a", pause_after_ms=200),
        ]
        vg._apply_voice_direction_to_segments(
            segs,
            [{"line_index": 0, "speaker": "Speaker 1", "emotion": "neutral", "pause_after_ms": 80}],
        )
        self.assertEqual(segs[0].pause_after_ms, 200)

        segs2 = [
            TranscriptSegment(speaker_index=0, text="a", pause_after_ms=80),
        ]
        vg._apply_voice_direction_to_segments(
            segs2,
            [{"line_index": 0, "speaker": "Speaker 1", "emotion": "neutral", "pause_after_ms": 200}],
        )
        self.assertEqual(segs2[0].pause_after_ms, 200)


if __name__ == "__main__":
    unittest.main()
