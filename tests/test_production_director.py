#!/usr/bin/env python3
"""Unit tests for app.services.production_director."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Project root (for `app` package) and src (for `vibevoice.config`).
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from pydantic import ValidationError

from app.services.production_director import (
    ProductionDirector,
    ProductionPlan,
    TrackEvent,
    build_fallback_production_plan,
    extract_json_object_from_llm_text,
)


def _minimal_valid_plan_dict() -> dict:
    return {
        "episode_id": "e-unit-test",
        "duration_target_seconds": 120.0,
        "genre": "News",
        "emotional_arc": [
            {"timestamp": 0.0, "valence": 0.0, "energy": 0.5},
            {"timestamp": 55.0, "valence": 0.2, "energy": 0.55},
            {"timestamp": 120.0, "valence": 0.1, "energy": 0.45},
        ],
        "tracks": [
            {
                "track_id": "tr_v",
                "track_role": "voice_main",
                "events": [
                    {
                        "event_id": "ev1",
                        "start_ms": 0,
                        "duration_ms": 5000,
                        "asset_ref": None,
                        "volume_db": 0.0,
                        "pan": 0.0,
                        "fade_in_ms": 0,
                        "fade_out_ms": 0,
                        "automation": None,
                        "trigger_word": None,
                    }
                ],
            }
        ],
        "voice_direction": [
            {
                "line_index": 0,
                "speaker": "Speaker 1",
                "style": "natural",
                "emotion": "neutral",
                "emphasis_words": [],
                "pause_after_ms": 0,
                "reverb_send": 0.1,
                "pan": 0.0,
            }
        ],
    }


class TestProductionDirectorJson(unittest.TestCase):
    def test_extract_json_strips_markdown_and_preamble(self):
        inner = _minimal_valid_plan_dict()
        blob = (
            "Sure — here is the plan.\n```json\n"
            + json.dumps(inner, indent=2)
            + "\n```\nThanks."
        )
        out = extract_json_object_from_llm_text(blob)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.get("episode_id"), "e-unit-test")

    def test_extract_json_balanced_braces(self):
        inner = {"episode_id": "x", "a": {"b": 1}}
        text = 'Noise before {"episode_id": "x", "a": {"b": 1}} trailing'
        out = extract_json_object_from_llm_text(text)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["episode_id"], "x")


class TestProductionDirectorSchema(unittest.TestCase):
    def test_track_event_pan_bounds(self):
        with self.assertRaises(ValidationError):
            TrackEvent(
                event_id="e",
                start_ms=0,
                duration_ms=100,
                pan=2.0,
            )

    def test_production_plan_requires_positive_duration(self):
        bad = _minimal_valid_plan_dict()
        bad["duration_target_seconds"] = 0.0
        with self.assertRaises(ValidationError):
            ProductionPlan.model_validate(bad)


class TestFallbackPlan(unittest.TestCase):
    def test_fallback_from_script_segments(self):
        script = "Speaker 1: Hello world.\nSpeaker 2: Reply here."
        segments = [
            {
                "segment_id": 1,
                "segment_type": "intro_music",
                "speaker": None,
                "text": None,
                "start_time_hint": 0.0,
                "duration_hint": 5.0,
                "energy_level": "high",
                "notes": None,
            },
            {
                "segment_id": 2,
                "segment_type": "dialogue",
                "speaker": "Speaker 1",
                "text": "Hello world.",
                "start_time_hint": 2.0,
                "duration_hint": 3.0,
                "energy_level": "medium",
                "notes": None,
            },
            {
                "segment_id": 3,
                "segment_type": "dialogue",
                "speaker": "Speaker 2",
                "text": "Reply here.",
                "start_time_hint": 6.0,
                "duration_hint": 3.0,
                "energy_level": "medium",
                "notes": None,
            },
            {
                "segment_id": 4,
                "segment_type": "outro_music",
                "speaker": None,
                "text": None,
                "start_time_hint": 10.0,
                "duration_hint": 8.0,
                "energy_level": "low",
                "notes": None,
            },
        ]
        plan = build_fallback_production_plan(
            script=script,
            script_segments=segments,
            genre="News",
            episode_id="fb-1",
            timing_hints=[{"end_ms": 20000}],
        )
        self.assertIsInstance(plan, ProductionPlan)
        self.assertEqual(plan.episode_id, "fb-1")
        self.assertEqual(plan.genre, "News")
        self.assertGreaterEqual(plan.duration_target_seconds, 30.0)
        self.assertEqual(len(plan.voice_direction), 2)
        roles = {t.track_role for t in plan.tracks}
        self.assertIn("voice_main", roles)
        self.assertIn("music_bed", roles)

    def test_fallback_dialogue_only_segments_adds_music_from_script_shape(self):
        """LLM segments often omit intro/outro; supplemental segments must yield mixable music."""
        script = "Speaker 1: Hello world.\nSpeaker 2: Reply here."
        segments = [
            {
                "segment_type": "dialogue",
                "speaker": "Speaker 1",
                "text": "Hello world.",
                "start_time_hint": 0.0,
                "duration_hint": 2.0,
                "energy_level": "medium",
            },
            {
                "segment_type": "dialogue",
                "speaker": "Speaker 2",
                "text": "Reply here.",
                "start_time_hint": 3.0,
                "duration_hint": 2.0,
                "energy_level": "medium",
            },
        ]
        plan = build_fallback_production_plan(
            script=script,
            script_segments=segments,
            genre="News",
            episode_id="fb-dialogue-only",
            timing_hints=[],
        )
        music_tr = next(t for t in plan.tracks if t.track_role == "music_bed")
        self.assertGreater(len(music_tr.events), 0)
        for ev in music_tr.events:
            self.assertIsNotNone(ev.asset_ref)
            self.assertIsNotNone(ev.asset_ref.generation_prompt)


class TestProductionDirectorAsync(unittest.IsolatedAsyncioTestCase):
    async def test_plan_validates_ollama_json(self):
        plan_json = json.dumps(_minimal_valid_plan_dict())
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": plan_json}}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("app.services.production_director.httpx.AsyncClient", return_value=mock_cm):
            director = ProductionDirector(base_url="http://127.0.0.1:9", model="test-model", timeout_seconds=5.0)
            plan = await director.plan(
                script="Speaker 1: Hi.",
                script_segments=[],
                genre="News",
                available_assets=[{"asset_id": "a1", "tags": ["bed"], "duration": 30.0, "mood": "calm", "bpm": 90}],
                timing_hints=[{"line_index": 0, "start_ms": 0, "end_ms": 500}],
            )
        self.assertEqual(plan.genre, "News")
        self.assertEqual(plan.episode_id, "e-unit-test")

    async def test_plan_fallback_on_bad_response(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "NOT JSON AT ALL {["}}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        script = "Speaker 1: Only line."
        with patch("app.services.production_director.httpx.AsyncClient", return_value=mock_cm):
            director = ProductionDirector(base_url="http://127.0.0.1:9", model="test-model", timeout_seconds=5.0)
            plan = await director.plan(
                script=script,
                script_segments=[],
                genre="General",
                available_assets=[],
                timing_hints=[],
            )
        self.assertIsInstance(plan, ProductionPlan)
        self.assertEqual(len(plan.voice_direction), 1)

    async def test_plan_fallback_on_timeout(self):
        async def slow_post(*_a, **_k):
            import asyncio

            await asyncio.sleep(10)
            return MagicMock()

        mock_client = MagicMock()
        mock_client.post = slow_post
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("app.services.production_director.httpx.AsyncClient", return_value=mock_cm):
            director = ProductionDirector(base_url="http://127.0.0.1:9", model="test-model", timeout_seconds=0.05)
            plan = await director.plan(
                script="Speaker 1: Hey.",
                script_segments=[],
                genre="General",
                available_assets=[],
                timing_hints=[],
            )
        self.assertIsInstance(plan, ProductionPlan)


if __name__ == "__main__":
    unittest.main()
