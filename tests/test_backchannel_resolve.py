"""Tests for mapping voice_backchannel plan events to library assets."""

from __future__ import annotations

import unittest

from app.services.backchannel_resolve import (
    patch_production_plan_voice_backchannels,
    resolve_voice_backchannel_asset_id,
)
from app.services.production_director import (
    AssetRef,
    EmotionalArcPoint,
    ProductionPlan,
    TimelineTrack,
    TrackEvent,
)


class _FakeLibrary:
    def __init__(self, ids: set[str]) -> None:
        self._ids = ids

    def get(self, asset_id: str) -> object:
        if asset_id not in self._ids:
            raise KeyError(asset_id)
        return type("A", (), {"duration_ms": 800})()


class BackchannelResolveTest(unittest.TestCase):
    def test_resolves_reactor_asset(self) -> None:
        from app.services.backchannel_synth import _phrase_slug, _slug

        name = "Host B"
        aid = f"bc_{_slug(name)}_{_phrase_slug('yeah')}"
        lib = _FakeLibrary({aid})
        got = resolve_voice_backchannel_asset_id(
            lib,
            ["Host A", name],
            trigger_word="yeah",
            anchor_speaker="Speaker 1",
        )
        self.assertEqual(got, aid)

    def test_patch_fills_missing_asset_id(self) -> None:
        from app.services.backchannel_synth import _phrase_slug, _slug

        name = "Host B"
        aid = f"bc_{_slug(name)}_{_phrase_slug('wow')}"
        lib = _FakeLibrary({aid})
        plan = ProductionPlan(
            episode_id="e1",
            duration_target_seconds=60.0,
            genre="News",
            emotional_arc=[
                EmotionalArcPoint(timestamp=0.0, valence=0.0, energy=0.5),
                EmotionalArcPoint(timestamp=30.0, valence=0.0, energy=0.5),
                EmotionalArcPoint(timestamp=60.0, valence=0.0, energy=0.5),
            ],
            tracks=[
                TimelineTrack(
                    track_id="bc",
                    track_role="voice_backchannel",
                    events=[
                        TrackEvent(
                            event_id="x1",
                            start_ms=5000,
                            duration_ms=500,
                            asset_ref=AssetRef(generation_prompt="should not mix alone"),
                            volume_db=-6.0,
                            trigger_word="wow",
                            anchor_speaker="Speaker 1",
                        )
                    ],
                )
            ],
            voice_direction=[],
        )
        out = patch_production_plan_voice_backchannels(plan, lib, ["Host A", name])
        ev = out.tracks[0].events[0]
        self.assertIsNotNone(ev.asset_ref)
        assert ev.asset_ref is not None
        self.assertEqual(ev.asset_ref.asset_id, aid)


if __name__ == "__main__":
    unittest.main()
