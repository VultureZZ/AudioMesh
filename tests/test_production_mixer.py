#!/usr/bin/env python3
"""Unit tests for app.services.production_mixer helpers and loudness export."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from app.services.production_director import (
    EmotionalArcPoint,
    ProductionPlan,
)
from app.services.production_mixer import (
    ProductionMixer,
    build_duck_gain_linear,
    interp_automation_linear,
    lufs_target_for_genre,
)


class _DummyLibrary:
    def resolve_path(self, asset_id: str) -> Path:  # noqa: ARG002
        return Path("/nonexistent")


def _mp3_to_lufs_integrated(path: Path) -> float:
    """Decode MP3 to stereo float [-1,1] and measure integrated loudness (LUFS)."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(path))
    seg = seg.set_channels(2)
    sr = seg.frame_rate
    raw = np.array(seg.get_array_of_samples(), dtype=np.float64)
    raw = raw.reshape((-1, seg.channels))
    denom = float(1 << (8 * seg.sample_width - 1))
    pcm = raw / denom
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(pcm))


class TestInterpAutomation(unittest.TestCase):
    def test_empty_returns_base(self):
        n = 1000
        out = interp_automation_linear(n, [], base_db=-3.0)
        self.assertEqual(out.shape, (n,))
        self.assertTrue(np.allclose(out, -3.0))

    def test_linear_between_two_points(self):
        n = 101
        out = interp_automation_linear(n, [(0, 0), (100, 10)], base_db=0.0)
        self.assertAlmostEqual(float(out[0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[50]), 5.0, places=5)
        self.assertAlmostEqual(float(out[100]), 10.0, places=5)

    def test_base_db_added(self):
        out = interp_automation_linear(50, [(0, 2), (49, 2)], base_db=1.0)
        self.assertTrue(np.allclose(out, 3.0))


class TestDuckingEnvelope(unittest.TestCase):
    def test_silence_vad_unity_gain(self):
        sr = 48000
        n = sr
        vad = np.zeros(n, dtype=np.float32)
        g = build_duck_gain_linear(n, vad, sr=sr)
        self.assertEqual(g.shape, (n,))
        self.assertTrue(np.allclose(g, 1.0, atol=1e-6))

    def test_speech_region_applies_duck_depth(self):
        sr = 48000
        n = sr
        vad = np.zeros(n, dtype=np.float32)
        vad[n // 2 : n // 2 + sr // 10] = 1.0
        g = build_duck_gain_linear(n, vad, sr=sr, duck_db=-12.0)
        duck_lin = 10.0 ** (-12.0 / 20.0)
        mid = g[n // 2 + sr // 20]
        self.assertLess(mid, 0.99)
        self.assertGreater(mid, duck_lin * 0.5)

    def test_min_duck_linear_floor_with_deep_duck_db(self):
        """Very deep duck_db must not push music below min_duck_linear during speech."""
        sr = 48000
        n = sr
        vad = np.ones(n, dtype=np.float32)
        g = build_duck_gain_linear(
            n,
            vad,
            sr=sr,
            duck_db=-24.0,
            min_duck_linear=0.4,
        )
        self.assertGreaterEqual(float(np.min(g)), 0.39)

    def test_lookahead_dips_before_speech_onset(self):
        sr = 48000
        n = 3 * sr
        vad = np.zeros(n, dtype=np.float32)
        onset = 2 * sr
        vad[onset : onset + sr // 20] = 1.0
        g = build_duck_gain_linear(
            n,
            vad,
            sr=sr,
            lookahead_ms=800.0,
            attack_ms=300.0,
            release_ms=500.0,
        )
        pre = onset - int(0.5 * sr)
        self.assertLess(g[pre], 1.0)
        self.assertGreater(g[pre], g[onset])


class TestLufsTargetTable(unittest.TestCase):
    def test_genre_overrides(self):
        self.assertEqual(lufs_target_for_genre("true_crime"), -18.0)
        self.assertEqual(lufs_target_for_genre("news"), -16.0)
        self.assertEqual(lufs_target_for_genre("comedy"), -15.0)
        self.assertEqual(lufs_target_for_genre("unknown"), -16.0)


class TestProductionMixerLoudnessFixture(unittest.TestCase):
    def test_render_hits_lufs_target_within_one(self):
        sr = 48000
        dur_s = 2.0
        t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
        mono = 0.15 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        stereo = np.vstack([mono, mono])

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "voice.wav"
            sf.write(str(wav), stereo.T, sr, subtype="PCM_24")

            plan = ProductionPlan(
                episode_id="fixture-lufs",
                duration_target_seconds=dur_s,
                genre="news",
                emotional_arc=[
                    EmotionalArcPoint(timestamp=0.0, valence=0.0, energy=0.5),
                    EmotionalArcPoint(timestamp=dur_s / 2.0, valence=0.0, energy=0.5),
                    EmotionalArcPoint(timestamp=dur_s, valence=0.0, energy=0.5),
                ],
                tracks=[],
                voice_direction=[],
            )
            out_mp3 = Path(td) / "out.mp3"
            mixer = ProductionMixer()
            mixer.render(
                plan,
                str(wav),
                str(out_mp3),
                library=_DummyLibrary(),
            )
            self.assertTrue(out_mp3.is_file())
            loud = _mp3_to_lufs_integrated(out_mp3)
            target = lufs_target_for_genre("news")
            self.assertAlmostEqual(loud, target, delta=1.0)


if __name__ == "__main__":
    unittest.main()
