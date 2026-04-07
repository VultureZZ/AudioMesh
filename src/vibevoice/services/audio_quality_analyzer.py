"""
Audio quality analysis service for voice training.

Analyzes audio for background music, background noise, recording quality,
and overall voice clone quality from clips.
Supports Qwen3-TTS (5-15s optimal) and legacy duration scoring.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config

logger = logging.getLogger(__name__)

# Clone quality levels
CLONE_QUALITY_EXCELLENT = "excellent"
CLONE_QUALITY_GOOD = "good"
CLONE_QUALITY_FAIR = "fair"
CLONE_QUALITY_POOR = "poor"

# Issue identifiers
ISSUE_BACKGROUND_MUSIC = "background_music"
ISSUE_BACKGROUND_NOISE = "background_noise"
ISSUE_LOW_RECORDING_QUALITY = "low_recording_quality"
ISSUE_BACKGROUND_AUDIO = "background_audio"

# Thresholds (tuned heuristically)
# Harmonic ratio above this suggests music presence
MUSIC_HARMONIC_RATIO_THRESHOLD = 0.6
# Spectral flatness above this suggests noise (noise is spectrally flat)
NOISE_FLATNESS_THRESHOLD = 0.5
# RMS below this (normalized) suggests too quiet
RMS_LOW_THRESHOLD = 0.02
# RMS above this suggests clipping/loud
RMS_HIGH_THRESHOLD = 0.95
# Duration thresholds for quality scoring (seconds) - legacy
DURATION_OPTIMAL_MIN = 60.0
DURATION_OPTIMAL_MAX = 180.0
DURATION_MIN = 30.0

# Qwen3-TTS duration scoring
QWEN3_DURATION_MIN = 3.0
QWEN3_DURATION_OPTIMAL_MIN = 5.0
QWEN3_DURATION_OPTIMAL_MAX = 15.0
QWEN3_DURATION_RECOMMENDED_MAX = 20.0
QWEN3_DURATION_HARD_MAX = 60.0


class AudioQualityAnalyzer:
    """Service for analyzing audio quality for voice cloning."""

    def analyze_quality(
        self,
        audio_files: List[Path],
        combined_path: Path,
        total_duration_seconds: float,
    ) -> Dict[str, Any]:
        """
        Analyze combined audio for background music, noise, and clone quality.

        Args:
            audio_files: List of source audio file paths
            combined_path: Path to combined WAV file
            total_duration_seconds: Total duration in seconds

        Returns:
            Dict with clone_quality, issues, recording_quality_score, etc.
        """
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(str(combined_path), sr=None, mono=True)

            individual_files: List[Dict[str, Any]] = []
            if audio_files:
                for af in audio_files:
                    try:
                        y_file, _ = librosa.load(str(af), sr=sr, mono=True)
                        file_analysis = self._analyze_audio_array(y_file, sr, af.name)
                        individual_files.append(file_analysis)
                    except Exception as e:
                        logger.warning("Could not analyze file %s: %s", af.name, e)
                        individual_files.append({
                            "filename": af.name,
                            "has_music": False,
                            "has_noise": False,
                            "recording_quality_score": 0.5,
                            "error": str(e),
                        })

            combined_analysis = self._analyze_audio_array(y, sr, "combined")
            combined_analysis["filename"] = "combined.wav"

            # Aggregate issues from individual files and combined
            all_has_music = any(f.get("has_music", False) for f in individual_files)
            all_has_noise = any(f.get("has_noise", False) for f in individual_files)
            if not all_has_music:
                all_has_music = combined_analysis.get("has_music", False)
            if not all_has_noise:
                all_has_noise = combined_analysis.get("has_noise", False)

            # Build issues list
            issues: List[str] = []
            if all_has_music:
                issues.append(ISSUE_BACKGROUND_MUSIC)
            if all_has_noise:
                issues.append(ISSUE_BACKGROUND_NOISE)
            if combined_analysis.get("low_recording_quality", False):
                issues.append(ISSUE_LOW_RECORDING_QUALITY)
            if all_has_music or all_has_noise:
                issues.append(ISSUE_BACKGROUND_AUDIO)

            # Deduplicate and order
            seen: set[str] = set()
            unique_issues: List[str] = []
            for issue in issues:
                if issue not in seen:
                    seen.add(issue)
                    unique_issues.append(issue)

            # Compute overall clone quality
            rec_score = combined_analysis.get("recording_quality_score", 0.5)
            duration_score = self._duration_quality_score(total_duration_seconds)
            music_penalty = 0.3 if all_has_music else 0.0
            noise_penalty = 0.2 if all_has_noise else 0.0
            composite = (
                (rec_score * 0.5) + (duration_score * 0.5) - music_penalty - noise_penalty
            )
            composite = max(0.0, min(1.0, composite))

            if composite >= 0.85:
                clone_quality = CLONE_QUALITY_EXCELLENT
            elif composite >= 0.65:
                clone_quality = CLONE_QUALITY_GOOD
            elif composite >= 0.4:
                clone_quality = CLONE_QUALITY_FAIR
            else:
                clone_quality = CLONE_QUALITY_POOR

            return {
                "clone_quality": clone_quality,
                "issues": unique_issues,
                "recording_quality_score": round(rec_score, 2),
                "background_music_detected": all_has_music,
                "background_noise_detected": all_has_noise,
                "individual_files": individual_files,
            }
        except Exception as e:
            logger.warning("Audio quality analysis failed: %s", e, exc_info=True)
            return {
                "clone_quality": CLONE_QUALITY_FAIR,
                "issues": [],
                "recording_quality_score": 0.5,
                "background_music_detected": False,
                "background_noise_detected": False,
                "individual_files": [],
                "analysis_error": str(e),
            }

    def _analyze_audio_array(
        self, y: "Any", sr: int, filename: str = ""
    ) -> Dict[str, Any]:
        """Analyze a numpy audio array for music, noise, and recording quality."""
        import librosa
        import numpy as np

        has_music = False
        has_noise = False
        low_recording_quality = False
        recording_quality_score = 0.5

        if len(y) == 0:
            return {
                "filename": filename,
                "has_music": False,
                "has_noise": False,
                "low_recording_quality": True,
                "recording_quality_score": 0.0,
                "error": "Empty audio",
            }

        # Harmonic/percussive separation - music has strong harmonic component
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = float(np.sqrt(np.mean(y_harmonic ** 2)))
            percussive_energy = float(np.sqrt(np.mean(y_percussive ** 2)))
            total_energy = float(np.sqrt(np.mean(y ** 2))) or 1e-10
            harmonic_ratio = harmonic_energy / total_energy
            if harmonic_ratio > MUSIC_HARMONIC_RATIO_THRESHOLD:
                has_music = True
        except Exception:
            harmonic_ratio = 0.0

        # Spectral flatness - noise tends to be flat
        try:
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            mean_flatness = float(np.mean(flatness))
            if mean_flatness > NOISE_FLATNESS_THRESHOLD:
                has_noise = True
        except Exception:
            mean_flatness = 0.0

        # RMS for recording level
        try:
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = float(np.mean(rms))
            if mean_rms < RMS_LOW_THRESHOLD:
                recording_quality_score = max(0, 0.5 - (RMS_LOW_THRESHOLD - mean_rms) * 10)
            elif mean_rms > RMS_HIGH_THRESHOLD:
                recording_quality_score = max(0, 0.5 - (mean_rms - RMS_HIGH_THRESHOLD) * 5)
            else:
                recording_quality_score = 0.5 + (mean_rms * 0.5)
                recording_quality_score = min(1.0, recording_quality_score)
            low_recording_quality = recording_quality_score < 0.4
        except Exception:
            recording_quality_score = 0.5

        return {
            "filename": filename,
            "has_music": has_music,
            "has_noise": has_noise,
            "low_recording_quality": low_recording_quality,
            "recording_quality_score": round(recording_quality_score, 2),
            "harmonic_ratio": round(harmonic_ratio, 3),
            "spectral_flatness": round(mean_flatness, 3),
        }

    def _duration_quality_score(self, duration_seconds: float) -> float:
        """Score 0-1 based on duration for voice cloning. Backend-aware (Qwen3 vs legacy)."""
        backend = (getattr(config, "TTS_BACKEND", "qwen3") or "qwen3").strip().lower()
        if backend == "qwen3":
            return self._duration_quality_score_qwen3(duration_seconds)
        return self._duration_quality_score_legacy(duration_seconds)

    def _duration_quality_score_qwen3(self, duration_seconds: float) -> float:
        """Qwen3: optimal 5-15s (1.0), min 3s, decay above 20s, cap at 60s."""
        if duration_seconds < QWEN3_DURATION_MIN:
            return max(0.0, duration_seconds / QWEN3_DURATION_MIN) * 0.5
        if duration_seconds < QWEN3_DURATION_OPTIMAL_MIN:
            return 0.5 + 0.5 * (duration_seconds - QWEN3_DURATION_MIN) / (
                QWEN3_DURATION_OPTIMAL_MIN - QWEN3_DURATION_MIN
            )
        if QWEN3_DURATION_OPTIMAL_MIN <= duration_seconds <= QWEN3_DURATION_OPTIMAL_MAX:
            return 1.0
        if duration_seconds <= QWEN3_DURATION_RECOMMENDED_MAX:
            return 0.9
        if duration_seconds <= QWEN3_DURATION_HARD_MAX:
            return max(0.5, 1.0 - (duration_seconds - QWEN3_DURATION_RECOMMENDED_MAX) / (
                QWEN3_DURATION_HARD_MAX - QWEN3_DURATION_RECOMMENDED_MAX
            ) * 0.4)
        return 0.5

    def _duration_quality_score_legacy(self, duration_seconds: float) -> float:
        """Legacy (e.g. VibeVoice): optimal 60-180s."""
        if duration_seconds < DURATION_MIN:
            return max(0, duration_seconds / DURATION_MIN) * 0.5
        if DURATION_OPTIMAL_MIN <= duration_seconds <= DURATION_OPTIMAL_MAX:
            return 1.0
        if duration_seconds < DURATION_OPTIMAL_MIN:
            return 0.5 + 0.5 * (duration_seconds - DURATION_MIN) / (
                DURATION_OPTIMAL_MIN - DURATION_MIN
            )
        return max(0.5, 1.0 - (duration_seconds - DURATION_OPTIMAL_MAX) / 300.0)


# Global analyzer instance
audio_quality_analyzer = AudioQualityAnalyzer()
