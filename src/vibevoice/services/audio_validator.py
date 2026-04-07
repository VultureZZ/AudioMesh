"""
Audio validation service for voice training.

Analyzes audio files and provides feedback on duration, quality, and recommendations.
Supports Qwen3-TTS (single reference 3-60s) and VibeVoice-style (longer multi-file) rules.
"""
from pathlib import Path
from typing import Dict, List, Optional

from ..config import config

# Audio duration recommendations (in seconds) - VibeVoice / legacy
MIN_DURATION_SECONDS = 30.0  # Minimum recommended duration
RECOMMENDED_MIN_DURATION = 60.0  # Recommended minimum (1 minute)
RECOMMENDED_MAX_DURATION = 180.0  # Recommended maximum (3 minutes)
MAX_RECOMMENDED_DURATION = 300.0  # Maximum recommended (5 minutes)

# Individual file recommendations (legacy)
RECOMMENDED_FILE_MIN_DURATION = 3.0  # Minimum per file (3 seconds)
RECOMMENDED_FILE_MAX_DURATION = 10.0  # Maximum per file (10 seconds)

# Qwen3-TTS: single reference clip duration (combined ref is used as one ref)
QWEN3_REF_MIN_S = 3
QWEN3_REF_OPTIMAL_MIN_S = 5
QWEN3_REF_OPTIMAL_MAX_S = 15
QWEN3_REF_RECOMMENDED_MAX_S = 20
QWEN3_REF_HARD_MAX_S = 60
QWEN3_REF_MAX_SIZE_MB = 10

# Standard audio settings
STANDARD_SAMPLE_RATE = 24000  # Hz
STANDARD_CHANNELS = 1  # Mono


def _is_qwen3_backend() -> bool:
    return (getattr(config, "TTS_BACKEND", "qwen3") or "qwen3").strip().lower() == "qwen3"


def get_qwen3_best_practice_tips() -> List[str]:
    """Best-practice recommendation strings for Qwen3-TTS voice cloning."""
    return [
        "One clean clip of 5-15 seconds is ideal.",
        "Providing a transcript of the reference improves clone quality.",
        "Use a quiet environment; no background music or other voices.",
        "Normal speech with varied intonation works best.",
        "Mono, 24 kHz or higher (handled by server); under 10 MB recommended.",
    ]


class AudioValidator:
    """Service for validating audio files for voice training."""

    def analyze_file(self, audio_path: Path) -> Dict:
        """
        Analyze a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with analysis results including duration, sample_rate, channels, file_size, warnings
        """
        try:
            from pydub import AudioSegment

            # Load audio file
            audio = AudioSegment.from_file(str(audio_path))

            duration_seconds = len(audio) / 1000.0  # pydub returns milliseconds
            sample_rate = audio.frame_rate
            channels = audio.channels
            file_size_bytes = audio_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            warnings = []

            # Check duration (Qwen3: only min 3s; no per-file max. Legacy: 3-10s per file)
            if _is_qwen3_backend():
                if duration_seconds < QWEN3_REF_MIN_S:
                    warnings.append(
                        f"File duration ({duration_seconds:.1f}s) is below minimum ({QWEN3_REF_MIN_S}s)."
                    )
            else:
                if duration_seconds < RECOMMENDED_FILE_MIN_DURATION:
                    warnings.append(
                        f"File duration ({duration_seconds:.1f}s) is below recommended minimum ({RECOMMENDED_FILE_MIN_DURATION}s)"
                    )
                elif duration_seconds > RECOMMENDED_FILE_MAX_DURATION:
                    warnings.append(
                        f"File duration ({duration_seconds:.1f}s) exceeds recommended maximum ({RECOMMENDED_FILE_MAX_DURATION}s)"
                    )

            # Check sample rate (informational, will be normalized)
            if sample_rate != STANDARD_SAMPLE_RATE:
                # Not a warning, just informational since we normalize
                pass

            # Check channels (informational, will be converted to mono)
            if channels > STANDARD_CHANNELS:
                # Not a warning, just informational since we convert
                pass

            return {
                "filename": audio_path.name,
                "duration_seconds": round(duration_seconds, 2),
                "sample_rate": sample_rate,
                "channels": channels,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(file_size_mb, 2),
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "filename": audio_path.name,
                "error": str(e),
                "warnings": [f"Failed to analyze file: {str(e)}"],
            }

    def validate_audio_files(
        self, audio_files: List[Path], combined_duration_seconds: Optional[float] = None
    ) -> Dict:
        """
        Validate multiple audio files and generate feedback.

        Args:
            audio_files: List of paths to audio files
            combined_duration_seconds: Optional combined duration (if already calculated)

        Returns:
            Dict with validation feedback including warnings, recommendations, and metrics
        """
        individual_analyses = []
        all_warnings = []
        total_duration = 0.0

        # Analyze each file
        for audio_file in audio_files:
            analysis = self.analyze_file(audio_file)
            individual_analyses.append(analysis)
            total_duration += analysis.get("duration_seconds", 0.0)
            all_warnings.extend(analysis.get("warnings", []))

        # Use provided combined duration if available, otherwise calculate from individual files
        if combined_duration_seconds is not None:
            total_duration = combined_duration_seconds

        # Generate warnings and recommendations (Qwen3 vs legacy)
        warnings = all_warnings.copy()
        recommendations = []
        file_count = len(audio_files)

        if _is_qwen3_backend():
            # Qwen3-TTS: combined ref 3-60s; 5-15s optimal. No 30s/60s or multi-file diversity messages.
            if total_duration < QWEN3_REF_MIN_S:
                warnings.append(
                    f"Total reference duration ({total_duration:.1f}s) is below minimum ({QWEN3_REF_MIN_S}s). "
                    "Use at least 3 seconds of clear speech."
                )
                recommendations.append("For best quality use 5-15 seconds of clear speech.")
            elif total_duration < QWEN3_REF_OPTIMAL_MIN_S:
                recommendations.append("For best quality use 5-15 seconds of clear speech.")
            elif QWEN3_REF_OPTIMAL_MIN_S <= total_duration <= QWEN3_REF_OPTIMAL_MAX_S:
                recommendations.append(
                    "Total duration is in the optimal range (5-15s) for voice cloning."
                )
            elif total_duration <= QWEN3_REF_RECOMMENDED_MAX_S:
                recommendations.append(
                    "Total duration is acceptable; 5-15s is optimal for best results."
                )
            elif total_duration <= QWEN3_REF_HARD_MAX_S:
                warnings.append(
                    f"Total reference duration ({total_duration:.1f}s) is above recommended 20s; "
                    "5-15s is optimal. Audio beyond 60s will be truncated."
                )
                recommendations.append("For best quality use 5-15 seconds of clear speech.")
            # If > 60s, voice_manager will truncate and add its own warning; we still validate here
            recommendations.extend(get_qwen3_best_practice_tips())

            # File size: warn if any file or total size > 10 MB
            total_size_mb = sum(
                a.get("file_size_mb", 0) or 0 for a in individual_analyses if "error" not in a
            )
            for a in individual_analyses:
                if "error" in a:
                    continue
                fmb = a.get("file_size_mb") or 0
                if fmb > QWEN3_REF_MAX_SIZE_MB:
                    warnings.append(
                        f"Reference audio over 10 MB ({fmb:.1f} MB); under 10 MB is recommended."
                    )
                    break
            if total_size_mb > QWEN3_REF_MAX_SIZE_MB and not any(
                (a.get("file_size_mb") or 0) > QWEN3_REF_MAX_SIZE_MB for a in individual_analyses if "error" not in a
            ):
                warnings.append(
                    "Combined reference size is over 10 MB; under 10 MB is recommended."
                )
        else:
            # Legacy VibeVoice-style rules
            if total_duration < MIN_DURATION_SECONDS:
                warnings.append(
                    f"Total audio duration ({total_duration:.1f}s) is below minimum recommended ({MIN_DURATION_SECONDS}s). "
                    "Voice quality may be poor."
                )
                recommendations.append(
                    f"Add more audio to reach at least {MIN_DURATION_SECONDS} seconds for better results."
                )
            elif total_duration < RECOMMENDED_MIN_DURATION:
                recommendations.append(
                    f"Total audio duration ({total_duration:.1f}s) is below recommended minimum ({RECOMMENDED_MIN_DURATION}s). "
                    "Consider adding more audio for better voice quality."
                )
            elif total_duration > MAX_RECOMMENDED_DURATION:
                warnings.append(
                    f"Total audio duration ({total_duration:.1f}s) exceeds recommended maximum ({MAX_RECOMMENDED_DURATION}s). "
                    "Additional audio beyond this point may not significantly improve quality."
                )
                recommendations.append(
                    f"Consider using {RECOMMENDED_MAX_DURATION}-{MAX_RECOMMENDED_DURATION} seconds for optimal results."
                )
            elif RECOMMENDED_MIN_DURATION <= total_duration <= RECOMMENDED_MAX_DURATION:
                recommendations.append(
                    f"Total audio duration ({total_duration:.1f}s) is within the optimal range "
                    f"({RECOMMENDED_MIN_DURATION}-{RECOMMENDED_MAX_DURATION}s)."
                )

            if file_count == 1:
                recommendations.append(
                    "Consider using multiple audio files (3-10 seconds each) for better voice diversity."
                )
            elif file_count > 20:
                recommendations.append(
                    "Using many small files is fine, but ensure total duration is within recommended range."
                )

        # Generate quality metrics summary
        quality_metrics = {
            "total_duration_seconds": round(total_duration, 2),
            "file_count": file_count,
            "standard_sample_rate": STANDARD_SAMPLE_RATE,
            "standard_channels": STANDARD_CHANNELS,
        }

        return {
            "total_duration_seconds": round(total_duration, 2),
            "individual_files": individual_analyses,
            "warnings": warnings,
            "recommendations": recommendations,
            "quality_metrics": quality_metrics,
        }