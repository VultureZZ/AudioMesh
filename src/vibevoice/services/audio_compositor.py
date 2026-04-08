"""
Audio composition service for production podcast output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ffmpeg
from pydub import AudioSegment

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class CuePlacement:
    cue_type: str
    file_path: str
    position_ms: int
    volume_db: float = 0.0
    duration_ms: Optional[int] = None


class AudioCompositor:
    """Composes voice + music cues into a production mix."""

    def mix_podcast(self, voice_path: str, cues: List[CuePlacement]) -> str:
        voice = AudioSegment.from_file(voice_path)
        voice_len = len(voice)

        intro_cues = [c for c in cues if c.cue_type == "intro"]
        outro_cues = [c for c in cues if c.cue_type == "outro"]
        transition_cues = [c for c in cues if c.cue_type == "transition"]
        bed_cues = [c for c in cues if c.cue_type == "bed"]
        dialogue_cues = [c for c in cues if c.cue_type == "dialogue"]

        total_len = max(
            voice_len,
            self._calculate_timeline_end(voice_len, intro_cues + outro_cues + transition_cues + bed_cues),
        )
        mix = (
            AudioSegment.silent(duration=total_len, frame_rate=voice.frame_rate)
            .set_channels(voice.channels)
            .set_sample_width(voice.sample_width)
        )
        logger.info(
            "Compositing production mix: voice_len_ms=%s, total_len_ms=%s, cues=intro:%s,outro:%s,transition:%s,bed:%s",
            voice_len,
            total_len,
            len(intro_cues),
            len(outro_cues),
            len(transition_cues),
            len(bed_cues),
        )

        if bed_cues:
            bed_track = self._build_bed_track(total_len, bed_cues[0], dialogue_cues)
            mix = mix.overlay(bed_track, position=max(bed_cues[0].position_ms, 0))

        for cue in intro_cues:
            intro = AudioSegment.from_file(cue.file_path).apply_gain(cue.volume_db).fade_out(2500)
            mix = mix.overlay(intro, position=max(cue.position_ms, 0))

        for cue in transition_cues:
            transition = AudioSegment.from_file(cue.file_path).apply_gain(cue.volume_db)
            mix = mix.overlay(transition, position=max(cue.position_ms, 0))

        for cue in outro_cues:
            outro = AudioSegment.from_file(cue.file_path).apply_gain(cue.volume_db).fade_in(1200)
            default_pos = max(voice_len - 500, 0)
            position = max(cue.position_ms if cue.position_ms > 0 else default_pos, 0)
            mix = mix.overlay(outro, position=position)

        final_mix = mix.overlay(voice, position=0)
        return self._export_mix(final_mix, expected_min_duration_ms=voice_len)

    def _build_bed_track(self, total_len: int, bed_cue: CuePlacement, dialogue_cues: List[CuePlacement]) -> AudioSegment:
        bed_source = AudioSegment.from_file(bed_cue.file_path).apply_gain(bed_cue.volume_db)
        if len(bed_source) <= 0:
            return AudioSegment.silent(duration=total_len, frame_rate=44100)

        # Target bed level: -6dB in gaps.
        bed = self._loop_to_length(bed_source, total_len).apply_gain(-6.0)

        # Duck to -18dB under dialogue (delta -12dB from base gap level).
        for cue in dialogue_cues:
            start = max(cue.position_ms, 0)
            if cue.duration_ms is None:
                continue
            end = min(start + cue.duration_ms, total_len)
            if end <= start:
                continue
            ducked = bed[start:end].apply_gain(-12.0)
            bed = bed[:start] + ducked + bed[end:]

        return bed

    def _loop_to_length(self, segment: AudioSegment, target_len: int) -> AudioSegment:
        if len(segment) >= target_len:
            return segment[:target_len]
        out = AudioSegment.empty()
        while len(out) < target_len:
            out += segment
        return out[:target_len]

    def _calculate_timeline_end(self, voice_len: int, cues: List[CuePlacement]) -> int:
        max_end = voice_len
        for cue in cues:
            try:
                cue_audio = AudioSegment.from_file(cue.file_path)
                max_end = max(max_end, cue.position_ms + len(cue_audio))
            except Exception:
                continue
        return max_end

    def _export_mix(self, mixed_audio: AudioSegment, expected_min_duration_ms: int) -> str:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        wav_path = config.OUTPUT_DIR / f"{stamp}_podcast_production_mix.wav"
        mp3_path = config.OUTPUT_DIR / f"{stamp}_podcast_production_mix.mp3"

        mixed_audio.export(str(wav_path), format="wav")

        try:
            (
                ffmpeg.input(str(wav_path))
                .output(str(mp3_path), audio_bitrate="192k")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            logger.error("ffmpeg conversion failed: %s", stderr)
            raise RuntimeError(f"Failed converting mix to mp3: {stderr}") from exc

        # Guard against accidental truncated exports (e.g. only intro cue length).
        rendered = AudioSegment.from_file(str(mp3_path))
        rendered_len = len(rendered)
        if rendered_len < max(expected_min_duration_ms - 3000, 1000):
            raise RuntimeError(
                f"Production mix appears truncated: rendered={rendered_len}ms expected>={expected_min_duration_ms}ms"
            )
        logger.info("Exported production mix: mp3=%s, duration_ms=%s", mp3_path, rendered_len)
        return str(mp3_path)


audio_compositor = AudioCompositor()

