"""
Production mix engine: pedalboard voice chains, timeline summing, sidechain ducking,
LUFS normalization, ffmpeg MP3 export.

Testable helpers: ``interp_automation_linear``, ``build_duck_gain_linear``,
``lufs_target_for_genre``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor,
    HighpassFilter,
    Limiter,
    NoiseGate,
    PeakFilter,
    Pedalboard,
    Reverb,
)

logger = logging.getLogger(__name__)

try:
    from vibevoice.config import config as _vibe_config  # type: ignore
except Exception:  # pragma: no cover
    _vibe_config = None


def _apply_line_energy_matching(
    mono: np.ndarray,
    sr: int,
    voice_direction: Sequence[Any],
    line_timing_ms: List[Tuple[int, float, float]],
) -> np.ndarray:
    """Gently scale each dialogue line toward an emotion-derived RMS target."""
    try:
        from vibevoice.config import config as _cfg  # type: ignore
    except Exception:
        return mono
    if not getattr(_cfg, "LINE_ENERGY_MATCHING", False):
        return mono
    if not line_timing_ms or not voice_direction:
        return mono
    from app.services.voice_prosody import emotion_line_energy_db

    out = mono.astype(np.float32).copy()
    by_li = {int(getattr(v, "line_index", -1)): v for v in voice_direction}
    rms_list: List[float] = []
    slices: List[Tuple[int, int, float]] = []
    for li, st_ms, en_ms in line_timing_ms:
        s = int(float(st_ms) * sr / 1000.0)
        e = int(float(en_ms) * sr / 1000.0)
        s = max(0, min(s, len(out) - 1))
        e = max(s + 1, min(e, len(out)))
        sl = out[s:e]
        if sl.size < 8:
            continue
        rms = float(np.sqrt(np.mean(sl**2) + 1e-12))
        rms_list.append(rms)
        emo = str(getattr(by_li.get(int(li)), "emotion", None) or "neutral")
        tgt_db = emotion_line_energy_db(emo)
        slices.append((s, e, tgt_db))
    if not rms_list:
        return out
    ref = float(np.median(rms_list))
    for s, e, tgt_db in slices:
        target_rms = ref * float(10.0 ** (tgt_db / 20.0))
        sl = out[s:e]
        cur = float(np.sqrt(np.mean(sl**2) + 1e-12))
        g = target_rms / max(cur, 1e-8)
        g = float(np.clip(g, 0.65, 1.45))
        out[s:e] *= g
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out /= peak
    return out

# Speaker-specific presence peaks (Hz) for spatial separation
_SPEAKER_PRESENCE_HZ: Dict[int, float] = {
    1: 2800.0,
    2: 3400.0,
    3: 2300.0,
    4: 3800.0,
}

_LUFS_BY_GENRE_KEY: Dict[str, float] = {
    "true_crime": -18.0,
    "true crime": -18.0,
    "news": -16.0,
    "comedy": -15.0,
}


def lufs_target_for_genre(genre: str) -> float:
    """Integrated loudness target (LUFS) for podcast mastering."""
    g = (genre or "").strip().lower()
    return _LUFS_BY_GENRE_KEY.get(g, -16.0)


def interp_automation_linear(
    n_samples: int,
    breakpoints: Sequence[Tuple[int, float]],
    *,
    base_db: float = 0.0,
) -> np.ndarray:
    """
    Per-sample gain in dB from automation breakpoints (offset_ms, volume_db relative to event start).
    Linear interpolation between points (``base_db`` added to the interpolated automation curve).
    """
    if not breakpoints:
        return np.full(n_samples, base_db, dtype=np.float64)
    pts = sorted(((int(p[0]), float(p[1])) for p in breakpoints), key=lambda x: x[0])
    offsets = np.array([p[0] for p in pts], dtype=np.float64)
    values = np.array([p[1] for p in pts], dtype=np.float64)
    x = np.arange(n_samples, dtype=np.float64)
    auto = np.interp(x, offsets, values, left=values[0], right=values[-1])
    return base_db + auto


def _db_to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 20.0)


def _linear_to_db(lin: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(lin, 1e-12))


def build_vad_speech_mask(mono: np.ndarray, sr: int, hop_ms: float = 50.0) -> np.ndarray:
    """Binary speech mask per sample (1 = speech) from energy in ``hop_ms`` frames."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    n = mono.shape[0]
    n_hops = (n + hop - 1) // hop
    e = np.zeros(n_hops, dtype=np.float64)
    for i in range(n_hops):
        sl = mono[i * hop : min((i + 1) * hop, n)]
        e[i] = float(np.sqrt(np.mean(sl**2) + 1e-18))
    th = max(float(np.percentile(e, 45)), 1e-4)
    hop_mask = (e > th).astype(np.float32)
    # upsample
    out = np.zeros(n, dtype=np.float32)
    for i in range(n_hops):
        out[i * hop : min((i + 1) * hop, n)] = hop_mask[i]
    return out


def build_duck_gain_linear(
    n_samples: int,
    vad_speech: np.ndarray,
    *,
    sr: int,
    duck_db: float = -12.0,
    min_duck_linear: Optional[float] = None,
    lookahead_ms: float = 800.0,
    attack_ms: float = 300.0,
    release_ms: float = 500.0,
) -> np.ndarray:
    """
    Multiplicative gain (linear 0..1) for music ducking from a speech VAD curve.
    Applies lookahead (max over future window), then exponential attack/release toward duck target.

    ``min_duck_linear``: floor for the ducked target (e.g. 0.35–0.45). Long-form dialogue
    keeps the energy VAD high most of the time; without a floor, beds can sit ~40 dB under
    VO and disappear after loudness normalization.
    """
    assert vad_speech.shape[0] == n_samples
    look = int(lookahead_ms * sr / 1000.0)
    look = max(0, min(look, n_samples))
    # Pre-roll: speech "seen" earlier
    future_max = np.copy(vad_speech.astype(np.float64))
    if look > 0:
        for i in range(n_samples):
            j1 = min(n_samples, i + look + 1)
            future_max[i] = float(np.max(vad_speech[i:j1]))
    target_lin = np.ones(n_samples, dtype=np.float64)
    duck_lin = float(10.0 ** (duck_db / 20.0))
    if min_duck_linear is not None and float(min_duck_linear) > 0.0:
        duck_lin = max(duck_lin, float(min_duck_linear))
    target_lin = np.where(future_max > 0.5, duck_lin, 1.0)

    att = np.exp(-1.0 / max(1.0, attack_ms * sr / 1000.0))
    rel = np.exp(-1.0 / max(1.0, release_ms * sr / 1000.0))
    out = np.ones(n_samples, dtype=np.float64)
    out[0] = target_lin[0]
    for i in range(1, n_samples):
        t = target_lin[i]
        c = out[i - 1]
        # moving toward quieter (duck) = attack; toward louder = release
        if t < c:
            out[i] = t + (c - t) * att
        else:
            out[i] = t + (c - t) * rel
    return np.clip(out, 0.0, 1.0)


def _parse_speaker_index(speaker_label: str) -> int:
    m = re.search(r"(\d+)", speaker_label or "")
    if not m:
        return 1
    idx = int(m.group(1))
    return max(1, min(4, idx))


def _speaker_chain(speaker_index: int, vc: Optional[Dict[str, float]] = None) -> Pedalboard:
    from app.services.genre_templates import merge_voice_chain_params

    p = merge_voice_chain_params(vc or {})
    peak_hz = _SPEAKER_PRESENCE_HZ.get(speaker_index, 2800.0)
    return Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=float(p["highpass_hz"])),
            NoiseGate(
                threshold_db=float(p["noise_gate_threshold_db"]),
                ratio=float(p["noise_gate_ratio"]),
                attack_ms=5,
                release_ms=250,
            ),
            Compressor(
                threshold_db=float(p["compressor_threshold_db"]),
                ratio=float(p["compressor_ratio"]),
                attack_ms=float(p["compressor_attack_ms"]),
                release_ms=float(p["compressor_release_ms"]),
            ),
            PeakFilter(
                cutoff_frequency_hz=peak_hz,
                gain_db=float(p["peak_filter_gain_db"]),
                q=float(p["peak_filter_q"]),
            ),
            Reverb(room_size=float(p["reverb_room_size"]), wet_level=float(p["reverb_wet_level"])),
        ]
    )


def _master_chain_from_params(p: Dict[str, float]) -> Pedalboard:
    return Pedalboard(
        [
            Compressor(
                threshold_db=float(p["master_compressor_threshold_db"]),
                ratio=float(p["master_compressor_ratio"]),
                attack_ms=float(p["master_compressor_attack_ms"]),
                release_ms=float(p["master_compressor_release_ms"]),
            ),
            Limiter(threshold_db=float(p["master_limiter_threshold_db"])),
        ]
    )


def _read_audio_float(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # (samples, ch) -> (ch, samples)
    x = data.T.copy()
    return x, int(sr)


def _to_stereo(ch2: np.ndarray) -> np.ndarray:
    if ch2.shape[0] == 1:
        return np.vstack([ch2, ch2])
    if ch2.shape[0] > 2:
        return ch2[:2]
    return ch2


def _mono_from_stereo(ch2: np.ndarray) -> np.ndarray:
    if ch2.shape[0] == 1:
        return ch2[0].copy()
    return (ch2[0] + ch2[1]) * 0.5


def _resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    if orig_sr == target_sr:
        return x
    n_src = x.shape[1]
    n_dst = int(round(n_src * target_sr / orig_sr))
    if n_dst <= 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, num=n_src, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    out = np.zeros((x.shape[0], n_dst), dtype=np.float32)
    for c in range(x.shape[0]):
        out[c] = np.interp(t_dst, t_src, x[c]).astype(np.float32)
    return out


def _pan_gains(pan: float) -> Tuple[float, float]:
    p = float(np.clip(pan, -1.0, 1.0))
    ang = (p + 1.0) * (np.pi / 4.0)
    return float(np.cos(ang)), float(np.sin(ang))


def _apply_fades(samples_ch: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    n = samples_ch.shape[1]
    out = samples_ch.copy()
    fi = min(fade_in, n)
    fo = min(fade_out, n)
    if fi > 0:
        env = np.linspace(0.0, 1.0, fi, endpoint=False, dtype=np.float32)
        out[:, :fi] *= env
    if fo > 0:
        env = np.linspace(1.0, 0.0, fo, endpoint=False, dtype=np.float32)
        out[:, n - fo :] *= env
    return out


class ProductionMixer:
    """Pedalboard + numpy timeline + pyloudnorm + ffmpeg MP3."""

    def render(
        self,
        plan: Any,
        voice_wav_path: str,
        output_path: str,
        *,
        library: Any,
        voice_line_timing_ms: Optional[List[Tuple[int, float, float]]] = None,
        asset_path_overrides: Optional[Dict[str, str]] = None,
        id3_title: Optional[str] = None,
        id3_genre: Optional[str] = None,
        episode_number: Optional[int] = None,
        word_index: Optional[List[Dict[str, Any]]] = None,
        timing_hints: Optional[List[Dict[str, Any]]] = None,
        genre_template: Any = None,
    ) -> str:
        """
        Mix ``plan`` with voice at ``voice_wav_path`` and write MP3 to ``output_path``.

        ``voice_line_timing_ms``: optional list of (line_index, start_ms, end_ms) for stem splits.
        ``asset_path_overrides``: optional map asset_id -> filesystem path (legacy cue mode).
        ``word_index`` / ``timing_hints``: used to resolve ``trigger_word`` placements at render time.
        """
        if word_index:
            try:
                from app.services.production_director import ProductionPlan as _ProdPlan
                from app.services.trigger_resolution import apply_trigger_word_resolution

                if isinstance(plan, _ProdPlan):
                    plan = apply_trigger_word_resolution(plan, word_index, timing_hints or [])
                else:
                    plan = apply_trigger_word_resolution(
                        _ProdPlan.model_validate(plan), word_index, timing_hints or []
                    )
            except Exception as exc:
                logger.warning("trigger_word resolution skipped: %s", exc)

        voice_path = Path(voice_wav_path)
        if not voice_path.is_file():
            raise FileNotFoundError(voice_path)

        samples, sr = _read_audio_float(voice_path)
        samples = _to_stereo(samples)
        n_samp = samples.shape[1]
        expected_min_ms = int(1000.0 * n_samp / sr)

        mono = _mono_from_stereo(samples).astype(np.float32)
        vad = build_vad_speech_mask(mono, sr)

        voice_proc = self._process_voice_tracks(
            samples,
            sr,
            plan,
            voice_line_timing_ms,
            genre_template=genre_template,
        )

        # Timeline length
        total_ms = int(1000.0 * n_samp / sr)
        for tr in plan.tracks:
            for ev in tr.events:
                total_ms = max(total_ms, int(ev.start_ms) + int(ev.duration_ms))
        total_samps = int(np.ceil(total_ms / 1000.0 * sr))

        master = np.zeros((2, total_samps), dtype=np.float32)
        master[:, : min(n_samp, total_samps)] += voice_proc[:, : min(n_samp, total_samps)]

        duck_db = -6.0
        min_duck_lin: Optional[float] = 0.36
        try:
            from vibevoice.config import config as _cfg

            duck_db = float(getattr(_cfg, "PRODUCTION_MUSIC_DUCK_DB", duck_db))
            min_duck_lin = getattr(_cfg, "PRODUCTION_MUSIC_DUCK_MIN_LINEAR", None)
            if min_duck_lin is not None:
                min_duck_lin = float(min_duck_lin)
        except Exception:
            pass

        overrides = asset_path_overrides or {}

        for tr in plan.tracks:
            role = str(getattr(tr, "track_role", "") or "")
            if role == "voice_main":
                continue
            for ev in tr.events:
                ref = getattr(ev, "asset_ref", None)
                aid = getattr(ref, "asset_id", None) if ref else None
                if not aid:
                    continue
                apath = self._resolve_asset_path(str(aid), library, overrides)
                if not apath.is_file():
                    logger.warning("Skip missing asset: %s", apath)
                    continue
                ev_samples, ev_sr = _read_audio_float(apath)
                ev_samples = _to_stereo(ev_samples)
                ev_samples = _resample_linear(ev_samples, ev_sr, sr)
                dur = int(ev.duration_ms)
                ev_samples = ev_samples[:, : min(ev_samples.shape[1], int(dur * sr / 1000.0))]
                if ev_samples.shape[1] == 0:
                    logger.warning("Skip empty asset audio after trim/resample: %s", apath)
                    continue
                ev_samples = _apply_fades(ev_samples, int(ev.fade_in_ms), int(ev.fade_out_ms))
                g_db = float(ev.volume_db or 0.0)
                g_lin_scalar = float(10.0 ** (g_db / 20.0))
                n_ev = ev_samples.shape[1]
                auto = getattr(ev, "automation", None) or []
                bp: List[Tuple[int, float]] = []
                if auto:
                    for a in auto:
                        bp.append((int(a.offset_ms * sr / 1000.0), float(a.volume_db)))
                auto_db = interp_automation_linear(n_ev, bp, base_db=0.0)
                auto_lin = _db_to_linear(auto_db.astype(np.float64)).astype(np.float32)

                if role == "sfx_ambience":
                    theta = np.linspace(0.0, 2.0 * np.pi, n_ev, endpoint=False, dtype=np.float64)
                    pan_curve = 0.3 * np.sin(theta)
                    ang = (pan_curve + 1.0) * (np.pi / 4.0)
                    gl = np.cos(ang).astype(np.float32)
                    gr = np.sin(ang).astype(np.float32)
                else:
                    pan = float(getattr(ev, "pan", 0.0) or 0.0)
                    gl_s, gr_s = _pan_gains(pan)
                    gl = np.full(n_ev, gl_s, dtype=np.float32)
                    gr = np.full(n_ev, gr_s, dtype=np.float32)
                ev_samples[0, :] *= gl * g_lin_scalar * auto_lin
                ev_samples[1, :] *= gr * g_lin_scalar * auto_lin

                start = max(0, int(ev.start_ms * sr / 1000.0))
                end = min(total_samps, start + n_ev)
                if end <= start:
                    continue

                region = slice(start, end)
                mix_slice = ev_samples[:, : end - start]

                if role in ("music_bed", "music_transition", "music_intro", "music_outro"):
                    seg_len = mix_slice.shape[1]
                    if seg_len == 0:
                        continue
                    vad_seg = vad[region]
                    if vad_seg.shape[0] == 0:
                        # No speech timeline in this region (e.g., cue after VO ends): keep full music level.
                        vad_seg = np.zeros(seg_len, dtype=np.float32)
                    elif vad_seg.shape[0] != seg_len:
                        vad_seg = np.interp(
                            np.linspace(0, 1, seg_len),
                            np.linspace(0, 1, vad_seg.shape[0]),
                            vad_seg.astype(np.float64),
                        ).astype(np.float32)
                    duck = build_duck_gain_linear(
                        seg_len,
                        vad_seg,
                        sr=sr,
                        duck_db=duck_db,
                        min_duck_linear=min_duck_lin,
                    )
                    mix_slice = mix_slice * duck[np.newaxis, :]

                master[:, region] += mix_slice

        from app.services.genre_templates import merge_voice_chain_params

        _vc_full = merge_voice_chain_params(
            (getattr(genre_template, "voice_chain_overrides", None) or {})
            if genre_template is not None
            else {}
        )
        if genre_template is not None:
            mt = getattr(genre_template, "mastering_targets", None) or {}
            if mt.get("peak_db") is not None:
                _vc_full["master_limiter_threshold_db"] = float(mt["peak_db"])
        master_pb = _master_chain_from_params(_vc_full)
        processed = master_pb(master.astype(np.float64), sr)
        if processed.ndim == 1:
            processed = np.stack([processed, processed])
        proc = np.asarray(processed, dtype=np.float32)
        if proc.shape[0] != 2:
            proc = _to_stereo(proc.reshape(1, -1))

        from app.services.genre_templates import mastering_lufs

        target = mastering_lufs(genre_template, str(getattr(plan, "genre", "") or ""))
        proc_interleaved = np.zeros((proc.shape[1], 2), dtype=np.float64)
        proc_interleaved[:, 0] = proc[0]
        proc_interleaved[:, 1] = proc[1]
        meter = pyln.Meter(sr)
        try:
            loud = meter.integrated_loudness(proc_interleaved)
            proc_interleaved = pyln.normalize.loudness(proc_interleaved, loud, target)
        except Exception as exc:
            logger.warning("pyloudnorm failed (%s); skipping LUFS step", exc)

        out2 = proc_interleaved.T.astype(np.float32)

        if _vibe_config is not None:
            _vibe_config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        tmp_wav = Path(tempfile.gettempdir()) / f"{stamp}_pmix_{uuid.uuid4().hex[:8]}.wav"
        sf.write(str(tmp_wav), out2.T, sr, subtype="PCM_24")

        out_mp3 = Path(output_path)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        meta = [
            "-metadata",
            f"title={id3_title or 'Podcast'}",
            "-metadata",
            f"genre={id3_genre or str(plan.genre) or 'Podcast'}",
        ]
        if episode_number is not None:
            meta.extend(["-metadata", f"track={episode_number}"])
        dur_sec = out2.shape[1] / float(sr)
        meta.extend(["-metadata", f"comment=duration_s={dur_sec:.2f}"])
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            *meta,
            str(out_mp3),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg mp3 failed: %s", exc.stderr.decode(errors="ignore") if exc.stderr else exc)
            raise RuntimeError("ffmpeg MP3 export failed") from exc
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except OSError:
                pass

        self._assert_not_truncated(out_mp3, expected_min_ms)
        return str(out_mp3.resolve())

    def _resolve_asset_path(self, asset_id: str, library: Any, overrides: Dict[str, str]) -> Path:
        if asset_id in overrides:
            return Path(overrides[asset_id])
        return library.resolve_path(asset_id)

    def _process_voice_tracks(
        self,
        samples_stereo: np.ndarray,
        sr: int,
        plan: Any,
        voice_line_timing_ms: Optional[List[Tuple[int, float, float]]],
        genre_template: Any = None,
    ) -> np.ndarray:
        n = samples_stereo.shape[1]
        mono = _mono_from_stereo(samples_stereo).astype(np.float32)
        vdir = list(getattr(plan, "voice_direction", []) or [])
        if voice_line_timing_ms and vdir:
            mono = _apply_line_energy_matching(mono, sr, vdir, voice_line_timing_ms)

        from app.services.genre_templates import merge_voice_chain_params

        vc_base: Optional[Dict[str, float]] = None
        if genre_template is not None and getattr(genre_template, "voice_chain_overrides", None):
            vc_base = merge_voice_chain_params(dict(genre_template.voice_chain_overrides))

        if not vdir:
            out_mono = np.zeros(n, dtype=np.float32)
            board = _speaker_chain(1, vc_base)
            chunk = mono[np.newaxis, :]
            proc = board(chunk.astype(np.float64), float(sr))
            pr = np.asarray(proc, dtype=np.float32).reshape(-1)
            if pr.shape[0] >= n:
                out_mono = pr[:n]
            else:
                out_mono[: pr.shape[0]] = pr
            return np.vstack([out_mono, out_mono])

        # Dry TTS baseline: only aligned line ranges get per-speaker processing; everything
        # else stays from the source buffer (Whisper timing often does not span full TTS length).
        out_mono = mono.copy()

        ranges: List[Tuple[int, int, int]] = []
        if voice_line_timing_ms:
            for line_idx, st_ms, en_ms in voice_line_timing_ms:
                s = int(st_ms * sr / 1000.0)
                e = int(en_ms * sr / 1000.0)
                s = max(0, min(s, n))
                e = max(s, min(e, n))
                sp = _parse_speaker_index(
                    next((v.speaker for v in vdir if v.line_index == line_idx), "Speaker 1")
                )
                ranges.append((sp, s, e))
        else:
            chunk_len = n // max(len(vdir), 1)
            for i, v in enumerate(vdir):
                s = i * chunk_len
                e = n if i == len(vdir) - 1 else min(n, (i + 1) * chunk_len)
                sp = _parse_speaker_index(v.speaker)
                ranges.append((sp, s, e))

        for sp, s, e in ranges:
            if e <= s:
                continue
            board = _speaker_chain(sp, vc_base)
            chunk = mono[s:e][np.newaxis, :]
            proc = board(chunk.astype(np.float64), float(sr))
            pr = np.asarray(proc, dtype=np.float32).reshape(-1)
            ln = e - s
            if pr.shape[0] < ln:
                pr = np.pad(pr, (0, ln - pr.shape[0]))
            elif pr.shape[0] > ln:
                pr = pr[:ln]
            out_mono[s:e] = pr

        vm = np.max(np.abs(out_mono))
        if vm > 1.0:
            out_mono /= vm
        return np.vstack([out_mono, out_mono])

    @staticmethod
    def _assert_not_truncated(mp3_path: Path, expected_min_duration_ms: int) -> None:
        try:
            from pydub import AudioSegment

            dur_ms = len(AudioSegment.from_file(str(mp3_path)))
        except Exception as exc:
            logger.warning("Could not verify MP3 duration: %s", exc)
            return
        if dur_ms < max(expected_min_duration_ms - 3000, 1000):
            raise RuntimeError(
                f"Production mix appears truncated: rendered={dur_ms}ms expected>={expected_min_duration_ms}ms"
            )


def legacy_cues_to_production_plan(
    *,
    voice_duration_seconds: float,
    cues: Sequence[Any],
) -> Tuple[Any, Dict[str, str]]:
    """Build a minimal ``ProductionPlan`` + asset path overrides for legacy ``CuePlacement`` mixing."""
    from app.services.production_director import (
        AssetRef,
        EmotionalArcPoint,
        ProductionPlan,
        TimelineTrack,
        TrackEvent,
    )

    overrides: Dict[str, str] = {}
    tracks: List[Any] = []
    eid = 0
    for c in cues:
        ct = getattr(c, "cue_type", "")
        fp = str(getattr(c, "file_path", "") or "")
        if not fp or ct == "dialogue":
            continue
        aid = f"legacy_{ct}_{eid}"
        overrides[aid] = fp
        pos = int(getattr(c, "position_ms", 0) or 0)
        vol = float(getattr(c, "volume_db", 0.0) or 0.0)
        dur_ms = getattr(c, "duration_ms", None)
        if dur_ms is None:
            try:
                x, srr = _read_audio_float(Path(fp))
                dur_ms = int(1000.0 * x.shape[1] / float(srr))
            except Exception:
                dur_ms = 5000
        if ct == "bed":
            tr_role = "music_bed"
        elif ct == "outro":
            tr_role = "music_outro"
        elif ct == "intro":
            tr_role = "music_intro"
        else:
            tr_role = "music_transition"
        ev = TrackEvent(
            event_id=f"e{eid}",
            start_ms=pos,
            duration_ms=int(dur_ms),
            asset_ref=AssetRef(asset_id=aid),
            volume_db=vol,
            pan=0.0,
            fade_in_ms=0,
            fade_out_ms=0,
        )
        tracks.append(TimelineTrack(track_id=f"legacy_{eid}", track_role=tr_role, events=[ev]))
        eid += 1

    d = max(1.0, float(voice_duration_seconds))
    plan = ProductionPlan(
        episode_id=str(uuid.uuid4()),
        duration_target_seconds=d,
        genre="General",
        emotional_arc=[
            EmotionalArcPoint(timestamp=0.0, valence=0.0, energy=0.5),
            EmotionalArcPoint(timestamp=d / 2.0, valence=0.0, energy=0.5),
            EmotionalArcPoint(timestamp=d, valence=0.0, energy=0.5),
        ],
        tracks=tracks,
        voice_direction=[],
    )
    return plan, overrides

