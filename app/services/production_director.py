"""
Production planning layer: maps scripts, segments, and assets to a validated ProductionPlan.

Uses Ollama as an audio-director persona; falls back to deterministic layout when the model
output is unusable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

TrackRole = Literal[
    "voice_main",
    "voice_backchannel",
    "music_bed",
    "music_transition",
    "sfx_impact",
    "sfx_riser",
    "sfx_whoosh",
    "sfx_ambience",
    "sfx_laugh",
    "sfx_reveal",
    "foley",
    "voice_backchannel",
]

MusicDensity = Literal["low", "medium", "high"]


class EmotionalArcPoint(BaseModel):
    """Valence / energy waypoint on the episode timeline."""

    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(description="Seconds from episode start")
    valence: float = Field(ge=-1.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)


class AutomationBreakpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_ms: int = Field(ge=0)
    volume_db: float


class AssetRef(BaseModel):
    """Library asset id or a generation prompt for ACE-Step / SAO-style fallbacks."""

    model_config = ConfigDict(extra="forbid")

    asset_id: Optional[str] = None
    generation_prompt: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self) -> AssetRef:
        if self.asset_id and self.generation_prompt:
            raise ValueError("asset_ref must set at most one of asset_id or generation_prompt")
        return self


class TrackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    asset_ref: Optional[AssetRef] = None
    volume_db: float = 0.0
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    fade_in_ms: int = Field(default=0, ge=0)
    fade_out_ms: int = Field(default=0, ge=0)
    automation: Optional[List[AutomationBreakpoint]] = None
    trigger_word: Optional[str] = None
    anchor_speaker: Optional[str] = Field(
        default=None,
        description="For voice_backchannel: main speaker line whose pause hosts the reaction (e.g. Speaker 1).",
    )


class TimelineTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    track_role: TrackRole
    events: List[TrackEvent] = Field(default_factory=list)


class VoiceDirectionLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_index: int = Field(ge=0)
    speaker: str
    style: str = "natural"
    emotion: str = "neutral"
    emphasis_words: List[str] = Field(default_factory=list)
    pause_after_ms: int = Field(default=0, ge=0)
    reverb_send: float = Field(default=0.0, ge=0.0, le=1.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)


class ProductionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    duration_target_seconds: float = Field(gt=0)
    genre: str
    emotional_arc: List[EmotionalArcPoint]
    tracks: List[TimelineTrack]
    voice_direction: List[VoiceDirectionLine]


class GenreRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    music_density: MusicDensity
    sfx_density: MusicDensity
    pacing_notes: str
    reverb_style: str
    example_patterns: List[Dict[str, Any]] = Field(
        min_length=2,
        max_length=3,
        description="Few-shot mini production plans for this genre",
    )


def _compact_asset_library(assets: List[Dict[str, Any]], max_items: int = 80) -> List[Dict[str, Any]]:
    """Shape a small JSON-safe summary for the Director prompt."""
    out: List[Dict[str, Any]] = []
    for raw in assets[:max_items]:
        if not isinstance(raw, dict):
            continue
        aid = raw.get("asset_id") or raw.get("id")
        tags = raw.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]
        out.append(
            {
                "asset_id": aid,
                "tags": tags or [],
                "duration": raw.get("duration"),
                "mood": raw.get("mood"),
                "bpm": raw.get("bpm"),
            }
        )
    return out


def extract_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull a single JSON object from model output: strips fences, preamble, trailing junk.
    Returns None if no object can be parsed.
    """
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str: Optional[str] = None
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = cleaned[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return _try_loose_json_object(chunk)
    return None


def _try_loose_json_object(chunk: str) -> Optional[Dict[str, Any]]:
    """Last-resort: trim to last complete brace or append closing braces."""
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Truncate trailing incomplete content after last balanced }
    fixed = chunk.rstrip()
    while fixed and fixed[-1] not in ("}", '"'):
        fixed = fixed[:-1]
    if fixed.endswith(","):
        fixed = fixed[:-1]
    if not fixed.endswith("}"):
        fixed = fixed + "}"
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


_SPEAKER_LINE = re.compile(r"^(Speaker\s+\d+):\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _fallback_segments_from_script(script: str) -> List[Dict[str, Any]]:
    """Deterministic segments when upstream segmentation is unavailable (mirrors podcast_generator)."""
    segments: List[Dict[str, Any]] = [
        {
            "segment_type": "intro_music",
            "speaker": None,
            "text": None,
            "start_time_hint": 0.0,
            "duration_hint": 5.0,
            "energy_level": "high",
            "notes": None,
        }
    ]
    current_time = 2.0
    for raw_line in script.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE.match(line)
        if not match:
            continue
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        if not text:
            continue
        word_count = max(len(text.split()), 1)
        dur_hint = round((word_count / 140.0) * 60.0, 2)
        segments.append(
            {
                "segment_type": "dialogue",
                "speaker": speaker,
                "text": text,
                "start_time_hint": round(current_time, 2),
                "duration_hint": dur_hint,
                "energy_level": "medium",
                "notes": None,
            }
        )
        current_time += max((word_count / 2.6), 1.0)

    if len(segments) > 3:
        midpoint = round(current_time / 2.0, 2)
        segments.insert(
            2,
            {
                "segment_type": "transition_sting",
                "speaker": None,
                "text": None,
                "start_time_hint": midpoint,
                "duration_hint": 2.0,
                "energy_level": "medium",
                "notes": None,
            },
        )
    segments.append(
        {
            "segment_type": "outro_music",
            "speaker": None,
            "text": None,
            "start_time_hint": round(max(current_time - 2.0, 0.0), 2),
            "duration_hint": 8.0,
            "energy_level": "low",
            "notes": None,
        }
    )
    for idx, seg in enumerate(segments, start=1):
        seg["segment_id"] = idx
    return segments


def _energy_to_arc(energy: Optional[str]) -> Tuple[float, float]:
    e = (energy or "medium").lower()
    if e == "high":
        return 0.45, 0.85
    if e == "low":
        return -0.15, 0.35
    return 0.05, 0.55


def _parse_dialogue_lines(script: str) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for raw in script.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            lines.append((m.group(1).strip(), m.group(2).strip()))
    return lines


def _infer_duration_seconds(
    segments: List[Dict[str, Any]],
    timing_hints: List[Dict[str, Any]],
) -> float:
    max_end = 0.0
    for seg in segments:
        try:
            st = float(seg.get("start_time_hint") or 0.0)
            du = float(seg.get("duration_hint") or 0.0)
            max_end = max(max_end, st + du)
        except (TypeError, ValueError):
            continue
    for hint in timing_hints:
        if not isinstance(hint, dict):
            continue
        for key in ("end_ms", "end_s", "end"):
            if key in hint:
                try:
                    v = float(hint[key])
                    if key == "end_ms":
                        max_end = max(max_end, v / 1000.0)
                    else:
                        max_end = max(max_end, v)
                except (TypeError, ValueError):
                    pass
    return max(30.0, max_end)


def build_fallback_production_plan(
    *,
    script: str,
    script_segments: List[Dict[str, Any]],
    genre: str,
    episode_id: str,
    timing_hints: Optional[List[Dict[str, Any]]] = None,
) -> ProductionPlan:
    """
    Build a ProductionPlan from segment dicts (or script-only segmentation), mirroring
    `_fallback_segments_from_script` timing heuristics.
    """
    segments = script_segments if script_segments else _fallback_segments_from_script(script)
    hints = timing_hints if timing_hints is not None else []
    duration_target_seconds = _infer_duration_seconds(segments, hints)

    emotional_arc: List[EmotionalArcPoint] = []
    for seg in segments:
        st = float(seg.get("start_time_hint") or 0.0)
        valence, energy = _energy_to_arc(seg.get("energy_level"))
        emotional_arc.append(EmotionalArcPoint(timestamp=st, valence=valence, energy=energy))

    voice_lines = _parse_dialogue_lines(script)
    voice_direction: List[VoiceDirectionLine] = []
    dialogue_seg_energy: List[str] = []
    for seg in segments:
        if str(seg.get("segment_type") or "").lower() == "dialogue":
            dialogue_seg_energy.append(str(seg.get("energy_level") or "medium"))

    for i, (speaker, _text) in enumerate(voice_lines):
        emo = "neutral"
        if i < len(dialogue_seg_energy):
            emo = "intense" if dialogue_seg_energy[i] == "high" else "warm" if dialogue_seg_energy[i] == "low" else "neutral"
        voice_direction.append(
            VoiceDirectionLine(
                line_index=i,
                speaker=speaker,
                style="conversational",
                emotion=emo,
                emphasis_words=[],
                pause_after_ms=120 if i < len(voice_lines) - 1 else 200,
                reverb_send=0.08,
                pan=0.0,
            )
        )

    voice_events: List[TrackEvent] = []
    music_events: List[TrackEvent] = []
    sfx_events: List[TrackEvent] = []
    ev = 0
    for seg in segments:
        st = seg.get("start_time_hint")
        du = seg.get("duration_hint")
        try:
            st_f = float(st or 0.0)
            du_f = float(du or 0.0)
        except (TypeError, ValueError):
            continue
        start_ms = int(max(0.0, st_f) * 1000)
        duration_ms = int(max(0.0, du_f) * 1000)
        seg_type = str(seg.get("segment_type") or "").lower()
        ev += 1
        eid = f"evt_{ev}"
        if seg_type == "dialogue":
            voice_events.append(
                TrackEvent(
                    event_id=eid,
                    start_ms=start_ms,
                    duration_ms=max(duration_ms, 1),
                    asset_ref=None,
                    volume_db=0.0,
                    pan=0.0,
                    fade_in_ms=20,
                    fade_out_ms=40,
                )
            )
        elif seg_type == "intro_music":
            music_events.append(
                TrackEvent(
                    event_id=eid,
                    start_ms=start_ms,
                    duration_ms=max(duration_ms, 500),
                    asset_ref=AssetRef(generation_prompt=f"{genre} podcast cold open — warm, inviting, NPR-style clarity"),
                    volume_db=-8.0,
                    pan=0.0,
                    fade_in_ms=400,
                    fade_out_ms=800,
                )
            )
        elif seg_type == "outro_music":
            music_events.append(
                TrackEvent(
                    event_id=eid,
                    start_ms=start_ms,
                    duration_ms=max(duration_ms, 500),
                    asset_ref=AssetRef(generation_prompt=f"{genre} episode resolve — gentle cadence, room tone tail"),
                    volume_db=-10.0,
                    pan=0.0,
                    fade_in_ms=600,
                    fade_out_ms=1200,
                )
            )
        elif seg_type == "transition_sting":
            sfx_events.append(
                TrackEvent(
                    event_id=eid,
                    start_ms=start_ms,
                    duration_ms=max(duration_ms, 200),
                    asset_ref=AssetRef(generation_prompt="short tonal rise, subtle tape noise, 1–2 seconds"),
                    volume_db=-6.0,
                    pan=0.0,
                    fade_in_ms=10,
                    fade_out_ms=120,
                )
            )
        elif seg_type in ("music_bed_in", "music_bed_out", "music_bed"):
            music_events.append(
                TrackEvent(
                    event_id=eid,
                    start_ms=start_ms,
                    duration_ms=max(duration_ms, 1),
                    asset_ref=None,
                    volume_db=-14.0,
                    pan=0.0,
                )
            )

    tracks: List[TimelineTrack] = [
        TimelineTrack(track_id="tr_voice_main", track_role="voice_main", events=voice_events),
        TimelineTrack(track_id="tr_music_bed", track_role="music_bed", events=music_events),
    ]
    if sfx_events:
        tracks.append(TimelineTrack(track_id="tr_sfx", track_role="sfx_riser", events=sfx_events))

    return ProductionPlan(
        episode_id=episode_id,
        duration_target_seconds=duration_target_seconds,
        genre=genre,
        emotional_arc=emotional_arc,
        tracks=tracks,
        voice_direction=voice_direction,
    )


GENRE_RULES: Dict[str, GenreRule] = {
    "News": GenreRule(
        music_density="low",
        sfx_density="low",
        pacing_notes=(
            "NPR-style clarity: music sits under narration, never competes; "
            "hard commits on facts; light room tone; avoid sensational stings."
        ),
        reverb_style="dry booth, short plate under VO only for scene recall",
        example_patterns=[
            {
                "episode_id": "fewshot-news-1",
                "duration_target_seconds": 600.0,
                "genre": "News",
                "tracks_hint": [
                    {"track_role": "music_bed", "note": "sparse piano under host intro only"},
                    {"track_role": "voice_main", "note": "forward, intimate proximity"},
                ],
            },
            {
                "episode_id": "fewshot-news-2",
                "duration_target_seconds": 900.0,
                "genre": "News",
                "tracks_hint": [
                    {"track_role": "sfx_ambience", "note": "quiet newsroom bed -24 LUFS under VO"},
                    {"track_role": "music_transition", "note": "single 1.5s harmonic bridge between blocks"},
                ],
            },
            {
                "episode_id": "fewshot-news-3",
                "duration_target_seconds": 420.0,
                "genre": "News",
                "tracks_hint": [
                    {"track_role": "voice_main", "note": "two-way interview: pan guest slightly R"},
                    {"track_role": "music_bed", "note": "duck -6 dB on emphasized stats"},
                ],
            },
        ],
    ),
    "True Crime": GenreRule(
        music_density="medium",
        sfx_density="high",
        pacing_notes=(
            "Ira Glass-style scoring: restrained motifs that accrue tension; "
            "slow reveals; low drones under exposition; sting only on turns."
        ),
        reverb_style="dark plate on narration; keep dialogue drier for realism",
        example_patterns=[
            {
                "episode_id": "fewshot-tc-1",
                "duration_target_seconds": 1200.0,
                "genre": "True Crime",
                "tracks_hint": [
                    {"track_role": "music_bed", "note": "pulse at 72 BPM, minor, under cold open"},
                    {"track_role": "sfx_riser", "note": "subtle 4s riser before reveal lines"},
                ],
            },
            {
                "episode_id": "fewshot-tc-2",
                "duration_target_seconds": 900.0,
                "genre": "True Crime",
                "tracks_hint": [
                    {"track_role": "sfx_ambience", "note": "rain/room tone low; sidechain to VO"},
                    {"track_role": "music_transition", "note": "one sting per act break, not per sentence"},
                ],
            },
            {
                "episode_id": "fewshot-tc-3",
                "duration_target_seconds": 780.0,
                "genre": "True Crime",
                "tracks_hint": [
                    {"track_role": "sfx_reveal", "note": "single low thud on ID reveal"},
                    {"track_role": "voice_main", "note": "whisper-to-close proximity on payoff line"},
                ],
            },
        ],
    ),
    "Comedy": GenreRule(
        music_density="medium",
        sfx_density="medium",
        pacing_notes=(
            "Snappy edits; music hits punchlines; avoid clutter under rapid banter; "
            "use foley for physical beats sparingly."
        ),
        reverb_style="bright room, short decay; slap only on host tags",
        example_patterns=[
            {
                "episode_id": "fewshot-com-1",
                "duration_target_seconds": 540.0,
                "genre": "Comedy",
                "tracks_hint": [
                    {"track_role": "voice_main", "note": "double-host ping-pong; tight crossfade"},
                    {"track_role": "music_transition", "note": "brass stab on segment return"},
                ],
            },
            {
                "episode_id": "fewshot-com-2",
                "duration_target_seconds": 660.0,
                "genre": "Comedy",
                "tracks_hint": [
                    {"track_role": "foley", "note": "single exaggerated swoosh per sketch beat"},
                    {"track_role": "music_bed", "note": "light funk bed -18 dB under ad reads only"},
                ],
            },
            {
                "episode_id": "fewshot-com-3",
                "duration_target_seconds": 480.0,
                "genre": "Comedy",
                "tracks_hint": [
                    {"track_role": "sfx_impact", "note": "rimshot-level hit on callback word"},
                ],
            },
        ],
    ),
    "General": GenreRule(
        music_density="medium",
        sfx_density="low",
        pacing_notes="Balanced podcast mix: clear voice priority, music supports arc, SFX only for emphasis.",
        reverb_style="medium plate on VO; keep music wide, voice mono-compatible",
        example_patterns=[
            {
                "episode_id": "fewshot-gen-1",
                "duration_target_seconds": 600.0,
                "genre": "General",
                "tracks_hint": [{"track_role": "voice_main", "note": "default single-host arc"}],
            },
            {
                "episode_id": "fewshot-gen-2",
                "duration_target_seconds": 720.0,
                "genre": "General",
                "tracks_hint": [
                    {"track_role": "music_bed", "note": "intro/outro only unless story demands"},
                    {"track_role": "music_transition", "note": "mid episode bridge at 50% time"},
                ],
            },
        ],
    ),
}


def _genre_rule_for(genre: str) -> GenreRule:
    g = (genre or "").strip()
    if g in GENRE_RULES:
        return GENRE_RULES[g]
    return GENRE_RULES["General"]


_VOICE_DIRECTION_GENRE_GUIDE = """
VOICE_DIRECTION (required — one object per dialogue line in script order, same count as speaker lines):
- emotion must be one of: neutral | excited | somber | curious | confident | amused | tense | warm
- emphasis_words: important tokens to stress (product names, numbers, punch words); empty list if none
- pause_after_ms: trailing silence after the line (0 allowed)

Genre-specific delivery targets (few-shot intent):
- tech_talk / Tech: mostly neutral and curious; use excited sparingly on product names and launches; short pauses (80–180ms).
- news / News: neutral and confident; never amused; crisp pacing; pause_after_ms typically 60–120ms.
- storytelling / Storytelling: wide emotional range; warm baseline; longer pauses for beats (150–400ms) on turns of phrase.
- true_crime / True Crime: somber or tense baseline; spike pause_after_ms (250–600ms) before/after reveals.
- comedy / Comedy: amused or excited baseline; fast banter; pause_after_ms often 0–80ms between rapid lines.

"""


def _build_director_system_prompt(genre: str, genre_template: Optional[Any] = None) -> str:
    if genre_template is not None:
        dr = genre_template.director_rules
        examples_json = json.dumps(dr.get("example_patterns", []), ensure_ascii=False, indent=2)
        sfx_line = ""
        cats = getattr(genre_template, "sfx_enabled_categories", None)
        if cats is not None:
            sfx_line = (
                "\nALLOWED SFX ASSET CATEGORIES (do not rely on disabled types such as laughs in true_crime): "
                + ", ".join(sorted(cats))
                + "\n"
            )
        genre_rules_block = f"""GENRE TEMPLATE ({genre_template.display_name}) — must obey:
- music_density: {dr.get("music_density")}
- sfx_density: {dr.get("sfx_density")}
- pacing_notes: {dr.get("pacing_notes")}
- reverb_style: {dr.get("reverb_style")}
{sfx_line}
Few-shot production patterns for this genre (structure and intent, not literal copy):
{examples_json}
"""
    else:
        rule = _genre_rule_for(genre)
        examples_json = json.dumps(rule.example_patterns, ensure_ascii=False, indent=2)
        genre_rules_block = f"""GENRE RULES (must obey):
- music_density: {rule.music_density}
- sfx_density: {rule.sfx_density}
- pacing_notes: {rule.pacing_notes}
- reverb_style: {rule.reverb_style}

Few-shot production patterns for this genre (structure and intent, not literal copy):
{examples_json}
"""

    return f"""You are a seasoned podcast audio director and mix engineer — think Ira Glass on story scoring, NPR broadcast clarity for news, and premium true-crime tension design where applicable. You output ONE JSON object that validates against the ProductionPlan schema — no markdown, no commentary.

{genre_rules_block}

{_VOICE_DIRECTION_GENRE_GUIDE.strip()}

OUTPUT SCHEMA (field names exact):
{{
  "episode_id": string,
  "duration_target_seconds": number,
  "genre": string,
  "emotional_arc": [ {{ "timestamp": number, "valence": number (-1..1), "energy": number (0..1) }} ],
  "tracks": [
    {{
      "track_id": string,
      "track_role": one of "voice_main"|"voice_backchannel"|"music_bed"|"music_transition"|"sfx_impact"|"sfx_riser"|"sfx_ambience"|"sfx_reveal"|"foley",
      "events": [
        {{
          "event_id": string,
          "start_ms": integer,
          "duration_ms": integer,
          "asset_ref": null | {{ "asset_id": string }} | {{ "generation_prompt": string }},
          "volume_db": number,
          "pan": number (-1..1),
          "fade_in_ms": integer,
          "fade_out_ms": integer,
          "automation": null | [ {{ "offset_ms": integer, "volume_db": number }} ],
          "trigger_word": null | string,
          "anchor_speaker": null | string
        }}
      ]
    }}
  ],
  "voice_direction": [
    {{
      "line_index": integer,
      "speaker": string,
      "style": string,
      "emotion": string (one of neutral|excited|somber|curious|confident|amused|tense|warm),
      "emphasis_words": [ string ],
      "pause_after_ms": integer,
      "reverb_send": number (0..1),
      "pan": number (-1..1)
    }}
  ]
}}

Rules:
1) Align start_ms roughly to WhisperX timing hints; never invent wild offsets if hints exist.
2) Prefer library asset_id from the provided catalog when mood/BPM fits; otherwise use generation_prompt on asset_ref (never both).
3) Keep voice_main events aligned to dialogue; music and sfx must duck under voice (-10 to -18 dB typical for beds).
4) emotional_arc should have at least 3 points spanning the episode.
5) One voice_direction row per spoken dialogue line (same count and order as script dialogue lines).

TRIGGER_WORD (exact script tokens from word_index_compact — match spelling/casing as spoken):
- sfx_impact: set trigger_word to a stressed/emphasis word; start_ms will snap to that word's start at mix time.
- sfx_riser: set trigger_word to the reveal/punch word; the riser will END 100ms BEFORE that word starts (build tension into the word).
- sfx_whoosh: set trigger_word to a word at a topic/segment boundary; the whoosh centers on the silence gap before that line.
- voice_backchannel: short reactions ("mm-hmm", "right", "exactly", "yeah", "wow") overlapping the other speaker. Set trigger_word to that phrase, and anchor_speaker to the MAIN speaker (the one being reacted to) so the mix can find a ≥400ms pause in their line; place asset_id from voice_backchannel category.

Respond with the JSON object only."""


def _build_user_prompt(
    script: str,
    timing_hints: List[Dict[str, Any]],
    asset_summary: List[Dict[str, Any]],
    word_index_compact: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "whisperx_timing_hints": timing_hints,
        "asset_library_summary": asset_summary,
        "script": script,
    }
    if word_index_compact:
        payload["word_index_compact"] = word_index_compact
    return (
        "Plan the episode using the following data. Respect timing hints; use word_index_compact "
        "for trigger_word anchors (exact tokens as aligned).\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


_ASSET_CATEGORY_ENUM = [
    "music_bed",
    "music_transition",
    "music_intro",
    "music_outro",
    "sfx_impact",
    "sfx_riser",
    "sfx_whoosh",
    "sfx_ambience",
    "sfx_laugh",
    "foley",
    "sfx_reveal",
]

DIRECTOR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_assets",
            "description": (
                "Search the local indexed audio library for music beds, transitions, intros/outros, "
                "or SFX. Use results to pick asset_id values in the final ProductionPlan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": _ASSET_CATEGORY_ENUM},
                    "genre": {"type": "string", "description": "e.g. news, true_crime, comedy"},
                    "mood": {"type": "string"},
                    "bpm_min": {"type": "integer"},
                    "bpm_max": {"type": "integer"},
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "min_duration_ms": {"type": "integer"},
                    "max_duration_ms": {"type": "integer"},
                    "limit": {"type": "integer", "description": "max rows (default 10)"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_generation",
            "description": (
                "Queue ACE-Step / Stable-Audio generation when no library asset fits. "
                "Executed after planning; returns a queue id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": _ASSET_CATEGORY_ENUM},
                    "prompt": {"type": "string"},
                    "duration_ms": {"type": "integer"},
                    "genre": {"type": "string"},
                    "mood": {"type": "string"},
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["category", "prompt", "duration_ms", "genre", "mood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_event",
            "description": (
                "Optional scratchpad: record a timeline event (for your reasoning). "
                "The final ProductionPlan JSON must still list all tracks and events explicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_role": {
                        "type": "string",
                        "enum": [
                            "voice_main",
                            "voice_backchannel",
                            "music_bed",
                            "music_transition",
                            "sfx_impact",
                            "sfx_riser",
                            "sfx_whoosh",
                            "sfx_ambience",
                            "sfx_laugh",
                            "sfx_reveal",
                            "foley",
                        ],
                    },
                    "start_ms": {"type": "integer"},
                    "duration_ms": {"type": "integer"},
                    "asset_ref": {
                        "type": "object",
                        "properties": {
                            "asset_id": {"type": "string"},
                            "generation_prompt": {"type": "string"},
                        },
                    },
                    "volume_db": {"type": "number"},
                    "pan": {"type": "number", "minimum": -1, "maximum": 1},
                    "fade_in_ms": {"type": "integer"},
                    "fade_out_ms": {"type": "integer"},
                    "trigger_word": {"type": "string"},
                    "anchor_speaker": {
                        "type": "string",
                        "description": "For voice_backchannel: main speaker label (e.g. Speaker 1).",
                    },
                },
                "required": [
                    "track_role",
                    "start_ms",
                    "duration_ms",
                    "volume_db",
                    "pan",
                    "fade_in_ms",
                    "fade_out_ms",
                ],
            },
        },
    },
]


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        return json.loads(s)
    return {}


def _director_dispatch_tool(
    name: str,
    raw_args: Any,
    library: Any,
    generation_queue: Any,
    placed_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    args = _parse_tool_arguments(raw_args)
    if name == "search_assets":
        cat = str(args.get("category") or "")
        if cat not in _ASSET_CATEGORY_ENUM:
            return {"error": f"invalid category: {cat}", "assets": []}
        lim = int(args.get("limit") or 10)
        bpm_range = None
        if args.get("bpm_min") is not None or args.get("bpm_max") is not None:
            bpm_range = (args.get("bpm_min"), args.get("bpm_max"))
        intensity = args.get("intensity")
        if intensity is not None:
            intensity = int(intensity)
        rows = library.search(
            cat,  # type: ignore[arg-type]
            genre=args.get("genre"),
            mood=args.get("mood"),
            min_duration_ms=args.get("min_duration_ms"),
            max_duration_ms=args.get("max_duration_ms"),
            bpm_range=bpm_range,
            intensity=intensity,
            limit=min(lim, 25),
        )
        compact = []
        for a in rows:
            compact.append(
                {
                    "asset_id": a.asset_id,
                    "category": a.category,
                    "genre_tags": a.genre_tags,
                    "mood_tags": a.mood_tags,
                    "bpm": a.bpm,
                    "duration_ms": a.duration_ms,
                    "intensity": a.intensity,
                    "key": a.key,
                }
            )
        return {"assets": compact, "count": len(compact)}

    if name == "request_generation":
        rid = generation_queue.enqueue_from_tool(
            category=str(args.get("category")),
            prompt=str(args.get("prompt") or ""),
            duration_ms=int(args.get("duration_ms") or 5000),
            genre=str(args.get("genre") or "news"),
            mood=str(args.get("mood") or "neutral"),
            intensity=int(args.get("intensity") or 3),
        )
        return {"queued": True, "request_id": rid}

    if name == "place_event":
        rec = {
            "track_role": args.get("track_role"),
            "start_ms": args.get("start_ms"),
            "duration_ms": args.get("duration_ms"),
            "asset_ref": args.get("asset_ref"),
            "volume_db": args.get("volume_db"),
            "pan": args.get("pan"),
            "fade_in_ms": args.get("fade_in_ms"),
            "fade_out_ms": args.get("fade_out_ms"),
            "trigger_word": args.get("trigger_word"),
            "anchor_speaker": args.get("anchor_speaker"),
        }
        placed_events.append(rec)
        return {"recorded": True, "index": len(placed_events)}

    return {"error": f"unknown tool: {name}"}


def _build_director_system_prompt_tools(genre: str, genre_template: Optional[Any] = None) -> str:
    base = _build_director_system_prompt(genre, genre_template=genre_template)
    return (
        base
        + "\n\n--- TOOL USE ---\n"
        "You have tools: search_assets, request_generation, place_event.\n"
        "1) Prefer search_assets to find real asset_id values from the catalog.\n"
        "2) If nothing fits, call request_generation with a precise prompt; it will be generated after planning.\n"
        "3) place_event is optional scratchpad only.\n"
        "When finished with tools, respond with a single JSON object matching ProductionPlan — no markdown, no tools.\n"
        "Cap: complete within reasonable tool use (hard limit 30 tool invocations server-side).\n"
    )


class ProductionDirector:
    """Calls Ollama in the Director role; validates JSON into ProductionPlan with fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        try:
            from vibevoice.config import config  # type: ignore

            self._base_url = base_url or config.OLLAMA_BASE_URL
            self._model = model or config.OLLAMA_MODEL
        except Exception:
            self._base_url = base_url or "http://127.0.0.1:11434"
            self._model = model or "llama3"
        self._timeout = timeout_seconds

    async def plan(
        self,
        script: str,
        script_segments: List[Dict[str, Any]],
        genre: str,
        available_assets: List[Dict[str, Any]],
        timing_hints: List[Dict[str, Any]],
        *,
        asset_library: Optional[Any] = None,
        generation_queue: Optional[Any] = None,
        use_tools: bool = True,
        word_index: Optional[List[Dict[str, Any]]] = None,
        genre_template: Optional[Any] = None,
    ) -> ProductionPlan:
        from app.services.asset_library import AssetLibrary
        from app.services.generation_queue import GenerationQueue

        episode_id = str(uuid.uuid4())
        lib = asset_library if asset_library is not None else AssetLibrary()
        gq = (
            generation_queue
            if generation_queue is not None
            else GenerationQueue(lib, genre_template=genre_template)
        )

        catalog_rows = lib.as_llm_catalog(limit=120, genre_template=genre_template)
        if not catalog_rows and available_assets:
            catalog_rows = _compact_asset_library(available_assets)
        from app.services.word_index import compact_word_index_for_llm

        wcompact = compact_word_index_for_llm(word_index) if word_index else None
        user = _build_user_prompt(script, timing_hints, catalog_rows, wcompact)

        if use_tools:
            try:
                system_tools = _build_director_system_prompt_tools(genre, genre_template=genre_template)
                plan = await asyncio.wait_for(
                    self._plan_with_tool_loop(system_tools, user, lib, gq),
                    timeout=self._timeout,
                )
                ep = plan.episode_id or episode_id
                return plan.model_copy(update={"episode_id": ep})
            except Exception as exc:
                _ed = str(exc).strip() or f"{type(exc).__name__}({exc!r})"
                logger.warning("Director tool loop failed (%s); trying single-shot JSON", _ed)

        try:
            system = _build_director_system_prompt(genre, genre_template=genre_template)
            raw = await asyncio.wait_for(
                self._ollama_generate_single(system, user),
                timeout=self._timeout,
            )
            data = extract_json_object_from_llm_text(raw)
            if not data:
                raise ValueError("Director returned no JSON object")
            data.setdefault("episode_id", episode_id)
            return ProductionPlan.model_validate(data)
        except Exception as exc:
            _ed = str(exc).strip() or f"{type(exc).__name__}({exc!r})"
            logger.warning("ProductionDirector falling back to segment-based plan: %s", _ed)
            return build_fallback_production_plan(
                script=script,
                script_segments=script_segments,
                genre=genre,
                episode_id=episode_id,
                timing_hints=timing_hints,
            )

    async def _plan_with_tool_loop(
        self,
        system: str,
        user: str,
        library: Any,
        generation_queue: Any,
    ) -> ProductionPlan:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        placed: List[Dict[str, Any]] = []
        tool_calls_total = 0
        url_base = self._base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for _ in range(36):
                payload: Dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "tools": DIRECTOR_TOOLS,
                    "stream": False,
                    "options": {"temperature": 0.25, "top_p": 0.9},
                }
                chat_resp = await client.post(f"{url_base}/api/chat", json=payload)
                if chat_resp.status_code == 400:
                    raise RuntimeError("Ollama rejected tool-calling payload (model may lack tool support)")
                chat_resp.raise_for_status()
                body = chat_resp.json()
                msg = body.get("message") or {}
                messages.append(msg)

                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        tool_calls_total += 1
                        if tool_calls_total > 30:
                            raise RuntimeError("Director tool call cap (30) exceeded")
                        fn = tc.get("function") or {}
                        tname = str(fn.get("name") or "")
                        raw_args = fn.get("arguments")
                        result = _director_dispatch_tool(
                            tname, raw_args, library, generation_queue, placed
                        )
                        tool_msg: Dict[str, Any] = {
                            "role": "tool",
                            "content": json.dumps(result),
                            "name": tname,
                        }
                        tid = tc.get("id")
                        if tid:
                            tool_msg["tool_call_id"] = str(tid)
                        messages.append(tool_msg)
                    continue

                content = (msg.get("content") or "").strip()
                if content:
                    data = extract_json_object_from_llm_text(content)
                    if data:
                        return ProductionPlan.model_validate(data)
                    raise ValueError("Director final message contained no ProductionPlan JSON")
                raise ValueError("Director returned empty content without tool_calls")
        raise RuntimeError("Director tool loop exceeded maximum rounds")

    async def _ollama_generate_single(self, system: str, user: str) -> str:
        url_base = self._base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            chat_resp = await client.post(
                f"{url_base}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.25, "top_p": 0.9},
                },
            )
            if chat_resp.status_code != 404:
                chat_resp.raise_for_status()
                body = chat_resp.json()
                msg = body.get("message") or {}
                text = (msg.get("content") or "").strip()
                if text:
                    return text

            prompt = (
                "SYSTEM INSTRUCTIONS (follow exactly):\n"
                + system
                + "\n\nUSER REQUEST:\n"
                + user
                + "\n\nRespond with the JSON object only."
            )
            gen_resp = await client.post(
                f"{url_base}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.25, "top_p": 0.9},
                },
            )
            gen_resp.raise_for_status()
            gen_body = gen_resp.json()
            return (gen_body.get("response") or "").strip()
