"""
Ollama client service for LLM script generation.
"""
import logging
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import config
from .podcast_pause_cues import contextual_pause_ms_for_handoff, redistribute_uniform_pause_markers

logger = logging.getLogger(__name__)


def _ad_scan_log_blob(prefix: str, text: str) -> None:
    max_c = int(getattr(config, "AD_SCAN_LOG_MAX_CHARS", 50000) or 0)
    if max_c <= 0 or len(text) <= max_c:
        logger.info("%s\n%s", prefix, text)
        return
    omitted = len(text) - max_c
    logger.info(
        "%s (truncated: %d chars total, %d omitted)\n%s",
        prefix,
        len(text),
        omitted,
        text[:max_c],
    )


# Per-block classification: timestamps come from our block boundaries, not model-invented ranges.
_AD_BLOCK_CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_classifications": {
            "type": "array",
            "description": "One entry per transcript block index; is_ad true only for sponsor/ad blocks.",
            "items": {
                "type": "object",
                "properties": {
                    "block_index": {"type": "integer"},
                    "is_ad": {"type": "boolean"},
                    "label": {
                        "type": "string",
                        "description": "Brand name if is_ad; otherwise short reason e.g. main episode",
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["block_index", "is_ad", "label", "confidence"],
            },
        },
    },
    "required": ["block_classifications"],
}

# Omit bulky fields from profile text sent to the script LLM (reference audio transcript, etc.)
_PROFILE_PROMPT_OMIT_KEYS = frozenset({"transcript"})


def _sanitize_profile_for_prompt(profile: Dict) -> Dict:
    """Return a shallow copy of profile suitable for embedding in prompts."""
    return {k: v for k, v in profile.items() if k not in _PROFILE_PROMPT_OMIT_KEYS}


_CUE_MARKER_BRACKET = re.compile(r"^\s*\[CUE:", re.IGNORECASE)
_INLINE_PRODUCTION_CUE = re.compile(r"\s*\[CUE:\s*[^\]]+\]\s*", re.IGNORECASE)
_SPEAKER_LINE_CANONICAL = re.compile(r"^\s*Speaker\s*(\d+)\s*:\s*", re.IGNORECASE)
# Inline TTS pause tokens (not spoken); must survive _remove_placeholder_brackets
_PAUSE_MS_INNER = re.compile(r"^\s*PAUSE_MS\s*:\s*(\d{1,4})\s*$", re.IGNORECASE)


def _remove_placeholder_brackets(script: str) -> str:
    """Strip bracketed placeholders (e.g. stage directions) but keep [CUE: ...] and [PAUSE_MS: ...] tokens."""

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if inner.strip().upper().startswith("CUE:"):
            return m.group(0)
        if _PAUSE_MS_INNER.match(inner):
            return m.group(0)
        return ""

    return re.sub(r"\s*\[([^\]]+)\]", repl, script)


def _inject_speaker_handoff_pauses(
    script: str,
    *,
    include_production_cues: bool = False,
) -> str:
    """
    Append ``[PAUSE_MS:N]`` at the end of speaker lines when the next speaker line is a different speaker.

    Skips lines that already contain a pause marker. Used for LLM-generated scripts so TTS inserts
    silence between turns (see ``strip_inline_pause_markers`` in ``tts/segments.py``).
    Pause duration is contextual, not a single fixed default.
    """
    from .tts.segments import INLINE_PAUSE_MS_PATTERN

    raw_lines = [ln.rstrip() for ln in script.split("\n")]
    entries: List[Tuple[str, Any]] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        sp_m = re.match(r"^(Speaker\s+(\d+)\s*:\s*)(.*)$", s, re.IGNORECASE)
        if sp_m:
            entries.append(
                (
                    "spk",
                    int(sp_m.group(2)),
                    sp_m.group(1),
                    sp_m.group(3).strip(),
                )
            )
        elif include_production_cues and _CUE_MARKER_BRACKET.match(s):
            entries.append(("cue", s))
        else:
            entries.append(("other", s))

    handoff_idx = 0
    for i, e in enumerate(entries):
        if e[0] != "spk":
            continue
        _, sn, prefix, body = e
        next_sn: Optional[int] = None
        for j in range(i + 1, len(entries)):
            if entries[j][0] == "spk":
                next_sn = entries[j][1]
                break
        if next_sn is None or next_sn == sn:
            continue
        if INLINE_PAUSE_MS_PATTERN.search(body):
            continue
        ms = contextual_pause_ms_for_handoff(
            body,
            handoff_index=handoff_idx,
            from_speaker=sn,
            to_speaker=next_sn,
        )
        handoff_idx += 1
        body = f"{body} [PAUSE_MS:{ms}]"
        entries[i] = ("spk", sn, prefix, body)

    out: List[str] = []
    for e in entries:
        if e[0] == "spk":
            _, _sn, prefix, body = e
            out.append(f"{prefix}{body}".strip())
        else:
            out.append(e[1])
    return "\n".join(out)


def normalize_podcast_speaker_labels(
    script: str,
    num_voices: int,
    *,
    include_production_cues: bool = False,
) -> str:
    """
    Rewrite ``Name:`` dialogue prefixes to ``Speaker 1:`` … ``Speaker N:`` in order of first appearance.

    TTS maps ``Speaker i`` to ``voices[i-1]``; persona or celebrity names in the margin break that mapping.
    """
    if num_voices < 1:
        return script
    lines = script.split("\n")
    label_to_index: Dict[str, int] = {}
    next_slot = 1
    out: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if include_production_cues and _CUE_MARKER_BRACKET.match(s):
            out.append(s)
            continue

        m = _SPEAKER_LINE_CANONICAL.match(s)
        if m:
            n = int(m.group(1))
            body = s[m.end() :].strip()
            n = max(1, min(n, num_voices))
            out.append(f"Speaker {n}: {body}".strip())
            continue

        if ":" not in s:
            out.append(s)
            continue

        prefix, _, rest = s.partition(":")
        prefix = prefix.strip()
        rest = rest.lstrip()
        if not prefix:
            out.append(s)
            continue

        key = prefix.casefold()
        if key not in label_to_index:
            if next_slot <= num_voices:
                label_to_index[key] = next_slot
                next_slot += 1
            else:
                label_to_index[key] = num_voices
        sn = label_to_index[key]
        out.append(f"Speaker {sn}: {rest}".strip())

    return "\n".join(out)


def _join_unique_phrases(phrases: List[str], max_chars: int = 600) -> str:
    """Join phrase list for the prompt, staying within a character budget."""
    if not phrases:
        return ""
    joined = ", ".join(str(p) for p in phrases if p)
    if len(joined) <= max_chars:
        return joined
    parts: List[str] = []
    total = 0
    for p in phrases:
        if not p:
            continue
        sep = ", " if parts else ""
        chunk = sep + str(p)
        if total + len(chunk) > max_chars:
            break
        parts.append(str(p))
        total += len(chunk)
    return ", ".join(parts)


# Piecewise-linear word-count bands aligned with discrete presets (5 / 10 / 15 / 30 min).
_DURATION_WORD_ANCHORS: List[Tuple[float, int, int]] = [
    (5.0, 500, 700),
    (10.0, 1000, 1400),
    (15.0, 1500, 2100),
    (30.0, 3000, 4200),
]

_DISCRETE_DURATION_TO_RANGE: Dict[str, str] = {
    "5 min": "500-700",
    "10 min": "1000-1400",
    "15 min": "1500-2100",
    "30 min": "3000-4200",
}


def _word_range_from_minutes(minutes: float) -> str:
    """Map a target length in minutes to a low-high word count string for the prompt."""
    m = max(0.25, float(minutes))
    anchors = _DURATION_WORD_ANCHORS
    m0, lo0, hi0 = anchors[0]
    if m <= m0:
        scale = m / m0 if m0 else 1.0
        lo = int(max(80, lo0 * scale))
        hi = int(max(120, hi0 * scale))
        return f"{lo}-{hi}"

    m_last, lo_last, hi_last = anchors[-1]
    if m >= m_last:
        scale = m / m_last
        lo = int(lo_last * scale)
        hi = int(hi_last * scale)
        return f"{lo}-{hi}"

    for i in range(len(anchors) - 1):
        ma, lo_a, hi_a = anchors[i]
        mb, lo_b, hi_b = anchors[i + 1]
        if ma <= m <= mb:
            t = (m - ma) / (mb - ma) if mb != ma else 0.0
            lo = int(lo_a + t * (lo_b - lo_a))
            hi = int(hi_a + t * (hi_b - hi_a))
            return f"{lo}-{hi}"

    return "1000-1400"


def _format_duration_label_minutes(minutes: float) -> str:
    rounded = round(float(minutes) * 10.0) / 10.0
    if abs(rounded - round(rounded)) < 1e-9:
        return f"~{int(round(rounded))} minutes"
    return f"~{rounded:g} minutes"


def resolve_script_duration_for_prompt(
    duration: Optional[str],
    approximate_duration_minutes: Optional[float],
) -> tuple[str, str]:
    """
    Compute word-count band and human-readable duration line for the script prompt.

    If approximate_duration_minutes is set, it takes precedence for the word target.
    Otherwise uses discrete labels in _DISCRETE_DURATION_TO_RANGE, then tries to parse "N min" strings.

    Returns:
        (word_range_str, duration_label) e.g. ("1000-1400", "~12 minutes")
    """
    if approximate_duration_minutes is not None:
        m = float(approximate_duration_minutes)
        return _word_range_from_minutes(m), _format_duration_label_minutes(m)

    ds = (duration or "").strip()
    if ds in _DISCRETE_DURATION_TO_RANGE:
        return _DISCRETE_DURATION_TO_RANGE[ds], ds

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b", ds, re.IGNORECASE)
    if match:
        m = float(match.group(1))
        return _word_range_from_minutes(m), _format_duration_label_minutes(m)

    if ds:
        return "1000-1400", ds

    return "1000-1400", "~10 minutes"


def _midpoint_word_target(word_range_str: str) -> int:
    """Midpoint integer from a 'lo-hi' word range string for structure budgeting in prompts."""
    s = (word_range_str or "").strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return max(1, (lo + hi) // 2)
    m2 = re.search(r"\d+", s)
    if m2:
        return max(1, int(m2.group()))
    return 1200


def estimated_duration_seconds_for_segmentation(
    script: str,
    duration: Optional[str] = None,
    approximate_duration_minutes: Optional[float] = None,
) -> float:
    """
    Align segmentation metadata with script-generation duration targeting (Step 1).
    When ``approximate_duration_minutes`` is set, it matches script generation exactly.
    Otherwise uses the same word-range resolution as script prompts (or script word count when substantial).
    """
    if approximate_duration_minutes is not None:
        return max(30.0, float(approximate_duration_minutes) * 60.0)
    word_range_str, _ = resolve_script_duration_for_prompt(duration, None)
    midpoint = _midpoint_word_target(word_range_str)
    from_script_words = len(script.split())
    if from_script_words > 80:
        return max(30.0, (from_script_words / 140.0) * 60.0)
    return max(30.0, (midpoint / 140.0) * 60.0)


def infer_num_voices_from_script(script: str) -> int:
    """Largest Speaker N label in the script, or 1 if none."""
    found: List[int] = []
    for m in re.finditer(r"(?i)^Speaker\s+(\d+)\s*:", script, re.MULTILINE):
        found.append(int(m.group(1)))
    return max(found) if found else 1


def _structure_word_budgets(total_words: int) -> tuple[int, int, int, int]:
    """Intro, body, analysis, close word counts that sum to total_words."""
    t = max(100, int(total_words))
    intro = max(50, round(t * 0.10))
    close = max(40, round(t * 0.10))
    analysis = max(80, round(t * 0.25))
    body = t - intro - close - analysis
    while body < 100 and analysis > 80:
        analysis -= 1
        body += 1
    while body < 100 and intro > 50:
        intro -= 1
        body += 1
    while body < 100 and close > 40:
        close -= 1
        body += 1
    if body < 100:
        body = max(100, t - intro - close - analysis)
    return intro, body, analysis, close


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL (defaults to config)
            model: Ollama model name (defaults to config)
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_MODEL
        self.timeout = 300  # 5 minutes for long-running requests
        self.client = httpx.Client(timeout=self.timeout)

        logger.info(f"OllamaClient initialized: {self.base_url}, model: {self.model}")

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    def identify_podcast_ad_segments(
        self,
        segments_payload: List[Dict[str, Any]],
        total_duration_seconds: float,
        custom_model: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ask the LLM to label advertisement/sponsor intervals from timestamped transcript segments.

        Returns:
            List of dicts with keys: start_seconds, end_seconds, label, confidence
        """
        jid = job_id or "-"
        model = custom_model or self.model
        if not segments_payload:
            return []
        indexed_blocks = [
            {
                "block_index": i,
                "start_seconds": round(float(b["start_seconds"]), 3),
                "end_seconds": round(float(b["end_seconds"]), 3),
                "text": (b.get("text") or "").strip(),
            }
            for i, b in enumerate(segments_payload)
        ]
        n_blocks = len(indexed_blocks)
        payload = {
            "total_duration_seconds": round(total_duration_seconds, 3),
            "block_count": n_blocks,
            "transcript_blocks": indexed_blocks,
        }
        user_text = (
            "Classify EACH transcript block by index. Blocks are short (typically under a minute); "
            "timestamps are fixed—you must NOT invent start/end times.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        last_idx = max(0, n_blocks - 1)
        system = f"""You classify fixed transcript blocks from a podcast. There are exactly {n_blocks} blocks with indices 0 through {last_idx}.

OUTPUT: Return block_classifications with EXACTLY one object per block_index (0..{last_idx}). Do not omit blocks.

For each block set:
- is_ad = true if the block is primarily: a paid third-party advertisement or sponsor read; military recruitment; insurance/product/retail pitch; OR the show's own subscription/premium upsell (Substack, Patreon, "become a subscriber", member perks) when it is clearly promotional.
- is_ad = false for main episode content: news, commentary, interviews, political discussion, reading long statements—even if it names brands in passing or says "Midas Touch Network" as the show.
- label: if is_ad, short brand or offer name (e.g. USAA, Shopify, MidasPlus). If not is_ad, use "Main content".

Heuristics:
- Many consecutive blocks of dense news/commentary are main content; short, brand-heavy blocks at the very start or very end are often ads.
- When unsure, prefer is_ad = false so we do not delete the episode audio.

Respond using the required JSON schema only."""

        request_body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "format": _AD_BLOCK_CLASSIFICATION_SCHEMA,
            "options": {
                "temperature": 0,
                "top_p": 0.9,
            },
        }

        logger.info(
            "[ad-scan] job=%s ollama_ad_request base_url=%s model=%s "
            "transcript_blocks_in_payload=%d user_message_chars=%d",
            jid,
            self.base_url,
            model,
            len(segments_payload),
            len(user_text),
        )
        _ad_scan_log_blob(f"[ad-scan] job={jid} ollama_system_prompt", system)
        _ad_scan_log_blob(f"[ad-scan] job={jid} ollama_user_message (JSON payload + task prefix)", user_text)

        try:
            t0 = time.perf_counter()
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=request_body,
            )
            if response.status_code == 400:
                # Older Ollama without schema support: fall back to JSON mode (object, not prose).
                logger.warning(
                    "[ad-scan] job=%s Ollama rejected structured format; retrying with format=json. "
                    "Upgrade Ollama for best results.",
                    jid,
                )
                request_body.pop("format", None)
                request_body["format"] = "json"
                response = self.client.post(
                    f"{self.base_url}/api/chat",
                    json=request_body,
                )
            response.raise_for_status()
            http_s = time.perf_counter() - t0
            result = response.json()
            message = result.get("message") or {}
            raw = (message.get("content") or "").strip()
            logger.info(
                "[ad-scan] job=%s ollama_ad_response http_wall_s=%.3f "
                "prompt_eval_count=%s eval_count=%s total_duration_ns=%s done_reason=%s",
                jid,
                http_s,
                result.get("prompt_eval_count"),
                result.get("eval_count"),
                result.get("total_duration"),
                result.get("done_reason"),
            )
            _ad_scan_log_blob(f"[ad-scan] job={jid} ollama_message_content_raw", raw or "(empty)")
            if not raw:
                raise RuntimeError("Ollama returned empty response for ad detection")
            parsed_obj = self._parse_json_payload(raw)
            if isinstance(parsed_obj, dict) and "block_classifications" in parsed_obj:
                validated = self._block_classifications_to_ad_segments(
                    segments_payload,
                    parsed_obj.get("block_classifications"),
                    total_duration_seconds,
                )
                logger.info(
                    "[ad-scan] job=%s ollama_block_classifications=%d validated_ad_intervals=%d",
                    jid,
                    len(parsed_obj.get("block_classifications") or []),
                    len(validated),
                )
            else:
                parsed_list = self._coerce_to_segment_list(parsed_obj)
                if parsed_list is None:
                    raise ValueError("Could not parse ad detection response (expected block_classifications or ad_segments)")
                validated = self._validate_ad_segment_dicts(parsed_list, total_duration_seconds)
                logger.info(
                    "[ad-scan] job=%s ollama_parsed_list_len=%d validated_ad_intervals=%d",
                    jid,
                    len(parsed_list),
                    len(validated),
                )
            _ad_scan_log_blob(
                f"[ad-scan] job={jid} ollama_validated_ad_segments (after time/clamp filter)",
                json.dumps(validated, ensure_ascii=False, indent=2),
            )
            return validated
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama ad detection request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama ad detection failed: {e}") from e

    @staticmethod
    def _strip_markdown_fence(raw: str) -> str:
        cleaned = raw.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _coerce_to_segment_list(parsed: Any) -> Optional[List[Any]]:
        """Accept either a JSON array or common wrapper objects models return."""
        if isinstance(parsed, list):
            return parsed
        if not isinstance(parsed, dict):
            return None
        for key in (
            "ad_segments",
            "ads",
            "segments",
            "advertisements",
            "items",
            "results",
            "data",
            "output",
            "response",
        ):
            val = parsed.get(key)
            if isinstance(val, list):
                return val
        for val in parsed.values():
            if isinstance(val, list):
                return val
        return None

    def _parse_json_payload(self, raw: str) -> Any:
        """Parse JSON object/array from model output (may be wrapped in fences or commentary)."""
        cleaned = self._strip_markdown_fence(raw)
        if not cleaned:
            raise ValueError("Empty ad detection response")

        for candidate in (cleaned, cleaned.lstrip()):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        dec = json.JSONDecoder()
        for opener in ("[", "{"):
            pos = 0
            while True:
                i = cleaned.find(opener, pos)
                if i < 0:
                    break
                try:
                    parsed, _ = dec.raw_decode(cleaned[i:])
                    return parsed
                except json.JSONDecodeError:
                    pass
                pos = i + 1

        snippet = cleaned[:800].replace("\n", " ")
        logger.warning(
            "Could not parse JSON (first ~800 chars): %s%s",
            snippet,
            "..." if len(cleaned) > 800 else "",
        )
        raise ValueError("Could not parse JSON from ad detection response")

    def _block_classifications_to_ad_segments(
        self,
        blocks: List[Dict[str, Any]],
        classifications: Any,
        total_duration_seconds: float,
    ) -> List[Dict[str, Any]]:
        """Map per-block is_ad flags to ad_segments using each block's start/end (authoritative)."""
        if not isinstance(classifications, list):
            return []
        td = max(0.0, float(total_duration_seconds))
        # Last classification wins if the model repeats a block_index.
        ad_by_index: Dict[int, Dict[str, Any]] = {}
        for item in classifications:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("block_index", -1))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(blocks):
                logger.debug("[ad-scan] block_classification skip bad index=%s blocks=%d", idx, len(blocks))
                continue
            is_ad = bool(item.get("is_ad", item.get("is_advertisement", False)))
            if not is_ad:
                continue
            label = str(item.get("label", "ad")).strip() or "ad"
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            blk = blocks[idx]
            try:
                start = float(blk["start_seconds"])
                end = float(blk["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            start = max(0.0, min(start, td))
            end = max(0.0, min(end, td))
            if end <= start or end - start < 0.05:
                continue
            ad_by_index[idx] = {
                "start_seconds": start,
                "end_seconds": end,
                "label": label,
                "confidence": conf,
            }
        out = list(ad_by_index.values())
        out.sort(key=lambda x: x["start_seconds"])
        return out

    def _validate_ad_segment_dicts(
        self, parsed: List[Any], total_duration_seconds: float
    ) -> List[Dict[str, Any]]:
        if not isinstance(parsed, list):
            raise ValueError("Ad detection response must be a JSON array")
        out: List[Dict[str, Any]] = []
        td = max(0.0, float(total_duration_seconds))
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start_seconds", item.get("start", 0)))
                end = float(item.get("end_seconds", item.get("end", 0)))
            except (TypeError, ValueError):
                continue
            label = str(item.get("label", "ad")).strip() or "ad"
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            start = max(0.0, min(start, td))
            end = max(0.0, min(end, td))
            if end <= start or end - start < 0.05:
                continue
            out.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "label": label,
                    "confidence": conf,
                }
            )
        out.sort(key=lambda x: x["start_seconds"])
        return out

    def generate_script(
        self,
        article_text: str,
        genre: str,
        duration: Optional[str],
        num_voices: int,
        custom_model: Optional[str] = None,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
        narrator_speaker_index: Optional[int] = None,
        approximate_duration_minutes: Optional[float] = None,
        include_production_cues: bool = False,
    ) -> str:
        """
        Generate podcast script from article using Ollama.

        Args:
            article_text: Scraped article content
            genre: Podcast genre (Comedy, Serious, News, etc.)
            duration: Target duration label (e.g. "10 min") when not using approximate_duration_minutes
            num_voices: Number of voices/speakers (1-4)
            custom_model: Optional custom model name (overrides default)
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order (maps to Speaker 1, 2, etc.)
            narrator_speaker_index: If set (1..num_voices), that Speaker is the episode narrator;
                others are co-hosts, experts, or reactors as appropriate.
            approximate_duration_minutes: Optional target episode length in minutes; drives word-count
                targets and overrides discrete ``duration`` when set.
            include_production_cues: If True, prompt includes ``[CUE: ...]`` markers for production mixing.
                Standard TTS flows should leave this False so scripts contain only spoken dialogue.

        Returns:
            Generated podcast script with speaker labels

        Raises:
            RuntimeError: If Ollama request fails
        """
        model = custom_model or self.model

        # Truncate article if too long (keep first ~8000 chars for context)
        max_article_length = 8000
        if len(article_text) > max_article_length:
            logger.info(f"Truncating article from {len(article_text)} to {max_article_length} characters")
            article_text = article_text[:max_article_length] + "\n[... article truncated ...]"

        # Build prompt for script generation
        prompt = self._build_prompt(
            article_text,
            genre,
            duration,
            num_voices,
            voice_profiles,
            voice_names,
            narrator_speaker_index=narrator_speaker_index,
            approximate_duration_minutes=approximate_duration_minutes,
            include_production_cues=include_production_cues,
        )

        logger.info(f"Generating script with Ollama (model: {model}, genre: {genre}, duration: {duration})")

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            script = result.get("response", "").strip()

            if not script:
                raise RuntimeError("Ollama returned empty script")

            # Clean up the script
            script = self._clean_script(script, num_voices, include_production_cues=include_production_cues)

            logger.info(f"Generated script: {len(script)} characters")
            return script

        except httpx.HTTPError as e:
            error_msg = f"Ollama API request failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during script generation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_script_segments(
        self,
        script: str,
        custom_model: Optional[str] = None,
        *,
        estimated_duration_seconds: Optional[float] = None,
        num_voices: Optional[int] = None,
        genre: Optional[str] = None,
        genre_style: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate structured production cue segments from a podcast script.

        Args:
            script: Speaker-labeled script text
            custom_model: Optional model override
            estimated_duration_seconds: Target length hint (defaults from script / duration metadata)
            num_voices: Speaker count (defaults from Speaker N labels in script)
            genre: Production genre label for the segmentation prompt
            genre_style: Tone/style line (often derived from production ``style`` or genre)

        Returns:
            List of segment dictionaries
        """
        model = custom_model or self.model
        est = estimated_duration_seconds
        if est is None:
            est = estimated_duration_seconds_for_segmentation(script, None, None)
        nv = num_voices if num_voices is not None else infer_num_voices_from_script(script)
        g = (genre or "").strip() or "General"
        gs = (genre_style or "").strip() or g
        prompt = self._build_segmentation_prompt(script, est, nv, g, gs)
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            raw = (result.get("response") or "").strip()
            if not raw:
                raise RuntimeError("Ollama returned empty segmentation response")
            parsed = self._parse_segment_json(raw)
            return self._validate_segments(parsed)
        except httpx.HTTPError as e:
            error_msg = f"Ollama API request failed for segmentation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during segmentation generation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _build_segmentation_prompt(
        self,
        script: str,
        estimated_duration_seconds: float,
        num_voices: int,
        genre: str,
        genre_style: str,
    ) -> str:
        return f"""You are a podcast post-production engineer. Your job is to parse a finished podcast script and emit a strict production JSON array that an audio pipeline will consume directly. Precision and schema adherence are non-negotiable — any malformed output breaks the pipeline.

--- INPUT SCRIPT ---
{script}

--- PRODUCTION METADATA (from Step 1) ---
Total estimated duration: {estimated_duration_seconds:.1f} seconds
Number of speakers: {num_voices}
Genre: {genre}
Tone: {genre_style}

--- OUTPUT REQUIREMENTS ---

Output ONLY a raw JSON array. No markdown fences. No commentary. No trailing text. Begin your response with [ and end with ].

--- SEGMENT SCHEMA ---

Each segment object must include ALL of the following fields:

{{
  "segment_id": <integer>,
  "segment_type": <string>,
  "speaker": <string or null>,
  "text": <string or null>,
  "start_time_hint": <number>,
  "duration_hint": <number>,
  "energy_level": <string>,
  "notes": <string or null>
}}

--- ALLOWED SEGMENT TYPES ---

| Type              | When to use                                                                 | Typical duration     |
|-------------------|-----------------------------------------------------------------------------|----------------------|
| intro_music       | Opening music before first spoken word. Always first segment.               | 3–6s                 |
| dialogue          | Any spoken segment by a named speaker. Split at speaker turns.              | 5–90s per turn       |
| transition_sting  | Short audio sting between major story blocks. Max 4 per episode.            | 1–3s                 |
| music_bed_in      | Marker for background score to begin (analysis sections). Not audible alone.| 0s (marker only)     |
| music_bed_out     | Marker for background score to end. Paired with every music_bed_in.        | 0s (marker only)     |
| outro_music       | Closing music after final spoken word. Always last segment.                 | 5–10s                |

--- TIMING RULES ---

1. start_time_hint must be monotonically non-decreasing across the array (equal is allowed for zero-duration markers)
2. Estimate dialogue duration at 140 words per minute: duration_hint = (word_count / 140) × 60
3. intro_music and outro_music durations are fixed: 5.0s and 8.0s respectively unless overridden by production metadata
4. transition_sting is always 2.0s
5. music_bed_in / music_bed_out markers carry duration_hint: 0.0 and do NOT advance start_time_hint
6. Do not create a transition_sting at every speaker change — only at major story block boundaries as marked by [CUE: TRANSITION_STING] in the script

--- ENERGY LEVEL ASSIGNMENT ---

- intro_music: high
- First dialogue segment (hook): high
- Story body segments: medium
- Analysis/deep-dive segments: low → medium
- Final close: medium
- outro_music: low
- transition_sting: use energy level of the segment that follows it
- music_bed_in / music_bed_out: match surrounding dialogue energy

--- DIALOGUE SPLITTING RULES ---

1. Split dialogue at every speaker change — one dialogue object per speaker turn
2. Preserve the verbatim text from the script — do not paraphrase, summarize, or trim
3. If a single speaker turn exceeds 90 seconds (≈210 words), split into consecutive dialogue segments with the same speaker and add note: "long turn — split for production"
4. Music cue markers ([CUE: ...]) in the script become their own segment objects and are removed from adjacent dialogue text

--- VALIDATION CHECKLIST (apply before outputting) ---

[ ] Array starts with segment_type: "intro_music"
[ ] Array ends with segment_type: "outro_music"
[ ] All start_time_hints are non-decreasing
[ ] Every music_bed_in has a paired music_bed_out later in the array
[ ] No more than 4 transition_sting segments total
[ ] All dialogue segments have non-null speaker and text fields
[ ] All non-dialogue segments have null speaker and null text fields
[ ] No [CUE: ...] tokens appear inside any text field
[ ] segment_id values are sequential integers starting at 1

Output the JSON array now:"""

    def _parse_segment_json(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)

    def _validate_segments(self, parsed: Any) -> List[Dict[str, Any]]:
        cue_in_text = re.compile(r"\[CUE:\s*[^\]]+\]", re.IGNORECASE)

        def _parse_opt_float(val: Any, default: Optional[float] = None) -> Optional[float]:
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        allowed_types = {
            "intro_music",
            "dialogue",
            "transition_sting",
            "music_bed_in",
            "music_bed_out",
            "outro_music",
        }
        if not isinstance(parsed, list):
            raise RuntimeError("Segment response must be a JSON array")

        normalized: List[Dict[str, Any]] = []
        last_time = 0.0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("segment_type") or "").strip()
            segment_type = "music_bed_in" if raw_type == "music_bed" else raw_type
            if segment_type not in allowed_types:
                continue

            start_time_hint_raw = item.get("start_time_hint", 0.0)
            try:
                start_time_hint = max(float(start_time_hint_raw), 0.0)
            except (TypeError, ValueError):
                start_time_hint = last_time
            if start_time_hint < last_time:
                start_time_hint = last_time
            last_time = start_time_hint

            duration_hint = _parse_opt_float(item.get("duration_hint"))
            energy_level = item.get("energy_level")
            energy_str = str(energy_level).strip() if energy_level is not None else None
            notes_val = item.get("notes")
            notes_str: Optional[str]
            if notes_val is None or notes_val == "":
                notes_str = None
            else:
                notes_str = str(notes_val).strip() or None

            segment: Dict[str, Any] = {
                "segment_type": segment_type,
                "start_time_hint": start_time_hint,
                "duration_hint": duration_hint,
                "energy_level": energy_str,
                "notes": notes_str,
            }

            if segment_type == "dialogue":
                speaker = str(item.get("speaker") or "").strip()
                text = str(item.get("text") or "").strip()
                if cue_in_text.search(text):
                    text = cue_in_text.sub("", text).strip()
                if not speaker or not text:
                    continue
                segment["speaker"] = speaker
                segment["text"] = text
            else:
                segment["speaker"] = None
                segment["text"] = None

            normalized.append(segment)

        for idx, seg in enumerate(normalized, start=1):
            seg["segment_id"] = idx

        if not normalized:
            raise RuntimeError("No valid segments returned by Ollama")
        return normalized

    def _build_prompt(
        self,
        article_text: str,
        genre: str,
        duration: Optional[str],
        num_voices: int,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
        narrator_speaker_index: Optional[int] = None,
        approximate_duration_minutes: Optional[float] = None,
        include_production_cues: bool = False,
    ) -> str:
        """
        Build prompt for Ollama script generation.

        Args:
            article_text: Article content
            genre: Podcast genre
            duration: Target duration label (optional if approximate_duration_minutes is set)
            num_voices: Number of voices
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order
            narrator_speaker_index: Optional 1-based index of which speaker is the narrator
            approximate_duration_minutes: Optional minutes target for word-count band
            include_production_cues: If True, instruct the model to emit ``[CUE: ...]`` production markers

        Returns:
            Formatted prompt string
        """
        word_target, duration_label = resolve_script_duration_for_prompt(duration, approximate_duration_minutes)
        midpoint = _midpoint_word_target(word_target)
        ert = midpoint / 140.0
        estimated_read_time = f"{ert:.1f}".rstrip("0").rstrip(".") if ert else "0"
        intro_word_count, body_word_count, analysis_word_count, close_word_count = _structure_word_budgets(midpoint)
        max_segment_words = min(150, max(80, midpoint // 7))

        # Genre-specific instructions
        genre_instructions = {
            "Comedy": "Make it light-hearted, humorous, and engaging. Include jokes and witty commentary.",
            "Serious": "Keep it professional, informative, and thoughtful. Maintain a serious tone throughout.",
            "News": "Present information clearly and objectively. Use a journalistic style with facts and analysis.",
            "Educational": "Focus on teaching and explaining concepts clearly. Use examples and analogies.",
            "Storytelling": "Narrate the content as a story with engaging descriptions and narrative flow.",
            "Interview": "Format as a conversation with questions and answers. Make it feel like a natural interview.",
            "Documentary": "Present information in a documentary style with detailed explanations and context.",
        }
        genre_style = genre_instructions.get(genre, "Keep it informative and engaging.")

        # Build speaker examples based on num_voices (1-4)
        speaker_examples = []
        for i in range(1, num_voices + 1):
            speaker_examples.append(f"Speaker {i}: [dialogue]")

        # Build voice profiles section if available
        voice_profiles_section = ""
        voice_adherence_block = ""
        if voice_profiles and voice_names:
            profile_lines: List[str] = []
            profile_lines.append(
                "Each speaker's lines must follow their profile for tone, cadence, vocabulary, and sentence structure. "
                "Podcast **Genre** (below) sets show format and how the topic is framed; these profiles govern how each speaker talks. "
                "Blend them: honor the genre's role while expressing dialogue in each speaker's voice from their profile.\n"
            )
            any_speaker_profile = False
            for i, voice_name in enumerate(voice_names[:num_voices], 1):
                raw = voice_profiles.get(voice_name)
                if not raw:
                    continue
                any_speaker_profile = True
                profile = _sanitize_profile_for_prompt(raw)
                profile_lines.append(f"\n### Speaker {i} ({voice_name})\n")
                vdp = profile.get("voice_design_prompt")
                if isinstance(vdp, str) and vdp.strip():
                    profile_lines.append(f"- Voice design (target delivery): {vdp.strip()}\n")
                pt = profile.get("profile_text")
                if isinstance(pt, str) and pt.strip():
                    profile_lines.append(f"- Full profile: {pt.strip()}\n")
                kws = profile.get("keywords")
                if isinstance(kws, list) and kws:
                    profile_lines.append(f"- Keywords / context: {', '.join(str(k) for k in kws if k)}\n")
                if profile.get("cadence"):
                    profile_lines.append(f"- Cadence: {profile['cadence']}\n")
                if profile.get("tone"):
                    profile_lines.append(f"- Tone: {profile['tone']}\n")
                if profile.get("vocabulary_style"):
                    profile_lines.append(f"- Vocabulary: {profile['vocabulary_style']}\n")
                if profile.get("sentence_structure"):
                    profile_lines.append(f"- Sentence structure: {profile['sentence_structure']}\n")
                phrases = profile.get("unique_phrases")
                if isinstance(phrases, list) and phrases:
                    line = _join_unique_phrases(phrases)
                    if line:
                        profile_lines.append(f"- Signature phrases (use sparingly, when natural): {line}\n")
                profile_lines.append(
                    f"\nFor **Speaker {i}** only: write dialogue that a TTS voice with this profile would sound natural reading aloud—"
                    f"match rhythm, word choice, and manner of speaking described above.\n"
                )
            if any_speaker_profile:
                voice_profiles_section = "".join(profile_lines)
                voice_adherence_block = """

**VOICE PROFILE ADHERENCE**
   - Apply the **VOICE PROFILES** section above to the matching Speaker number; lines must reflect each profile (tone, cadence, structure, vocabulary, and signature phrases when natural).
   - Genre and production specifications set format; profiles set how each person talks. Do not substitute generic dialogue for a defined profile.
   - Profiles are for **delivery and chemistry**, not for stripping names: co-hosts can still address each other and reference people in the story inside the dialogue.

"""
        if not voice_profiles_section.strip():
            voice_profiles_section = (
                "(No voice profiles supplied — use balanced, broadcast-appropriate delivery for each speaker label.)"
            )

        narrator_line = ""
        if narrator_speaker_index is not None:
            n = narrator_speaker_index
            vnames = voice_names or []
            narrator_voice = ""
            if 1 <= n <= len(vnames):
                narrator_voice = str(vnames[n - 1]).strip()
            nv = f' (voice name: "{narrator_voice}")' if narrator_voice else ""
            narrator_line = f"\n- Narrator: Speaker {n}{nv} — carries the main through-line of the episode."

        music_cue_section = ""
        if include_production_cues:
            music_cue_section = """
6. MUSIC CUE MARKERS (embed inline as single-line comments)
   Insert these exact tokens on their own line where audio production cues should occur:
   [CUE: INTRO_MUSIC]         ← before first spoken word
   [CUE: TRANSITION_STING]    ← at major story shifts (limit to 3-4 per episode)
   [CUE: MUSIC_BED_IN]        ← where background score should swell (analysis section)
   [CUE: MUSIC_BED_OUT]       ← where background score should fade
   [CUE: OUTRO_MUSIC]         ← after final spoken word

"""
        do_not_spoken_line = (
            "- Do not include any text that isn't meant to be spoken or is a music cue marker\n"
            if include_production_cues
            else "- Do not include any text that isn't meant to be spoken aloud\n"
        )
        script_format_cue_rule = (
            ""
            if include_production_cues
            else "\nDo not include [CUE: ...] lines or any non-dialogue production markers — only speaker lines for text-to-speech.\n"
        )

        prompt = f"""You are an expert podcast script writer specializing in producing broadcast-quality, audio-native dialogue. Your scripts are written for human ears, not human eyes — every word choice, sentence rhythm, and transition must work when spoken aloud.

--- VOICE PROFILES ---
{voice_profiles_section}

--- SOURCE MATERIAL ---
{article_text}

--- PRODUCTION SPECIFICATIONS ---
- Genre: {genre}
- Tone/Style: {genre_style}
- Target word count: {word_target} words ({duration_label})
- Number of speakers: {num_voices}{narrator_line}
- Estimated read time: {estimated_read_time} minutes at 140 WPM

--- SCRIPT FORMAT ---
Use ONLY this format. No markdown, no stage directions in prose, no header labels:

{chr(10).join(speaker_examples)}

Each speaker turn must be on its own line, prefixed with exactly ``Speaker 1:``, ``Speaker 2:``, … ``Speaker {num_voices}:`` (the word Speaker, a space, the digit, then a colon). Never use names, roles, or persona labels as the **prefix** before the colon — only ``Speaker N:``.
Inside the spoken text after the colon, write **personable, natural dialogue**: co-hosts may address each other by first name or informal reference, react to one another ("yeah, but—", "exactly"), and cite people and organizations from the source material by name. The VOICE PROFILES describe *how* each numbered speaker sounds, not what text must appear at the start of a line.
Blank lines between speaker turns only — no other blank lines.
When the next line is a different speaker, end the current line with exactly one inline pause token after the spoken words: ``[PAUSE_MS:N]`` where **N varies by moment** (allowed range roughly **120–480**). Treat N as performance timing, not decoration: use **shorter** gaps (e.g. 130–200) for quick banter, reactions, or completions; **medium** (200–320) for normal handoffs; **longer** (320–480) after a direct question, a punchline, a heavy fact, or a topic pivot. **Never** use the same N on every line — scan your script and ensure values differ across turns (no copy-pasting one number). These tokens are stripped before text-to-speech and become silence between turns — they must never be read as dialogue. If you omit them, the system may insert contextual pauses at each speaker handoff.
Do NOT include stage directions, scene headers, act labels, or any non-dialogue text.{script_format_cue_rule}

--- AUDIO WRITING RULES ---

1. NATURAL SPEECH PATTERNS
   - Use contractions consistently (don't, we're, that's, I'd, you'll)
   - Sentence length should vary: mix short punchy statements with longer explanatory ones
   - Avoid sentences over 30 words — they lose listeners
   - Use appositives and mid-sentence pauses naturally: "The crater — named after Wiseman's late wife — sits on the near-side boundary"
   - Never start consecutive sentences with the same word
   - Avoid passive voice; keep subject-verb order tight

2. PACING AND ENERGY
   - Open with a strong hook in the first 15 words of dialogue
   - Each major story transition should reset energy — don't let momentum flatline
   - Use rhetorical questions sparingly to re-engage listeners at the 1/3 and 2/3 marks
   - Include at least one "moment of weight" per major story: a pause beat where significance lands
   - Lighter, faster exchanges for transitions between stories; slower, more deliberate delivery signals for complex analysis
   - **Turn gaps:** make ``[PAUSE_MS:N]`` feel musical — alternate tighter and looser beats; match the emotional temperature of the line that just ended (not a metronome).

3. SPEAKER DYNAMICS
   - No speaker should dominate more than 60% of any 2-minute segment
   - Build in natural interruptions, completions, or agreements ("Exactly." / "Right, and what's interesting is—") to signal interplay
   - Keep the show **human and personable**: let hosts acknowledge each other, ask direct questions of one another, and use names or casual address inside lines when it fits the genre (still only ``Speaker N:`` as the line prefix).
   - Each speaker must have a consistent verbal identity maintained throughout:
     * Speaker 1 (anchor): sets context, asks the critical question, summarizes takeaways
     * Speaker 2 (analyst/co-host): adds texture, nuance, and "what this means" framing
     * Narrator (if present): only for scene-setting prose passages — kept brief, maximum 3 sentences
   - Avoid "talking head" blocks — no single speaker should hold the floor for more than ~100 words

4. CONTENT FIDELITY
   - Every factual claim must originate from the provided source material
   - Include specific details: names, numbers, dates, locations, direct quotes where available
   - Do NOT add speculation, opinion, or context not present in the articles
   - Paraphrase quotes into conversational speech — quoted text that reads as written prose sounds unnatural when spoken
   - Prioritize the most newsworthy facts; don't bury the lede

5. STRUCTURE (applied to {duration_label} format)
   - INTRO (first {intro_word_count} words): Hook + scope teaser. No lengthy preamble. Get to the first story within 20 words.
   - BODY (~{body_word_count} words): Cover stories by importance. Use verbal signposts: "But here's where it gets interesting..." / "Connecting back to what we covered earlier..." / "And this is the part that surprised me..."
   - EXTENDED ANALYSIS (~{analysis_word_count} words): Deeper dive on the lead story or a cross-story theme. Slower pace, more considered language.
   - CLOSE (final {close_word_count} words): Thematic recap — 2-3 sentences max per key story. End on a forward-looking note. No "thanks for listening" filler.
{music_cue_section}{voice_adherence_block}
--- DO NOT ---
- Do not write "(pause)" or "(beat)" in prose. For silence between speaker turns use ``[PAUSE_MS:N]`` tokens as described above; inside a single speaker line you may still use em dashes and ellipses for rhythm
- Do not write sound effect instructions
- Do not use anything other than ``Speaker 1:`` … ``Speaker {num_voices}:`` as the **start of a dialogue line** (before the first colon). Names and roles belong **inside** the line, not in the margin.
{do_not_spoken_line}- Do not repeat the episode title or date mid-script
- Do not let any story segment exceed {max_segment_words} words without a speaker exchange

--- QUALITY CHECK (apply before outputting) ---
Read the first and last line of your script. The first must contain a hook. The last must be a complete, resolved statement — not a trailing question.
Scan for any sentence over 30 words and break it.
Confirm {num_voices} distinct Speaker labels (Speaker 1..{num_voices}) are present with roughly balanced turns.
Re-read every ``[PAUSE_MS:N]`` token: values must **not** all be identical — vary N intentionally across the episode.

Generate the complete podcast script now:"""

        return prompt

    def _clean_script(self, script: str, num_voices: int, include_production_cues: bool = False) -> str:
        """
        Clean and format generated script.

        Args:
            script: Raw script from Ollama
            num_voices: Expected number of voices
            include_production_cues: When False, drop ``[CUE: ...]`` lines and inline markers

        Returns:
            Cleaned script
        """
        # Remove markdown code blocks if present
        if script.startswith("```"):
            lines = script.split("\n")
            # Remove first line if it's ``` or ```markdown
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            script = "\n".join(lines)

        script = _remove_placeholder_brackets(script)

        # Ensure proper speaker labels
        lines = script.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if _CUE_MARKER_BRACKET.match(line):
                if include_production_cues:
                    cleaned_lines.append(line)
                continue

            if not include_production_cues:
                line = _INLINE_PRODUCTION_CUE.sub(" ", line).strip()
                if not line:
                    continue

            # Check if line starts with a speaker label
            has_speaker = any(line.startswith(f"Speaker {i}:") for i in range(1, num_voices + 1))

            if not has_speaker and cleaned_lines:
                # If previous line was a speaker line, append to it
                if cleaned_lines and any(cleaned_lines[-1].startswith(f"Speaker {i}:") for i in range(1, num_voices + 1)):
                    cleaned_lines[-1] += " " + line
                    continue

            cleaned_lines.append(line)

        # Clean up any extra whitespace that might have been left after removing placeholders
        cleaned_script = "\n".join(cleaned_lines)
        cleaned_script = normalize_podcast_speaker_labels(
            cleaned_script, num_voices, include_production_cues=include_production_cues
        )
        # Remove multiple spaces and clean up spacing around punctuation
        cleaned_script = re.sub(r' +', ' ', cleaned_script)
        cleaned_script = re.sub(r' \.', '.', cleaned_script)
        cleaned_script = re.sub(r' ,', ',', cleaned_script)
        cleaned_script = re.sub(r' \?', '?', cleaned_script)
        cleaned_script = re.sub(r' !', '!', cleaned_script)

        cleaned_script = _inject_speaker_handoff_pauses(
            cleaned_script,
            include_production_cues=include_production_cues,
        )
        cleaned_script = redistribute_uniform_pause_markers(cleaned_script)

        return cleaned_script

    def infer_speaker_display_name_from_transcript(self, transcript: str) -> Optional[str]:
        """
        Extract a single self-introduced name from a short transcript, or None.
        Used by Speaker Voice Isolator when SPEAKER_NAME_INFERENCE includes ollama/both.
        """
        text = (transcript or "").strip()
        if len(text) < 4:
            return None
        system = (
            "You read a short transcript of one person speaking. "
            "If they clearly introduce themselves by name (examples: 'I'm Jordan', "
            "'My name is Maria Garcia', 'Call me Sam'), reply with ONLY that name "
            "using letters and spaces (1-4 words). "
            "If there is no clear self-introduction, or you are unsure, reply with exactly: NONE"
        )
        user = f"Transcript:\n{text[:2500]}"
        request_body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9},
        }
        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=request_body, timeout=90)
            response.raise_for_status()
            raw = ((response.json() or {}).get("message") or {}).get("content") or ""
            raw = raw.strip().strip('"').strip("'")
            if not raw or raw.upper() == "NONE":
                return None
            return raw
        except Exception as exc:
            logger.warning("Ollama speaker name inference failed: %s", exc)
            return None

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


# Global Ollama client instance
ollama_client = OllamaClient()
