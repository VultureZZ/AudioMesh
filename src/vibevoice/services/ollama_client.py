"""
Ollama client service for LLM script generation.
"""
import logging
import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from ..config import config

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
        duration: str,
        num_voices: int,
        custom_model: Optional[str] = None,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate podcast script from article using Ollama.

        Args:
            article_text: Scraped article content
            genre: Podcast genre (Comedy, Serious, News, etc.)
            duration: Target duration (e.g., "5 min", "30 min")
            num_voices: Number of voices/speakers (1-4)
            custom_model: Optional custom model name (overrides default)
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order (maps to Speaker 1, 2, etc.)

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
        prompt = self._build_prompt(article_text, genre, duration, num_voices, voice_profiles, voice_names)

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
            script = self._clean_script(script, num_voices)

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
    ) -> List[Dict[str, Any]]:
        """
        Generate structured production cue segments from a podcast script.

        Args:
            script: Speaker-labeled script text
            custom_model: Optional model override

        Returns:
            List of segment dictionaries
        """
        model = custom_model or self.model
        prompt = self._build_segmentation_prompt(script)
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

    def _build_segmentation_prompt(self, script: str) -> str:
        return f"""You are a production assistant for podcast post-processing.

Convert the script into a strict JSON array of segments with cue injection points.

Allowed segment_type values:
- intro_music
- dialogue
- transition_sting
- music_bed
- outro_music

Rules:
1. Output ONLY JSON, no markdown fences or commentary.
2. For dialogue segments, include:
   - speaker (e.g., "Speaker 1")
   - text
   - start_time_hint (number, seconds from 0; monotonic non-decreasing)
3. Non-dialogue segments should still include start_time_hint.
4. Include intro_music near start (0-2s), outro_music near ending, transition_sting at major topic shifts, and optional music_bed markers.
5. Keep dialogue text faithful to the script.

SCRIPT:
{script}
"""

    def _parse_segment_json(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)

    def _validate_segments(self, parsed: Any) -> List[Dict[str, Any]]:
        allowed_types = {"intro_music", "dialogue", "transition_sting", "music_bed", "outro_music"}
        if not isinstance(parsed, list):
            raise RuntimeError("Segment response must be a JSON array")

        normalized: List[Dict[str, Any]] = []
        last_time = 0.0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            segment_type = str(item.get("segment_type") or "").strip()
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

            segment: Dict[str, Any] = {
                "segment_type": segment_type,
                "start_time_hint": start_time_hint,
            }
            if segment_type == "dialogue":
                speaker = str(item.get("speaker") or "").strip()
                text = str(item.get("text") or "").strip()
                if not speaker or not text:
                    continue
                segment["speaker"] = speaker
                segment["text"] = text
            normalized.append(segment)
        if not normalized:
            raise RuntimeError("No valid segments returned by Ollama")
        return normalized

    def _build_prompt(
        self,
        article_text: str,
        genre: str,
        duration: str,
        num_voices: int,
        voice_profiles: Optional[Dict[str, Dict]] = None,
        voice_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for Ollama script generation.

        Args:
            article_text: Article content
            genre: Podcast genre
            duration: Target duration
            num_voices: Number of voices
            voice_profiles: Optional dict mapping voice names to profile data
            voice_names: Optional list of voice names in order

        Returns:
            Formatted prompt string
        """
        # Map duration to approximate word count (without "words" suffix)
        duration_map = {
            "5 min": "500-700",
            "10 min": "1000-1400",
            "15 min": "1500-2100",
            "30 min": "3000-4200",
        }
        word_target = duration_map.get(duration, "1000-1400")

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
                voice_profiles_section = "\n\n**VOICE PROFILES:**\n" + "".join(profile_lines)
                voice_adherence_block = """
6. **Voice profile adherence**
   - Apply the **VOICE PROFILES** section to the matching Speaker number: that speaker's lines must reflect their profile (tone, cadence, structure, vocabulary, and signature phrases when natural).
   - Genre and podcast specifications set format and framing; speaker profiles set how each person talks. Do not ignore a speaker's profile in favor of generic dialogue.

"""

        # Build conditional speaker differentiation section
        speaker_differentiation = ""
        if num_voices == 1:
            speaker_differentiation = "   - Speaker 1: Solo host, presents all information and guides the narrative"
        elif num_voices == 2:
            speaker_differentiation = """   - Speaker 1: Primary host, guides the conversation, asks questions
   - Speaker 2: Co-host/expert, provides insights and elaboration"""
        else:
            # num_voices > 2
            speaker_differentiation = """   - Speaker 1: Primary host, guides the conversation, asks questions
   - Speaker 2: Co-host/expert, provides insights and elaboration
   - Additional speakers: Contribute unique perspectives or counterpoints"""

        prompt = f"""You are an expert podcast script writer specializing in creating natural, engaging dialogue for text-to-speech synthesis.
{voice_profiles_section}

**ARTICLE TO ADAPT:**
{article_text}

**PODCAST SPECIFICATIONS:**
- Genre: {genre}
- Tone/Style: {genre_style}
- Target word count: {word_target} words (~{duration})
- Speakers: {num_voices}

**OUTPUT FORMAT:**
Use exactly this format with no additional markup:

{chr(10).join(speaker_examples)}

**SCRIPT REQUIREMENTS:**

1. **Natural Speech Patterns**
   - Use contractions (don't, we're, that's)
   - Include verbal fillers sparingly (well, you know, I mean)
   - Add brief reactions (Exactly, Right, Interesting)
   - Vary sentence length—mix short punchy lines with longer explanations

2. **Speaker Differentiation**
{speaker_differentiation}

3. **Structure**
   - Opening hook (grab attention in first 2-3 exchanges)
   - Brief intro of topic
   - Main content covering key article points
   - Smooth transitions between subtopics
   - Memorable conclusion with takeaway

4. **TTS Optimization**
   - Avoid abbreviations (write "versus" not "vs.")
   - Spell out numbers under 10
   - Use em-dashes for natural pauses
   - Avoid complex punctuation that may confuse TTS

5. **Pacing**
   - Balance information density with breathing room
   - Include moments of agreement/reaction between substantive points
   - Aim for 130-150 words per minute of target duration
{voice_adherence_block}**DO NOT:**
- Include stage directions or notes in parentheses or brackets
- Use placeholders like [Host Name], [Co-host Name], [Name], etc. - write actual dialogue only
- Use asterisks or other formatting
- Add sound effect cues
- Add or imply background music, theme music, intro/outro music, ambiance, jingles, or any non-speech audio
- Include timestamps
- Include any bracketed placeholders or notes
- Start with greeting-style introductions (e.g., "Welcome to...", "Hello and welcome...", "Thanks for joining us...") or any "show intro" language

Generate the complete podcast script now:"""

        return prompt

    def _clean_script(self, script: str, num_voices: int) -> str:
        """
        Clean and format generated script.

        Args:
            script: Raw script from Ollama
            num_voices: Expected number of voices

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

        # Remove placeholders in brackets (e.g., [Host Name], [Co-host Name], [Name], etc.)
        # Pattern matches brackets with text inside, but preserves speaker labels like "Speaker 1:"
        # Remove placeholder and any space before it if followed by punctuation
        placeholder_pattern = r'\s*\[[^\]]+\]'
        script = re.sub(placeholder_pattern, '', script)

        # Ensure proper speaker labels
        lines = script.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
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
        # Remove multiple spaces and clean up spacing around punctuation
        cleaned_script = re.sub(r' +', ' ', cleaned_script)
        cleaned_script = re.sub(r' \.', '.', cleaned_script)
        cleaned_script = re.sub(r' ,', ',', cleaned_script)
        cleaned_script = re.sub(r' \?', '?', cleaned_script)
        cleaned_script = re.sub(r' !', '!', cleaned_script)

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
