"""
OpenAI Chat Completions for podcast script and segmentation.
Reuses prompt construction and post-processing from OllamaClient.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from .ollama_client import ollama_client

logger = logging.getLogger(__name__)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _resolve_model(model: Optional[str]) -> str:
    m = (model or "").strip()
    return m if m else DEFAULT_OPENAI_MODEL


def _chat_message_content(
    api_key: str,
    model: str,
    user_content: str,
    *,
    temperature: float,
    timeout: float = 300.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(OPENAI_CHAT_COMPLETIONS_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return (text or "").strip()


def generate_script_openai(
    article_text: str,
    genre: str,
    duration: Optional[str],
    num_voices: int,
    *,
    api_key: str,
    model: Optional[str] = None,
    voice_profiles: Optional[Dict[str, Dict]] = None,
    voice_names: Optional[List[str]] = None,
    narrator_speaker_index: Optional[int] = None,
    approximate_duration_minutes: Optional[float] = None,
    include_production_cues: bool = False,
) -> str:
    """Generate a podcast script via OpenAI using the same prompt as Ollama."""
    resolved_model = _resolve_model(model)

    max_article_length = 8000
    text = article_text
    if len(text) > max_article_length:
        logger.info(
            "Truncating article from %d to %d characters",
            len(text),
            max_article_length,
        )
        text = text[:max_article_length] + "\n[... article truncated ...]"

    prompt = ollama_client._build_prompt(
        text,
        genre,
        duration,
        num_voices,
        voice_profiles,
        voice_names,
        narrator_speaker_index=narrator_speaker_index,
        approximate_duration_minutes=approximate_duration_minutes,
        include_production_cues=include_production_cues,
    )

    logger.info(
        "Generating script with OpenAI (model=%s, genre=%s, duration=%s)",
        resolved_model,
        genre,
        duration,
    )
    try:
        raw = _chat_message_content(api_key, resolved_model, prompt, temperature=0.7)
    except httpx.HTTPError as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e
    if not raw:
        raise RuntimeError("OpenAI returned empty script")
    return ollama_client._clean_script(
        raw,
        num_voices,
        include_production_cues=include_production_cues,
    )


def generate_script_segments_openai(
    script: str,
    *,
    api_key: str,
    model: Optional[str] = None,
    estimated_duration_seconds: float,
    num_voices: int,
    genre: str,
    genre_style: str,
) -> List[Dict[str, Any]]:
    """Structured production segments via OpenAI, same prompt/schema as Ollama."""
    resolved_model = _resolve_model(model)
    prompt = ollama_client._build_segmentation_prompt(
        script,
        estimated_duration_seconds,
        num_voices,
        genre,
        genre_style,
    )
    try:
        raw = _chat_message_content(api_key, resolved_model, prompt, temperature=0.2)
    except httpx.HTTPError as e:
        raise RuntimeError(f"OpenAI API request failed for segmentation: {e}") from e
    if not raw:
        raise RuntimeError("OpenAI returned empty segmentation response")
    parsed = ollama_client._parse_segment_json(raw)
    return ollama_client._validate_segments(parsed)
