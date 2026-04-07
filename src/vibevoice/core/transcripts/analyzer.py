"""
LLM transcript analysis service.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from ...config import config

logger = logging.getLogger(__name__)


def _duration_formatted(duration_seconds: Optional[float]) -> str:
    if not duration_seconds:
        return ""
    total = int(duration_seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


class TranscriptAnalyzer:
    async def analyze(
        self,
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        recording_type: str,
        duration_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        transcript_text = self._format_transcript(segments, speakers)
        provider = (config.LLM_PROVIDER or "anthropic").lower().strip()
        try:
            if provider == "anthropic":
                parsed = await self._analyze_with_anthropic(transcript_text, speakers, recording_type)
            elif provider == "openai":
                parsed = await self._analyze_with_openai(transcript_text, speakers, recording_type)
            else:
                parsed = await self._analyze_with_ollama(transcript_text, speakers, recording_type)
        except Exception as exc:
            logger.warning("LLM analysis failed, falling back to heuristic summary: %s", exc)
            parsed = self._heuristic_fallback(segments, speakers, recording_type)

        parsed.setdefault("summary", "No summary generated.")
        parsed.setdefault("action_items", [])
        parsed.setdefault("key_decisions", [])
        parsed.setdefault("open_questions", [])
        parsed.setdefault("topics_discussed", [])
        parsed.setdefault("sentiment", "neutral")
        parsed["duration_formatted"] = _duration_formatted(duration_seconds)
        return parsed

    def _format_transcript(self, segments: list[dict[str, Any]], speakers: list[dict[str, Any]]) -> str:
        labels = {s.get("id"): s.get("label") or s.get("id") for s in speakers}
        rows = []
        for seg in segments:
            sid = seg.get("speaker_id") or "SPEAKER_00"
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            rows.append(f"{labels.get(sid, sid)}: {text}")
        return "\n".join(rows)

    def _build_prompt(self, transcript_text: str, speakers: list[dict[str, Any]], recording_type: str) -> str:
        speaker_names = [s.get("label") or s.get("id") for s in speakers]
        focus = {
            "meeting": "Include action items, decisions, and open questions.",
            "call": "Include action items, commitments, and unresolved questions.",
            "memo": "Prioritize concise summary and topics. Action items are optional.",
            "interview": "Prioritize key points, speaker viewpoints, and major takeaways.",
            "other": "Provide a clear summary and useful extracted items.",
        }.get(recording_type, "Provide a clear summary and useful extracted items.")
        return (
            "You are an expert transcript analyst. Return valid JSON only.\n"
            f"Recording type: {recording_type}\n"
            f"Speakers: {speaker_names}\n"
            f"Guidance: {focus}\n\n"
            "Return this shape:\n"
            "{"
            '"summary": string, '
            '"action_items": [{"action": string, "owner": string|null, "due_hint": string|null, "priority": "low|medium|high"}], '
            '"key_decisions": [string], '
            '"open_questions": [string], '
            '"topics_discussed": [string], '
            '"sentiment": string'
            "}\n\n"
            f"Transcript:\n{transcript_text}\n"
        )

    async def _analyze_with_anthropic(
        self,
        transcript_text: str,
        speakers: list[dict[str, Any]],
        recording_type: str,
    ) -> dict[str, Any]:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        prompt = self._build_prompt(transcript_text, speakers, recording_type)
        headers = {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": config.LLM_MODEL,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        chunks = data.get("content") or []
        text = "".join([c.get("text", "") for c in chunks if isinstance(c, dict)])
        return self._parse_json(text)

    async def _analyze_with_openai(
        self,
        transcript_text: str,
        speakers: list[dict[str, Any]],
        recording_type: str,
    ) -> dict[str, Any]:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        prompt = self._build_prompt(transcript_text, speakers, recording_type)
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "content-type": "application/json",
        }
        body = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return self._parse_json(text)

    async def _analyze_with_ollama(
        self,
        transcript_text: str,
        speakers: list[dict[str, Any]],
        recording_type: str,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(transcript_text, speakers, recording_type)
        payload = {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{config.OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return self._parse_json(data.get("response", ""))

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return json.loads(text)

    def _heuristic_fallback(
        self,
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        recording_type: str,
    ) -> dict[str, Any]:
        lines = []
        for s in segments[:15]:
            txt = (s.get("text") or "").strip()
            if txt:
                lines.append(txt)
        summary = " ".join(lines)
        if len(summary) > 600:
            summary = summary[:600] + "..."
        topics = []
        if recording_type in {"meeting", "call"}:
            topics = ["discussion", "follow-up", "next steps"]
        elif recording_type == "memo":
            topics = ["personal notes"]
        elif recording_type == "interview":
            topics = ["questions", "responses"]
        else:
            topics = ["transcript overview"]
        return {
            "summary": summary or "Transcript processed successfully.",
            "action_items": [],
            "key_decisions": [],
            "open_questions": [],
            "topics_discussed": topics,
            "sentiment": "neutral",
        }


transcript_analyzer = TranscriptAnalyzer()

