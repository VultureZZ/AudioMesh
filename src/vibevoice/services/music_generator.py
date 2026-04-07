"""
Music generation orchestration service for ACE-Step integration.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ..config import config
from .music_process import music_process_manager
from .ollama_client import ollama_client

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Orchestrates ACE-Step task submission, polling, and local file storage."""

    def __init__(self) -> None:
        self._task_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _base_url(host: str, port: int) -> str:
        return f"http://{host}:{port}"

    @staticmethod
    def _unwrap_response(payload: dict[str, Any]) -> Any:
        code = payload.get("code")
        if code != 200:
            raise RuntimeError(payload.get("error") or f"ACE-Step API error (code={code})")
        return payload.get("data")

    @staticmethod
    def _normalize_status(raw_status: int) -> str:
        if raw_status == 1:
            return "succeeded"
        if raw_status == 2:
            return "failed"
        return "running"

    async def _acestep_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = music_process_manager.ensure_running()
        music_process_manager.touch_activity()
        url = f"{self._base_url(cfg.host, cfg.port)}{path}"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload)
        music_process_manager.touch_activity()
        response.raise_for_status()
        return response.json()

    async def _acestep_get_bytes(self, path_or_url: str) -> bytes:
        cfg = music_process_manager.ensure_running()
        music_process_manager.touch_activity()
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"{self._base_url(cfg.host, cfg.port)}{path_or_url}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
        music_process_manager.touch_activity()
        response.raise_for_status()
        return response.content

    @staticmethod
    def _extract_extension(file_url: str, fallback: str = "mp3") -> str:
        parsed = urlparse(file_url)
        query_path = parse_qs(parsed.query).get("path", [""])[0]
        if query_path:
            suffix = Path(unquote(query_path)).suffix
            if suffix:
                return suffix.lstrip(".")
        suffix = Path(parsed.path).suffix
        if suffix:
            return suffix.lstrip(".")
        return fallback

    @staticmethod
    def _parse_result_payload(raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, list):
            return [x for x in raw_result if isinstance(x, dict)]
        if isinstance(raw_result, dict):
            return [raw_result]
        if isinstance(raw_result, str):
            text = raw_result.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        return []

    def _cache_task_result(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._task_cache[task_id] = payload

    def _get_cached_task_result(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._task_cache.get(task_id)

    async def _download_and_store(
        self,
        task_id: str,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out_dir = config.MUSIC_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        stored: list[dict[str, Any]] = []
        for idx, item in enumerate(items, 1):
            file_url = item.get("file")
            if not isinstance(file_url, str) or not file_url:
                continue

            ext = self._extract_extension(file_url, fallback="mp3")
            filename = f"music_{task_id}_{idx}.{ext}"
            out_path = out_dir / filename

            content = await self._acestep_get_bytes(file_url)
            out_path.write_bytes(content)
            stored.append(
                {
                    "filename": filename,
                    "audio_url": f"/api/v1/music/download/{filename}",
                    "file_path": str(out_path),
                    "seed_value": item.get("seed_value"),
                    "prompt": item.get("prompt"),
                    "lyrics": item.get("lyrics"),
                    "metas": item.get("metas") or {},
                    "dit_model": item.get("dit_model"),
                    "lm_model": item.get("lm_model"),
                }
            )
        return stored

    async def generate_music(self, params: dict[str, Any]) -> str:
        payload = await self._acestep_post("/release_task", params)
        data = self._unwrap_response(payload)
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected ACE-Step response format for task submission")
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("ACE-Step did not return a task_id")
        return task_id

    async def generate_cover(
        self,
        *,
        src_audio_path: Path,
        prompt: str = "",
        lyrics: str = "",
        duration: Optional[float] = None,
        audio_cover_strength: float = 0.6,
        vocal_language: Optional[str] = None,
        instrumental: bool = False,
        thinking: bool = True,
        inference_steps: int = 8,
        batch_size: int = 1,
        seed: int = -1,
        audio_format: str = "mp3",
    ) -> str:
        if not src_audio_path.exists():
            raise ValueError(f"Reference audio file not found: {src_audio_path}")

        normalized_strength = max(0.0, min(float(audio_cover_strength), 1.0))
        payload: dict[str, Any] = {
            "task_type": "cover",
            "src_audio_path": str(src_audio_path.resolve()),
            "audio_cover_strength": normalized_strength,
            "prompt": self._clean_text(prompt),
            "lyrics": self._clean_text(lyrics),
            "duration": duration,
            "vocal_language": self._clean_text(vocal_language),
            "instrumental": instrumental,
            "thinking": thinking,
            "inference_steps": inference_steps,
            "batch_size": max(1, min(batch_size, 4)),
            "seed": seed,
            "audio_format": audio_format,
        }

        if instrumental:
            payload.pop("lyrics", None)
            payload.pop("vocal_language", None)

        clean_payload = {k: v for k, v in payload.items() if v not in (None, "")}
        return await self.generate_music(clean_payload)

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        return (value or "").strip()

    @staticmethod
    def _parse_ollama_json(response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        if not text:
            raise RuntimeError("Ollama returned an empty refinement response")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError("Ollama refinement response was not valid JSON")
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError("Ollama refinement response could not be parsed as JSON") from exc

    def _refine_prompt_with_ollama(
        self,
        *,
        description: str,
        instrumental: bool,
        vocal_language: Optional[str],
        duration: Optional[float],
    ) -> dict[str, Any]:
        language = (vocal_language or "en").strip().lower() if not instrumental else None
        refinement_prompt = (
            "You are a music prompt refiner for ACE-Step.\n"
            "Convert user intent into strict JSON only.\n"
            "Do not include markdown or explanations.\n\n"
            "Output schema:\n"
            "{\n"
            '  "caption": "string",\n'
            '  "lyrics": "string (optional, empty for instrumental)",\n'
            '  "bpm": 120,\n'
            '  "keyscale": "C Major",\n'
            '  "timesignature": "4"\n'
            "}\n\n"
            "Rules:\n"
            "- Keep places, names, and narrative details faithful to user input.\n"
            "- Never switch language.\n"
            "- Keep vocals language aligned with requested language.\n"
            "- Keep output concise and model-friendly.\n"
            "- Return empty lyrics for instrumental tracks.\n\n"
            f"User description: {description}\n"
            f"Instrumental: {instrumental}\n"
            f"Vocal language: {language or 'n/a'}\n"
            f"Target duration seconds: {int(duration) if duration and duration > 0 else 'auto'}\n"
        )

        response = ollama_client.client.post(
            f"{ollama_client.base_url}/api/generate",
            json={
                "model": ollama_client.model,
                "prompt": refinement_prompt,
                "stream": False,
                "options": {"temperature": 0.4, "top_p": 0.9},
            },
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        parsed = self._parse_ollama_json(raw_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Ollama refinement output had unexpected shape")
        return parsed

    def prepare_simple_payload(
        self,
        *,
        description: str,
        input_mode: Literal["refine", "exact"] = "refine",
        instrumental: bool,
        vocal_language: Optional[str] = None,
        duration: Optional[float] = None,
        batch_size: int = 1,
        exact_caption: Optional[str] = None,
        exact_lyrics: Optional[str] = None,
        exact_bpm: Optional[int] = None,
        exact_keyscale: Optional[str] = None,
        exact_timesignature: Optional[str] = None,
    ) -> dict[str, Any]:
        description_text = self._clean_text(description)

        if input_mode not in {"refine", "exact"}:
            raise ValueError("input_mode must be 'refine' or 'exact'")

        normalized_language = (vocal_language or "en").strip().lower() if not instrumental else None
        payload: dict[str, Any] = {
            "thinking": True,
            "batch_size": max(1, min(batch_size, 4)),
            "instrumental": instrumental,
            "use_cot_caption": False,
            "use_cot_language": False,
            "use_cot_metas": False,
            "use_format": False,
        }

        if duration and duration > 0:
            payload["duration"] = duration
        if normalized_language:
            payload["vocal_language"] = normalized_language

        if input_mode == "exact":
            caption = self._clean_text(exact_caption)
            lyrics = self._clean_text(exact_lyrics)
            if not caption and not lyrics:
                raise ValueError("Exact mode requires at least one of exact_caption or exact_lyrics")

            payload["prompt"] = caption or description_text or ("instrumental track" if instrumental else "song")
            if not instrumental and lyrics:
                payload["lyrics"] = lyrics
            if exact_bpm is not None:
                payload["bpm"] = exact_bpm
            if exact_keyscale:
                payload["keyscale"] = exact_keyscale.strip()
            if exact_timesignature:
                payload["timesignature"] = exact_timesignature.strip()
        else:
            if not description_text:
                raise ValueError("Description is required in refine mode")
            refined = self._refine_prompt_with_ollama(
                description=description_text,
                instrumental=instrumental,
                vocal_language=normalized_language,
                duration=duration,
            )
            caption = self._clean_text(str(refined.get("caption", "")))
            lyrics = self._clean_text(str(refined.get("lyrics", "")))
            payload["prompt"] = caption or description_text
            if not instrumental and lyrics:
                payload["lyrics"] = lyrics

            bpm = refined.get("bpm")
            if isinstance(bpm, int) and 30 <= bpm <= 300:
                payload["bpm"] = bpm
            keyscale = refined.get("keyscale")
            if isinstance(keyscale, str) and keyscale.strip():
                payload["keyscale"] = keyscale.strip()
            timesignature = refined.get("timesignature")
            if isinstance(timesignature, str) and timesignature.strip():
                payload["timesignature"] = timesignature.strip()

        if instrumental:
            payload.pop("lyrics", None)
            payload.pop("vocal_language", None)

        logger.info(
            "Prepared simple music payload: mode=%s instrumental=%s keys=%s",
            input_mode,
            instrumental,
            sorted(payload.keys()),
        )
        return payload

    async def create_sample(
        self,
        description: str,
        input_mode: Literal["refine", "exact"] = "refine",
        instrumental: bool = False,
        vocal_language: Optional[str] = None,
        duration: Optional[float] = None,
        batch_size: int = 1,
        exact_caption: Optional[str] = None,
        exact_lyrics: Optional[str] = None,
        exact_bpm: Optional[int] = None,
        exact_keyscale: Optional[str] = None,
        exact_timesignature: Optional[str] = None,
        prepared_payload: Optional[dict[str, Any]] = None,
    ) -> str:
        payload = prepared_payload or self.prepare_simple_payload(
            description=description,
            input_mode=input_mode,
            instrumental=instrumental,
            vocal_language=vocal_language,
            duration=duration,
            batch_size=batch_size,
            exact_caption=exact_caption,
            exact_lyrics=exact_lyrics,
            exact_bpm=exact_bpm,
            exact_keyscale=exact_keyscale,
            exact_timesignature=exact_timesignature,
        )
        return await self.generate_music(payload)

    async def get_status(self, task_id: str) -> dict[str, Any]:
        payload = await self._acestep_post("/query_result", {"task_id_list": [task_id]})
        data = self._unwrap_response(payload)
        if not isinstance(data, list) or not data:
            raise RuntimeError("ACE-Step returned an empty status response")
        task_data = data[0] if isinstance(data[0], dict) else {}
        raw_status = int(task_data.get("status", 0))
        status = self._normalize_status(raw_status)

        if status != "succeeded":
            return {
                "task_id": task_id,
                "status": status,
                "audios": [],
                "metadata": [],
                "error": task_data.get("error"),
            }

        cached = self._get_cached_task_result(task_id)
        if cached is not None:
            return cached

        raw_result = task_data.get("result")
        parsed_items = self._parse_result_payload(raw_result)
        local_items = await self._download_and_store(task_id, parsed_items)
        result = {
            "task_id": task_id,
            "status": "succeeded",
            "audios": [item["audio_url"] for item in local_items],
            "metadata": local_items,
            "error": None,
        }
        self._cache_task_result(task_id, result)
        return result

    async def generate_lyrics(
        self,
        description: str,
        genre: str,
        mood: str,
        language: str,
        duration_hint: Optional[str] = None,
    ) -> dict[str, str]:
        prompt = (
            "You are a songwriting assistant for a music generation app.\n"
            "Generate lyrics only and an optional style caption.\n\n"
            f"Description: {description}\n"
            f"Genre: {genre}\n"
            f"Mood: {mood}\n"
            f"Language: {language}\n"
            f"Duration hint: {duration_hint or 'auto'}\n\n"
            "Output format strictly:\n"
            "Caption: <short production/style tags for model prompt>\n"
            "Lyrics:\n"
            "[Verse 1]\n"
            "... lyrics ...\n"
            "[Chorus]\n"
            "... lyrics ...\n"
            "[Verse 2]\n"
            "... lyrics ...\n"
            "[Bridge]\n"
            "... lyrics ...\n"
            "[Chorus]\n"
            "... lyrics ...\n\n"
            "Do not include explanations."
        )

        response = ollama_client.client.post(
            f"{ollama_client.base_url}/api/generate",
            json={
                "model": ollama_client.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8, "top_p": 0.9},
            },
        )
        response.raise_for_status()
        body = response.json().get("response", "").strip()
        if not body:
            raise RuntimeError("Lyrics generation returned an empty response")

        caption = ""
        lyrics = body
        lines = body.splitlines()
        if lines and lines[0].lower().startswith("caption:"):
            caption = lines[0].split(":", 1)[1].strip()
            remaining = "\n".join(lines[1:]).strip()
            if remaining.lower().startswith("lyrics:"):
                remaining = remaining.split(":", 1)[1].strip()
            lyrics = remaining.strip() or lyrics

        return {"caption": caption, "lyrics": lyrics}

    def health(self) -> dict[str, Any]:
        probe = music_process_manager.check_health_if_running()
        return {
            "available": bool(config.ACESTEP_REPO_DIR.exists()),
            "running": probe is not None,
            "service": "acestep",
            "host": config.ACESTEP_HOST,
            "port": config.ACESTEP_PORT,
        }


music_generator = MusicGenerator()
