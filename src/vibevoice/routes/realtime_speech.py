"""
Realtime speech generation WebSocket endpoint.

This endpoint bridges the browser to the upstream VibeVoice-Realtime demo server.

Browser protocol (JSON control messages, binary audio frames):
  - start: { "type": "start", "cfg_scale"?: number, "inference_steps"?: number, "voice"?: string }
  - text:  { "type": "text", "text": string }   (buffered)
  - flush: { "type": "flush" }                  (start generating current buffer)
  - stop:  { "type": "stop" }                   (cancel current generation)

Server -> browser:
  - binary frames: PCM16LE mono @ 24000Hz chunks
  - JSON status/log frames: { "type": "status" | "error" | "end", ... }

Upstream server notes:
  - It expects the full text at WS connect time via the `text` query param.
  - It may interleave JSON log frames (text) and raw audio bytes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import websockets
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from ..config import config
from ..services.realtime_process import realtime_process_manager


router = APIRouter(prefix="/api/v1/speech", tags=["speech"])
logger = logging.getLogger(__name__)


@dataclass
class _WsLimiter:
    requests_per_minute: int
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._timestamps: dict[str, list[float]] = {}

    async def allow(self, key: str) -> tuple[bool, int, int]:
        now = time.time()
        cutoff = now - self.window_seconds
        async with self._lock:
            ts = self._timestamps.get(key, [])
            ts = [t for t in ts if t > cutoff]
            if len(ts) >= self.requests_per_minute:
                self._timestamps[key] = ts
                return False, self.requests_per_minute, 0
            ts.append(now)
            self._timestamps[key] = ts
            remaining = self.requests_per_minute - len(ts)
            return True, self.requests_per_minute, remaining


# One shared limiter instance for WS connections/messages.
_connection_limiter = _WsLimiter(requests_per_minute=config.RATE_LIMIT_PER_MINUTE)
_message_limiter = _WsLimiter(requests_per_minute=config.RATE_LIMIT_PER_MINUTE * 10)


def _extract_api_key(ws: WebSocket) -> Optional[str]:
    # Prefer query param (easy for browser WS), fall back to header for advanced clients.
    key = ws.query_params.get("api_key")
    if key:
        return key
    return ws.headers.get("x-api-key")


@router.websocket("/realtime")
async def realtime_speech(ws: WebSocket) -> None:
    conn_id = f"ws-{id(ws)}"
    api_key = _extract_api_key(ws)
    if not config.validate_api_key(api_key):
        logger.info("[%s] Rejecting websocket: invalid/missing API key", conn_id)
        await ws.close(code=1008, reason="Invalid or missing API key")
        return

    # Track key for rate limiting.
    rate_key = api_key or "anonymous"
    allowed, limit, remaining = await _connection_limiter.allow(rate_key)
    if not allowed:
        logger.info("[%s] Rejecting websocket: rate limit exceeded (key=%s)", conn_id, rate_key)
        await ws.accept()
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "message": f"Rate limit exceeded: {limit} connections per minute",
                    "retry_after_seconds": 60,
                }
            )
        )
        await ws.close(code=1013, reason="Rate limit exceeded")
        return

    await ws.accept()
    logger.info(
        "[%s] WebSocket accepted. client=%s rate_key=%s remaining_connections=%s",
        conn_id,
        getattr(ws.client, "host", None),
        rate_key,
        remaining,
    )

    # Session state
    buffered_text = ""
    cfg_scale: float = 1.5
    inference_steps: Optional[int] = None
    voice: Optional[str] = None

    upstream_ws: Optional[websockets.WebSocketClientProtocol] = None
    upstream_task: Optional[asyncio.Task[None]] = None

    async def send_status(event: str, data: Optional[dict[str, Any]] = None) -> None:
        logger.info("[%s] status=%s data=%s", conn_id, event, data)
        await ws.send_text(
            json.dumps(
                {
                    "type": "status",
                    "event": event,
                    "data": data or {},
                }
            )
        )

    async def send_error(message: str) -> None:
        logger.warning("[%s] error=%s", conn_id, message)
        await ws.send_text(json.dumps({"type": "error", "message": message}))

    async def close_upstream() -> None:
        nonlocal upstream_ws, upstream_task
        task = upstream_task
        upstream_task = None
        if task:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        if upstream_ws:
            try:
                await upstream_ws.close()
            except Exception:
                pass
            upstream_ws = None

    async def start_generation(text_to_generate: str) -> None:
        nonlocal upstream_ws, upstream_task
        if not text_to_generate.strip():
            await send_error("No text to generate. Send {type:'text'} then {type:'flush'}.")
            return
        if upstream_task:
            await send_error("Generation already in progress.")
            return

        logger.info("[%s] Starting generation. text_len=%s", conn_id, len(text_to_generate))
        srv = realtime_process_manager.ensure_running()
        upstream_cfg = realtime_process_manager.get_cached_upstream_config() or {}
        available_voices = upstream_cfg.get("voices", []) if isinstance(upstream_cfg, dict) else []
        default_voice = upstream_cfg.get("default_voice") if isinstance(upstream_cfg, dict) else None

        # Share upstream available presets with the client (for UI selection/debugging).
        if isinstance(available_voices, list) and available_voices:
            await send_status(
                "upstream_voice_presets",
                {"voices": available_voices, "default_voice": default_voice},
            )

        params = [f"text={quote(text_to_generate)}"]
        if cfg_scale:
            params.append(f"cfg={quote(str(cfg_scale))}")
        if inference_steps is not None:
            params.append(f"steps={quote(str(inference_steps))}")
        if voice:
            if isinstance(available_voices, list) and voice not in available_voices:
                await send_status(
                    "voice_not_found",
                    {
                        "requested_voice": voice,
                        "default_voice": default_voice,
                        "available_voice_count": len(available_voices),
                        "note": "Requested voice isn't supported by the upstream realtime demo; it will fall back to default.",
                    },
                )
            else:
                params.append(f"voice={quote(voice)}")
                await send_status("voice_selected", {"voice": voice})

        upstream_url = f"ws://{srv.host}:{srv.port}/stream?{'&'.join(params)}"
        await send_status("upstream_connecting", {"url": upstream_url, "host": srv.host, "port": srv.port})

        try:
            upstream_ws = await websockets.connect(
                upstream_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=10,
                close_timeout=5,
            )
        except Exception as e:
            logger.exception("[%s] Upstream websocket connect failed", conn_id)
            raise RuntimeError(f"Upstream websocket connect failed: {e}") from e

        async def _forward() -> None:
            bytes_sent = 0
            binary_frames = 0
            text_frames = 0
            first_binary_at: Optional[float] = None
            started_at = time.time()
            try:
                await send_status("upstream_connected")
                async for message in upstream_ws:
                    # Upstream interleaves JSON logs (text) and audio bytes.
                    if isinstance(message, (bytes, bytearray)):
                        payload = bytes(message)
                        if first_binary_at is None:
                            first_binary_at = time.time()
                            logger.info(
                                "[%s] First audio bytes from upstream after %.3fs",
                                conn_id,
                                first_binary_at - started_at,
                            )
                        binary_frames += 1
                        bytes_sent += len(payload)
                        await ws.send_bytes(payload)
                        continue

                    # Forward upstream log frames as status for UI visibility.
                    text_frames += 1
                    try:
                        payload = json.loads(message)
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "status",
                                    "event": "upstream_log",
                                    "data": payload,
                                }
                            )
                        )
                    except Exception:
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "status",
                                    "event": "upstream_text",
                                    "data": {"message": message},
                                }
                            )
                        )
            finally:
                close_code = getattr(upstream_ws, "close_code", None)
                close_reason = getattr(upstream_ws, "close_reason", None)
                logger.info(
                    "[%s] Upstream stream ended. binary_frames=%s bytes=%s text_frames=%s close_code=%s close_reason=%s",
                    conn_id,
                    binary_frames,
                    bytes_sent,
                    text_frames,
                    close_code,
                    close_reason,
                )
                await send_status("generation_complete")
                await ws.send_text(json.dumps({"type": "end"}))
                await close_upstream()

        upstream_task = asyncio.create_task(_forward())

    try:
        await send_status(
            "connected",
            {
                "sample_rate": 24000,
                "note": "Upstream VibeVoice-Realtime demo accepts full text at connect time. "
                "This API buffers text until you send {type:'flush'}.",
                "rate_limit_remaining_connections": remaining,
            },
        )

        while True:
            allowed, _, _ = await _message_limiter.allow(rate_key)
            if not allowed:
                await send_error("Rate limit exceeded for messages. Try again later.")
                await ws.close(code=1013, reason="Rate limit exceeded")
                return

            raw = await ws.receive_text()
            logger.info("[%s] received_text=%s", conn_id, raw[:500])
            try:
                msg = json.loads(raw)
            except Exception:
                await send_error("Invalid JSON message.")
                continue

            msg_type = msg.get("type")
            if msg_type == "start":
                # Optional session parameters; generation starts on flush.
                cfg_scale = float(msg.get("cfg_scale", cfg_scale))
                steps_val = msg.get("inference_steps", inference_steps)
                inference_steps = int(steps_val) if steps_val is not None else None
                voice = msg.get("voice", voice)
                await send_status(
                    "session_started",
                    {"cfg_scale": cfg_scale, "inference_steps": inference_steps, "voice": voice},
                )
            elif msg_type == "text":
                text = msg.get("text")
                if not isinstance(text, str):
                    await send_error("Expected {type:'text', text:string}.")
                    continue
                buffered_text += text
                await send_status("text_buffered", {"length": len(buffered_text)})
            elif msg_type == "flush":
                to_generate = buffered_text
                buffered_text = ""
                await start_generation(to_generate)
            elif msg_type == "stop":
                buffered_text = ""
                await close_upstream()
                await send_status("stopped")
            else:
                await send_error(f"Unknown message type: {msg_type!r}")

    except WebSocketDisconnect:
        logger.info("[%s] WebSocketDisconnect", conn_id)
        await close_upstream()
    except Exception as e:
        logger.exception("[%s] WebSocket handler error", conn_id)
        try:
            await send_error(str(e))
        except Exception:
            pass
        await close_upstream()
        try:
            await ws.close(code=1011, reason="Server error")
        except Exception:
            pass
    finally:
        logger.info("[%s] WebSocket handler exit", conn_id)

