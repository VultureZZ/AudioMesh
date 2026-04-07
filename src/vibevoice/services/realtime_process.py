"""
Realtime model server subprocess manager.

This backend does NOT run VibeVoice-Realtime-0.5B in-process. Instead, it launches
the official VibeVoice realtime websocket demo server as a subprocess and bridges
client WebSocket connections to it.

Upstream docs:
  - https://raw.githubusercontent.com/microsoft/VibeVoice/main/docs/vibevoice-realtime-0.5b.md
  - Usage 1: python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B

The upstream demo server exposes:
  - WS: /stream?text=...&cfg=...&steps=...&voice=...
  - GET: /config
"""

from __future__ import annotations

import logging
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..config import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeServerConfig:
    host: str
    port: int
    model_id: str
    device: str
    repo_dir: str
    startup_timeout_seconds: float
    server_command: Optional[str]


class RealtimeProcessManager:
    """
    Manages a single local realtime server subprocess.

    Notes:
    - The upstream demo server is not designed for multi-tenant concurrency; it
      uses a global lock. We treat it as a single shared resource.
    - We keep this manager process-global (module singleton) for simplicity.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._last_start_time: Optional[float] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_config_payload: Optional[dict] = None
        self._last_config_at: Optional[float] = None

    def _current_cfg(self) -> RealtimeServerConfig:
        return RealtimeServerConfig(
            host=config.REALTIME_HOST,
            port=config.REALTIME_PORT,
            model_id=config.REALTIME_MODEL_ID,
            device=config.REALTIME_DEVICE,
            repo_dir=str(config.REALTIME_VIBEVOICE_REPO_DIR),
            startup_timeout_seconds=config.REALTIME_STARTUP_TIMEOUT_SECONDS,
            server_command=config.REALTIME_SERVER_COMMAND,
        )

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _probe_upstream_http_config(host: str, port: int) -> dict:
        """
        Verify the upstream demo server is actually running by calling GET /config.

        The upstream demo (microsoft/VibeVoice demo/web/app.py) exposes:
          - GET /config  -> {"voices": [...], "default_voice": "..."}
        """
        url = f"http://{host}:{port}/config"
        req = Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=2) as resp:
                body = resp.read(4096).decode("utf-8", "replace")
                if resp.status != 200:
                    raise RuntimeError(f"Unexpected status {resp.status} from {url}: {body[:200]}")
                try:
                    import json

                    return json.loads(body)
                except Exception as e:
                    raise RuntimeError(f"Non-JSON response from {url}: {body[:200]}") from e
        except URLError as e:
            raise RuntimeError(f"Failed to GET {url}: {e}") from e

    def _build_default_command(self, cfg: RealtimeServerConfig) -> list[str]:
        script_path = (
            config.REALTIME_VIBEVOICE_REPO_DIR / "demo" / "vibevoice_realtime_demo.py"
        )
        if not script_path.exists():
            raise RuntimeError(
                "Realtime demo entrypoint not found. Expected "
                f"{script_path}. Clone microsoft/VibeVoice (or a compatible fork) "
                f"into {config.REALTIME_VIBEVOICE_REPO_DIR} and ensure the demo exists."
            )
        return [
            sys.executable,
            str(script_path),
            "--port",
            str(cfg.port),
            "--model_path",
            cfg.model_id,
            "--device",
            cfg.device,
        ]

    def _start_locked(self, cfg: RealtimeServerConfig) -> None:
        # If something is already listening, assume it is the realtime server.
        if self._is_port_open(cfg.host, cfg.port):
            if self._process and self._process.poll() is None:
                logger.info(
                    "Realtime server already running (pid=%s) on %s:%s (not starting another).",
                    self._process.pid,
                    cfg.host,
                    cfg.port,
                )
            else:
                logger.warning(
                    "Realtime port already open on %s:%s (not starting subprocess). "
                    "If realtime isn't working, another process may be occupying this port.",
                    cfg.host,
                    cfg.port,
                )
            return

        if self._process and self._process.poll() is None:
            # Process is alive but port isn't ready yet; fall through to wait.
            logger.info(
                "Realtime server process running (pid=%s) but port not ready yet.",
                self._process.pid,
            )
        else:
            self._process = None

        if cfg.server_command:
            cmd = shlex.split(cfg.server_command)
        else:
            cmd = self._build_default_command(cfg)

        logger.info(
            "Starting realtime server subprocess. cwd=%s host=%s port=%s model=%s device=%s cmd=%s",
            cfg.repo_dir,
            cfg.host,
            cfg.port,
            cfg.model_id,
            cfg.device,
            cmd,
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Prefer importing the upstream `vibevoice` package from the repo itself to avoid
        # conflicts with other installed `vibevoice` packages.
        env["PYTHONPATH"] = cfg.repo_dir + os.pathsep + env.get("PYTHONPATH", "")

        capture = env.get("REALTIME_CAPTURE_SUBPROCESS_LOGS", "").strip() == "1"
        stdout = subprocess.PIPE if capture else None
        stderr = subprocess.PIPE if capture else None

        self._process = subprocess.Popen(
            cmd,
            cwd=cfg.repo_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        self._last_start_time = time.time()

        if capture:
            self._start_log_threads_locked()

        logger.info("Realtime server subprocess started (pid=%s).", self._process.pid)

    def _start_log_threads_locked(self) -> None:
        proc = self._process
        if not proc:
            return

        def _pump(stream, level: int, name: str) -> None:
            try:
                if not stream:
                    return
                for raw in iter(stream.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip()
                    if line:
                        logger.log(level, "[realtime_subprocess:%s pid=%s] %s", name, proc.pid, line)
            except Exception:
                logger.exception("Failed reading realtime subprocess %s", name)

        if proc.stdout and not self._stdout_thread:
            self._stdout_thread = threading.Thread(
                target=_pump, args=(proc.stdout, logging.INFO, "stdout"), daemon=True
            )
            self._stdout_thread.start()
        if proc.stderr and not self._stderr_thread:
            self._stderr_thread = threading.Thread(
                target=_pump, args=(proc.stderr, logging.WARNING, "stderr"), daemon=True
            )
            self._stderr_thread.start()

    def ensure_running(self) -> RealtimeServerConfig:
        """
        Ensure the realtime server is running and accepting connections.

        Returns:
            The server config used for the running server.
        """
        cfg = self._current_cfg()
        with self._lock:
            self._start_locked(cfg)

        # Wait for readiness (port open) outside lock.
        logger.info(
            "Waiting for realtime server readiness on %s:%s (timeout=%ss).",
            cfg.host,
            cfg.port,
            cfg.startup_timeout_seconds,
        )
        deadline = time.time() + cfg.startup_timeout_seconds
        while time.time() < deadline:
            if self._is_port_open(cfg.host, cfg.port):
                logger.info("Realtime server port is open on %s:%s.", cfg.host, cfg.port)
                try:
                    cfg_payload = self._probe_upstream_http_config(cfg.host, cfg.port)
                    logger.info(
                        "Realtime server /config probe OK. default_voice=%s voices=%s",
                        cfg_payload.get("default_voice"),
                        len(cfg_payload.get("voices", [])) if isinstance(cfg_payload.get("voices"), list) else "unknown",
                    )
                    with self._lock:
                        self._last_config_payload = cfg_payload
                        self._last_config_at = time.time()
                except Exception as e:
                    # This is the key diagnostic for the "handshake timed out" case:
                    # port is open but the expected upstream server isn't responding correctly.
                    logger.error(
                        "Realtime server probe failed even though port is open: %s. "
                        "This usually means REALTIME_PORT is occupied by a different service.",
                        e,
                    )
                    raise
                return cfg

            # If the process died, surface a useful error early.
            with self._lock:
                if self._process and self._process.poll() is not None:
                    code = self._process.returncode
                    logger.error(
                        "Realtime server process exited early (code=%s).", code
                    )
                    self._process = None
                    raise RuntimeError(
                        f"Realtime server process exited early with code {code}. "
                        "Check logs above for details."
                    )

            time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for realtime server on {cfg.host}:{cfg.port} "
            f"after {cfg.startup_timeout_seconds}s."
        )

    def get_cached_upstream_config(self) -> Optional[dict]:
        with self._lock:
            return self._last_config_payload

    def stop(self) -> None:
        """Stop the realtime server subprocess (if started by this process)."""
        with self._lock:
            proc = self._process
            self._process = None

        if not proc or proc.poll() is not None:
            return

        logger.info("Stopping realtime server subprocess (pid=%s).", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.info("Realtime server subprocess stopped.")


# Global singleton
realtime_process_manager = RealtimeProcessManager()

