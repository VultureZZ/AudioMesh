"""
ACE-Step API subprocess manager.

Starts and stops the ACE-Step API server on demand, and shuts it down after an
idle timeout so GPU memory is not kept allocated long-term.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..config import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MusicServerConfig:
    host: str
    port: int
    repo_dir: Path
    startup_timeout_seconds: float
    idle_shutdown_seconds: int
    server_command: Optional[str]
    config_path: str
    lm_model_path: str
    lm_backend: str
    device: str


class MusicProcessManager:
    """Manages a single local ACE-Step API subprocess."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._idle_timer: Optional[threading.Timer] = None

    def _current_cfg(self) -> MusicServerConfig:
        return MusicServerConfig(
            host=config.ACESTEP_HOST,
            port=config.ACESTEP_PORT,
            repo_dir=config.ACESTEP_REPO_DIR,
            startup_timeout_seconds=config.ACESTEP_STARTUP_TIMEOUT_SECONDS,
            idle_shutdown_seconds=config.ACESTEP_IDLE_SHUTDOWN_SECONDS,
            server_command=config.ACESTEP_SERVER_COMMAND,
            config_path=config.ACESTEP_CONFIG_PATH,
            lm_model_path=config.ACESTEP_LM_MODEL_PATH,
            lm_backend=config.ACESTEP_LM_BACKEND,
            device=config.ACESTEP_DEVICE,
        )

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _probe_health(host: str, port: int) -> dict:
        url = f"http://{host}:{port}/health"
        req = Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=2.0) as resp:
                body = resp.read(4096).decode("utf-8", "replace")
                if resp.status != 200:
                    raise RuntimeError(f"Unexpected status {resp.status} from {url}: {body[:200]}")
                try:
                    payload = json.loads(body)
                except Exception as exc:
                    raise RuntimeError(f"Non-JSON response from {url}: {body[:200]}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Unexpected health payload from {url}: {payload}")
                return payload
        except URLError as exc:
            raise RuntimeError(f"Failed to GET {url}: {exc}") from exc

    @staticmethod
    def _find_uv_executable(repo_dir: Path) -> str:
        project_uv = repo_dir / ".venv" / "bin" / "uv"
        if project_uv.exists():
            return str(project_uv)
        return "uv"

    def _build_default_command(self, cfg: MusicServerConfig) -> list[str]:
        uv_exec = self._find_uv_executable(cfg.repo_dir)
        return [
            uv_exec,
            "run",
            "acestep-api",
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
        ]

    def _build_env(self, cfg: MusicServerConfig) -> dict:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["ACESTEP_CONFIG_PATH"] = cfg.config_path
        env["ACESTEP_LM_MODEL_PATH"] = cfg.lm_model_path
        env["ACESTEP_LM_BACKEND"] = cfg.lm_backend
        env["ACESTEP_DEVICE"] = cfg.device
        return env

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_shutdown_locked(self, cfg: MusicServerConfig) -> None:
        self._cancel_idle_timer_locked()
        if cfg.idle_shutdown_seconds <= 0:
            return
        self._idle_timer = threading.Timer(cfg.idle_shutdown_seconds, self.stop)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def touch_activity(self) -> None:
        """Reset idle shutdown timer after music-related requests."""
        cfg = self._current_cfg()
        with self._lock:
            self._schedule_idle_shutdown_locked(cfg)

    def _start_locked(self, cfg: MusicServerConfig) -> None:
        if self._process and self._process.poll() is None:
            return

        if self._is_port_open(cfg.host, cfg.port):
            logger.warning(
                "ACE-Step port is already open on %s:%s. Reusing external process.",
                cfg.host,
                cfg.port,
            )
            self._schedule_idle_shutdown_locked(cfg)
            return

        if not cfg.repo_dir.exists():
            raise RuntimeError(
                f"ACE-Step repo not found at {cfg.repo_dir}. "
                "Run `python scripts/setup_vibevoice.py` to provision it, "
                "or clone https://github.com/ace-step/ACE-Step-1.5 and set ACESTEP_REPO_DIR."
            )

        if cfg.server_command:
            cmd = shlex.split(cfg.server_command)
        else:
            cmd = self._build_default_command(cfg)

        logger.info(
            "Starting ACE-Step API subprocess. cwd=%s host=%s port=%s cmd=%s",
            cfg.repo_dir,
            cfg.host,
            cfg.port,
            cmd,
        )

        self._process = subprocess.Popen(
            cmd,
            cwd=str(cfg.repo_dir),
            env=self._build_env(cfg),
        )
        self._schedule_idle_shutdown_locked(cfg)
        logger.info("ACE-Step API subprocess started (pid=%s).", self._process.pid)

    def ensure_running(self) -> MusicServerConfig:
        """Ensure ACE-Step API server is running and healthy."""
        cfg = self._current_cfg()
        with self._lock:
            self._start_locked(cfg)

        deadline = time.time() + cfg.startup_timeout_seconds
        while time.time() < deadline:
            if self._is_port_open(cfg.host, cfg.port):
                self._probe_health(cfg.host, cfg.port)
                self.touch_activity()
                return cfg

            with self._lock:
                if self._process and self._process.poll() is not None:
                    code = self._process.returncode
                    self._process = None
                    raise RuntimeError(
                        f"ACE-Step API process exited early with code {code}. "
                        "Check backend logs for subprocess errors."
                    )
            time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for ACE-Step API on {cfg.host}:{cfg.port} "
            f"after {cfg.startup_timeout_seconds}s."
        )

    def is_running(self) -> bool:
        cfg = self._current_cfg()
        with self._lock:
            if self._process and self._process.poll() is None:
                return True
        return self._is_port_open(cfg.host, cfg.port)

    def check_health_if_running(self) -> Optional[dict]:
        cfg = self._current_cfg()
        if not self._is_port_open(cfg.host, cfg.port):
            return None
        try:
            return self._probe_health(cfg.host, cfg.port)
        except Exception:
            return None

    def stop(self) -> None:
        """Stop ACE-Step API subprocess if it was started by this process."""
        with self._lock:
            self._cancel_idle_timer_locked()
            proc = self._process
            self._process = None

        if not proc or proc.poll() is not None:
            return

        logger.info("Stopping ACE-Step API subprocess (pid=%s).", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.info("ACE-Step API subprocess stopped.")


music_process_manager = MusicProcessManager()
