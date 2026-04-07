"""
Music presets and generation history storage using JSON files.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import config
from .music_presets import DEFAULT_MUSIC_PRESETS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MusicStorage:
    """Thread-safe storage for music presets and generation history."""

    def __init__(self, storage_file: Optional[Path] = None) -> None:
        if storage_file is None:
            storage_file = config.MUSIC_OUTPUT_DIR / "music_library.json"
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        with self.lock:
            payload: Dict[str, Any] = {"presets": {}, "history": {}}

            if self.storage_file.exists():
                try:
                    loaded = json.loads(self.storage_file.read_text())
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = {"presets": {}, "history": {}}

            payload.setdefault("presets", {})
            payload.setdefault("history", {})

            if not payload["presets"]:
                now = _utc_now_iso()
                seeded_presets: Dict[str, Dict[str, Any]] = {}
                for preset in DEFAULT_MUSIC_PRESETS:
                    preset_id = str(uuid4())
                    seeded_presets[preset_id] = {
                        "name": str(preset.get("name", "")).strip(),
                        "mode": str(preset.get("mode", "custom")).strip(),
                        "values": dict(preset.get("values", {})),
                        "created_at": now,
                        "updated_at": now,
                    }
                payload["presets"] = seeded_presets

            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            self.storage_file.write_text(json.dumps(payload, indent=2))

    def _load(self) -> Dict[str, Any]:
        try:
            if self.storage_file.exists():
                payload = json.loads(self.storage_file.read_text())
                if not isinstance(payload, dict):
                    return {"presets": {}, "history": {}}
                payload.setdefault("presets", {})
                payload.setdefault("history", {})
                return payload
        except Exception:
            pass
        return {"presets": {}, "history": {}}

    def _save(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            payload.setdefault("presets", {})
            payload.setdefault("history", {})
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            self.storage_file.write_text(json.dumps(payload, indent=2))

    def list_presets(self) -> List[Dict[str, Any]]:
        payload = self._load()
        items: List[Dict[str, Any]] = []
        for preset_id, data in payload.get("presets", {}).items():
            if not isinstance(data, dict):
                continue
            item = data.copy()
            item["id"] = preset_id
            items.append(item)
        items.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
        return items

    def create_preset(self, name: str, mode: str, values: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._load()
        preset_id = str(uuid4())
        now = _utc_now_iso()
        payload["presets"][preset_id] = {
            "name": name,
            "mode": mode,
            "values": values,
            "created_at": now,
            "updated_at": now,
        }
        self._save(payload)
        return {"id": preset_id, **payload["presets"][preset_id]}

    def update_preset(
        self,
        preset_id: str,
        name: Optional[str] = None,
        mode: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload = self._load()
        item = payload.get("presets", {}).get(preset_id)
        if not isinstance(item, dict):
            return None
        if name is not None:
            item["name"] = name
        if mode is not None:
            item["mode"] = mode
        if values is not None:
            item["values"] = values
        item["updated_at"] = _utc_now_iso()
        self._save(payload)
        return {"id": preset_id, **item}

    def delete_preset(self, preset_id: str) -> bool:
        payload = self._load()
        presets = payload.get("presets", {})
        if preset_id not in presets:
            return False
        del presets[preset_id]
        self._save(payload)
        return True

    def list_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        payload = self._load()
        items: List[Dict[str, Any]] = []
        for history_id, data in payload.get("history", {}).items():
            if not isinstance(data, dict):
                continue
            item = data.copy()
            item["id"] = history_id
            items.append(item)
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items[: max(1, limit)]

    def create_history_entry(
        self,
        task_id: str,
        mode: str,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = self._load()
        history_id = str(uuid4())
        now = _utc_now_iso()
        payload["history"][history_id] = {
            "task_id": task_id,
            "mode": mode,
            "status": "running",
            "request_payload": request_payload,
            "audios": [],
            "metadata": [],
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._save(payload)
        return {"id": history_id, **payload["history"][history_id]}

    def update_history_by_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        audios: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
        payload = self._load()
        history = payload.get("history", {})
        changed = False
        for item in history.values():
            if not isinstance(item, dict):
                continue
            if item.get("task_id") != task_id:
                continue
            if status is not None:
                item["status"] = status
            if audios is not None:
                item["audios"] = audios
            if metadata is not None:
                item["metadata"] = metadata
            if error is not None:
                item["error"] = error
            item["updated_at"] = _utc_now_iso()
            changed = True
        if changed:
            self._save(payload)

    def delete_history(self, history_id: str) -> bool:
        payload = self._load()
        history = payload.get("history", {})
        if history_id not in history:
            return False
        del history[history_id]
        self._save(payload)
        return True


music_storage = MusicStorage()
