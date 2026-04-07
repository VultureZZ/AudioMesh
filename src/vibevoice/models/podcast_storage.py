"""
Podcast metadata storage using JSON files.

Stores a persistent library of generated podcasts that can be listed, searched, and deleted.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..config import config


class PodcastStorage:
    """Thread-safe podcast metadata storage."""

    def __init__(self, storage_file: Optional[Path] = None):
        if storage_file is None:
            storage_file = config.PODCASTS_DIR / "podcast_metadata.json"
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        if not self.storage_file.exists():
            with self.lock:
                self.storage_file.parent.mkdir(parents=True, exist_ok=True)
                initial_data = {"podcasts": {}}
                self.storage_file.write_text(json.dumps(initial_data, indent=2))

    def _load(self) -> Dict:
        try:
            if self.storage_file.exists():
                content = self.storage_file.read_text()
                data = json.loads(content)
                if not isinstance(data, dict):
                    return {"podcasts": {}}
                data.setdefault("podcasts", {})
                return data
            return {"podcasts": {}}
        except (json.JSONDecodeError, IOError):
            return {"podcasts": {}}

    def _save(self, data: Dict) -> None:
        with self.lock:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            data.setdefault("podcasts", {})
            self.storage_file.write_text(json.dumps(data, indent=2))

    def add_podcast(
        self,
        podcast_id: str,
        title: str,
        voices: List[str],
        audio_path: Path,
        script_path: Optional[Path] = None,
        source_url: Optional[str] = None,
        genre: Optional[str] = None,
        duration: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        data = self._load()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload: Dict = {
            "title": title,
            "voices": voices,
            "audio_path": str(audio_path),
            "audio_filename": audio_path.name,
            "script_path": str(script_path) if script_path else None,
            "source_url": source_url,
            "genre": genre,
            "duration": duration,
            "created_at": now,
        }
        if extra:
            payload.update(extra)
        data["podcasts"][podcast_id] = payload
        self._save(data)

    def get_podcast(self, podcast_id: str) -> Optional[Dict]:
        data = self._load()
        item = data.get("podcasts", {}).get(podcast_id)
        if item and isinstance(item, dict):
            item = item.copy()
            item["id"] = podcast_id
            return item
        return None

    def list_podcasts(self) -> List[Dict]:
        data = self._load()
        items: List[Dict] = []
        for pid, payload in data.get("podcasts", {}).items():
            if not isinstance(payload, dict):
                continue
            item = payload.copy()
            item["id"] = pid
            items.append(item)
        # Most recent first
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def delete_podcast(self, podcast_id: str) -> Optional[Dict]:
        data = self._load()
        if podcast_id not in data.get("podcasts", {}):
            return None
        item = data["podcasts"].pop(podcast_id)
        self._save(data)
        if isinstance(item, dict):
            item = item.copy()
            item["id"] = podcast_id
        return item


podcast_storage = PodcastStorage()

