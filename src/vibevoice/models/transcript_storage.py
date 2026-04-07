"""
Thread-safe JSON storage for transcript metadata/results.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import config


class TranscriptStorage:
    """Simple file-backed storage for transcript jobs."""

    def __init__(self, storage_file: Optional[Path] = None) -> None:
        if storage_file is None:
            storage_file = config.TRANSCRIPTS_DIR / "transcript_metadata.json"
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        if not self.storage_file.exists():
            with self.lock:
                self.storage_file.parent.mkdir(parents=True, exist_ok=True)
                self.storage_file.write_text(json.dumps({"transcripts": {}}, indent=2))

    def _load(self) -> dict[str, Any]:
        try:
            if self.storage_file.exists():
                data = json.loads(self.storage_file.read_text())
                if not isinstance(data, dict):
                    return {"transcripts": {}}
                data.setdefault("transcripts", {})
                return data
            return {"transcripts": {}}
        except (json.JSONDecodeError, OSError):
            return {"transcripts": {}}

    def _save(self, data: dict[str, Any]) -> None:
        with self.lock:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            data.setdefault("transcripts", {})
            self.storage_file.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def create_transcript(
        self,
        transcript_id: str,
        *,
        title: str,
        file_name: str,
        file_size_bytes: int,
        language: str = "en",
        recording_type: str = "meeting",
        upload_path: Optional[str] = None,
    ) -> dict[str, Any]:
        data = self._load()
        now = self._now_iso()
        entry = {
            "id": transcript_id,
            "title": title,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "duration_seconds": None,
            "file_name": file_name,
            "file_size_bytes": int(file_size_bytes),
            "language": language,
            "recording_type": recording_type,
            "upload_path": upload_path,
            "converted_path": None,
            "speakers": [],
            "transcript": [],
            "analysis": None,
            "error": None,
            "progress_pct": 0,
            "current_stage": "Queued for processing",
        }
        data["transcripts"][transcript_id] = entry
        self._save(data)
        return entry

    def get_transcript(self, transcript_id: str) -> Optional[dict[str, Any]]:
        data = self._load()
        return data["transcripts"].get(transcript_id)

    def update_transcript(self, transcript_id: str, **updates: Any) -> Optional[dict[str, Any]]:
        data = self._load()
        transcript = data["transcripts"].get(transcript_id)
        if not transcript:
            return None
        transcript.update(updates)
        transcript["updated_at"] = self._now_iso()
        data["transcripts"][transcript_id] = transcript
        self._save(data)
        return transcript

    def set_status(
        self,
        transcript_id: str,
        *,
        status: str,
        progress_pct: Optional[int] = None,
        current_stage: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {"status": status}
        if progress_pct is not None:
            payload["progress_pct"] = int(progress_pct)
        if current_stage is not None:
            payload["current_stage"] = current_stage
        if error is not None:
            payload["error"] = error
        return self.update_transcript(transcript_id, **payload)

    def list_transcripts(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        recording_type: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        data = self._load()
        items = list(data["transcripts"].values())
        if status:
            items = [x for x in items if x.get("status") == status]
        if recording_type:
            items = [x for x in items if x.get("recording_type") == recording_type]
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        total = len(items)
        return items[offset : offset + limit], total

    def delete_transcript(self, transcript_id: str) -> bool:
        data = self._load()
        if transcript_id not in data["transcripts"]:
            return False
        del data["transcripts"][transcript_id]
        self._save(data)
        return True


transcript_storage = TranscriptStorage()

