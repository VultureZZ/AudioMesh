"""
Pydantic models for transcript service entities.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


RecordingType = Literal["meeting", "call", "memo", "interview", "other"]


class TranscriptStatus(str, Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MATCHING = "matching"
    AWAITING_LABELS = "awaiting_labels"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class SpeakerSegment(BaseModel):
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float = 0.0


class Speaker(BaseModel):
    id: str
    label: Optional[str] = None
    voice_library_match: Optional[str] = None
    match_confidence: Optional[float] = None
    talk_time_seconds: float = 0.0
    segment_count: int = 0
    summary: Optional[str] = None
    audio_segment_path: Optional[str] = None


class ActionItem(BaseModel):
    action: str
    owner: Optional[str] = None
    due_hint: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "medium"


class TranscriptAnalysis(BaseModel):
    summary: str
    action_items: list[ActionItem] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    topics_discussed: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    duration_formatted: str = ""


class Transcript(BaseModel):
    id: str
    title: str
    status: TranscriptStatus
    created_at: datetime
    updated_at: datetime
    duration_seconds: Optional[float] = None
    file_name: str
    file_size_bytes: int
    language: str = "en"
    recording_type: RecordingType = "meeting"
    upload_path: Optional[str] = None
    converted_path: Optional[str] = None
    speakers: list[Speaker] = Field(default_factory=list)
    transcript: list[SpeakerSegment] = Field(default_factory=list)
    analysis: Optional[TranscriptAnalysis] = None
    error: Optional[str] = None
    progress_pct: int = 0
    current_stage: Optional[str] = None

