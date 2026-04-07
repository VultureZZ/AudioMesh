# AudioMesh — Meeting Intelligence Module
## Project Outline for Cursor Implementation

**Module Name:** `mesh-meetings`
**Version:** 1.0.0
**Parent Project:** AudioMesh API & WebUI
**Purpose:** Extend AudioMesh with AI-powered meeting/conversation processing — transcribe uploaded audio, identify and profile speakers, generate summaries, and extract action items.

---

## 1. Overview

The Meeting Intelligence Module adds a new functional layer to the existing AudioMesh platform. Where AudioMesh synthesizes voice *from* text, this module works in reverse — it ingests recorded audio (meetings, calls, interviews, conversations) and produces structured intelligence: who spoke, what was said, what was decided, and what needs to happen next.

### Core Value Proposition
- Upload any audio recording (meeting, call, podcast interview, voice memo)
- Automatic speech-to-text with speaker diarization (who said what)
- Match detected speakers against the existing AudioMesh voice library
- Label unknown speakers via an interactive UI
- AI-generated meeting summary, action items, key decisions, and open questions
- Auto-extract speaker audio segments and offer to add them to the voice library
- Export reports as PDF or structured JSON

### Integration Points with Existing AudioMesh System
- **Voice Library**: Detected speaker embeddings compared against stored AudioMesh voice profiles for auto-identification
- **Voice Creation**: Extracted speaker audio segments can be sent directly to the existing voice creation pipeline
- **API**: New `/api/v1/meetings/*` routes added alongside existing `/api/v1/generate` routes
- **WebUI**: New "Meetings" tab in the existing React frontend

---

## 2. Technology Stack

### New Backend Dependencies
```
# requirements-meetings.txt additions
whisperx>=3.1.0              # Transcription + forced alignment + diarization integration
pyannote.audio>=3.1.0        # Speaker diarization
pyannote.core>=5.0.0         # Core pyannote structures
speechbrain>=1.0.0           # Speaker embedding extraction for voice matching
pydub>=0.25.0                # Audio segment extraction and manipulation
anthropic>=0.25.0            # LLM analysis (Claude API) — or use local LLM
reportlab>=4.0.0             # PDF report generation
ffmpeg-python>=0.2.0         # Audio format conversion
torch>=2.0.0                 # Already required by AudioMesh
torchaudio>=2.0.0            # Audio processing utilities
```

### HuggingFace Models Required
```
# Speaker Diarization (requires HF token + license acceptance)
pyannote/speaker-diarization-3.1
pyannote/segmentation-3.0

# Transcription
openai/whisper-large-v3       # Best accuracy, ~3GB VRAM
# OR
openai/whisper-medium         # Faster, less accurate, ~1.5GB VRAM

# Speaker Embeddings (for matching against AudioMesh library)
speechbrain/spkrec-ecapa-voxceleb
```

### Configuration Additions (config.yaml)
```yaml
meetings:
  # HuggingFace token for pyannote models (set in .env)
  hf_token: ${HF_TOKEN}

  # Whisper model size: tiny, base, small, medium, large-v2, large-v3
  whisper_model: "large-v3"

  # Max audio file size for upload (MB)
  max_upload_mb: 500

  # Supported input formats
  supported_formats: ["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"]

  # Speaker matching threshold (cosine similarity, 0-1)
  # Higher = more strict matching against AudioMesh library
  speaker_match_threshold: 0.75

  # LLM provider for analysis: "anthropic", "openai", or "local"
  llm_provider: "anthropic"
  llm_model: "claude-opus-4-6"

  # Max concurrent meeting processing jobs
  max_concurrent_jobs: 2

  # Cleanup: hours to retain meeting files after processing
  retention_hours: 72

  # Auto-extract speaker audio segments
  extract_speaker_audio: true
  min_segment_duration_seconds: 3.0
```

---

## 3. Project Structure (Additions to Existing AudioMesh)

```
audiomesh-api/
├── src/
│   ├── api/
│   │   ├── routers/
│   │   │   ├── generate.py          # EXISTING
│   │   │   ├── voices.py            # EXISTING
│   │   │   ├── realtime.py          # EXISTING
│   │   │   └── meetings.py          # NEW — meeting endpoints
│   ├── core/
│   │   ├── meetings/                # NEW MODULE
│   │   │   ├── __init__.py
│   │   │   ├── transcriber.py       # WhisperX transcription service
│   │   │   ├── diarizer.py          # PyAnnote speaker diarization
│   │   │   ├── speaker_matcher.py   # Match speakers against voice library
│   │   │   ├── audio_extractor.py   # Extract per-speaker audio segments
│   │   │   ├── analyzer.py          # LLM analysis (summary, actions, etc.)
│   │   │   ├── reporter.py          # PDF and JSON report generation
│   │   │   └── pipeline.py          # Orchestrates full processing pipeline
│   ├── models/
│   │   ├── meeting.py               # NEW — Pydantic models for meetings
│   │   └── ...                      # EXISTING
│   └── services/
│       ├── meeting_service.py       # NEW — Meeting CRUD + job management
│       └── ...                      # EXISTING
│
├── frontend/src/
│   ├── components/
│   │   ├── meetings/                # NEW — Meeting UI components
│   │   │   ├── MeetingUpload.tsx    # Drag-and-drop upload with progress
│   │   │   ├── MeetingStatus.tsx    # Processing progress tracker
│   │   │   ├── SpeakerLabeler.tsx   # Interactive speaker identification UI
│   │   │   ├── TranscriptViewer.tsx # Speaker-attributed transcript display
│   │   │   ├── MeetingSummary.tsx   # Summary, actions, decisions display
│   │   │   ├── ActionItems.tsx      # Action item list with owner/priority
│   │   │   └── MeetingExport.tsx    # PDF/JSON export controls
│   │   └── ...                      # EXISTING
│   ├── pages/
│   │   └── MeetingsPage.tsx         # NEW — Meetings tab/page
│   └── services/
│       └── meetingsApi.ts           # NEW — API client for meetings endpoints
│
├── meetings/                        # NEW — File storage
│   ├── uploads/                     # Raw uploaded audio files
│   ├── segments/                    # Extracted per-speaker audio segments
│   ├── transcripts/                 # JSON transcripts
│   └── reports/                     # Generated PDF/JSON reports
│
└── tests/
    └── test_meetings/               # NEW
        ├── test_transcriber.py
        ├── test_diarizer.py
        ├── test_analyzer.py
        └── test_pipeline.py
```

---

## 4. Data Models

### Meeting (meetings/models/meeting.py)

```python
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from datetime import datetime

class MeetingStatus(str, Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"       # WhisperX running
    DIARIZING = "diarizing"             # PyAnnote running
    MATCHING = "matching"               # Speaker matching against voice library
    AWAITING_LABELS = "awaiting_labels" # Waiting for user to name speakers
    ANALYZING = "analyzing"             # LLM generating summary/actions
    COMPLETE = "complete"
    FAILED = "failed"

class SpeakerSegment(BaseModel):
    speaker_id: str           # "SPEAKER_00", "SPEAKER_01", etc.
    start_ms: int
    end_ms: int
    text: str
    confidence: float

class Speaker(BaseModel):
    id: str                           # "SPEAKER_00"
    label: Optional[str] = None       # User-assigned name: "John Smith"
    voice_library_match: Optional[str] = None  # Matched AudioMesh voice ID
    match_confidence: Optional[float] = None
    talk_time_seconds: float
    segment_count: int
    summary: Optional[str] = None     # LLM-generated speaker profile
    audio_segment_path: Optional[str] = None  # Extracted audio for this speaker

class ActionItem(BaseModel):
    action: str
    owner: Optional[str] = None       # Speaker label
    due_hint: Optional[str] = None    # "by end of week", "next meeting", etc.
    priority: str = "medium"          # low, medium, high

class MeetingAnalysis(BaseModel):
    summary: str
    action_items: list[ActionItem]
    key_decisions: list[str]
    open_questions: list[str]
    topics_discussed: list[str]
    sentiment: str                    # overall: positive, neutral, mixed, tense
    duration_formatted: str           # "1h 23m"

class Meeting(BaseModel):
    id: str
    title: str
    status: MeetingStatus
    created_at: datetime
    updated_at: datetime
    duration_seconds: Optional[float] = None
    file_name: str
    file_size_bytes: int
    language: str = "en"
    speakers: list[Speaker] = []
    transcript: list[SpeakerSegment] = []
    analysis: Optional[MeetingAnalysis] = None
    error: Optional[str] = None
    progress_pct: int = 0
    current_stage: Optional[str] = None
```

---

## 5. API Endpoints

### Base Path: `/api/v1/meetings`

---

#### `POST /api/v1/meetings/upload`
Upload an audio file and start processing.

**Request:** `multipart/form-data`
```
audio_file: <binary>
title: "Q4 Planning Meeting"          (optional, defaults to filename)
language: "en"                         (optional, default: "en")
known_speakers: '[{"voice_id": "uuid", "name": "John"}]'  (optional JSON string)
```

**Response: 202 Accepted**
```json
{
  "meeting_id": "uuid",
  "status": "queued",
  "message": "File uploaded successfully. Processing queued.",
  "estimated_wait_seconds": 30
}
```

**Validation Rules:**
- Max file size: configurable (default 500MB)
- Supported formats: mp3, wav, m4a, mp4, webm, ogg, flac
- Audio converted to WAV 16kHz mono before processing

---

#### `GET /api/v1/meetings/{meeting_id}/status`
Poll processing status and progress.

**Response: 200 OK**
```json
{
  "meeting_id": "uuid",
  "status": "transcribing",
  "progress_pct": 35,
  "current_stage": "Running speech-to-text transcription...",
  "duration_seconds": 3842,
  "speakers_detected": null,
  "error": null
}
```

---

#### `GET /api/v1/meetings/{meeting_id}`
Get full meeting result including transcript, speakers, and analysis.

**Response: 200 OK**
```json
{
  "meeting_id": "uuid",
  "title": "Q4 Planning Meeting",
  "status": "awaiting_labels",
  "duration_seconds": 3842,
  "speakers": [
    {
      "id": "SPEAKER_00",
      "label": null,
      "voice_library_match": "voice-uuid-john",
      "match_confidence": 0.91,
      "talk_time_seconds": 1823,
      "segment_count": 47,
      "summary": null,
      "audio_segment_path": "/api/v1/meetings/uuid/speakers/SPEAKER_00/audio"
    }
  ],
  "transcript": [
    {
      "speaker_id": "SPEAKER_00",
      "start_ms": 0,
      "end_ms": 4200,
      "text": "Alright everyone, let's get started.",
      "confidence": 0.97
    }
  ],
  "analysis": null
}
```

---

#### `PATCH /api/v1/meetings/{meeting_id}/speakers`
Assign labels to identified speakers. Triggers LLM analysis once all speakers are labeled (or user skips).

**Request Body:**
```json
{
  "speakers": [
    {"id": "SPEAKER_00", "label": "John Smith"},
    {"id": "SPEAKER_01", "label": "Sarah Chen"},
    {"id": "SPEAKER_02", "label": "Unknown"}
  ],
  "proceed_to_analysis": true
}
```

**Response: 200 OK**
```json
{
  "meeting_id": "uuid",
  "status": "analyzing",
  "message": "Speaker labels saved. LLM analysis started."
}
```

---

#### `GET /api/v1/meetings/{meeting_id}/speakers/{speaker_id}/audio`
Stream the extracted audio for a specific speaker (useful for voice library enrollment).

**Response:** `audio/wav` binary stream

---

#### `POST /api/v1/meetings/{meeting_id}/speakers/{speaker_id}/add-to-library`
Add a detected speaker's audio to the AudioMesh voice library.

**Request Body:**
```json
{
  "voice_name": "John Smith",
  "description": "Auto-extracted from Q4 Planning Meeting"
}
```

**Response: 201 Created**
```json
{
  "voice_id": "new-voice-uuid",
  "message": "Voice 'John Smith' added to library successfully."
}
```

---

#### `GET /api/v1/meetings/{meeting_id}/report`
Download a formatted meeting report.

**Query Params:**
- `format`: `pdf` | `json` | `markdown` (default: `pdf`)

**Response:** File download

---

#### `GET /api/v1/meetings`
List all meetings with pagination.

**Query Params:**
- `limit`: int (default: 20)
- `offset`: int (default: 0)
- `status`: filter by status

**Response: 200 OK**
```json
{
  "meetings": [...],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

---

#### `DELETE /api/v1/meetings/{meeting_id}`
Delete a meeting and all associated files.

---

## 6. Processing Pipeline

### pipeline.py — Full Orchestration

```python
# Pseudocode — implement in src/core/meetings/pipeline.py

async def process_meeting(meeting_id: str):

    meeting = await meeting_service.get(meeting_id)
    audio_path = meeting.upload_path

    try:
        # ── STAGE 1: CONVERT & VALIDATE ─────────────────────
        await update_status(meeting_id, "transcribing", 5, "Converting audio format...")
        wav_path = await convert_to_wav(audio_path)  # ffmpeg: 16kHz mono WAV
        duration = get_audio_duration(wav_path)
        await meeting_service.set_duration(meeting_id, duration)

        # ── STAGE 2: TRANSCRIBE ──────────────────────────────
        await update_status(meeting_id, "transcribing", 10, "Transcribing audio...")
        # WhisperX: transcribe with word-level timestamps
        raw_transcript = await transcriber.transcribe(wav_path, language=meeting.language)
        # WhisperX: forced alignment for precise word timestamps
        aligned = await transcriber.align(raw_transcript, wav_path)

        # ── STAGE 3: DIARIZE ─────────────────────────────────
        await update_status(meeting_id, "diarizing", 40, "Identifying speakers...")
        # PyAnnote: who spoke when
        diarization = await diarizer.run(wav_path)
        # WhisperX: merge transcript + diarization → speaker-attributed segments
        segments = await diarizer.assign_speakers(aligned, diarization)

        # ── STAGE 4: MATCH KNOWN SPEAKERS ────────────────────
        await update_status(meeting_id, "matching", 60, "Matching against voice library...")
        speaker_ids = extract_unique_speakers(segments)
        matches = await speaker_matcher.match_all(speaker_ids, wav_path, diarization)
        # matches = [{"speaker_id": "SPEAKER_00", "voice_id": "uuid", "confidence": 0.91}]

        # ── STAGE 5: EXTRACT SPEAKER AUDIO ───────────────────
        await update_status(meeting_id, "matching", 70, "Extracting speaker segments...")
        speaker_audio_paths = await audio_extractor.extract_all(
            wav_path, speaker_ids, diarization, meeting_id
        )

        # ── STAGE 6: SAVE TRANSCRIPT + SPEAKERS ──────────────
        speakers = build_speaker_objects(speaker_ids, matches, segments, speaker_audio_paths)
        await meeting_service.save_transcript(meeting_id, segments, speakers)

        # ── STAGE 7: CHECK IF AUTO-COMPLETE OR AWAIT LABELS ──
        all_matched = all(s.voice_library_match for s in speakers)
        if all_matched or len(speakers) == 0:
            # Auto-proceed if all speakers matched from library
            await run_analysis(meeting_id, segments, speakers)
        else:
            # Wait for user to label unknown speakers via UI
            await update_status(meeting_id, "awaiting_labels", 75,
                "Speaker identification complete. Please label unrecognized speakers.")

    except Exception as e:
        await meeting_service.set_error(meeting_id, str(e))
        raise


async def run_analysis(meeting_id: str, segments: list, speakers: list):
    """Run LLM analysis — called after speaker labels confirmed."""
    await update_status(meeting_id, "analyzing", 80, "Generating meeting intelligence...")

    transcript_text = format_transcript_for_llm(segments, speakers)
    analysis = await analyzer.analyze(transcript_text, speakers)

    await update_status(meeting_id, "analyzing", 95, "Generating report...")
    await reporter.generate_pdf(meeting_id, analysis, segments, speakers)

    await meeting_service.save_analysis(meeting_id, analysis)
    await update_status(meeting_id, "complete", 100, "Processing complete.")
```

---

## 7. Core Services

### transcriber.py

```python
# Uses whisperx for transcription + alignment
# Key methods:
async def transcribe(audio_path: str, language: str) -> dict
async def align(transcript: dict, audio_path: str) -> dict

# WhisperX handles:
# - Model: openai/whisper-large-v3 (configurable)
# - Word-level timestamps for accurate speaker assignment
# - Language detection if not specified
# - Batch processing for GPU efficiency
```

### diarizer.py

```python
# Uses pyannote.audio 3.1 for speaker diarization
# Key methods:
async def run(audio_path: str) -> Annotation  # pyannote Annotation object
async def assign_speakers(aligned_transcript: dict, diarization: Annotation) -> list[SpeakerSegment]

# Notes:
# - Requires HF_TOKEN env var
# - Requires accepting pyannote model license on HuggingFace
# - Returns speaker turn annotations with timestamps
# - WhisperX diarize_audio() used to merge with transcript
```

### speaker_matcher.py

```python
# Uses SpeechBrain ECAPA-TDNN embeddings to match detected speakers
# against existing AudioMesh voice library profiles
# Key methods:
async def extract_embedding(audio_path: str, speaker_id: str, diarization: Annotation) -> Tensor
async def match_against_library(embedding: Tensor) -> Optional[tuple[str, float]]
  # Returns (voice_id, confidence) or None if no match above threshold
async def match_all(speaker_ids: list, audio_path: str, diarization: Annotation) -> list[dict]

# Integration:
# - AudioMesh voice creation should ALSO store speaker embedding
#   so future meetings can auto-identify known voices
# - Threshold configurable: meetings.speaker_match_threshold
```

### audio_extractor.py

```python
# Extracts audio segments per speaker using pydub
# Key methods:
async def extract_all(audio_path: str, speaker_ids: list, diarization: Annotation, meeting_id: str) -> dict
async def extract_speaker(audio_path: str, speaker_id: str, segments: list) -> str
  # Returns path to extracted WAV file

# Notes:
# - Concatenates all non-overlapping segments for each speaker
# - Minimum segment duration filter (config: min_segment_duration_seconds)
# - Output format: WAV 24kHz (compatible with AudioMesh voice library)
# - These files can be directly passed to existing voice creation API
```

### analyzer.py

```python
# LLM-powered meeting intelligence generation
# Key methods:
async def analyze(transcript_text: str, speakers: list[Speaker]) -> MeetingAnalysis

# Prompt structure covers:
# 1. SPEAKER_PROFILES — role/style/focus per speaker
# 2. MEETING_SUMMARY — paragraph narrative of the meeting
# 3. ACTION_ITEMS — [{action, owner, due_hint, priority}]
# 4. KEY_DECISIONS — list of decided items
# 5. OPEN_QUESTIONS — unresolved questions
# 6. TOPICS_DISCUSSED — bullet list of agenda items covered
# 7. SENTIMENT — overall tone assessment

# Provider support: Anthropic Claude (default), OpenAI, or local via Ollama
# Long transcript handling: chunked summarization for transcripts > context window
```

### reporter.py

```python
# Generates formatted reports
# Key methods:
async def generate_pdf(meeting_id: str, analysis: MeetingAnalysis, ...) -> str
async def generate_json(meeting_id: str, analysis: MeetingAnalysis, ...) -> str
async def generate_markdown(meeting_id: str, analysis: MeetingAnalysis, ...) -> str

# PDF Report Sections:
# - Cover: Meeting title, date, duration, attendees
# - Executive Summary
# - Attendee Profiles
# - Action Items Table (owner, action, priority, due)
# - Key Decisions
# - Open Questions
# - Full Transcript (optional, can be excluded)
```

---

## 8. Frontend Components

### MeetingsPage.tsx
Main page with three views:
1. **Upload View** — drop zone for new meetings
2. **Processing View** — live status tracker
3. **Results View** — transcript, summary, action items

### MeetingUpload.tsx
```
- Drag-and-drop audio file upload
- File type + size validation (client-side)
- Meeting title input
- Optional: known speakers pre-assignment (select from voice library)
- Upload progress bar with estimated processing time display
- Supported format badges: MP3, WAV, M4A, MP4, WEBM
```

### MeetingStatus.tsx
```
- Auto-polling GET /status every 3 seconds
- Animated progress bar with current stage label
- Stage icons: 🎙️ Transcribing → 👥 Diarizing → 🔍 Matching → ⏳ Awaiting Labels → 🤖 Analyzing → ✅ Complete
- Cancel button (for queued jobs)
- Error display with retry option
```

### SpeakerLabeler.tsx
```
- Shows when status = "awaiting_labels"
- Card per detected speaker showing:
  - Speaker ID + talk time + segment count
  - 10-second audio preview (play button)
  - Auto-match badge if matched from library (with confidence %)
  - Text input for name (pre-filled if matched)
  - "Skip / Mark as Unknown" option
- "Proceed to Analysis" button
- "Add all new speakers to voice library" option
```

### TranscriptViewer.tsx
```
- Color-coded by speaker (consistent colors throughout)
- Timestamps shown on hover
- Search/filter by speaker
- Click segment to play that audio portion
- Export to .txt or .srt (subtitle format)
```

### MeetingSummary.tsx
```
Tab navigation:
  [Summary] [Action Items] [Decisions] [Questions] [Speakers]

Summary Tab:
  - Meeting narrative paragraph
  - Topics discussed tags
  - Sentiment indicator badge

Action Items Tab:
  - Sortable table: Priority | Owner | Action | Due
  - Priority color coding (red/yellow/green)
  - Check-off capability (local UI state)
  - Copy all button

Decisions Tab + Questions Tab:
  - Clean bulleted lists

Speakers Tab:
  - Speaker profile cards
  - Talk time bar chart
  - "Add to Voice Library" button per speaker
```

### MeetingExport.tsx
```
- Download PDF Report button
- Download JSON button
- Copy transcript to clipboard
- Future: Push action items to task manager (Lodestar BPM integration hook)
```

---

## 9. Implementation Phases

### Phase 1 — Backend Core (Week 1-2)
- [ ] Install and configure WhisperX + PyAnnote
- [ ] Implement `transcriber.py` with WhisperX
- [ ] Implement `diarizer.py` with PyAnnote
- [ ] Basic `pipeline.py` orchestration
- [ ] `POST /upload` and `GET /status` endpoints
- [ ] File storage management (upload, cleanup)
- [ ] Basic transcript storage (JSON)

### Phase 2 — Speaker Intelligence (Week 2-3)
- [ ] Implement `speaker_matcher.py` with SpeechBrain embeddings
- [ ] Modify AudioMesh voice creation to store speaker embeddings
- [ ] Implement `audio_extractor.py`
- [ ] `GET /{id}` full result endpoint
- [ ] `PATCH /{id}/speakers` label endpoint
- [ ] `POST /{id}/speakers/{id}/add-to-library` integration

### Phase 3 — LLM Analysis (Week 3)
- [ ] Implement `analyzer.py` with Claude API
- [ ] Design and test analysis prompts
- [ ] Long transcript chunking strategy
- [ ] Implement `reporter.py` (PDF + JSON + Markdown)
- [ ] `GET /{id}/report` endpoint with format selection

### Phase 4 — Frontend (Week 4)
- [ ] Add "Meetings" navigation tab to existing WebUI
- [ ] `MeetingUpload.tsx` with drag-and-drop
- [ ] `MeetingStatus.tsx` with polling
- [ ] `SpeakerLabeler.tsx` interactive UI
- [ ] `TranscriptViewer.tsx`
- [ ] `MeetingSummary.tsx` with tabs
- [ ] `MeetingExport.tsx`

### Phase 5 — Polish & Production (Week 5)
- [ ] Meeting list view with search/filter
- [ ] Background cleanup job for old meetings
- [ ] Error handling and retry logic throughout pipeline
- [ ] Rate limiting on upload endpoint
- [ ] Progress estimation (per audio duration)
- [ ] API authentication (if not already in AudioMesh)
- [ ] Docker compose update to include meeting dependencies
- [ ] README documentation for new module

---

## 10. Environment Variables (Additions)

```bash
# .env additions for Meeting Intelligence Module

# Required: HuggingFace token for pyannote models
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create token with read access
# 3. Accept license at https://hf.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LLM Analysis Provider
LLM_PROVIDER=anthropic          # anthropic | openai | local
ANTHROPIC_API_KEY=sk-ant-xxxxx  # If using Anthropic
OPENAI_API_KEY=sk-xxxxx         # If using OpenAI

# Optional: Override meeting-specific settings
MEETING_WHISPER_MODEL=large-v3
MEETING_MAX_UPLOAD_MB=500
MEETING_RETENTION_HOURS=72
MEETING_SPEAKER_MATCH_THRESHOLD=0.75
```

---

## 11. LLM Analysis Prompt

```python
SYSTEM_PROMPT = """You are an expert meeting analyst. You extract structured intelligence
from meeting transcripts with precision and clarity. Always respond with valid JSON only."""

ANALYSIS_PROMPT = """Analyze the following meeting transcript and provide structured output.

SPEAKERS:
{speaker_list}

TRANSCRIPT:
{transcript}

Respond with this exact JSON structure:
{{
  "speaker_profiles": [
    {{
      "speaker_id": "SPEAKER_00",
      "name": "John Smith",
      "role_inference": "Appears to be project manager, focused on timelines and resource allocation",
      "communication_style": "Direct, asks clarifying questions, frequently summarizes decisions",
      "key_concerns": ["Q4 deadline", "budget constraints"]
    }}
  ],
  "summary": "A comprehensive 2-3 paragraph narrative summary of the meeting...",
  "topics_discussed": ["Q4 roadmap planning", "Budget review", "Hiring timeline"],
  "action_items": [
    {{
      "action": "Finalize vendor contract and send to legal for review",
      "owner": "Sarah Chen",
      "due_hint": "by end of week",
      "priority": "high"
    }}
  ],
  "key_decisions": [
    "Agreed to delay feature X to Q1 to meet Q4 deadline",
    "Approved additional headcount request for engineering team"
  ],
  "open_questions": [
    "What is the exact budget ceiling for Q1 hiring?",
    "Does the vendor contract include SLA guarantees?"
  ],
  "sentiment": "productive",
  "sentiment_notes": "Meeting was focused and collaborative with clear outcomes"
}}"""
```

---

## 12. Key Technical Notes for Cursor

### WhisperX vs Raw Whisper
Use **WhisperX** (`pip install whisperx`), not raw Whisper. WhisperX provides:
- Forced alignment (word-level timestamps, not just segment-level)
- Built-in diarization pipeline integration
- Significantly faster processing via batching
- The `assign_word_speakers()` function is critical for accurate attribution

### PyAnnote License Requirement
PyAnnote models require accepting a license on HuggingFace before they can be downloaded. The pipeline should detect a missing/invalid HF_TOKEN and return a clear error with setup instructions rather than a cryptic model download failure.

### VRAM Considerations
Running WhisperX large-v3 + PyAnnote simultaneously requires ~8-10GB VRAM. The meeting processing pipeline should be aware of the existing VibeVoice model VRAM usage and either:
1. Unload VibeVoice models before processing meetings (use existing VRAM manager)
2. Or use a separate VRAM budget calculation

### Long Audio Handling
For recordings over 60 minutes:
- WhisperX processes in chunks automatically
- LLM analysis may exceed context window — implement hierarchical summarization:
  1. Summarize each 15-minute chunk
  2. Synthesize chunk summaries into final analysis

### Speaker Embedding Storage
When a new voice is added to the AudioMesh library (existing pipeline), also compute and store a SpeechBrain ECAPA embedding in the voice metadata. This enables future meeting speaker auto-identification without any UI labeling.

### Audio Extraction Quality
Extracted speaker audio segments should:
- Be concatenated from all segments (not just longest)
- Have short silences inserted between concatenated segments
- Be at 24kHz WAV (matching AudioMesh voice library requirements)
- Filter out segments shorter than `min_segment_duration_seconds`

---

## 13. Future Enhancements (Post-MVP)

- **Real-time meeting mode**: Pipe live audio (microphone) through the pipeline with rolling transcription and live action item extraction
- **Calendar integration**: Auto-pull meeting title, attendees from calendar invite
- **Lodestar BPM integration**: Push action items directly to BPM task system via webhook
- **Speaker training mode**: Upload multiple recordings of the same person to improve matching accuracy
- **Meeting templates**: Pre-configured analysis prompts for standup, sales call, client review, etc.
- **Multi-language support**: Translate non-English meetings + analyze in English
- **Recurring meeting intelligence**: Track action item completion across recurring meetings

---

*Document Version: 1.0 | Module: mesh-meetings | Parent: AudioMesh API & WebUI*
