# AudioMesh API Reference

This document describes the REST API exposed by the AudioMesh FastAPI backend.

- **Base URL**: `http://localhost:8000`
- **API version prefix**: `/api/v1`
- **Interactive OpenAPI**: `http://localhost:8000/docs` (no auth / no rate-limit)

## Authentication

API key auth is controlled by the `API_KEY` environment variable:

- If **`API_KEY` is unset/empty**, all requests are accepted (even with no key).
- If **`API_KEY` is set**, every request (except `/health`, `/docs`, `/openapi.json`, `/redoc`) must include an API key:
  - Header: `X-API-Key: <key>`
  - Or query param: `?api_key=<key>`

## Rate limiting

Rate limiting is **per API key** (or `"anonymous"` if no key is provided) and is configured by `RATE_LIMIT_PER_MINUTE` (default `100`).

- Responses include:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
- On limit exceeded you’ll receive **HTTP 429** with:
  - `Retry-After: 60`

## Endpoints

### Health / Root

- `GET /health` (no auth / no rate-limit)
- `GET /` (subject to auth / rate-limit)

### Speech

- `POST /api/v1/speech/generate`
  - Body (JSON): `{ transcript: string, speakers: string[], settings?: { language, output_format, sample_rate } }`
  - Notes:
    - When `TTS_BACKEND=qwen3` (default), `settings.language` is applied (e.g. `en`, `zh`). When `TTS_BACKEND=vibevoice`, generation uses VibeVoice defaults.
  - Returns: `{ success, message, audio_url, file_path }`

- `WS /api/v1/speech/realtime`
  - Description: **Realtime streaming** text-to-speech over WebSocket (PCM audio chunks).
  - Auth:
    - Query param: `?api_key=<key>` (recommended for browsers)
    - Or header: `X-API-Key: <key>`
  - Client → server messages (JSON text frames):
    - `start`: `{ "type": "start", "cfg_scale"?: number, "inference_steps"?: number, "voice"?: string }`
    - `text`: `{ "type": "text", "text": string }` (buffered)
    - `flush`: `{ "type": "flush" }` (begins generation of the buffered text)
    - `stop`: `{ "type": "stop" }` (cancels current generation and clears buffer)
  - Server → client messages:
    - **Binary frames**: raw **PCM16LE mono @ 24000 Hz** audio chunks
    - JSON text frames:
      - `status`: `{ "type": "status", "event": string, "data"?: any }`
      - `error`: `{ "type": "error", "message": string }`
      - `end`: `{ "type": "end" }`
  - Notes / limitations:
    - The upstream VibeVoice realtime demo accepts the **full text at connect time**, so this API **buffers** `text` messages until you send `flush`.
    - Only one generation runs at a time per connection; additional `flush` calls during generation return an error.

- `GET /api/v1/speech/download/{filename}`
  - Returns: audio file (`audio/wav`)

### Voices

- `GET /api/v1/voices`
  - Returns: `{ voices: Voice[], total: number }`

- `POST /api/v1/voices` (multipart form)
  - Fields:
    - `name` (required)
    - `description` (optional)
    - `keywords` (optional, comma-separated)
    - `audio_files` (required, repeated: multiple files)
  - Returns: `{ success, message, voice?, validation_feedback? }`

- `POST /api/v1/voices/from-audio-clips` (multipart form)
  - Fields:
    - `name` (required)
    - `description` (optional)
    - `keywords` (optional, comma-separated)
    - `audio_file` (required, single file)
    - `clip_ranges` (required, JSON string)
      - Example: `[{"start_seconds":0.0,"end_seconds":4.2},{"start_seconds":10.0,"end_seconds":15.0}]`
  - Guardrails:
    - max clips: `50`
    - min clip duration: `0.5s`
    - max total selected duration: `60s` when `TTS_BACKEND=qwen3`, else `600s`
  - Notes:
    - Clips are normalized to **mono, 24kHz** WAV before training.

- **Voice cloning (Qwen3-TTS):** When `TTS_BACKEND=qwen3`, reference length should be 3–60s (5–15s optimal). Combined audio over 60s is truncated. Provide a transcript in the voice profile when possible. See [SETUP.md](SETUP.md) for full best practices.

- `PUT /api/v1/voices/{voice_id}`
  - Body (JSON): `{ name?: string, description?: string }`

- `DELETE /api/v1/voices/{voice_id}`

### Voice profiles

Voice profiles are style metadata (cadence/tone/etc.) used by Ollama-assisted features and podcast script generation.

- `POST /api/v1/voices/profile/analyze-audio` (multipart form)
  - Fields:
    - `audio_file` (required)
    - `keywords` (optional, comma-separated)
    - `ollama_url` (optional)
    - `ollama_model` (optional)
  - Returns: `{ success, message, profile?, transcript?, validation_feedback? }`
  - Notes:
    - Transcription may be unavailable on some Python versions; in that case `transcript` may be missing/null.
    - If Ollama is unreachable, this may return **HTTP 503**.

- `GET /api/v1/voices/{voice_id}/profile`
  - Returns: `{ success, message, profile? }` (profile may be `null`)

- `POST /api/v1/voices/{voice_id}/profile`
  - Create/update a profile (optionally with keywords and Ollama overrides).
  - Body (JSON): `{ keywords?: string[], ollama_url?: string, ollama_model?: string }`

- `PUT /api/v1/voices/{voice_id}/profile/keywords`
  - Update profile keywords (keywords required) and re-profile.
  - Body (JSON): `{ keywords: string[], ollama_url?: string, ollama_model?: string }`

- `POST /api/v1/voices/{voice_id}/profile/generate`
  - Force profile generation (optionally with keywords and Ollama overrides).
  - Body (JSON): `{ keywords?: string[], ollama_url?: string, ollama_model?: string }`

- `PUT /api/v1/voices/{voice_id}/profile`
  - Apply a full profile payload (works for default voices too).
  - Body (JSON): `{ cadence?, tone?, vocabulary_style?, sentence_structure?, unique_phrases?, keywords?, profile_text? }`

### Transcripts

- `POST /api/v1/transcripts/upload` (multipart form, returns `202 Accepted`)
  - Fields:
    - `audio_file` (required)
    - `title` (optional)
    - `language` (optional, default `en`)
    - `recording_type` (optional, default `meeting`)
  - Returns queued transcript metadata including `transcript_id` and initial processing status.

- `GET /api/v1/transcripts/{transcript_id}/status`
  - Returns: `{ transcript_id, status, progress_pct, current_stage, duration_seconds, speakers_detected, error }`

- `GET /api/v1/transcripts/{transcript_id}`
  - Returns the full transcript record (segments, speakers, analysis, and metadata when available).

- `PATCH /api/v1/transcripts/{transcript_id}/speakers`
  - Body (JSON): `{ speakers: [{ id, label }], proceed_to_analysis?: boolean }`
  - Notes:
    - Updates speaker labels by speaker `id`.
    - If `proceed_to_analysis=true`, analysis is triggered after labels are saved.

- `GET /api/v1/transcripts/{transcript_id}/speakers/{speaker_id}/audio`
  - Returns extracted speaker audio (`audio/wav`) when speaker segment extraction is available.

- `POST /api/v1/transcripts/{transcript_id}/speakers/{speaker_id}/add-to-library` (returns `201 Created`)
  - Body (JSON): `{ voice_name?: string, description?: string }`
  - Creates a custom voice in the voice library from the extracted speaker audio.

- `GET /api/v1/transcripts/{transcript_id}/report?format=pdf|json|markdown`
  - Returns generated report file (`application/pdf`, `application/json`, or `text/markdown`).

- `GET /api/v1/transcripts?limit=20&offset=0&status=<optional>&recording_type=<optional>`
  - Returns paginated transcript list: `{ transcripts, total, limit, offset }`.

- `DELETE /api/v1/transcripts/{transcript_id}`
  - Deletes transcript metadata and associated files.

### Music

- `POST /api/v1/music/generate`
  - Body (JSON): `MusicGenerateRequest` fields including `caption`, `lyrics`, `bpm`, `keyscale`, `timesignature`, `duration`, `vocal_language`, `instrumental`, `thinking`, `inference_steps`, `batch_size`, `seed`, `audio_format`.
  - Returns: `{ success, message, task_id }`
  - Notes:
    - Submits a custom ACE-Step generation task.
    - Returns immediately; poll status endpoint for completion.

- `POST /api/v1/music/cover-generate`
  - Body (`multipart/form-data`):
    - Required file: `reference_audio` (audio file to mimic structure from)
    - Optional fields: `prompt`, `lyrics`, `duration`, `audio_cover_strength` (0.0-1.0), `vocal_language`, `instrumental`, `thinking`, `inference_steps`, `batch_size`, `seed`, `audio_format`
  - Returns: `{ success, message, task_id }`
  - Notes:
    - Submits ACE-Step cover mode generation with `task_type=cover`.
    - Uses reference audio as the structural source and applies prompt/lyrics modifications.

- `POST /api/v1/music/simple-generate`
  - Body (JSON): `MusicSimpleGenerateRequest` fields including `description`, `input_mode`, `instrumental`, `vocal_language`, optional overrides (`exact_caption`, `exact_lyrics`, `exact_bpm`, `exact_keyscale`, `exact_timesignature`), `duration`, and `batch_size`.
  - Returns: `{ success, message, task_id }`

- `POST /api/v1/music/generate-lyrics`
  - Body (JSON): `{ description, genre?, mood?, language?, duration_hint? }`
  - Returns: `{ success, message, lyrics, caption }`
  - Notes:
    - Uses Ollama-backed prompting to draft lyrics and a generation caption.

- `GET /api/v1/music/status/{task_id}`
  - Returns: `{ success, message, task_id, status, audios, metadata, error? }`
  - Notes:
    - `audios` contains generated output paths/URLs when ready.
    - Status polling also updates server-side music history metadata.

- `GET /api/v1/music/download/{filename}`
  - Returns generated audio file by filename (`audio/mpeg`, `audio/wav`, or `audio/flac` depending on extension).

- `GET /api/v1/music/health`
  - Returns ACE-Step service readiness and runtime state.

- `GET /api/v1/music/presets`
  - Returns: `{ presets, total }`

- `POST /api/v1/music/presets`
  - Body (JSON): `{ name, mode, values }`
  - Returns created preset.

- `PUT /api/v1/music/presets/{preset_id}`
  - Body (JSON): `{ name, mode, values }`
  - Updates existing preset.

- `DELETE /api/v1/music/presets/{preset_id}`
  - Deletes a saved preset.

- `GET /api/v1/music/history?limit=50`
  - Returns: `{ history, total }`

- `GET /api/v1/music/history/{history_id}`
  - Returns one history item.

- `DELETE /api/v1/music/history/{history_id}`
  - Deletes one history item.

### Podcast generation

- `POST /api/v1/podcast/generate-script`
  - Body (JSON): `{ url, voices: string[], genre, duration, ollama_url?, ollama_model? }`
  - Returns: `{ success, message, script? }`

- `POST /api/v1/podcast/generate`
  - Body (JSON): `{ script, voices: string[], settings?, title?, source_url?, genre?, duration?, save_to_library? }`
  - Notes:
    - When `TTS_BACKEND=qwen3`, `settings.language` is applied; when `vibevoice`, VibeVoice defaults are used.
    - If `save_to_library` is true (default), response `audio_url` will be a library download URL.
  - Returns: `{ success, message, audio_url?, file_path?, script?, podcast_id? }`

- `GET /api/v1/podcast/download/{filename}`
  - Legacy filename-based download for items generated into the `outputs/` folder.

### Podcast library

- `GET /api/v1/podcasts?query=<optional>`
  - Returns: `{ podcasts: PodcastItem[], total }`

- `GET /api/v1/podcasts/{podcast_id}`
  - Returns JSON including metadata and (if available) `script` text.

- `GET /api/v1/podcasts/{podcast_id}/download`
  - Returns: audio file (`audio/wav`)

- `DELETE /api/v1/podcasts/{podcast_id}`
  - Deletes metadata and best-effort deletes associated audio/script files under `podcasts/`.

## Curl examples

### Generate speech

```bash
curl -sS -X POST "http://localhost:8000/api/v1/speech/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "transcript": "Speaker 1: Hello.\nSpeaker 2: Hi there.",
    "speakers": ["Alice", "Frank"]
  }'
```

### Create voice (multi-file upload)

```bash
curl -sS -X POST "http://localhost:8000/api/v1/voices" \
  -H "X-API-Key: ${API_KEY}" \
  -F 'name=MyVoice' \
  -F 'description=Demo voice' \
  -F 'keywords=podcast,tech' \
  -F 'audio_files=@/path/to/clip1.wav' \
  -F 'audio_files=@/path/to/clip2.wav'
```

### Create voice from clips (single file + ranges)

```bash
curl -sS -X POST "http://localhost:8000/api/v1/voices/from-audio-clips" \
  -H "X-API-Key: ${API_KEY}" \
  -F 'name=MyClippedVoice' \
  -F 'audio_file=@/path/to/long_recording.mp3' \
  -F 'clip_ranges=[{"start_seconds":0.0,"end_seconds":5.0},{"start_seconds":12.0,"end_seconds":18.0}]'
```

### Analyze audio → apply profile to a voice

```bash
curl -sS -X POST "http://localhost:8000/api/v1/voices/profile/analyze-audio" \
  -H "X-API-Key: ${API_KEY}" \
  -F 'audio_file=@/path/to/sample.wav' \
  -F 'keywords=finance,markets'
```

Then apply the returned profile payload:

```bash
curl -sS -X PUT "http://localhost:8000/api/v1/voices/<voice_id>/profile" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "tone": "calm and confident",
    "cadence": "medium pace, clear pauses",
    "keywords": ["finance", "markets"]
  }'
```

### Generate podcast script → audio (saved to library) → download

```bash
curl -sS -X POST "http://localhost:8000/api/v1/podcast/generate-script" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "url": "https://example.com/article",
    "voices": ["Alice", "Frank"],
    "genre": "News",
    "duration": "10 min"
  }'
```

```bash
curl -sS -X POST "http://localhost:8000/api/v1/podcast/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "script": "Speaker 1: Welcome...\nSpeaker 2: Today we discuss...",
    "voices": ["Alice", "Frank"],
    "title": "My Podcast",
    "save_to_library": true
  }'
```

If the response includes `podcast_id`, download via:

```bash
curl -L "http://localhost:8000/api/v1/podcasts/<podcast_id>/download" \
  -H "X-API-Key: ${API_KEY}" \
  --output podcast.wav
```

