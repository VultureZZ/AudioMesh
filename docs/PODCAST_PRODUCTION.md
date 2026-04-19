# Podcast Production Pipeline

This document describes how AudioMesh produces a fully mixed production podcast: the technologies involved at each stage, how the stages are sequenced, and how audio segments are layered and overlap in the final mix.

---

## Two Production Modes

### Basic Mode (`POST /api/v1/podcast/generate`)

Generates TTS audio directly from a script. No music cues or mixing. Produces a single WAV output and optionally saves it to the podcast library.

### Production Mode (`POST /api/v1/podcast/generate-production`)

Full pipeline with script segmentation, multi-voice TTS, WhisperX timing alignment, ACE-Step music cue generation, and a pydub/ffmpeg mix. Returns a 192 kbps MP3. The task runs asynchronously; callers poll `GET /api/v1/podcast/status/{task_id}` for progress.

---

## Stage-by-Stage Pipeline

### Stage 1 - Script Generation (8% progress)

**Trigger:** Either directly via `POST /api/v1/podcast/generate-script` (URL or raw article text) or as part of a production run.

**What happens:**

1. If a URL is provided, `ArticleScraper` fetches and extracts plain text from the article.
2. Voice profiles are loaded from storage for each named voice. These profiles contain style/tone information built when the voice was created.
3. `OllamaClient.generate_script()` sends the article text plus voice profiles to a local Ollama LLM (default `llama3.2`). The prompt targets a specific word count derived from the requested duration or `approximate_duration_minutes` field.
4. The LLM returns a dialogue script in `Speaker N: text` format, with 1-4 speakers.
5. When `include_production_cues=true`, the script may also contain `[CUE: ...]` markers; these are stripped before TTS and used only for segmentation.

**Script segmentation** also runs at this stage (or as its own sub-step in production mode). `OllamaClient.generate_script_segments()` asks the LLM to return a structured JSON list of segments, each with a `segment_type`, timing hints, energy level, and notes. Segment types are:

| Type | Meaning |
|---|---|
| `intro_music` | Opening music window |
| `dialogue` | A single speaker line |
| `transition_sting` | Short hit between topic sections |
| `music_bed` / `music_bed_in` / `music_bed_out` | Ambient underscore start/end markers |
| `outro_music` | Closing music window |

If Ollama JSON segmentation fails, a deterministic fallback (`_fallback_segments_from_script`) parses speaker lines, estimates word-count durations at 140 wpm, and synthesises an intro + optional midpoint transition + outro structure.

**Technologies:** `requests` (article scraping), Ollama HTTP API, local LLM (default `llama3.2`).

---

### Stage 2 - Voice Track Generation (25% progress)

**What happens:**

1. All `[CUE: ...]` markers are stripped from the script (`strip_production_cue_markers`).
2. `PodcastGenerator._format_script_for_voices()` normalises `Speaker N:` labels and assigns any unlabelled lines to Speaker 1.
3. `VoiceGenerator.generate_speech()` synthesises the full multi-speaker dialogue as a single concatenated WAV. Each `Speaker N` label maps to the corresponding voice in the request's `voices` list (index 0 = Speaker 1, etc.).
4. The output WAV is written to `outputs/`.

**TTS backends:**

- **Qwen3-TTS** (default): `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` for named/cloned voices; base and VoiceDesign variants for built-in speakers. Runs on CUDA.
- **VibeVoice** (legacy, `TTS_BACKEND=vibevoice`): `microsoft/VibeVoice-1.5B` via the upstream inference scripts.

**Technologies:** PyTorch, Qwen3-TTS or VibeVoice model, CUDA.

---

### Stage 2b - Timing Alignment (between stages 2 and 3)

**What happens:**

After the voice WAV is produced, `PodcastTimingService.build_dialogue_timing()` derives precise per-line timestamps:

1. The voice WAV is passed to `transcript_transcriber.transcribe()` (WhisperX with `large-v3` model by default).
2. `transcript_transcriber.align()` runs forced phoneme alignment to get word- and segment-level timestamps.
3. The aligned segments are matched one-to-one with parsed `Speaker N: text` lines from the script.
4. Each line gets a `start_time_hint` (seconds) and `duration_ms` derived from WhisperX alignment.
5. If WhisperX is unavailable (no `requirements-transcripts.txt` environment or model missing), a fallback timer estimates duration at 2.6 words/second with a 2-second initial offset.

These per-line timing values are merged back into the segment list from Stage 1, replacing the LLM's coarser estimates.

**Technologies:** WhisperX, `faster-whisper`, `pyannote.audio` (for diarization when transcripts worker runs), phoneme alignment.

---

### Stage 2c - GPU Memory Hand-off

After timing alignment, the system explicitly releases GPU memory held by WhisperX and the TTS model (`transcript_transcriber.unload_models()`, `voice_generator.release_gpu_memory_after_speech()`, `release_torch_cuda_memory()`). If music cues are requested, the system then polls `wait_for_cuda_memory()` until at least `ACESTEP_MIN_FREE_VRAM_MIB` MiB are free on the ACE-Step device before proceeding. This prevents OOM when the TTS and music models would otherwise both reside in VRAM simultaneously.

---

### Stage 3 - Music Cue Generation (45% progress)

**What happens:**

Up to four named cue groups are requested. Each enabled cue runs sequentially (not in parallel) to prevent ACE-Step from being called twice at once.

| Cue group flag | Cue type | Default duration |
|---|---|---|
| `intro` | `intro` | 12 s |
| `bed` | `bed` (looping underscore) | 45 s source |
| `transitions` | `transition` | 3 s |
| `outro` | `outro` | 10 s |

For each cue, `PodcastMusicService.generate_cue()`:

1. Selects a text-to-music prompt from `_PROMPT_PRESETS` keyed on `(cue_type, style)`. The `style` field is one of `tech_talk`, `casual`, `news`, or `storytelling`.
   Example: `("intro", "tech_talk")` -> `"upbeat modern podcast intro, energetic synth, 12 seconds, broadcast quality"`.
2. Submits the prompt to `music_generator.generate_music()` (the existing ACE-Step runtime) as an instrumental task with `thinking=False`.
3. Polls ACE-Step status every 2.5 seconds until `succeeded` (timeout: 300 s per cue).
4. Returns the absolute path to the generated WAV file.

If ACE-Step is not configured (model not downloaded, or subprocess not started), this stage is skipped and a warning is added; the final output will be voice-only.

**Technologies:** ACE-Step (`acestep-v15-turbo` DiT, `acestep-5Hz-lm-0.6B` LM), pydub for format reading, asyncio for polling, CUDA.

---

### Stage 4 - Production Mix (75% progress)

**What happens:**

`AudioCompositor.mix_podcast()` composes all audio elements onto a single pydub timeline:

1. **Timeline length** is set to `max(voice_len, end_of_last_music_cue)`.
2. **Bed track** (if present): the source WAV is looped to cover the full timeline length, then applied at -6 dB. Under every dialogue segment, the bed is ducked to -18 dB (a -12 dB delta) to keep speech intelligible. The ducking regions are derived from the `start_time_hint` + `duration_ms` values from Stage 2b.
3. **Intro**: placed at position 0, volume -1.5 dB, with a 2500 ms fade-out at its end.
4. **Transition stings**: placed at the `start_time_hint` positions identified in the segment list for `transition_sting` segments, volume -1 dB.
5. **Outro**: placed at `voice_len - 500 ms` (or at its own `position_ms` if non-zero), volume -2 dB, with a 1200 ms fade-in at its start.
6. **Voice track**: overlaid on top of everything at position 0, full volume, as the final layer.

The mix is exported to WAV via pydub, then converted to a 192 kbps MP3 with ffmpeg. A guard check ensures the rendered MP3 duration is at least `voice_len - 3000 ms` to catch accidental truncation.

**Segment overlap summary:**

| Layer | Position | Volume | Transition |
|---|---|---|---|
| Music bed | Full timeline start | -6 dB gap, -18 dB under dialogue | No fade |
| Intro music | 0 ms | -1.5 dB | 2500 ms fade-out at end |
| Transition sting | At `transition_sting` start_time_hint | -1 dB | None |
| Outro music | Near voice end | -2 dB | 1200 ms fade-in at start |
| Voice track | 0 ms | 0 dB | None |

**Technologies:** pydub (`AudioSegment`), ffmpeg-python, soundfile.

---

### Stage 5 - Library Save and Delivery (100% progress)

If `save_to_library=true`:

1. A UUID podcast ID is generated.
2. The MP3 is copied to `PODCASTS_DIR/{podcast_id}.mp3`.
3. The script text is saved as `PODCASTS_DIR/{podcast_id}.txt`.
4. Metadata (title, voices, source URL, genre, duration, file size) is written to the podcast storage index.
5. The download URL changes from `/api/v1/podcast/download/{filename}` to `/api/v1/podcasts/{podcast_id}/download`.

---

## Technology Summary

| Concern | Technology |
|---|---|
| Web framework / API | FastAPI (Python), uvicorn |
| Script / segmentation LLM | Ollama (local), default model `llama3.2` |
| Article scraping | `requests` + HTML parsing |
| Text-to-speech (default) | Qwen3-TTS 1.7B on CUDA |
| Text-to-speech (legacy) | VibeVoice 1.5B on CUDA |
| Timing alignment | WhisperX (`large-v3`) + forced phoneme alignment |
| Music generation | ACE-Step (`acestep-v15-turbo` DiT + `acestep-5Hz-lm-0.6B` LM) |
| Audio composition | pydub (`AudioSegment`) |
| MP3 encoding | ffmpeg via `ffmpeg-python` |
| GPU memory management | PyTorch CUDA APIs + custom VRAM polling |
| Podcast library storage | JSON flat-file storage (`PodcastStorage`) |
| Frontend | React + TypeScript (Vite), served separately |

---

## Configuration Reference

Key environment variables that affect podcast production:

| Variable | Default | Effect |
|---|---|---|
| `TTS_BACKEND` | `qwen3` | `qwen3` or `vibevoice` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for script and segmentation |
| `OLLAMA_MODEL` | `llama3.2` | LLM used for script and segment generation |
| `DIRECTOR_TIMEOUT_SECONDS` | `240` | Ollama timeout (seconds) per ProductionDirector tool loop or JSON plan attempt |
| `TRANSCRIPT_WHISPER_MODEL` | `large-v3` | WhisperX model for timing alignment |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | ACE-Step DiT model |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | ACE-Step LM model |
| `ACESTEP_DEVICE` | (auto) | CUDA device index for ACE-Step |
| `ACESTEP_MIN_FREE_VRAM_MIB` | (config default) | Min free VRAM before ACE-Step starts |
| `GPU_VRAM_WAIT_TIMEOUT_SECONDS` | (config default) | Max wait for VRAM to free up |
| `PODCASTS_DIR` | `outputs/podcasts` | Library storage directory |
| `TRANSCRIPT_PROCESSOR_MODE` | `inline` | `subprocess` isolates transcript deps in `.venv-transcripts` |
