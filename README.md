# AudioMesh

A FastAPI + React project for generating speech, managing voices, and producing multi-voice podcasts. Default TTS backend is **Qwen3-TTS**; legacy **VibeVoice** is supported via `TTS_BACKEND=vibevoice`.

## Overview

- **Qwen3-TTS** (default): [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for voice cloning and built-in speakers. No separate repo clone; install `pip install qwen-tts` and use.
- **VibeVoice** (legacy): Community VibeVoice-1.5B with cloned repo and model when `TTS_BACKEND=vibevoice`.

## Features

- Multi-speaker voice generation (Qwen3-TTS or VibeVoice)
- Built-in default voices (Alice, Frank, Mary, Carter, Maya) and custom voice cloning
- Support for English and other languages (Qwen3-TTS: Chinese, Japanese, Korean, etc.)
- Music generation with ACE-Step (custom prompts, lyrics-assisted mode, presets, and history)
- Transcript service for meetings, calls, voice memos, and interviews
- GPU acceleration (CUDA) recommended
- FastAPI backend with a web UI (React + TypeScript)
- Optional realtime streaming TTS over WebSocket (VibeVoice-Realtime demo)

## Prerequisites

- Python 3.8 or higher (3.12 recommended for Qwen3-TTS)
- Git
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- ~6GB disk space for models

## Quick Start

### 1. Clone this repository

```bash
git clone <your-repo-url>
cd AudioMesh
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

With default `TTS_BACKEND=qwen3`, this installs `qwen-tts` and `soundfile`. Model weights download on first use.

### 4. Provision model repos (recommended)

Run setup once to provision both:
- legacy VibeVoice assets (for `TTS_BACKEND=vibevoice`)
- ACE-Step repository and dependencies (for Music generation)

```bash
python scripts/setup_vibevoice.py
```

If you only use default Qwen3-TTS speech and do not use the Music feature, this step is optional.

### 5. Run the test script

```bash
python tests/test_voice_generation.py
```

Or with speakers:

```bash
python tests/test_voice_generation.py Alice Frank
```

Default voices (Qwen3 CustomVoice): Alice, Frank, Mary, Carter, Maya. Custom voices can be created from audio in the web UI. Generated audio is in `outputs/`.

## Project Structure

```
AudioMesh/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── src/
│   └── vibevoice/           # FastAPI backend (routes, services, middleware)
├── frontend/                # React + TypeScript web UI
├── tests/
│   ├── test_voice_generation.py  # Test script
│   └── sample_transcript.txt     # Sample transcript
├── scripts/
│   └── setup_vibevoice.py   # Setup script
├── docs/
│   └── SETUP.md             # Detailed setup guide
├── outputs/                  # Generated audio files
├── models/                   # Downloaded models
└── VibeVoice/               # Cloned upstream VibeVoice repository (legacy backend)
```

## Models Utilized

| Area | Models / components |
| ---- | ------------------- |
| **TTS (default)** | Qwen3-TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (`QWEN_TTS_*`). |
| **TTS (legacy)** | VibeVoice: `microsoft/VibeVoice-1.5B` (`TTS_BACKEND=vibevoice`, `MODEL_PATH`, `VIBEVOICE_REPO_DIR`). |
| **Realtime TTS** | `microsoft/VibeVoice-Realtime-0.5B` (`REALTIME_VIBEVOICE_REPO_DIR`, `REALTIME_MODEL_ID`). |
| **Music** | ACE-Step runtime (`ACESTEP_CONFIG_PATH=acestep-v15-turbo`, `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B`). Optional Ollama prompt refinement (`OLLAMA_*`, default `llama3.2`). |
| **Transcripts** | Whisper (`TRANSCRIPT_WHISPER_MODEL`, default `large-v3`) with speaker analysis tooling, plus configurable LLM analysis (`LLM_PROVIDER`, `LLM_MODEL`). |
| **Podcast / lyrics** | Ollama for podcast script and lyric/caption assistance (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`). |
| **Voice profiles** | Ollama-assisted profile analysis and keyword enrichment (`OLLAMA_*`). |

For full setup and API details, see `docs/SETUP.md` and `docs/API.md`.

## Usage

### Generating Voice from Text

1. Create a transcript file with the format:
```
Speaker 1: Your text here.
Speaker 2: Response text here.
Speaker 1: More dialogue.
```

2. Run inference:
```bash
python VibeVoice/demo/inference_from_file.py \
  --model_path models/VibeVoice-1.5B \
  --txt_path path/to/your/transcript.txt \
  --speaker_names Alice Frank
```

### Custom Transcripts

Edit `tests/sample_transcript.txt` or create your own transcript file following the format above.

## Troubleshooting

### FlashAttention2 Warning

If you see a warning about FlashAttention2 not being installed, this is normal and not an error. The model will automatically use SDPA (scaled dot product attention) instead, which works fine. FlashAttention2 is optional and provides faster inference. To install it (optional):

```bash
pip install flash-attn --no-build-isolation
```

### CUDA Out of Memory

- Reduce the length of your input text
- Use a smaller batch size if the script supports it
- Close other GPU-intensive applications

### Model Not Found

Run the setup script again:
```bash
python scripts/setup_vibevoice.py
```

If the Music page shows `ACE-Step Not Configured`, ensure the setup script completed successfully and that `ACESTEP_REPO_DIR` points to the cloned ACE-Step folder.

### Slow Generation

- Ensure you're using a GPU runtime (CUDA)
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- CPU generation is significantly slower

### Missing Voices

The script will list available voices on startup. Use the exact names shown (e.g., "Alice", "Frank", not "en-Alice_woman").

## Running the API

After installing dependencies, you can run the API server in several ways:

### Option 1: Using the run script (recommended)
```bash
python run_api.py
```

### Option 2: Using uvicorn directly
```bash
PYTHONPATH=src uvicorn vibevoice.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Using Python module
```bash
PYTHONPATH=src python -m vibevoice.main
```

Once running, access:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- API Root: http://localhost:8000/

## Running the Web UI

The frontend lives in `frontend/`. See `frontend/README.md` for setup, development, and build instructions.

## API Endpoints

- Full API reference: `docs/API.md`

### Speech

- `POST /api/v1/speech/generate` - Generate speech from text
- `GET /api/v1/speech/download/{filename}` - Download generated audio

### Voices

- `GET /api/v1/voices` - List all voices (default + custom)
- `POST /api/v1/voices` - Create custom voice (multipart: `audio_files`[])
- `POST /api/v1/voices/from-audio-clips` - Create custom voice from multiple clips within a single uploaded file
- `PUT /api/v1/voices/{voice_id}` - Update a voice (name/description)
- `DELETE /api/v1/voices/{voice_id}` - Delete custom voice

### Voice profiles

- `POST /api/v1/voices/profile/analyze-audio` - Analyze an audio file and derive a style profile (Ollama-assisted)
- `GET /api/v1/voices/{voice_id}/profile` - Get voice profile
- `POST /api/v1/voices/{voice_id}/profile` - Create/update voice profile
- `PUT /api/v1/voices/{voice_id}/profile` - Apply a full voice profile payload to a voice
- `PUT /api/v1/voices/{voice_id}/profile/keywords` - Update profile keywords and re-profile
- `POST /api/v1/voices/{voice_id}/profile/generate` - Force profile generation

### Transcripts

- `POST /api/v1/transcripts/upload` - Upload audio and start transcript pipeline
- `GET /api/v1/transcripts/{transcript_id}/status` - Poll transcript progress
- `GET /api/v1/transcripts/{transcript_id}` - Get transcript, speakers, and analysis
- `PATCH /api/v1/transcripts/{transcript_id}/speakers` - Label speakers and trigger analysis
- `GET /api/v1/transcripts/{transcript_id}/speakers/{speaker_id}/audio` - Download extracted speaker audio
- `POST /api/v1/transcripts/{transcript_id}/speakers/{speaker_id}/add-to-library` - Create voice from extracted speaker audio
- `GET /api/v1/transcripts/{transcript_id}/report` - Download report (`pdf`, `json`, `markdown`)
- `GET /api/v1/transcripts` - List transcripts
- `DELETE /api/v1/transcripts/{transcript_id}` - Delete transcript and files

### Music

- `POST /api/v1/music/generate` - Submit custom music generation task
- `POST /api/v1/music/simple-generate` - Generate from a description-driven prompt
- `POST /api/v1/music/generate-lyrics` - Generate lyrics/caption with Ollama assistance
- `GET /api/v1/music/status/{task_id}` - Poll task status and output files
- `GET /api/v1/music/download/{filename}` - Download generated audio
- `GET /api/v1/music/health` - Check ACE-Step runtime readiness
- `GET /api/v1/music/presets` - List saved presets
- `POST /api/v1/music/presets` - Create preset
- `PUT /api/v1/music/presets/{preset_id}` - Update preset
- `DELETE /api/v1/music/presets/{preset_id}` - Delete preset
- `GET /api/v1/music/history` - List generation history
- `GET /api/v1/music/history/{history_id}` - Get one history item
- `DELETE /api/v1/music/history/{history_id}` - Delete history item

### Transcript Worker Environment (recommended)

To avoid dependency conflicts between TTS backends and transcript libraries, run transcript processing in a separate Python environment:

```bash
python -m venv .venv-transcripts
source .venv-transcripts/bin/activate
pip install -r requirements-transcripts.txt
```

Set runtime mode in `.env`:

```bash
TRANSCRIPT_PROCESSOR_MODE=subprocess
TRANSCRIPT_WORKER_PYTHON=.venv-transcripts/bin/python
```

With this setup, the API process remains on your main `.venv` dependencies, while transcript jobs run in the dedicated worker environment.

### Podcast generation + library

- `POST /api/v1/podcast/generate-script` - Generate a script from an article URL (Ollama-assisted)
- `POST /api/v1/podcast/generate` - Generate podcast audio from a script
- `GET /api/v1/podcast/download/{filename}` - Download generated podcast audio (filename-based)
- `GET /api/v1/podcasts` - List/search saved podcasts
- `GET /api/v1/podcasts/{podcast_id}` - Get podcast metadata (and script, if available)
- `GET /api/v1/podcasts/{podcast_id}/download` - Download saved podcast audio (id-based)
- `DELETE /api/v1/podcasts/{podcast_id}` - Delete saved podcast

### Auth and rate limiting

Authentication and rate limiting are configured via environment variables:

- `API_KEY`: if set, clients must send `X-API-Key: <key>` (docs/openapi/health are exempt).
- `RATE_LIMIT_PER_MINUTE`: per-key limit (default `100`), returned via `X-RateLimit-*` headers.

See the interactive API documentation at `/docs` for detailed request/response schemas.

## Future Development

This project structure is designed to support:

- **API Development**: REST API endpoints in `src/vibevoice/`
- **Web Interface**: Build a frontend for voice generation
- **Testing**: Expand test suite in `tests/`
- **Documentation**: Add API documentation in `docs/`

## Resources

- [VibeVoice GitHub](https://github.com/vibevoice-community/VibeVoice) (legacy backend)
- [VibeVoice Model on Hugging Face](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [VibeVoice Realtime Model on Hugging Face](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Beginner's Guide](https://www.kdnuggets.com/beginners-guide-to-vibevoice)

## License

This repository is licensed under the MIT License. See `LICENSE`.

AudioMesh integrates with third-party software and models (for example the upstream VibeVoice repositories and models downloaded from Hugging Face) which are governed by their own licenses and terms. Review the upstream projects for details.

## Disclaimer

AudioMesh is intended for research and development purposes. Please use responsibly and in compliance with all applicable laws and regulations. Always disclose the use of AI when sharing AI-generated content.
