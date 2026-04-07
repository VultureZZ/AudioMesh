# Detailed Setup Guide

This guide provides step-by-step instructions for setting up AudioMesh from scratch.

The default TTS backend is **Qwen3-TTS** (set `TTS_BACKEND=qwen3`). You can use legacy **VibeVoice** by setting `TTS_BACKEND=vibevoice` and running the VibeVoice setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup (Qwen3-TTS)](#initial-setup-qwen3-tts)
3. [Legacy VibeVoice Setup](#legacy-vibevoice-setup)
4. [Verification](#verification)
5. [Feature Setup: Music and Transcription](#feature-setup-music-and-transcription)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher (3.12 recommended for Qwen3-TTS)
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: ~6GB for Qwen3-TTS models (or VibeVoice model if using legacy)
- **GPU**: CUDA-capable GPU recommended (NVIDIA with CUDA 11.8+)

### Software Requirements

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **ffmpeg** (required for audio decoding via `pydub`, and recommended for audio profiling/transcription)
   - macOS (Homebrew):
   ```bash
   brew install ffmpeg
   ```
   - Linux:
   ```bash
   # Example (Debian/Ubuntu)
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```

2. **Git**
   ```bash
   git --version
   ```

3. **pip** (usually comes with Python)
   ```bash
   pip --version
   ```

4. **CUDA Toolkit** (for GPU support)
   - Check if CUDA is available: `nvidia-smi`
   - Install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

5. **Ollama** (recommended for podcast scripting and voice profiling)
   - Install and run Ollama, then ensure it’s reachable at the configured URL (default `http://localhost:11434`).
   - Pull a model (default `llama3.2`): `ollama pull llama3.2`

## Initial Setup (Qwen3-TTS)

Default backend is Qwen3-TTS. No separate repo clone is required.

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd AudioMesh
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Base Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- On Python 3.13+, `audioop` is removed from stdlib. This project installs `audioop-lts` automatically for `pydub` compatibility.
- Audio transcription for “Voice Profile from Audio” requires `faster-whisper`, which is installed automatically on Python < 3.13. On newer Python versions, transcription is disabled and the API will return a minimal profile.

### Step 4: Environment (optional)

- `TTS_BACKEND=qwen3` (default) or `vibevoice` for legacy
- `QWEN_TTS_DEVICE=cuda:0` or `cpu`
- `QWEN_TTS_CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `QWEN_TTS_BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `TTS_MODEL_IDLE_UNLOAD_SECONDS=15` — unload TTS models after this many seconds of no use (0 = never unload)

Optional FlashAttention 2 for lower GPU memory:
```bash
pip install flash-attn --no-build-isolation
```

### Voice cloning best practices (Qwen3-TTS)

When creating custom voices with `TTS_BACKEND=qwen3`, use these guidelines for best clone quality:

- **Reference length:** 3–60 seconds total (combined from one or more files). Optimal is 5–15 seconds of clear speech. Audio over 60 seconds is truncated automatically.
- **Transcript:** Providing a transcript of the reference audio (e.g. in the voice profile) improves cloning accuracy.
- **Content:** Use clear speech only: no background music, no other voices, no singing. Normal speech with varied intonation works best.
- **Technical:** Mono, 24 kHz or higher (the server normalizes). Keep reference files under 10 MB when possible.
- **Environment:** Record in a quiet space; avoid long pauses (over 2 seconds) and very short phrases.

**Legacy VibeVoice:** To use VibeVoice instead, set `TTS_BACKEND=vibevoice` and run:
```bash
python scripts/setup_vibevoice.py
```
**Expected output (legacy only):**
```
============================================================
VibeVoice Setup Script
============================================================

[1/4] Cloning VibeVoice repository...
Cloning VibeVoice repository to /path/to/AudioMesh/VibeVoice...
✓ Repository cloned successfully

[2/4] Installing VibeVoice package...
Installing VibeVoice package...
✓ VibeVoice package installed successfully

[3/4] Downloading model from Hugging Face...
Downloading model microsoft/VibeVoice-1.5B to /path/to/models/VibeVoice-1.5B...
This may take a while (model is ~5.4 GB)...
✓ Model downloaded successfully

[4/4] Verifying installation...
✓ VibeVoice repository
✓ Model directory
✓ VibeVoice package importable

Setup completed successfully!
```

### Step 5: Verify Installation

Run the test script to verify everything works:

```bash
python tests/test_voice_generation.py
```

You should see output indicating successful voice generation, and an audio file in the `outputs/` directory.

## Manual Setup (Alternative)

If the automated setup script fails, you can set up manually:

### 1. Clone VibeVoice Repository

```bash
git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git
```

### 2. Install VibeVoice Package

```bash
pip install -e VibeVoice/
```

### 3. Download Model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "microsoft/VibeVoice-1.5B",
    local_dir="models/VibeVoice-1.5B",
    local_dir_use_symlinks=False
)
```

Or using the command line:

```bash
huggingface-cli download microsoft/VibeVoice-1.5B --local-dir models/VibeVoice-1.5B
```

## Verification

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Verify Model Files

Check that the model directory contains the necessary files:

```bash
ls -lh models/VibeVoice-1.5B/
```

You should see files like:
- `config.json`
- `model.safetensors.index.json`
- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- `model-00003-of-00003.safetensors`
- `preprocessor_config.json`

### Test Import

```python
import vibevoice
print("VibeVoice imported successfully")
```

## API Reference

- **Interactive API docs**: `http://localhost:8000/docs`
- **Full endpoint reference**: [`docs/API.md`](API.md)

## Frontend (Web UI)

The React + TypeScript web UI lives in `frontend/`.

- Quick start:
  - `cd frontend`
  - `npm install`
  - `cp .env.example .env` and set `VITE_API_URL` (default `http://localhost:8000`)
  - `npm run dev`
- See [`frontend/README.md`](../frontend/README.md) for more details.

## Features: Podcast Library + Voice Profile from Audio

### Podcast Library
- Generated podcasts can be saved to the on-disk library under `podcasts/` (metadata JSON + audio + script).
- The frontend “Podcast Library” page lists/searches/deletes saved items.

### Voice Profile from Audio
- In the UI, go to **Voices → Analyze Audio → Profile**.
- Upload an audio file, the backend will:
  - validate audio (duration/quality warnings)
  - attempt transcription (if supported by your Python version)
  - generate a style-oriented profile (cadence/tone/vocabulary) via Ollama
  - allow you to apply/copy the profile onto any voice

### Create Voice from Clips (single file → multiple ranges)
- In the UI, go to **Voices → Create from Clips**.
- Upload one audio file, then select multiple time ranges (clips).
- Recommended clip selection:
  - multiple clips of **3–10 seconds** each
  - keep the total selected duration under a few minutes for best results
- API endpoint used by the UI:
  - `POST /api/v1/voices/from-audio-clips` (multipart form)
  - fields:
    - `name` (required)
    - `description` (optional)
    - `keywords` (optional, comma-separated)
    - `audio_file` (required)
    - `clip_ranges` (required JSON array like `[{"start_seconds": 0.0, "end_seconds": 4.2}, ...]`)

## Feature Setup: Music and Transcription

### Music Setup (ACE-Step)

Music generation endpoints (`/api/v1/music/*`) require an ACE-Step checkout and runtime.

Recommended one-time setup from the repo root:

```bash
python scripts/setup_vibevoice.py
```

This provisions legacy VibeVoice assets and ACE-Step dependencies used by the Music feature.

Set or verify these environment variables (for example in `.env`):

```bash
ACESTEP_REPO_DIR=ACE-Step-1.5
ACESTEP_HOST=127.0.0.1
ACESTEP_PORT=8001
ACESTEP_DEVICE=cuda
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_LM_BACKEND=pt
ACESTEP_STARTUP_TIMEOUT_SECONDS=120
ACESTEP_IDLE_SHUTDOWN_SECONDS=120
ACESTEP_SERVER_COMMAND=
MUSIC_OUTPUT_DIR=outputs/music
```

Optional (used by lyrics generation and prompt refinement):

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Quick verification:
- Start backend and call `GET /api/v1/music/health`.
- If the UI/API reports `ACE-Step Not Configured`, check `ACESTEP_REPO_DIR` and rerun setup.

### Transcription Setup (Dedicated Worker Environment)

Transcript processing can run in a separate Python environment to avoid dependency conflicts with TTS backends.

Create and install transcript worker dependencies:

```bash
python -m venv .venv-transcripts
source .venv-transcripts/bin/activate
pip install -r requirements-transcripts.txt
```

Configure transcript runtime mode:

```bash
TRANSCRIPT_PROCESSOR_MODE=subprocess
TRANSCRIPT_WORKER_PYTHON=.venv-transcripts/bin/python
```

Core transcript configuration:

```bash
TRANSCRIPT_WHISPER_MODEL=large-v3
TRANSCRIPT_MAX_UPLOAD_MB=500
TRANSCRIPT_SUPPORTED_FORMATS=mp3,wav,m4a,mp4,webm,ogg,flac
TRANSCRIPT_SPEAKER_MATCH_THRESHOLD=0.75
TRANSCRIPT_MAX_CONCURRENT_JOBS=2
TRANSCRIPT_RETENTION_HOURS=72
TRANSCRIPT_EXTRACT_SPEAKER_AUDIO=true
TRANSCRIPT_MIN_SEGMENT_DURATION_SECONDS=3.0
```

Required tokens/keys by feature:

```bash
HF_TOKEN=
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-6
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

Quick verification:
- Upload audio using `POST /api/v1/transcripts/upload`.
- Poll `GET /api/v1/transcripts/{transcript_id}/status` until complete.
- If jobs fail immediately in `subprocess` mode, verify `.venv-transcripts` exists and `TRANSCRIPT_WORKER_PYTHON` points to a valid Python executable.

## Optional: Realtime TTS (VibeVoice-Realtime-0.5B)

This repo supports a realtime streaming TTS mode via `WS /api/v1/speech/realtime`.

Under the hood, the backend launches the official **microsoft/VibeVoice** realtime websocket demo as a subprocess (it is *not* bundled in this repo).

### Step A: Clone the Microsoft VibeVoice repo (realtime demo server)

From the project root:

```bash
git clone --depth 1 https://github.com/microsoft/VibeVoice.git VibeVoice-Microsoft
```

### Step B: Install its Python package (in the same `.venv`)

```bash
pip install -e VibeVoice-Microsoft/
```

Note: Both the community repo and the Microsoft repo install a Python package named `vibevoice`. If you install both into the same `.venv`, whichever is installed last can “win” imports for subprocess demos. This project now forces subprocesses to import from the intended repo via `PYTHONPATH`, so both features can coexist in one `.venv`.

### Step C: Configure the backend to find/run it

Set environment variables (e.g. in your `.env`):

- `REALTIME_VIBEVOICE_REPO_DIR=VibeVoice-Microsoft`
- `REALTIME_MODEL_ID=microsoft/VibeVoice-Realtime-0.5B`
- `REALTIME_DEVICE=cuda` (or `cpu`, `mps`)
- `REALTIME_PORT=6767` (recommended; choose any free port)

Then start the backend and open the frontend **Realtime** page.

Important:
- If you set env vars in the shell, they must be **exported** to affect `run_api.py`.
  - Good: `export REALTIME_PORT=6767` then `./run_api.py`
  - Also good (one command): `REALTIME_PORT=6767 ./run_api.py`
  - Not enough by itself: `REALTIME_PORT=6767` (won’t propagate to child processes)

Notes:
- The upstream demo server streams **PCM16LE mono @ 24kHz**.
- The upstream demo currently expects the full text at connection time; this repo’s WebSocket API buffers text until you send `flush`.

### Changing the realtime port later

You can change the port at any time by setting `REALTIME_PORT` to a different unused port and restarting the backend.

If you ever run the upstream demo server manually, its port must match `REALTIME_PORT` as well (the backend connects to `http://REALTIME_HOST:REALTIME_PORT` and `ws://REALTIME_HOST:REALTIME_PORT/stream`).

## Troubleshooting

### Issue: FlashAttention2 Warning

**Symptoms:**
- Warning: "FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed"
- Model falls back to SDPA (scaled dot product attention)

**Note:** This is not an error! The model will work fine with SDPA. FlashAttention2 is optional and provides faster inference, but SDPA is a good fallback.

**Solutions (Optional - for faster inference):**
1. Install FlashAttention2 (requires matching CUDA/PyTorch versions):
   ```bash
   pip install flash-attn --no-build-isolation
   ```
2. Verify installation: `python -c "import flash_attn; print('FlashAttention2 installed')"`
3. If installation fails, continue using SDPA - it works fine

### Issue: CUDA Not Available

**Symptoms:**
- Warning: "CUDA not available - will use CPU (slower)"
- Very slow generation

**Solutions:**
1. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
2. Verify CUDA installation: `nvidia-smi`
3. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Process killed during generation

**Solutions:**
1. Reduce input text length
2. Close other GPU applications
3. Use CPU mode (slower but works):
   ```bash
   CUDA_VISIBLE_DEVICES="" python tests/test_voice_generation.py
   ```
4. Process in smaller chunks

### Issue: Model Download Fails

**Symptoms:**
- `ConnectionError` or timeout during download
- Incomplete model files

**Solutions:**
1. Check internet connection
2. Use Hugging Face CLI with resume:
   ```bash
   huggingface-cli download microsoft/VibeVoice-1.5B --local-dir models/VibeVoice-1.5B --resume-download
   ```
3. Download manually from [Hugging Face](https://huggingface.co/microsoft/VibeVoice-1.5B)

### Issue: Voice Names Not Found

**Symptoms:**
- `Error: Voice not found`
- Available voices list is empty

**Solutions:**
1. Check that voice files exist in `VibeVoice/demo/voices/`
2. Use exact names from the "Available voices" list
3. Common names: `Alice`, `Frank`, `Mary`, `Carter`, `Maya`

### Issue: Inference Script Not Found

**Symptoms:**
- `FileNotFoundError: inference_from_file.py`

**Solutions:**
1. Verify VibeVoice repo is cloned: `ls VibeVoice/demo/`
2. Re-clone the repository:
   ```bash
   rm -rf VibeVoice
   git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git
   ```

### Issue: Package Installation Fails

**Symptoms:**
- `pip install -e VibeVoice/` fails
- Missing dependencies

**Solutions:**
1. Check Python version: `python --version` (need 3.8+)
2. Upgrade pip: `pip install --upgrade pip`
3. Install build tools:
   - Linux: `sudo apt-get install build-essential`
   - macOS: `xcode-select --install`
4. Check VibeVoice's `pyproject.toml` or `setup.py` for requirements

### Issue: Slow Generation on CPU

**Expected behavior:** CPU generation is 10-50x slower than GPU.

**Solutions:**
1. Use shorter text for testing
2. Consider using cloud GPU (Google Colab, etc.)
3. Wait longer - CPU generation can take several minutes

## Next Steps

After successful setup:

1. **Run the test script:**
   ```bash
   python tests/test_voice_generation.py
   ```

2. **Create custom transcripts:**
   - Edit `tests/sample_transcript.txt`
   - Or create new transcript files

3. **Experiment with different speakers:**
   ```bash
   python tests/test_voice_generation.py Mary Carter
   ```

4. **Explore the VibeVoice repository:**
   - Check `VibeVoice/demo/` for more examples
   - Read the VibeVoice documentation

5. **Start developing:**
   - Plan your API structure in `src/vibevoice/`
   - Design your web interface
   - Write additional tests

## Running as a Linux service (systemd)

This repo includes an installer script that can set up `systemd` services to run the API (and optionally the Web UI) in the background and on boot.

### 1) Install the API service

From the repo root:

```bash
sudo ./scripts/install_systemd_service.sh
```

This creates:
- `audiomesh-api.service` (enabled + started)
- `/etc/audiomesh/audiomesh.env` (created if missing)

Useful commands:

```bash
systemctl status audiomesh-api
journalctl -u audiomesh-api -f
```

### 2) Optional: install the Web UI service

To run the UI as a service too (Vite preview after a build):

```bash
sudo ./scripts/install_systemd_service.sh --with-web --web-mode preview
```

Or, to run the Vite dev server (HMR; better for development than 24/7 use):

```bash
sudo ./scripts/install_systemd_service.sh --with-web --web-mode dev
```

The UI service will listen on:
- `http://<your-linux-pc-ip>:3001`

### 3) Configure ports and settings

Edit `/etc/audiomesh/audiomesh.env` and restart services:

```bash
sudo nano /etc/audiomesh/audiomesh.env
sudo systemctl restart audiomesh-api
sudo systemctl restart audiomesh-web
```

Notes:
- The API defaults to `0.0.0.0:8000` (set `HOST` and `PORT` to change).
- The UI defaults to `0.0.0.0:3001` (set `AUDIOMESH_WEB_HOST` and `AUDIOMESH_WEB_PORT` to change).

### Common issue: EACCES when rebuilding the frontend

If you previously installed with `sudo` and the installer built the UI as root, `frontend/dist/` may be owned by root, causing errors like:
- `EACCES: permission denied, unlink ... frontend/dist/...`

Fix by changing ownership of the frontend build output back to your user (or removing it), then rebuild.

## Additional Resources

- [VibeVoice Community Repository](https://github.com/vibevoice-community/VibeVoice)
- [VibeVoice Model Card](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Getting Help

If you encounter issues not covered here:

1. Check the VibeVoice repository issues
2. Review the beginner's guide: https://www.kdnuggets.com/beginners-guide-to-vibevoice
3. Check Hugging Face model discussions
4. Review error messages carefully - they often contain helpful information
