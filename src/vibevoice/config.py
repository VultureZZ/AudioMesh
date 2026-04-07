"""
Configuration management for AudioMesh API.

Loads configuration from environment variables or .env file.
"""
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file if it exists
    load_dotenv()
except Exception:
    # Keep imports optional for minimal environments; production should install python-dotenv.
    pass

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Config:
    """Application configuration."""

    PROJECT_ROOT = PROJECT_ROOT

    # API Key (optional - if not set, any key is accepted)
    API_KEY: Optional[str] = os.getenv("API_KEY", None)

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

    # Paths
    CUSTOM_VOICES_DIR: Path = Path(os.getenv("CUSTOM_VOICES_DIR", "custom_voices"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    PODCASTS_DIR: Path = Path(os.getenv("PODCASTS_DIR", "podcasts"))
    TRANSCRIPTS_DIR: Path = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))

    # TTS backend: "qwen3" (Qwen3-TTS), "vibevoice" (legacy subprocess), or "xtts"/"bark" when implemented
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "qwen3")

    # Qwen3-TTS (when TTS_BACKEND=qwen3)
    QWEN_TTS_CUSTOM_VOICE_MODEL: str = os.getenv(
        "QWEN_TTS_CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    QWEN_TTS_BASE_MODEL: str = os.getenv(
        "QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    QWEN_TTS_VOICE_DESIGN_MODEL: str = os.getenv(
        "QWEN_TTS_VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    )
    QWEN_TTS_DEVICE: str = os.getenv("QWEN_TTS_DEVICE", "cuda:0")
    QWEN_TTS_DTYPE: str = os.getenv("QWEN_TTS_DTYPE", "bfloat16")
    # Seconds of idle time after which TTS models are unloaded to free memory. 0 = never unload.
    TTS_MODEL_IDLE_UNLOAD_SECONDS: int = int(os.getenv("TTS_MODEL_IDLE_UNLOAD_SECONDS", "15"))

    # Legacy VibeVoice (when TTS_BACKEND=vibevoice)
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "models/VibeVoice-1.5B"))
    VIBEVOICE_REPO_DIR: Path = Path(os.getenv("VIBEVOICE_REPO_DIR", "VibeVoice"))

    # Realtime TTS (VibeVoice-Realtime-0.5B demo server)
    # These settings are used by the backend to launch/manage a local realtime model server
    # and bridge WebSocket audio chunks to the frontend.
    REALTIME_VIBEVOICE_REPO_DIR: Path = Path(
        os.getenv("REALTIME_VIBEVOICE_REPO_DIR", os.getenv("VIBEVOICE_REPO_DIR", "VibeVoice"))
    )
    REALTIME_MODEL_ID: str = os.getenv(
        "REALTIME_MODEL_ID", "microsoft/VibeVoice-Realtime-0.5B"
    )
    REALTIME_DEVICE: str = os.getenv("REALTIME_DEVICE", "cuda")
    REALTIME_HOST: str = os.getenv("REALTIME_HOST", "127.0.0.1")
    # Default chosen to avoid common dev servers on 3000.
    REALTIME_PORT: int = int(os.getenv("REALTIME_PORT", "6767"))
    REALTIME_STARTUP_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_STARTUP_TIMEOUT_SECONDS", "60")
    )
    # Optional explicit command for starting the realtime server. If set, overrides defaults.
    # Example:
    #   REALTIME_SERVER_COMMAND="python /path/to/VibeVoice/demo/vibevoice_realtime_demo.py --port 3000 --model_path microsoft/VibeVoice-Realtime-0.5B"
    REALTIME_SERVER_COMMAND: Optional[str] = os.getenv("REALTIME_SERVER_COMMAND", None)

    # ACE-Step music generation API server (subprocess-managed)
    ACESTEP_REPO_DIR: Path = Path(os.getenv("ACESTEP_REPO_DIR", "ACE-Step-1.5"))
    ACESTEP_HOST: str = os.getenv("ACESTEP_HOST", "127.0.0.1")
    ACESTEP_PORT: int = int(os.getenv("ACESTEP_PORT", "8001"))
    ACESTEP_DEVICE: str = os.getenv("ACESTEP_DEVICE", "cuda")
    ACESTEP_CONFIG_PATH: str = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    ACESTEP_LM_MODEL_PATH: str = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
    ACESTEP_LM_BACKEND: str = os.getenv("ACESTEP_LM_BACKEND", "pt")
    ACESTEP_STARTUP_TIMEOUT_SECONDS: float = float(
        os.getenv("ACESTEP_STARTUP_TIMEOUT_SECONDS", "120")
    )
    ACESTEP_IDLE_SHUTDOWN_SECONDS: int = int(
        os.getenv("ACESTEP_IDLE_SHUTDOWN_SECONDS", "120")
    )
    # Optional explicit command. Example:
    #   ACESTEP_SERVER_COMMAND="uv run acestep-api --host 127.0.0.1 --port 8001"
    ACESTEP_SERVER_COMMAND: Optional[str] = os.getenv("ACESTEP_SERVER_COMMAND", None)

    # Server
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Ollama configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Transcript service configuration
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    TRANSCRIPT_WHISPER_MODEL: str = os.getenv("TRANSCRIPT_WHISPER_MODEL", "large-v3")
    TRANSCRIPT_MAX_UPLOAD_MB: int = int(os.getenv("TRANSCRIPT_MAX_UPLOAD_MB", "500"))
    TRANSCRIPT_SUPPORTED_FORMATS: list[str] = [
        x.strip().lower()
        for x in os.getenv(
            "TRANSCRIPT_SUPPORTED_FORMATS",
            "mp3,wav,m4a,mp4,webm,ogg,flac",
        ).split(",")
        if x.strip()
    ]
    TRANSCRIPT_SPEAKER_MATCH_THRESHOLD: float = float(
        os.getenv("TRANSCRIPT_SPEAKER_MATCH_THRESHOLD", "0.75")
    )
    TRANSCRIPT_MAX_CONCURRENT_JOBS: int = int(os.getenv("TRANSCRIPT_MAX_CONCURRENT_JOBS", "2"))
    TRANSCRIPT_RETENTION_HOURS: int = int(os.getenv("TRANSCRIPT_RETENTION_HOURS", "72"))
    TRANSCRIPT_EXTRACT_SPEAKER_AUDIO: bool = (
        os.getenv("TRANSCRIPT_EXTRACT_SPEAKER_AUDIO", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    TRANSCRIPT_MIN_SEGMENT_DURATION_SECONDS: float = float(
        os.getenv("TRANSCRIPT_MIN_SEGMENT_DURATION_SECONDS", "3.0")
    )
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-opus-4-6")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    # Transcript processor mode:
    # - inprocess: run transcript pipeline in API process
    # - subprocess: run transcript pipeline in external worker python env
    TRANSCRIPT_PROCESSOR_MODE: str = os.getenv("TRANSCRIPT_PROCESSOR_MODE", "subprocess").strip().lower()
    # Python executable for subprocess worker mode (separate venv recommended).
    TRANSCRIPT_WORKER_PYTHON: str = os.getenv(
        "TRANSCRIPT_WORKER_PYTHON",
        str((PROJECT_ROOT / ".venv-transcripts" / "bin" / "python")),
    )
    MUSIC_OUTPUT_DIR: Path = Path(os.getenv("MUSIC_OUTPUT_DIR", "outputs/music"))
    MUSIC_REFERENCE_DIR: Path = Path(os.getenv("MUSIC_REFERENCE_DIR", "outputs/music/reference"))

    def __init__(self):
        """Initialize configuration and ensure directories exist."""
        # Convert relative paths to absolute paths relative to project root
        if not self.MODEL_PATH.is_absolute():
            self.MODEL_PATH = PROJECT_ROOT / self.MODEL_PATH
        if not self.VIBEVOICE_REPO_DIR.is_absolute():
            self.VIBEVOICE_REPO_DIR = PROJECT_ROOT / self.VIBEVOICE_REPO_DIR
        if not self.CUSTOM_VOICES_DIR.is_absolute():
            self.CUSTOM_VOICES_DIR = PROJECT_ROOT / self.CUSTOM_VOICES_DIR
        if not self.OUTPUT_DIR.is_absolute():
            self.OUTPUT_DIR = PROJECT_ROOT / self.OUTPUT_DIR
        if not self.PODCASTS_DIR.is_absolute():
            self.PODCASTS_DIR = PROJECT_ROOT / self.PODCASTS_DIR
        if not self.TRANSCRIPTS_DIR.is_absolute():
            self.TRANSCRIPTS_DIR = PROJECT_ROOT / self.TRANSCRIPTS_DIR
        if not self.REALTIME_VIBEVOICE_REPO_DIR.is_absolute():
            self.REALTIME_VIBEVOICE_REPO_DIR = PROJECT_ROOT / self.REALTIME_VIBEVOICE_REPO_DIR
        if not self.ACESTEP_REPO_DIR.is_absolute():
            self.ACESTEP_REPO_DIR = PROJECT_ROOT / self.ACESTEP_REPO_DIR
        if not self.MUSIC_OUTPUT_DIR.is_absolute():
            self.MUSIC_OUTPUT_DIR = PROJECT_ROOT / self.MUSIC_OUTPUT_DIR
        if not self.MUSIC_REFERENCE_DIR.is_absolute():
            self.MUSIC_REFERENCE_DIR = PROJECT_ROOT / self.MUSIC_REFERENCE_DIR

        # If the user didn't explicitly set REALTIME_VIBEVOICE_REPO_DIR, try a sensible default:
        # prefer a microsoft/VibeVoice checkout if present (it contains the realtime demo server).
        realtime_repo_env = os.getenv("REALTIME_VIBEVOICE_REPO_DIR")
        if not realtime_repo_env:
            candidate = PROJECT_ROOT / "VibeVoice-Microsoft"
            expected_demo = candidate / "demo" / "vibevoice_realtime_demo.py"
            current_expected_demo = (
                self.REALTIME_VIBEVOICE_REPO_DIR / "demo" / "vibevoice_realtime_demo.py"
            )
            if not current_expected_demo.exists() and expected_demo.exists():
                self.REALTIME_VIBEVOICE_REPO_DIR = candidate

        # Ensure directories exist
        self.CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
        self.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MUSIC_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        (self.TRANSCRIPTS_DIR / "uploads").mkdir(parents=True, exist_ok=True)
        (self.TRANSCRIPTS_DIR / "segments").mkdir(parents=True, exist_ok=True)
        (self.TRANSCRIPTS_DIR / "json").mkdir(parents=True, exist_ok=True)
        (self.TRANSCRIPTS_DIR / "reports").mkdir(parents=True, exist_ok=True)

    @property
    def requires_api_key(self) -> bool:
        """Check if API key validation is required."""
        return self.API_KEY is not None and self.API_KEY != ""

    def validate_api_key(self, provided_key: Optional[str]) -> bool:
        """
        Validate provided API key.

        Args:
            provided_key: The API key provided by the client

        Returns:
            True if key is valid (or no key required), False otherwise
        """
        if not self.requires_api_key:
            # No API key required, accept any key or no key
            return True
        # API key required, validate against configured key
        return provided_key == self.API_KEY


# Global configuration instance
config = Config()
