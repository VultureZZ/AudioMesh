"""
Pydantic models for request/response validation.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SpeechSettings(BaseModel):
    """Settings for speech generation."""

    language: str = Field(default="en", description="Language code (en, zh, etc.)")
    output_format: str = Field(default="wav", description="Output audio format")
    sample_rate: int = Field(default=24000, description="Sample rate in Hz")


class SpeechGenerateRequest(BaseModel):
    """Request model for speech generation."""

    transcript: str = Field(..., description="Transcript text with speaker labels (e.g., 'Speaker 1: Hello')")
    speakers: List[str] = Field(..., min_length=1, description="List of speaker names")
    speaker_instructions: Optional[List[str]] = Field(
        None,
        description="Optional style/emotion instruction per speaker (e.g. 'speak in a happy tone'). Length must match speakers.",
    )
    settings: Optional[SpeechSettings] = Field(default_factory=SpeechSettings, description="Speech generation settings")


class SpeechGenerateResponse(BaseModel):
    """Response model for speech generation."""

    success: bool = Field(..., description="Whether generation was successful")
    message: str = Field(..., description="Status message")
    audio_url: Optional[str] = Field(None, description="URL to generated audio file")
    file_path: Optional[str] = Field(None, description="Path to generated audio file")


class VoiceQualityAnalysis(BaseModel):
    """Audio quality analysis for a voice clone."""

    clone_quality: str = Field(..., description="Overall clone quality: excellent, good, fair, poor")
    issues: List[str] = Field(default_factory=list, description="Detected issues (e.g., background_music, background_noise)")
    recording_quality_score: float = Field(0.5, description="Recording quality score 0-1")
    background_music_detected: bool = Field(False, description="Whether background music was detected")
    background_noise_detected: bool = Field(False, description="Whether background noise was detected")


class VoiceResponse(BaseModel):
    """Response model for a single voice."""

    id: str = Field(..., description="Voice identifier")
    name: str = Field(..., description="Voice name")
    display_name: Optional[str] = Field(
        None, description="Human-friendly display name (may differ from `name` for default voices)"
    )
    language_code: Optional[str] = Field(None, description="Optional voice language code (e.g., en, zh, in)")
    language_label: Optional[str] = Field(None, description="Optional human-friendly language label (e.g., English)")
    gender: Optional[str] = Field(
        None, description="Optional voice gender: male, female, neutral, or unknown"
    )
    description: Optional[str] = Field(None, description="Voice description")
    type: str = Field(..., description="Voice type: 'default' or 'custom'")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    audio_files: Optional[List[str]] = Field(None, description="List of audio file names")
    image_url: Optional[str] = Field(
        None, description="Relative URL to voice avatar image (e.g. /api/v1/voices/{id}/image) when present"
    )
    quality_analysis: Optional[VoiceQualityAnalysis] = Field(
        None, description="Audio quality analysis (custom voices only)"
    )


class VoiceListResponse(BaseModel):
    """Response model for voice list."""

    voices: List[VoiceResponse] = Field(..., description="List of available voices")
    total: int = Field(..., description="Total number of voices")


class VoiceCreateRequest(BaseModel):
    """Request model for creating a custom voice."""

    name: str = Field(..., min_length=1, description="Voice name (must be unique)")
    description: Optional[str] = Field(None, description="Voice description")


class AudioClipRange(BaseModel):
    """Time range (in seconds) for selecting audio clips from a larger file."""

    start_seconds: float = Field(..., ge=0.0, description="Clip start time in seconds (inclusive)")
    end_seconds: float = Field(..., gt=0.0, description="Clip end time in seconds (exclusive)")


class IndividualFileAnalysis(BaseModel):
    """Analysis result for a single audio file."""

    filename: str = Field(..., description="Name of the audio file")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    file_size_mb: Optional[float] = Field(None, description="File size in megabytes")
    warnings: List[str] = Field(default_factory=list, description="Warnings for this file")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class AudioValidationFeedback(BaseModel):
    """Feedback from audio file validation."""

    total_duration_seconds: float = Field(..., description="Total duration of all audio files in seconds")
    individual_files: List[IndividualFileAnalysis] = Field(default_factory=list, description="Analysis of each individual file")
    warnings: List[str] = Field(default_factory=list, description="Warnings about audio quality or duration")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for better results")
    quality_metrics: Dict = Field(default_factory=dict, description="Audio quality metrics summary")


class VoiceCreateResponse(BaseModel):
    """Response model for voice creation."""

    success: bool = Field(..., description="Whether creation was successful")
    message: str = Field(..., description="Status message")
    voice: Optional[VoiceResponse] = Field(None, description="Created voice details")
    validation_feedback: Optional[AudioValidationFeedback] = Field(None, description="Audio validation feedback and recommendations")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class PodcastScriptDurationInputs(BaseModel):
    """Shared duration targeting for podcast script generation."""

    duration: Optional[str] = Field(
        None,
        description="Discrete target duration label: '5 min', '10 min', '15 min', or '30 min'. Optional if approximate_duration_minutes is set.",
    )
    approximate_duration_minutes: Optional[float] = Field(
        None,
        ge=0.25,
        le=180.0,
        description="Approximate target episode length in minutes; drives script word-count targets (takes precedence over duration when set).",
    )

    @model_validator(mode="after")
    def _duration_or_minutes(self) -> "PodcastScriptDurationInputs":
        has_d = self.duration is not None and str(self.duration).strip() != ""
        has_m = self.approximate_duration_minutes is not None
        if not has_d and not has_m:
            raise ValueError(
                "Provide either duration (e.g. '10 min') or approximate_duration_minutes (e.g. 12.5)"
            )
        return self


class PodcastScriptRequest(PodcastScriptDurationInputs):
    """Request model for generating podcast script from URL."""

    url: str = Field(..., description="URL of the article to convert to podcast")
    voices: List[str] = Field(..., min_length=1, max_length=4, description="List of voice names (1-4 voices)")
    genre: str = Field(..., description="Podcast genre (Comedy, Serious, News, Educational, Storytelling, Interview, Documentary)")
    ollama_url: Optional[str] = Field(None, description="Optional custom Ollama server URL")
    ollama_model: Optional[str] = Field(None, description="Optional custom Ollama model name")
    include_production_cues: bool = Field(
        False,
        description="If true, script may include [CUE: ...] markers for production mixing; keep false for standard TTS.",
    )


class PodcastArticleScriptRequest(PodcastScriptDurationInputs):
    """Request model for generating a podcast script from raw article text supplied by the client."""

    article_text: str = Field(..., min_length=1, description="Raw article body (plain text or lightly formatted)")
    title: Optional[str] = Field(None, description="Optional headline shown to the model for context")
    voices: List[str] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Voice names in speaker order: index 0 is Speaker 1, etc.",
    )
    narrator_speaker_index: int = Field(
        ...,
        ge=1,
        le=4,
        description="Which speaker is the narrator: 1 = first entry in `voices`, 2 = second, up to the number of voices",
    )
    genre: str = Field(..., description="Podcast genre (Comedy, Serious, News, Educational, Storytelling, Interview, Documentary)")
    ollama_url: Optional[str] = Field(None, description="Optional custom Ollama server URL")
    ollama_model: Optional[str] = Field(None, description="Optional custom Ollama model name")
    include_production_cues: bool = Field(
        False,
        description="If true, script may include [CUE: ...] markers for production mixing; keep false for standard TTS.",
    )

    @model_validator(mode="after")
    def narrator_index_matches_voice_count(self) -> "PodcastArticleScriptRequest":
        n = len(self.voices)
        if self.narrator_speaker_index > n:
            raise ValueError(
                f"narrator_speaker_index ({self.narrator_speaker_index}) must be between 1 and {n} "
                f"(the number of voices)"
            )
        return self


class DialogueSegment(BaseModel):
    """Dialogue segment metadata for production cue planning."""

    speaker: str = Field(..., description="Speaker label, e.g. Speaker 1")
    text: str = Field(..., description="Dialogue text for this segment")
    start_time_hint: float = Field(..., ge=0.0, description="Approximate start time in seconds from voice track start")


class PodcastSegment(BaseModel):
    """Structured script segment for production mode cue placement."""

    segment_type: Literal[
        "intro_music",
        "dialogue",
        "transition_sting",
        "music_bed",
        "music_bed_in",
        "music_bed_out",
        "outro_music",
    ] = Field(
        ...,
        description="Segment type for music/voice orchestration",
    )
    segment_id: Optional[int] = Field(None, description="1-based index in the production segment list")
    speaker: Optional[str] = Field(None, description="Speaker label for dialogue segments")
    text: Optional[str] = Field(None, description="Dialogue content for dialogue segments")
    start_time_hint: Optional[float] = Field(
        None,
        ge=0.0,
        description="Approximate segment start time in seconds from voice track start",
    )
    duration_hint: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated segment duration in seconds (0 for marker-only cues)",
    )
    energy_level: Optional[str] = Field(None, description="Production energy label (e.g. high, medium, low)")
    notes: Optional[str] = Field(None, description="Optional production notes from segmentation")


class PodcastScriptResponse(BaseModel):
    """Response model for podcast script generation."""

    success: bool = Field(..., description="Whether script generation was successful")
    message: str = Field(..., description="Status message")
    script: Optional[str] = Field(None, description="Generated podcast script with speaker labels")
    script_segments: List[PodcastSegment] = Field(default_factory=list, description="Structured production cue segments")
    warnings: List[str] = Field(default_factory=list, description="Optional warnings (e.g., background music risk)")


class PodcastGenerateRequest(BaseModel):
    """Request model for generating podcast audio from script."""

    script: str = Field(..., description="Podcast script with speaker labels")
    voices: List[str] = Field(..., min_length=1, max_length=4, description="List of voice names (1-4 voices)")
    settings: Optional[SpeechSettings] = Field(default_factory=SpeechSettings, description="Speech generation settings")
    title: Optional[str] = Field(None, description="Optional title for saving into the podcast library")
    source_url: Optional[str] = Field(None, description="Optional source URL (e.g., article URL)")
    genre: Optional[str] = Field(None, description="Optional genre metadata")
    duration: Optional[str] = Field(None, description="Optional duration metadata (e.g., '10 min')")
    save_to_library: bool = Field(default=True, description="Whether to save the generated podcast to the library")


class PodcastGenerateResponse(BaseModel):
    """Response model for podcast audio generation."""

    success: bool = Field(..., description="Whether generation was successful")
    message: str = Field(..., description="Status message")
    audio_url: Optional[str] = Field(None, description="URL to generated audio file")
    file_path: Optional[str] = Field(None, description="Path to generated audio file")
    script: Optional[str] = Field(None, description="Script used for generation")
    script_segments: List[PodcastSegment] = Field(default_factory=list, description="Structured production cue segments")
    podcast_id: Optional[str] = Field(None, description="Podcast library identifier (if saved)")
    warnings: List[str] = Field(default_factory=list, description="Optional warnings (e.g., background music risk)")


ProductionGenre = Literal["tech_talk", "news", "storytelling", "true_crime", "comedy"]


class PodcastProductionRequest(BaseModel):
    """Request model for production-mode podcast generation with music cues."""

    script: str = Field(..., description="Podcast script with speaker labels")
    voices: List[str] = Field(..., min_length=1, max_length=4, description="List of voice names (1-4 voices)")
    settings: Optional[SpeechSettings] = Field(default_factory=SpeechSettings, description="Speech generation settings")
    title: Optional[str] = Field(None, description="Optional title for saving into the podcast library")
    source_url: Optional[str] = Field(None, description="Optional source URL (e.g., article URL)")
    genre: Optional[str] = Field(None, description="Optional genre metadata")
    duration: Optional[str] = Field(None, description="Optional duration metadata (e.g., '10 min')")
    save_to_library: bool = Field(default=True, description="Whether to save the generated podcast to the library")
    production_mode: bool = Field(default=True, description="Whether production mode is enabled")
    production_genre: Optional[ProductionGenre] = Field(
        None,
        description=(
            "Production template id (Director, library filter, generation prompts, mastering). "
            "Takes precedence over legacy `style` when set."
        ),
    )
    style: Literal["tech_talk", "casual", "news", "storytelling"] = Field(
        default="casual",
        description="Legacy alias: maps to a default production template when `production_genre` is omitted",
    )
    enabled_cues: List[Literal["intro", "outro", "transitions", "bed"]] = Field(
        default_factory=lambda: ["intro", "outro", "transitions"],
        description="Enabled cue groups for production mixing",
    )
    ollama_url: Optional[str] = Field(None, description="Optional custom Ollama server URL")
    ollama_model: Optional[str] = Field(None, description="Optional custom Ollama model name")


class PodcastProductionSubmitResponse(BaseModel):
    """Response model for accepted production-mode generation tasks."""

    success: bool = Field(..., description="Whether request was accepted")
    message: str = Field(..., description="Status message")
    task_id: str = Field(..., description="Podcast production task identifier")
    status: str = Field(..., description="Initial task status")


class PodcastProductionStatusResponse(BaseModel):
    """Response model for production-mode task status polling."""

    success: bool = Field(..., description="Whether status query succeeded")
    message: str = Field(..., description="Status message")
    task_id: str = Field(..., description="Podcast production task identifier")
    status: str = Field(..., description="Task status: queued/running/succeeded/failed")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    progress_pct: int = Field(default=0, ge=0, le=100, description="Task progress percentage")
    stage_progress: Dict[str, str] = Field(default_factory=dict, description="Stage status map")
    cue_status: Dict[str, str] = Field(default_factory=dict, description="Per-cue generation status map")
    audio_url: Optional[str] = Field(None, description="Final mixed audio URL when complete")
    file_path: Optional[str] = Field(None, description="Final mixed audio path when complete")
    podcast_id: Optional[str] = Field(None, description="Podcast library identifier (if saved)")
    script_segments: List[PodcastSegment] = Field(default_factory=list, description="Structured production cue segments")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered during production")
    error: Optional[str] = Field(default=None, description="Failure reason when task fails")
    mix_qa_error: Optional[str] = Field(
        default=None,
        description="Set when post-mix QA threw (mix still completed if present)",
    )


class RegenerateEventRequest(BaseModel):
    """Re-run ACE-Step / SAO for a single TrackEvent and remix (no TTS / Director)."""

    track_id: str = Field(..., description="Timeline track_id from ProductionPlan")
    event_id: str = Field(..., description="TrackEvent.event_id to regenerate")


class PodcastCompareRequest(BaseModel):
    """A/B: same voice track, two production genre templates."""

    script: str = Field(..., description="Podcast script with speaker labels")
    voices: List[str] = Field(..., min_length=1, max_length=4, description="Voice names")
    genres: List[ProductionGenre] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two production_genre template ids",
    )
    settings: Optional[SpeechSettings] = Field(default_factory=SpeechSettings, description="Speech generation settings")
    ollama_url: Optional[str] = Field(None, description="Optional Ollama URL for Director")
    ollama_model: Optional[str] = Field(None, description="Optional Ollama model")
    genre: Optional[str] = Field(None, description="Optional metadata for segmentation")
    duration: Optional[str] = Field(None, description="Optional duration hint for segmentation")
    style: Literal["tech_talk", "casual", "news", "storytelling"] = Field(
        default="casual",
        description="Legacy style for script segmentation when genre metadata is absent",
    )

    @field_validator("genres")
    @classmethod
    def _two_distinct(cls, v: List[str]) -> List[str]:
        if len(v) != 2:
            raise ValueError("genres must have exactly two entries")
        return v


class PodcastCompareSubmitResponse(BaseModel):
    compare_id: str
    status: str
    message: str = ""


class PodcastCompareStatusResponse(BaseModel):
    compare_id: str
    status: str
    message: str = ""
    audio_url_a: Optional[str] = None
    audio_url_b: Optional[str] = None
    file_path_a: Optional[str] = None
    file_path_b: Optional[str] = None
    qa_a: Optional[Dict[str, Any]] = None
    qa_b: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class PodcastItem(BaseModel):
    """Podcast library item metadata."""

    id: str = Field(..., description="Podcast identifier")
    title: str = Field(..., description="Podcast title")
    voices: List[str] = Field(default_factory=list, description="Voices used in this podcast")
    source_url: Optional[str] = Field(None, description="Source URL (if any)")
    genre: Optional[str] = Field(None, description="Genre metadata (if any)")
    duration: Optional[str] = Field(None, description="Target duration metadata (if any)")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    audio_url: Optional[str] = Field(None, description="Download URL for the podcast audio")


class PodcastListResponse(BaseModel):
    """Podcast library list response."""

    podcasts: List[PodcastItem] = Field(default_factory=list, description="Podcast library items")
    total: int = Field(..., description="Total number of items")


class MusicGenerateRequest(BaseModel):
    """Request model for custom ACE-Step music generation."""

    caption: str = Field(default="", description="Music style prompt/caption")
    lyrics: str = Field(default="", description="Lyrics text")
    bpm: Optional[int] = Field(default=None, ge=30, le=300, description="Tempo in BPM")
    keyscale: str = Field(default="", description="Musical key/scale (e.g., C Major)")
    timesignature: str = Field(default="", description="Time signature (2,3,4,6)")
    duration: Optional[float] = Field(default=None, ge=10, le=600, description="Target duration in seconds")
    vocal_language: str = Field(default="en", description="Vocal language code")
    instrumental: bool = Field(default=False, description="Generate instrumental-only music")
    thinking: bool = Field(default=True, description="Enable 5Hz LM-assisted reasoning")
    inference_steps: int = Field(default=8, ge=1, le=200, description="Diffusion inference steps")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of variations")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    audio_format: str = Field(default="mp3", description="Output audio format: mp3/wav/flac")


class MusicCoverGenerateRequest(BaseModel):
    """Request model for ACE-Step cover-mode generation."""

    prompt: str = Field(default="", description="Cover mode prompt/caption")
    lyrics: str = Field(default="", description="Optional lyrics text")
    duration: Optional[float] = Field(default=None, ge=10, le=600, description="Target duration in seconds")
    audio_cover_strength: float = Field(default=0.6, ge=0.0, le=1.0, description="Cover guidance strength")
    vocal_language: Optional[str] = Field(default=None, description="Optional vocal language code")
    instrumental: bool = Field(default=False, description="Generate instrumental-only music")
    thinking: bool = Field(default=True, description="Enable 5Hz LM-assisted reasoning")
    inference_steps: int = Field(default=8, ge=1, le=200, description="Diffusion inference steps")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of variations")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    audio_format: str = Field(default="mp3", description="Output audio format: mp3/wav/flac")


class MusicGenerateResponse(BaseModel):
    """Response model for submitted music generation tasks."""

    success: bool = Field(..., description="Whether request was accepted")
    message: str = Field(..., description="Status message")
    task_id: str = Field(..., description="ACE-Step task identifier")


class MusicStatusResponse(BaseModel):
    """Response model for music generation task status."""

    success: bool = Field(..., description="Whether status query succeeded")
    message: str = Field(..., description="Status message")
    task_id: str = Field(..., description="ACE-Step task identifier")
    status: str = Field(..., description="Task status: running/succeeded/failed")
    audios: List[str] = Field(default_factory=list, description="Generated audio URLs when complete")
    metadata: List[Dict] = Field(default_factory=list, description="Generated metadata records")
    error: Optional[str] = Field(default=None, description="Failure reason when task fails")


class MusicLyricsRequest(BaseModel):
    """Request model for LLM-assisted lyrics generation."""

    description: str = Field(..., description="User idea/description for the song")
    genre: str = Field(default="Pop", description="Target genre")
    mood: str = Field(default="Neutral", description="Target mood")
    language: str = Field(default="English", description="Target lyrics language")
    duration_hint: Optional[str] = Field(default=None, description="Optional duration hint")


class MusicLyricsResponse(BaseModel):
    """Response model for generated lyrics."""

    success: bool = Field(..., description="Whether lyrics generation was successful")
    message: str = Field(..., description="Status message")
    lyrics: str = Field(..., description="Generated lyrics with structure tags")
    caption: str = Field(default="", description="Suggested style caption/prompt")


class MusicSimpleGenerateRequest(BaseModel):
    """Request model for simple description-driven generation."""

    description: str = Field(default="", description="Natural language music description (required for refine mode)")
    input_mode: Literal["refine", "exact"] = Field(
        default="refine",
        description="Simple input mode: 'refine' uses Ollama shaping, 'exact' forwards ACE-Step-ready fields",
    )
    instrumental: bool = Field(default=False, description="Generate instrumental-only output")
    vocal_language: Optional[str] = Field(default=None, description="Optional vocal language override")
    duration: Optional[float] = Field(default=None, ge=10, le=600, description="Optional target duration in seconds")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of generated variations")
    exact_caption: Optional[str] = Field(default=None, description="Exact mode caption to send to ACE-Step")
    exact_lyrics: Optional[str] = Field(default=None, description="Exact mode lyrics to send to ACE-Step")
    exact_bpm: Optional[int] = Field(default=None, ge=30, le=300, description="Exact mode BPM override")
    exact_keyscale: Optional[str] = Field(default=None, description="Exact mode key/scale override")
    exact_timesignature: Optional[str] = Field(default=None, description="Exact mode time signature override")


class MusicHealthResponse(BaseModel):
    """Response model for ACE-Step service health."""

    available: bool = Field(..., description="Whether ACE-Step repo/config is available")
    running: bool = Field(..., description="Whether ACE-Step API server is currently running")
    service: str = Field(..., description="Service identifier")
    host: str = Field(..., description="ACE-Step host")
    port: int = Field(..., description="ACE-Step port")


class AceStepRuntimeSettingsUpdateRequest(BaseModel):
    """Request model for updating global ACE-Step runtime model settings."""

    acestep_config_path: str = Field(..., min_length=1, description="ACE-Step DiT model ID")
    acestep_lm_model_path: str = Field(..., min_length=1, description="ACE-Step LM model ID")


class AceStepRuntimeSettingsResponse(BaseModel):
    """Response model for current ACE-Step runtime model settings."""

    acestep_config_path: str = Field(..., description="ACE-Step DiT model ID")
    acestep_lm_model_path: str = Field(..., description="ACE-Step LM model ID")
    source: str = Field(..., description="settings_file or env_defaults")
    restart_required: bool = Field(
        ..., description="Whether a running ACE-Step subprocess must restart to apply changes"
    )
    settings_file: str = Field(..., description="Path to persisted runtime settings file")


class AceStepModelCatalogResponse(BaseModel):
    """Response model for supported ACE-Step model catalog."""

    dit_models: List[str] = Field(default_factory=list, description="Supported ACE-Step DiT model IDs")
    lm_models: List[str] = Field(default_factory=list, description="Supported ACE-Step LM model IDs")
    defaults: Dict[str, str] = Field(default_factory=dict, description="Environment-backed default model IDs")
    current: Dict[str, str] = Field(default_factory=dict, description="Currently effective runtime settings")


class MusicPresetRequest(BaseModel):
    """Request model for saving/updating a music preset."""

    name: str = Field(..., min_length=1, description="Preset display name")
    mode: str = Field(default="custom", description="Preset mode: simple/custom")
    values: Dict = Field(default_factory=dict, description="Preset parameter values")


class MusicPresetResponse(BaseModel):
    """Response model for a single music preset."""

    id: str = Field(..., description="Preset identifier")
    name: str = Field(..., description="Preset display name")
    mode: str = Field(..., description="Preset mode")
    values: Dict = Field(default_factory=dict, description="Stored preset values")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class MusicPresetListResponse(BaseModel):
    """Response model for music presets list."""

    presets: List[MusicPresetResponse] = Field(default_factory=list, description="Saved music presets")
    total: int = Field(..., description="Total number of presets")


class MusicHistoryItemResponse(BaseModel):
    """Response model for a music generation history item."""

    id: str = Field(..., description="History item identifier")
    task_id: str = Field(..., description="Associated ACE-Step task id")
    mode: str = Field(..., description="Generation mode: simple/custom")
    status: str = Field(..., description="Generation status")
    request_payload: Dict = Field(default_factory=dict, description="Original generation request payload")
    audios: List[str] = Field(default_factory=list, description="Generated audio URLs")
    metadata: List[Dict] = Field(default_factory=list, description="Generated metadata entries")
    error: Optional[str] = Field(default=None, description="Error text if generation failed")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class MusicHistoryListResponse(BaseModel):
    """Response model for music generation history list."""

    history: List[MusicHistoryItemResponse] = Field(default_factory=list, description="History items")
    total: int = Field(..., description="Total number of returned history items")


class VoiceProfile(BaseModel):
    """Structured voice profile with speech pattern characteristics."""

    cadence: Optional[str] = Field(None, description="Description of speech rhythm/pace")
    tone: Optional[str] = Field(None, description="Emotional tone and delivery style")
    vocabulary_style: Optional[str] = Field(None, description="Word choice patterns (formal, casual, technical, etc.)")
    sentence_structure: Optional[str] = Field(None, description="Typical sentence patterns (short, long, complex)")
    unique_phrases: List[str] = Field(default_factory=list, description="Common phrases or expressions")
    keywords: List[str] = Field(default_factory=list, description="Keywords for context (e.g., person names)")
    profile_text: Optional[str] = Field(None, description="Full text description of the voice")
    created_at: Optional[datetime] = Field(None, description="Profile creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Profile last update timestamp")


class VoiceProfileApplyRequest(BaseModel):
    """Request model for applying a full voice profile payload to a voice."""

    cadence: Optional[str] = Field(None, description="Description of speech rhythm/pace")
    tone: Optional[str] = Field(None, description="Emotional tone and delivery style")
    vocabulary_style: Optional[str] = Field(None, description="Word choice patterns (formal, casual, technical, etc.)")
    sentence_structure: Optional[str] = Field(None, description="Typical sentence patterns (short, long, complex)")
    unique_phrases: List[str] = Field(default_factory=list, description="Common phrases or expressions")
    keywords: List[str] = Field(default_factory=list, description="Keywords for context (e.g., person names)")
    profile_text: Optional[str] = Field(None, description="Full text description of the voice")


class VoiceProfileRequest(BaseModel):
    """Request model for creating/updating voice profiles."""

    keywords: Optional[List[str]] = Field(None, description="Keywords to enhance profile (e.g., person names)")
    ollama_url: Optional[str] = Field(None, description="Optional custom Ollama server URL")
    ollama_model: Optional[str] = Field(None, description="Optional custom Ollama model name")


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile data."""

    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Status message")
    profile: Optional[VoiceProfile] = Field(None, description="Voice profile data")


class VoiceProfileFromAudioRequest(BaseModel):
    """Request model (logical) for deriving a profile from audio."""

    keywords: Optional[List[str]] = Field(None, description="Optional keywords/context to help profiling")
    ollama_url: Optional[str] = Field(None, description="Optional custom Ollama server URL")
    ollama_model: Optional[str] = Field(None, description="Optional custom Ollama model name")


class VoiceProfileFromAudioResponse(BaseModel):
    """Response model for audio-derived voice profile."""

    success: bool = Field(..., description="Whether profiling was successful")
    message: str = Field(..., description="Status message")
    profile: Optional[VoiceProfile] = Field(None, description="Derived voice profile")
    transcript: Optional[str] = Field(None, description="Transcript derived from the audio")
    validation_feedback: Optional[AudioValidationFeedback] = Field(
        None, description="Audio validation feedback and recommendations"
    )


class VoiceUpdateRequest(BaseModel):
    """Request model for updating voice details."""

    name: Optional[str] = Field(None, min_length=1, description="New voice name")
    description: Optional[str] = Field(None, description="New voice description")
    language_code: Optional[str] = Field(None, description="Optional voice language code (e.g., en, zh, in)")
    gender: Optional[str] = Field(None, description="Optional voice gender: male, female, neutral, unknown")


class VoiceUpdateResponse(BaseModel):
    """Response model for voice update."""

    success: bool = Field(..., description="Whether update was successful")
    message: str = Field(..., description="Status message")
    voice: Optional[VoiceResponse] = Field(None, description="Updated voice details")


class AdSegmentItem(BaseModel):
    """A detected advertisement or sponsor segment."""

    start_seconds: float = Field(..., ge=0, description="Segment start time in seconds")
    end_seconds: float = Field(..., ge=0, description="Segment end time in seconds")
    label: str = Field(..., description="Short description of the ad or sponsor read")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")


class PodcastAdScanStatusResponse(BaseModel):
    """Job status for podcast ad scanning."""

    job_id: str = Field(..., description="Scan job identifier")
    status: str = Field(
        ...,
        description="queued | transcribing | analyzing | complete | failed",
    )
    progress_pct: int = Field(0, ge=0, le=100, description="Approximate completion percentage")
    current_stage: Optional[str] = Field(None, description="Human-readable stage message")
    ad_segments: Optional[List[AdSegmentItem]] = Field(None, description="Detected ads when complete")
    duration_seconds: Optional[float] = Field(None, description="Total audio duration when known")
    error: Optional[str] = Field(None, description="Error message when status is failed")


class PodcastAdScanSubmitResponse(BaseModel):
    """Immediate response after submitting an ad scan job."""

    job_id: str
    status: str
    progress_pct: int = 0


class PodcastAdExportRequest(BaseModel):
    """Export edited audio after ad scan."""

    job_id: str = Field(..., description="Completed scan job id")
    export_mode: Literal["clean", "ads_only"] = Field(
        ...,
        description="clean removes ads; ads_only keeps only detected ad audio",
    )


class PodcastAdExportResponse(BaseModel):
    """Result of an export operation."""

    download_url: str = Field(..., description="Relative URL to download the MP3")
    filename: str = Field(..., description="Safe basename of the exported file")
    duration_seconds: float = Field(..., description="Duration of the exported audio")
    file_size_bytes: int = Field(..., description="Exported file size on disk")


class SpeakerIsolationClipItem(BaseModel):
    """A single extracted preview clip for one speaker."""

    clip_id: str = Field(..., description="Stable id, e.g. speaker_1_clip_2")
    filename: str = Field(..., description="Basename of the mp3 file on disk")
    start_seconds: float = Field(..., description="Approximate start in source audio (seconds)")
    end_seconds: float = Field(..., description="Approximate end in source audio (seconds)")
    duration_seconds: float = Field(..., description="Exported clip duration (seconds, 10–15)")
    download_url: str = Field(..., description="Relative URL to stream or download the clip")


class SpeakerIsolationSpeakerItem(BaseModel):
    """One diarized speaker with up to three clips."""

    speaker_id: str = Field(..., description="Internal diarization label (e.g. SPEAKER_00)")
    label: str = Field(..., description='Display label, e.g. "Speaker 1" or an inferred name')
    total_speaking_seconds: float = Field(..., description="Sum of speech time for this speaker")
    clips: List[SpeakerIsolationClipItem] = Field(default_factory=list)
    label_source: Optional[str] = Field(
        None,
        description='optional: "inferred" if name was detected from speech, else "default"',
    )


class SpeakerIsolationStatusResponse(BaseModel):
    """Job status for speaker isolation / clip extraction."""

    job_id: str
    status: str = Field(
        ...,
        description="queued | diarizing | extracting | inferring_names | complete | failed",
    )
    progress_pct: int = Field(0, ge=0, le=100)
    current_stage: Optional[str] = None
    speakers: Optional[List[SpeakerIsolationSpeakerItem]] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class SpeakerIsolationSubmitResponse(BaseModel):
    """Immediate response after submitting a speaker isolation job."""

    job_id: str
    status: str = "queued"
    progress_pct: int = 0


class CreateVoiceFromIsolationClipRequest(BaseModel):
    """Create a custom voice from one isolation job clip."""

    job_id: str = Field(..., description="Isolation job id")
    clip_id: str = Field(..., description="Clip id from job results, e.g. speaker_1_clip_2")
    voice_name: str = Field(..., min_length=1, description="Name for the new voice")
    voice_description: Optional[str] = Field(None, description="Optional description")
