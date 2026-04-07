"""
Voice management endpoints.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from ..config import config
from ..models.schemas import (
    AudioValidationFeedback,
    ErrorResponse,
    IndividualFileAnalysis,
    VoiceCreateResponse,
    VoiceListResponse,
    VoiceProfileApplyRequest,
    VoiceProfileFromAudioResponse,
    VoiceProfile,
    VoiceProfileRequest,
    VoiceProfileResponse,
    VoiceQualityAnalysis,
    VoiceResponse,
    VoiceUpdateRequest,
    VoiceUpdateResponse,
)
from ..services.voice_manager import voice_manager
from ..services.voice_profile_from_audio import voice_profile_from_audio

router = APIRouter(prefix="/api/v1/voices", tags=["voices"])

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


def _build_voice_create_response(voice_data: dict) -> VoiceCreateResponse:
    """Build VoiceCreateResponse from voice_data dict."""
    created_at = voice_data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = None
    image_url = f"/api/v1/voices/{voice_data['id']}/image" if voice_data.get("image_filename") else None
    quality_analysis = None
    if voice_data.get("quality_analysis"):
        qa = voice_data["quality_analysis"]
        quality_analysis = VoiceQualityAnalysis(
            clone_quality=qa.get("clone_quality", "fair"),
            issues=qa.get("issues", []),
            recording_quality_score=qa.get("recording_quality_score", 0.5),
            background_music_detected=qa.get("background_music_detected", False),
            background_noise_detected=qa.get("background_noise_detected", False),
        )
    voice_response = VoiceResponse(
        id=voice_data["id"],
        name=voice_data["name"],
        display_name=voice_data.get("display_name"),
        language_code=voice_data.get("language_code"),
        language_label=voice_data.get("language_label"),
        gender=voice_data.get("gender"),
        description=voice_data.get("description"),
        type=voice_data.get("type", "custom"),
        created_at=created_at,
        audio_files=voice_data.get("audio_files"),
        image_url=image_url,
        quality_analysis=quality_analysis,
    )
    validation_feedback = None
    if "validation_feedback" in voice_data:
        feedback_data = voice_data["validation_feedback"]
        individual_files = [
            IndividualFileAnalysis(**f) for f in feedback_data.get("individual_files", [])
        ]
        validation_feedback = AudioValidationFeedback(
            total_duration_seconds=feedback_data.get("total_duration_seconds", 0.0),
            individual_files=individual_files,
            warnings=feedback_data.get("warnings", []),
            recommendations=feedback_data.get("recommendations", []),
            quality_metrics=feedback_data.get("quality_metrics", {}),
        )
    message = f"Voice '{voice_data.get('name', '')}' created successfully"
    if validation_feedback and validation_feedback.warnings:
        message += f" ({len(validation_feedback.warnings)} warning(s) about audio quality/duration)"
    return VoiceCreateResponse(
        success=True,
        message=message,
        voice=voice_response,
        validation_feedback=validation_feedback,
    )


async def _create_voice_from_prompt(
    name: str,
    description: str,
    voice_design_prompt: str,
    image: UploadFile | None,
    language_code: str | None,
    gender: str | None,
) -> VoiceCreateResponse:
    """Create a VoiceDesign voice from text description."""
    temp_image_path = None
    if image and image.filename:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image must be JPEG, PNG, or WebP. Got: {content_type}",
            )
        image_content = await image.read()
        if len(image_content) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image must be at most {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB",
            )
        ext = Path(image.filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image must have extension .jpg, .png, or .webp",
            )
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_image.write(image_content)
        temp_image.close()
        temp_image_path = Path(temp_image.name)
    try:
        voice_data = voice_manager.create_voice_from_prompt(
            name=name,
            description=description,
            voice_design_prompt=voice_design_prompt.strip(),
            language_code=language_code,
            gender=gender,
            image_path=temp_image_path,
        )
        return _build_voice_create_response(voice_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    finally:
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink(missing_ok=True)


async def _create_voice_from_clips_impl(
    name: str,
    description: str,
    audio_file: UploadFile,
    clip_ranges: str,
    image: UploadFile,
    keywords: str,
    language_code: str,
    gender: str,
) -> VoiceCreateResponse:
    """Create voice from clips within a single audio file (shared implementation)."""
    clip_ranges_raw = json.loads(clip_ranges)
    if not isinstance(clip_ranges_raw, list) or len(clip_ranges_raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="clip_ranges must be a non-empty JSON array",
        )
    max_clips = 50
    is_qwen3 = (config.TTS_BACKEND or "qwen3").strip().lower() == "qwen3"
    max_total_selected_seconds = 60.0 if is_qwen3 else 600.0
    min_clip_seconds = 0.5
    if len(clip_ranges_raw) > max_clips:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many clips requested (max {max_clips})",
        )
    temp_input_path = None
    temp_clip_paths = []
    temp_image_path = None
    if image and image.filename:
        content_type = (image.content_type or "").lower()
        if content_type in ALLOWED_IMAGE_CONTENT_TYPES:
            image_content = await image.read()
            if len(image_content) <= MAX_IMAGE_SIZE_BYTES:
                ext = Path(image.filename).suffix.lower()
                if ext in (".jpg", ".jpeg", ".png", ".webp"):
                    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    temp_image.write(image_content)
                    temp_image.close()
                    temp_image_path = Path(temp_image.name)

    try:
        suffix = Path(audio_file.filename).suffix
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input_path = Path(temp_input.name)
        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty")
        temp_input.write(content)
        temp_input.close()

        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(temp_input_path))
        duration_seconds = len(audio) / 1000.0
        if duration_seconds <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file has zero duration")
        parsed_ranges = []
        total_selected_seconds = 0.0
        for idx, item in enumerate(clip_ranges_raw):
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] must be an object with start_seconds and end_seconds",
                )
            start_seconds = float(item.get("start_seconds"))
            end_seconds = float(item.get("end_seconds"))
            if start_seconds < 0 or end_seconds <= 0 or start_seconds >= end_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] invalid start_seconds/end_seconds",
                )
            if end_seconds > duration_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] end_seconds exceeds audio duration",
                )
            clip_duration = end_seconds - start_seconds
            if clip_duration < min_clip_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] too short (min {min_clip_seconds}s)",
                )
            total_selected_seconds += clip_duration
            if total_selected_seconds > max_total_selected_seconds:
                detail = (
                    "For Qwen3-TTS, total selected clip duration must be at most 60 seconds."
                    if is_qwen3
                    else f"Total selected clip duration exceeds {max_total_selected_seconds:.0f}s"
                )
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
            parsed_ranges.append((start_seconds, end_seconds))

        for start_seconds, end_seconds in parsed_ranges:
            start_ms = int(start_seconds * 1000)
            end_ms = int(end_seconds * 1000)
            segment = audio[start_ms:end_ms]
            if segment.frame_rate != 24000:
                segment = segment.set_frame_rate(24000)
            if segment.channels > 1:
                segment = segment.set_channels(1)
            clip_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            clip_path = Path(clip_temp.name)
            clip_temp.close()
            segment.export(str(clip_path), format="wav", parameters=["-ar", "24000"])
            temp_clip_paths.append(clip_path)

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
        voice_data = voice_manager.create_custom_voice(
            name=name,
            description=description,
            audio_files=temp_clip_paths,
            keywords=keywords_list,
            ollama_url=None,
            ollama_model=None,
            language_code=language_code,
            gender=gender,
            image_path=temp_image_path,
        )
        return _build_voice_create_response(voice_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid clip_ranges JSON: {e}",
        ) from e
    finally:
        for p in temp_clip_paths:
            if p.exists():
                p.unlink(missing_ok=True)
        if temp_input_path and temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink(missing_ok=True)


@router.post(
    "",
    response_model=VoiceCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_voice(
    name: str = Form(...),
    description: str = Form(None),
    creation_source: str = Form("audio"),
    audio_files: list[UploadFile] = File(None),
    audio_file: UploadFile = File(None),
    clip_ranges: str = Form(None),
    voice_design_prompt: str = Form(None),
    image: UploadFile = File(None),
    keywords: str = Form(None),
    language_code: str = Form(None),
    gender: str = Form(None),
) -> VoiceCreateResponse:
    """
    Create a voice: from audio files, from clips in one file, or from a text description (VoiceDesign).

    creation_source: "audio" (default) | "clips" | "prompt"
    - audio: provide audio_files (multiple files combined)
    - clips: provide audio_file and clip_ranges (JSON array of {start_seconds, end_seconds})
    - prompt: provide voice_design_prompt (natural-language voice description)
    """
    source = (creation_source or "audio").strip().lower()
    if source not in ("audio", "clips", "prompt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="creation_source must be one of: audio, clips, prompt",
        )

    if source == "prompt":
        if not voice_design_prompt or not voice_design_prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="voice_design_prompt is required when creation_source is prompt",
            )
        return await _create_voice_from_prompt(
            name=name,
            description=description or "",
            voice_design_prompt=voice_design_prompt,
            image=image,
            language_code=language_code or None,
            gender=gender or None,
        )

    if source == "clips":
        if not audio_file or not audio_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_file is required when creation_source is clips",
            )
        if not clip_ranges or not clip_ranges.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="clip_ranges is required when creation_source is clips",
            )
        return await _create_voice_from_clips_impl(
            name=name,
            description=description,
            audio_file=audio_file,
            clip_ranges=clip_ranges,
            image=image,
            keywords=keywords,
            language_code=language_code,
            gender=gender,
        )

    # source == "audio"
    if not audio_files or len(audio_files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file is required when creation_source is audio",
        )

    # Optional avatar image validation
    temp_image_path = None
    if image and image.filename:
        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image must be JPEG, PNG, or WebP. Got: {content_type}",
            )
        image_content = await image.read()
        if len(image_content) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image must be at most {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB",
            )
        ext = Path(image.filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image must have extension .jpg, .png, or .webp",
            )
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_image.write(image_content)
        temp_image.close()
        temp_image_path = Path(temp_image.name)

    # Save uploaded files temporarily
    temp_files = []
    try:
        for audio_file in audio_files:
            # Validate file type
            if not audio_file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All files must have a filename",
                )

            # Create temporary file
            import tempfile

            suffix = Path(audio_file.filename).suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_files.append(Path(temp_file.name))

            # Write uploaded content
            content = await audio_file.read()
            temp_file.write(content)
            temp_file.close()

        # Parse keywords if provided
        keywords_list = None
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

        # Get Ollama settings from form (optional)
        ollama_url = None  # Could be added as form field if needed
        ollama_model = None  # Could be added as form field if needed

        # Create voice (audio source)
        try:
            voice_data = voice_manager.create_custom_voice(
                name=name,
                description=description,
                audio_files=temp_files,
                keywords=keywords_list,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                language_code=language_code,
                gender=gender,
                image_path=temp_image_path,
            )
            return _build_voice_create_response(voice_data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()


@router.post(
    "/from-audio-clips",
    response_model=VoiceCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_voice_from_audio_clips(
    name: str = Form(...),
    description: str = Form(None),
    audio_file: UploadFile = File(...),
    clip_ranges: str = Form(...),
    image: UploadFile = File(None),
    keywords: str = Form(None),
    language_code: str = Form(None),
    gender: str = Form(None),
) -> VoiceCreateResponse:
    """
    Create a voice from clips within a single audio file (alias for POST / with creation_source=clips).
    """
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file must have a filename")
    return await _create_voice_from_clips_impl(
        name=name,
        description=description,
        audio_file=audio_file,
        clip_ranges=clip_ranges,
        image=image,
        keywords=keywords or "",
        language_code=language_code,
        gender=gender,
    )


async def _create_voice_from_audio_clips_deprecated(
    name: str,
    description: str,
    audio_file: UploadFile,
    clip_ranges: str,
    keywords: str,
    language_code: str,
    gender: str,
) -> VoiceCreateResponse:
    """Deprecated: kept only for reference. Use _create_voice_from_clips_impl."""
    try:
        clip_ranges_raw = json.loads(clip_ranges)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"clip_ranges must be valid JSON: {str(e)}",
        ) from e

    if not isinstance(clip_ranges_raw, list) or len(clip_ranges_raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="clip_ranges must be a non-empty JSON array",
        )

    # Guardrails to limit abuse
    max_clips = 50
    is_qwen3 = (config.TTS_BACKEND or "qwen3").strip().lower() == "qwen3"
    max_total_selected_seconds = 60.0 if is_qwen3 else 600.0  # Qwen3: 60s ref; legacy: 10 min
    min_clip_seconds = 0.5

    if len(clip_ranges_raw) > max_clips:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many clips requested (max {max_clips})",
        )

    # Save uploaded audio file temporarily
    temp_input_path: Path | None = None
    temp_clip_paths: list[Path] = []

    try:
        suffix = Path(audio_file.filename).suffix
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input_path = Path(temp_input.name)

        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty")
        temp_input.write(content)
        temp_input.close()

        # Load audio to determine duration and slice clips
        from pydub import AudioSegment

        try:
            audio = AudioSegment.from_file(str(temp_input_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load audio file for clipping: {str(e)}",
            ) from e

        duration_seconds = len(audio) / 1000.0
        if duration_seconds <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file has zero duration")

        parsed_ranges: list[tuple[float, float]] = []
        total_selected_seconds = 0.0
        for idx, item in enumerate(clip_ranges_raw):
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] must be an object with start_seconds and end_seconds",
                )

            try:
                start_seconds = float(item.get("start_seconds"))
                end_seconds = float(item.get("end_seconds"))
            except (TypeError, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] start_seconds/end_seconds must be numbers",
                ) from e

            if start_seconds < 0 or end_seconds <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] must be >= 0 and end_seconds > 0",
                )
            if start_seconds >= end_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] must satisfy start_seconds < end_seconds",
                )
            if end_seconds > duration_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] end_seconds exceeds audio duration ({duration_seconds:.2f}s)",
                )

            clip_duration = end_seconds - start_seconds
            if clip_duration < min_clip_seconds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clip_ranges[{idx}] is too short (min {min_clip_seconds:.1f}s)",
                )

            total_selected_seconds += clip_duration
            if total_selected_seconds > max_total_selected_seconds:
                detail = (
                    "For Qwen3-TTS, total selected clip duration must be at most 60 seconds."
                    if is_qwen3
                    else f"Total selected clip duration exceeds {max_total_selected_seconds:.0f}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail,
                )

            parsed_ranges.append((start_seconds, end_seconds))

        # Slice clips and export to temp WAV files
        for idx, (start_seconds, end_seconds) in enumerate(parsed_ranges):
            start_ms = int(start_seconds * 1000)
            end_ms = int(end_seconds * 1000)
            segment = audio[start_ms:end_ms]

            # Normalize to standard settings early to avoid surprises
            if segment.frame_rate != 24000:
                segment = segment.set_frame_rate(24000)
            if segment.channels > 1:
                segment = segment.set_channels(1)

            clip_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            clip_path = Path(clip_temp.name)
            clip_temp.close()
            segment.export(str(clip_path), format="wav", parameters=["-ar", "24000"])
            temp_clip_paths.append(clip_path)

        # Parse keywords if provided
        keywords_list = None
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

        # Create voice using existing pipeline
        voice_data = voice_manager.create_custom_voice(
            name=name,
            description=description,
            audio_files=temp_clip_paths,
            keywords=keywords_list,
            ollama_url=None,
            ollama_model=None,
            language_code=language_code,
            gender=gender,
        )

        # Parse created_at if it's a string
        created_at = voice_data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        image_url = f"/api/v1/voices/{voice_data['id']}/image" if voice_data.get("image_filename") else None
        quality_analysis = None
        if voice_data.get("quality_analysis"):
            qa = voice_data["quality_analysis"]
            quality_analysis = VoiceQualityAnalysis(
                clone_quality=qa.get("clone_quality", "fair"),
                issues=qa.get("issues", []),
                recording_quality_score=qa.get("recording_quality_score", 0.5),
                background_music_detected=qa.get("background_music_detected", False),
                background_noise_detected=qa.get("background_noise_detected", False),
            )
        voice_response = VoiceResponse(
            id=voice_data["id"],
            name=voice_data["name"],
            display_name=voice_data.get("display_name"),
            language_code=voice_data.get("language_code"),
            language_label=voice_data.get("language_label"),
            gender=voice_data.get("gender"),
            description=voice_data.get("description"),
            type=voice_data.get("type", "custom"),
            created_at=created_at,
            audio_files=voice_data.get("audio_files"),
            image_url=image_url,
            quality_analysis=quality_analysis,
        )

        # Parse validation feedback if present
        validation_feedback = None
        if "validation_feedback" in voice_data:
            feedback_data = voice_data["validation_feedback"]
            individual_files = [
                IndividualFileAnalysis(**file_data) for file_data in feedback_data.get("individual_files", [])
            ]
            validation_feedback = AudioValidationFeedback(
                total_duration_seconds=feedback_data.get("total_duration_seconds", 0.0),
                individual_files=individual_files,
                warnings=feedback_data.get("warnings", []),
                recommendations=feedback_data.get("recommendations", []),
                quality_metrics=feedback_data.get("quality_metrics", {}),
            )

        message = f"Voice '{name}' created successfully"
        if validation_feedback and validation_feedback.warnings:
            warning_count = len(validation_feedback.warnings)
            message += f" ({warning_count} warning{'s' if warning_count > 1 else ''} about audio quality/duration)"

        return VoiceCreateResponse(
            success=True,
            message=message,
            voice=voice_response,
            validation_feedback=validation_feedback,
        )
    except ValueError as e:
        # Used for validation errors from voice manager
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    finally:
        # Clean up temporary files
        for p in temp_clip_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        if temp_input_path is not None:
            try:
                if temp_input_path.exists():
                    temp_input_path.unlink()
            except Exception:
                pass


@router.post(
    "/profile/analyze-audio",
    response_model=VoiceProfileFromAudioResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_voice_profile_from_audio(
    audio_file: UploadFile = File(...),
    keywords: str = Form(None),
    ollama_url: str = Form(None),
    ollama_model: str = Form(None),
) -> VoiceProfileFromAudioResponse:
    """
    Analyze an audio file and derive a style-oriented VoiceProfile.

    This does NOT create a new voice; it returns a profile that can be applied to any voice.
    """
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file must have a filename")

    import tempfile
    from pathlib import Path

    suffix = Path(audio_file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)

    try:
        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty")
        temp_file.write(content)
        temp_file.close()

        keywords_list = None
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

        profile_dict, validation_dict, transcript = voice_profile_from_audio.analyze(
            audio_path=temp_path,
            keywords=keywords_list,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

        # Parse validation feedback
        individual_files = [
            IndividualFileAnalysis(**file_data) for file_data in validation_dict.get("individual_files", [])
        ]
        validation_feedback = AudioValidationFeedback(
            total_duration_seconds=validation_dict.get("total_duration_seconds", 0.0),
            individual_files=individual_files,
            warnings=validation_dict.get("warnings", []),
            recommendations=validation_dict.get("recommendations", []),
            quality_metrics=validation_dict.get("quality_metrics", {}),
        )

        profile = VoiceProfile(
            cadence=profile_dict.get("cadence"),
            tone=profile_dict.get("tone"),
            vocabulary_style=profile_dict.get("vocabulary_style"),
            sentence_structure=profile_dict.get("sentence_structure"),
            unique_phrases=profile_dict.get("unique_phrases", []),
            keywords=profile_dict.get("keywords", []),
            profile_text=profile_dict.get("profile_text"),
            created_at=None,
            updated_at=None,
        )

        return VoiceProfileFromAudioResponse(
            success=True,
            message="Voice profile generated from audio",
            profile=profile,
            transcript=transcript or None,
            validation_feedback=validation_feedback,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        # RuntimeError is used for Ollama unavailable / model missing, etc.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze audio: {str(e)}",
        ) from e
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.get(
    "",
    response_model=VoiceListResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def list_voices() -> VoiceListResponse:
    """
    List all available voices (default + custom).

    Returns:
        List of all voices with metadata
    """
    try:
        voices_data = voice_manager.list_all_voices()

        voices = []
        for voice_data in voices_data:
            # Parse created_at if it's a string
            created_at = voice_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    created_at = None

            image_url = None
            if voice_data.get("image_filename"):
                image_url = f"/api/v1/voices/{voice_data['id']}/image"
            quality_analysis = None
            if voice_data.get("quality_analysis"):
                qa = voice_data["quality_analysis"]
                quality_analysis = VoiceQualityAnalysis(
                    clone_quality=qa.get("clone_quality", "fair"),
                    issues=qa.get("issues", []),
                    recording_quality_score=qa.get("recording_quality_score", 0.5),
                    background_music_detected=qa.get("background_music_detected", False),
                    background_noise_detected=qa.get("background_noise_detected", False),
                )
            voices.append(
                VoiceResponse(
                    id=voice_data["id"],
                    name=voice_data["name"],
                    display_name=voice_data.get("display_name"),
                    language_code=voice_data.get("language_code"),
                    language_label=voice_data.get("language_label"),
                    gender=voice_data.get("gender"),
                    description=voice_data.get("description"),
                    type=voice_data.get("type", "default"),
                    created_at=created_at,
                    audio_files=voice_data.get("audio_files"),
                    image_url=image_url,
                    quality_analysis=quality_analysis,
                )
            )

        return VoiceListResponse(voices=voices, total=len(voices))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list voices: {str(e)}",
        ) from e


def _media_type_for_image(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


@router.get(
    "/{voice_id}/image",
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_voice_image(voice_id: str):
    """
    Return the avatar image for a custom voice.

    Returns 404 if the voice has no image or is not a custom voice.
    """
    image_path = voice_manager.get_voice_image_path(voice_id)
    if not image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No image for voice '{voice_id}'",
        )
    return FileResponse(
        path=str(image_path),
        media_type=_media_type_for_image(image_path),
    )


@router.put(
    "/{voice_id}/image",
    response_model=VoiceUpdateResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def update_voice_image(
    voice_id: str,
    image: UploadFile = File(...),
) -> VoiceUpdateResponse:
    """
    Set or replace the avatar image for a custom voice.
    """
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is required",
        )
    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image must be JPEG, PNG, or WebP. Got: {content_type}",
        )
    image_content = await image.read()
    if len(image_content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image must be at most {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB",
        )
    ext = Path(image.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must have extension .jpg, .png, or .webp",
        )
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_image.write(image_content)
    temp_image.close()
    temp_image_path = Path(temp_image.name)
    try:
        updated_voice = voice_manager.update_voice_image(voice_id, temp_image_path)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=msg,
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg,
        ) from e
    finally:
        if temp_image_path.exists():
            temp_image_path.unlink()

    created_at = updated_voice.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = None
    image_url = f"/api/v1/voices/{voice_id}/image" if updated_voice.get("image_filename") else None
    quality_analysis = None
    if updated_voice.get("quality_analysis"):
        qa = updated_voice["quality_analysis"]
        quality_analysis = VoiceQualityAnalysis(
            clone_quality=qa.get("clone_quality", "fair"),
            issues=qa.get("issues", []),
            recording_quality_score=qa.get("recording_quality_score", 0.5),
            background_music_detected=qa.get("background_music_detected", False),
            background_noise_detected=qa.get("background_noise_detected", False),
        )
    voice_response = VoiceResponse(
        id=updated_voice["id"],
        name=updated_voice["name"],
        display_name=updated_voice.get("display_name"),
        language_code=updated_voice.get("language_code"),
        language_label=updated_voice.get("language_label"),
        gender=updated_voice.get("gender"),
        description=updated_voice.get("description"),
        type=updated_voice.get("type", "custom"),
        created_at=created_at,
        audio_files=updated_voice.get("audio_files"),
        image_url=image_url,
        quality_analysis=quality_analysis,
    )
    return VoiceUpdateResponse(
        success=True,
        message=f"Voice image updated for '{voice_id}'",
        voice=voice_response,
    )


@router.delete(
    "/{voice_id}",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_voice(voice_id: str) -> JSONResponse:
    """
    Delete a custom voice.

    Args:
        voice_id: Voice identifier

    Returns:
        Success response
    """
    try:
        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Delete voice
        deleted = voice_manager.delete_custom_voice(voice_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": f"Voice '{voice_id}' deleted successfully",
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {str(e)}",
        ) from e


@router.put(
    "/{voice_id}",
    response_model=VoiceUpdateResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def update_voice(
    voice_id: str,
    request: VoiceUpdateRequest,
) -> VoiceUpdateResponse:
    """
    Update voice details (name and/or description).

    Args:
        voice_id: Voice identifier
        request: Update request with name and/or description

    Returns:
        Updated voice response
    """
    try:
        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Update voice
        updated_voice = voice_manager.update_voice(
            voice_id=voice_id,
            name=request.name,
            description=request.description,
            language_code=request.language_code,
            gender=request.gender,
        )

        # Parse created_at if it's a string
        created_at = updated_voice.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        image_url = None
        if updated_voice.get("image_filename"):
            image_url = f"/api/v1/voices/{updated_voice['id']}/image"
        quality_analysis = None
        if updated_voice.get("quality_analysis"):
            qa = updated_voice["quality_analysis"]
            quality_analysis = VoiceQualityAnalysis(
                clone_quality=qa.get("clone_quality", "fair"),
                issues=qa.get("issues", []),
                recording_quality_score=qa.get("recording_quality_score", 0.5),
                background_music_detected=qa.get("background_music_detected", False),
                background_noise_detected=qa.get("background_noise_detected", False),
            )
        voice_response = VoiceResponse(
            id=updated_voice["id"],
            name=updated_voice["name"],
            display_name=updated_voice.get("display_name"),
            language_code=updated_voice.get("language_code"),
            language_label=updated_voice.get("language_label"),
            gender=updated_voice.get("gender"),
            description=updated_voice.get("description"),
            type=updated_voice.get("type", "custom"),
            created_at=created_at,
            audio_files=updated_voice.get("audio_files"),
            image_url=image_url,
            quality_analysis=quality_analysis,
        )

        return VoiceUpdateResponse(
            success=True,
            message=f"Voice '{voice_id}' updated successfully",
            voice=voice_response,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update voice: {str(e)}",
        ) from e


@router.get(
    "/{voice_id}/profile",
    response_model=VoiceProfileResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_voice_profile(voice_id: str) -> VoiceProfileResponse:
    """
    Get voice profile.

    Args:
        voice_id: Voice identifier

    Returns:
        Voice profile response
    """
    try:
        from ..models.voice_storage import voice_storage

        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Get profile
        profile_data = voice_storage.get_voice_profile(voice_id)

        if not profile_data:
            return VoiceProfileResponse(
                success=True,
                message="No profile found for this voice",
                profile=None,
            )

        # Parse timestamps if present
        created_at = profile_data.get("created_at")
        updated_at = profile_data.get("updated_at")

        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                updated_at = None

        profile = VoiceProfile(
            cadence=profile_data.get("cadence"),
            tone=profile_data.get("tone"),
            vocabulary_style=profile_data.get("vocabulary_style"),
            sentence_structure=profile_data.get("sentence_structure"),
            unique_phrases=profile_data.get("unique_phrases", []),
            keywords=profile_data.get("keywords", []),
            profile_text=profile_data.get("profile_text"),
            created_at=created_at,
            updated_at=updated_at,
        )

        return VoiceProfileResponse(
            success=True,
            message="Profile retrieved successfully",
            profile=profile,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get voice profile: {str(e)}",
        ) from e


@router.put(
    "/{voice_id}/profile",
    response_model=VoiceProfileResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def apply_voice_profile(
    voice_id: str,
    request: VoiceProfileApplyRequest,
) -> VoiceProfileResponse:
    """
    Apply a provided VoiceProfile payload to a voice.

    This supports both custom voices and default voices.
    """
    try:
        # Ensure voice exists (default voices are valid here too)
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Voice '{voice_id}' not found")

        from ..models.voice_storage import voice_storage

        profile_payload = request.model_dump()
        voice_storage.update_voice_profile(voice_id, profile_payload)

        profile_data = voice_storage.get_voice_profile(voice_id)
        if not profile_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Profile applied but no profile data available",
            )

        created_at = profile_data.get("created_at")
        updated_at = profile_data.get("updated_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                updated_at = None

        profile = VoiceProfile(
            cadence=profile_data.get("cadence"),
            tone=profile_data.get("tone"),
            vocabulary_style=profile_data.get("vocabulary_style"),
            sentence_structure=profile_data.get("sentence_structure"),
            unique_phrases=profile_data.get("unique_phrases", []),
            keywords=profile_data.get("keywords", []),
            profile_text=profile_data.get("profile_text"),
            created_at=created_at,
            updated_at=updated_at,
        )

        return VoiceProfileResponse(success=True, message="Profile applied successfully", profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply profile: {str(e)}",
        ) from e


@router.post(
    "/{voice_id}/profile",
    response_model=VoiceProfileResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_or_update_voice_profile(
    voice_id: str,
    request: VoiceProfileRequest,
) -> VoiceProfileResponse:
    """
    Create or update voice profile with optional keywords.

    Args:
        voice_id: Voice identifier
        request: Profile request with optional keywords

    Returns:
        Voice profile response
    """
    try:
        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Get existing profile
        from ..models.voice_storage import voice_storage

        existing_profile = voice_storage.get_voice_profile(voice_id)

        # Enhance or create profile
        from ..services.voice_profiler import voice_profiler
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if existing_profile and request.keywords:
                # Enhance existing profile
                logger.info(f"Enhancing existing profile for voice {voice_id} with keywords: {request.keywords}")
                updated_voice = voice_manager.enhance_voice_profile(
                    voice_id=voice_id,
                    keywords=request.keywords or [],
                )
                message = "Profile enhanced successfully"
            else:
                # Create new profile
                logger.info(f"Creating new profile for voice {voice_id} with keywords: {request.keywords}")
                profile = voice_profiler.profile_voice_from_audio(
                    voice_name=voice_data.get("name", voice_id),
                    voice_description=voice_data.get("description"),
                    keywords=request.keywords,
                    ollama_url=request.ollama_url,
                    ollama_model=request.ollama_model,
                )

                if profile:
                    # Ensure keywords are saved
                    if request.keywords:
                        profile["keywords"] = request.keywords
                    voice_storage.update_voice_profile(voice_id, profile)
                    logger.info(f"Profile created and saved for voice {voice_id}")
                    message = "Profile created successfully"
                else:
                    logger.warning(f"Profile creation returned empty profile for voice {voice_id}")
                    message = "Profile creation attempted but returned empty result"
        except RuntimeError as e:
            # RuntimeError means Ollama is not available or model missing
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

        # Get updated profile
        profile_data = voice_storage.get_voice_profile(voice_id)

        if not profile_data:
            return VoiceProfileResponse(
                success=True,
                message="Profile operation completed but no profile data available",
                profile=None,
            )

        # Parse timestamps
        created_at = profile_data.get("created_at")
        updated_at = profile_data.get("updated_at")

        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                updated_at = None

        profile = VoiceProfile(
            cadence=profile_data.get("cadence"),
            tone=profile_data.get("tone"),
            vocabulary_style=profile_data.get("vocabulary_style"),
            sentence_structure=profile_data.get("sentence_structure"),
            unique_phrases=profile_data.get("unique_phrases", []),
            keywords=profile_data.get("keywords", []),
            profile_text=profile_data.get("profile_text"),
            created_at=created_at,
            updated_at=updated_at,
        )

        return VoiceProfileResponse(
            success=True,
            message=message,
            profile=profile,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create/update voice profile: {str(e)}",
        ) from e


@router.put(
    "/{voice_id}/profile/keywords",
    response_model=VoiceProfileResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def update_voice_profile_keywords(
    voice_id: str,
    request: VoiceProfileRequest,
) -> VoiceProfileResponse:
    """
    Update voice profile keywords and re-profile.

    Args:
        voice_id: Voice identifier
        request: Profile request with keywords

    Returns:
        Voice profile response
    """
    try:
        if not request.keywords:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Keywords are required",
            )

        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Enhance profile with keywords
        try:
            updated_voice = voice_manager.enhance_voice_profile(
                voice_id=voice_id,
                keywords=request.keywords,
                ollama_url=request.ollama_url,
                ollama_model=request.ollama_model,
            )
        except RuntimeError as e:
            # RuntimeError means Ollama is not available or model missing
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

        # Get updated profile
        from ..models.voice_storage import voice_storage

        profile_data = voice_storage.get_voice_profile(voice_id)

        if not profile_data:
            return VoiceProfileResponse(
                success=True,
                message="Profile updated but no profile data available",
                profile=None,
            )

        # Parse timestamps
        created_at = profile_data.get("created_at")
        updated_at = profile_data.get("updated_at")

        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                updated_at = None

        profile = VoiceProfile(
            cadence=profile_data.get("cadence"),
            tone=profile_data.get("tone"),
            vocabulary_style=profile_data.get("vocabulary_style"),
            sentence_structure=profile_data.get("sentence_structure"),
            unique_phrases=profile_data.get("unique_phrases", []),
            keywords=profile_data.get("keywords", []),
            profile_text=profile_data.get("profile_text"),
            created_at=created_at,
            updated_at=updated_at,
        )

        return VoiceProfileResponse(
            success=True,
            message="Profile keywords updated successfully",
            profile=profile,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile keywords: {str(e)}",
        ) from e


@router.post(
    "/{voice_id}/profile/generate",
    response_model=VoiceProfileResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate_voice_profile(
    voice_id: str,
    request: VoiceProfileRequest,
) -> VoiceProfileResponse:
    """
    Manually trigger voice profile generation.

    Args:
        voice_id: Voice identifier
        request: Profile request with optional keywords

    Returns:
        Voice profile response
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if voice exists
        voice_data = voice_manager.get_voice(voice_id)
        if not voice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        # Get existing profile
        from ..models.voice_storage import voice_storage
        from ..services.voice_profiler import voice_profiler

        existing_profile = voice_storage.get_voice_profile(voice_id)

        logger.info(f"Manually generating profile for voice {voice_id} with keywords: {request.keywords}")

        # Generate or enhance profile
        try:
            if existing_profile and request.keywords:
                # Enhance existing profile
                profile = voice_profiler.enhance_profile_with_keywords(
                    voice_name=voice_data.get("name", voice_id),
                    existing_profile=existing_profile,
                    keywords=request.keywords or [],
                    ollama_url=request.ollama_url,
                    ollama_model=request.ollama_model,
                )
                message = "Profile enhanced successfully"
            else:
                # Create new profile
                profile = voice_profiler.profile_voice_from_audio(
                    voice_name=voice_data.get("name", voice_id),
                    voice_description=voice_data.get("description"),
                    keywords=request.keywords,
                    ollama_url=request.ollama_url,
                    ollama_model=request.ollama_model,
                )
                message = "Profile generated successfully"
        except RuntimeError as e:
            # RuntimeError means Ollama is not available or model missing
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Profile generation returned empty result",
            )

        # Ensure keywords are saved
        if request.keywords:
            profile["keywords"] = request.keywords

        # Save profile
        voice_storage.update_voice_profile(voice_id, profile)
        logger.info(f"Profile saved for voice {voice_id}")

        # Get updated profile
        profile_data = voice_storage.get_voice_profile(voice_id)

        if not profile_data:
            return VoiceProfileResponse(
                success=True,
                message=message,
                profile=None,
            )

        # Parse timestamps
        created_at = profile_data.get("created_at")
        updated_at = profile_data.get("updated_at")

        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = None

        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                updated_at = None

        profile_response = VoiceProfile(
            cadence=profile_data.get("cadence"),
            tone=profile_data.get("tone"),
            vocabulary_style=profile_data.get("vocabulary_style"),
            sentence_structure=profile_data.get("sentence_structure"),
            unique_phrases=profile_data.get("unique_phrases", []),
            keywords=profile_data.get("keywords", []),
            profile_text=profile_data.get("profile_text"),
            created_at=created_at,
            updated_at=updated_at,
        )

        return VoiceProfileResponse(
            success=True,
            message=message,
            profile=profile_response,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to generate profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate profile: {str(e)}",
        ) from e
