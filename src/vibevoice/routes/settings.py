"""
Runtime settings endpoints.
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, status

from ..models.schemas import (
    AceStepModelCatalogResponse,
    AceStepRuntimeSettingsResponse,
    AceStepRuntimeSettingsUpdateRequest,
    ErrorResponse,
    OpenAIListModelsRequest,
    OpenAIListModelsResponse,
)
from ..services.acestep_settings import acestep_settings_service
from ..services.music_process import music_process_manager
from ..services.openai_models_filter import openai_model_id_for_chat_completions

logger = logging.getLogger(__name__)

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def _compute_source() -> str:
    return "settings_file" if acestep_settings_service.storage_file.exists() else "env_defaults"


@router.get(
    "/acestep",
    response_model=AceStepRuntimeSettingsResponse,
    responses={500: {"model": ErrorResponse}},
)
async def get_acestep_runtime_settings() -> AceStepRuntimeSettingsResponse:
    values = acestep_settings_service.get_current()
    return AceStepRuntimeSettingsResponse(
        acestep_config_path=values["acestep_config_path"],
        acestep_lm_model_path=values["acestep_lm_model_path"],
        source=_compute_source(),
        restart_required=music_process_manager.is_running(),
        settings_file=str(acestep_settings_service.storage_file),
    )


@router.put(
    "/acestep",
    response_model=AceStepRuntimeSettingsResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_acestep_runtime_settings(
    request: AceStepRuntimeSettingsUpdateRequest,
) -> AceStepRuntimeSettingsResponse:
    previous = acestep_settings_service.get_current()
    try:
        updated = acestep_settings_service.update(
            acestep_config_path=request.acestep_config_path,
            acestep_lm_model_path=request.acestep_lm_model_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    changed = updated != previous
    was_running = music_process_manager.is_running()
    if changed and was_running:
        music_process_manager.stop()

    return AceStepRuntimeSettingsResponse(
        acestep_config_path=updated["acestep_config_path"],
        acestep_lm_model_path=updated["acestep_lm_model_path"],
        source=_compute_source(),
        restart_required=changed and was_running,
        settings_file=str(acestep_settings_service.storage_file),
    )


@router.get(
    "/acestep/models",
    response_model=AceStepModelCatalogResponse,
    responses={500: {"model": ErrorResponse}},
)
async def get_acestep_model_catalog() -> AceStepModelCatalogResponse:
    catalog = acestep_settings_service.get_catalog()
    return AceStepModelCatalogResponse(**catalog)


@router.post(
    "/openai/models",
    response_model=OpenAIListModelsResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def list_openai_chat_models(body: OpenAIListModelsRequest) -> OpenAIListModelsResponse:
    """
    List model IDs from OpenAI ``GET /v1/models`` for the given API key, filtered for chat-style models.
    """
    key = (body.openai_api_key or "").strip()
    if not key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="openai_api_key is required")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                OPENAI_MODELS_URL,
                headers={"Authorization": f"Bearer {key}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("OpenAI models list request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach OpenAI to list models",
        ) from exc

    if resp.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OpenAI rejected the API key (unauthorized)",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI models API returned HTTP {resp.status_code}",
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid JSON from OpenAI models API",
        ) from exc

    raw_ids: list[str] = []
    for entry in payload.get("data") or []:
        if isinstance(entry, dict):
            mid = entry.get("id")
            if isinstance(mid, str) and mid.strip():
                raw_ids.append(mid.strip())

    filtered = sorted({m for m in raw_ids if openai_model_id_for_chat_completions(m)})
    return OpenAIListModelsResponse(models=filtered)
