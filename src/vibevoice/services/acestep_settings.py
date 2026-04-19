"""
Runtime settings and validation for ACE-Step model selection.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..config import config

# Canonical model IDs aligned with ACE-Step 1.5 docs.
SUPPORTED_DIT_MODELS: tuple[str, ...] = (
    "ACE-Step/acestep-v15-base",
    "ACE-Step/acestep-v15-sft",
    "ACE-Step/acestep-v15-turbo",
    "ACE-Step/acestep-v15-xl-base",
    "ACE-Step/acestep-v15-xl-sft",
    "ACE-Step/acestep-v15-xl-turbo",
)

# Backward compatibility: accept bare model ids (e.g. "acestep-v15-xl-sft")
# and normalize them to canonical Hugging Face ids.
DIT_MODEL_ALIASES: dict[str, str] = {
    model_id.split("/", 1)[1]: model_id for model_id in SUPPORTED_DIT_MODELS
}

SUPPORTED_LM_MODELS: tuple[str, ...] = (
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
)


class AceStepSettingsService:
    """Reads/writes app-level ACE-Step runtime settings."""

    @staticmethod
    def _normalize_dit_model_id(model_id: str) -> str:
        normalized = model_id.strip()
        return DIT_MODEL_ALIASES.get(normalized, normalized)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._storage_file = config.OUTPUT_DIR / "config" / "acestep_runtime_settings.json"

    def _default_values(self) -> dict[str, str]:
        return {
            "acestep_config_path": config.ACESTEP_CONFIG_PATH,
            "acestep_lm_model_path": config.ACESTEP_LM_MODEL_PATH,
        }

    def _validate(self, *, acestep_config_path: str, acestep_lm_model_path: str) -> None:
        canonical_dit = self._normalize_dit_model_id(acestep_config_path)
        if canonical_dit not in SUPPORTED_DIT_MODELS:
            raise ValueError(
                f"Unsupported ACE-Step DiT model: {acestep_config_path}. "
                f"Supported models: {', '.join(SUPPORTED_DIT_MODELS)}"
            )
        if acestep_lm_model_path not in SUPPORTED_LM_MODELS:
            raise ValueError(
                f"Unsupported ACE-Step LM model: {acestep_lm_model_path}. "
                f"Supported models: {', '.join(SUPPORTED_LM_MODELS)}"
            )

    def _read_raw(self) -> dict[str, Any]:
        if not self._storage_file.exists():
            return {}
        try:
            payload = json.loads(self._storage_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def get_current(self) -> dict[str, str]:
        defaults = self._default_values()
        with self._lock:
            payload = self._read_raw()
            acestep_config_path = self._normalize_dit_model_id(
                str(payload.get("acestep_config_path") or defaults["acestep_config_path"]).strip()
            )
            acestep_lm_model_path = str(
                payload.get("acestep_lm_model_path") or defaults["acestep_lm_model_path"]
            ).strip()

        # If persisted values drift out of supported list (e.g., old file), fall back to defaults.
        if acestep_config_path not in SUPPORTED_DIT_MODELS:
            acestep_config_path = defaults["acestep_config_path"]
        if acestep_lm_model_path not in SUPPORTED_LM_MODELS:
            acestep_lm_model_path = defaults["acestep_lm_model_path"]

        return {
            "acestep_config_path": acestep_config_path,
            "acestep_lm_model_path": acestep_lm_model_path,
        }

    def update(self, *, acestep_config_path: str, acestep_lm_model_path: str) -> dict[str, str]:
        acestep_config_path = self._normalize_dit_model_id(acestep_config_path.strip())
        acestep_lm_model_path = acestep_lm_model_path.strip()
        self._validate(
            acestep_config_path=acestep_config_path,
            acestep_lm_model_path=acestep_lm_model_path,
        )

        with self._lock:
            self._storage_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "acestep_config_path": acestep_config_path,
                "acestep_lm_model_path": acestep_lm_model_path,
            }
            self._storage_file.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return payload

    def get_catalog(self) -> dict[str, Any]:
        current = self.get_current()
        return {
            "dit_models": list(SUPPORTED_DIT_MODELS),
            "lm_models": list(SUPPORTED_LM_MODELS),
            "defaults": self._default_values(),
            "current": current,
        }

    @property
    def storage_file(self) -> Path:
        return self._storage_file


acestep_settings_service = AceStepSettingsService()
