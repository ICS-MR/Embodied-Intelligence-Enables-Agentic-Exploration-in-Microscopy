"""Minimal public API for object localization."""

from .pipeline import (
    LocalizationConfig,
    compare_localizations,
    run_model_localization,
    run_vlm_localization,
)

__all__ = [
    "LocalizationConfig",
    "run_model_localization",
    "run_vlm_localization",
    "compare_localizations",
]
