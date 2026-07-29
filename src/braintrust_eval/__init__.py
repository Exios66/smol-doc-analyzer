"""Braintrust + OpenRouter RVL-CDIP vision classification experiments.

Replicates the 10-per-class / 1024×1024 Kimi K3 evaluation workflow and adds:

- Capture of OpenRouter ``reasoning`` / ``reasoning_content`` for prompt analysis
- DeepSeek R1 (text) prompt-improvement pass over misclassifications
- Cost projections for flagship models (Sonnet / Opus) on the same dataset

CLI: ``python -m src.braintrust_eval --help``
"""

from __future__ import annotations

from src.braintrust_eval.classifier import (
    CLASSIFICATION_PROMPT,
    UNDERSCORE_LABELS,
    VALID_CLASSES,
    ClassificationResult,
    classify_image,
    default_reasoning_config,
    normalize_capstone_label,
)
from src.braintrust_eval.dataset import (
    DEFAULT_N_PER_CLASS,
    DEFAULT_SEED,
    DEFAULT_SIZE,
    build_fixed_size_sampled,
)

__all__ = [
    "CLASSIFICATION_PROMPT",
    "ClassificationResult",
    "DEFAULT_N_PER_CLASS",
    "DEFAULT_SEED",
    "DEFAULT_SIZE",
    "UNDERSCORE_LABELS",
    "VALID_CLASSES",
    "build_fixed_size_sampled",
    "classify_image",
    "default_reasoning_config",
    "normalize_capstone_label",
]
