"""
Document Image Classification and Information Extraction (DICIE) pipeline.

Implements the Fig. 1 processing chain from Raj, Dickinson & Fung,
"Document Classification and Information Extraction framework for
Insurance Applications":

  Input (PDF / page images)
    → Stage 1 Document Processing
    → Stage 2 Document Classification (+ page aggregation)
    → Stage 3 Information Extraction
    → Output (aggregated prediction / downstream response)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.docie.pipeline import DociePipeline, process_document
    from src.docie.types import DociePrediction, ProcessedDocument

__all__ = [
    "DociePipeline",
    "DociePrediction",
    "ProcessedDocument",
    "process_document",
]


def __getattr__(name: str) -> Any:
    if name in {"DociePipeline", "process_document"}:
        from src.docie import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in {"DociePrediction", "ProcessedDocument"}:
        from src.docie import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
