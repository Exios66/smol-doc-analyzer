"""Chained document-analysis pipeline (classify → extract → vision → summarize)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.orchestrator import DocumentAnalysisOrchestrator
    from src.pipeline.types import AnalysisContext, AnalysisDocument, StageResult

__all__ = [
    "AnalysisContext",
    "AnalysisDocument",
    "DocumentAnalysisOrchestrator",
    "StageResult",
    "analyze_document",
]


def __getattr__(name: str) -> Any:
    if name in {"DocumentAnalysisOrchestrator", "analyze_document"}:
        from src.pipeline import orchestrator as _orch

        return getattr(_orch, name)
    if name in {"AnalysisContext", "AnalysisDocument", "StageResult"}:
        from src.pipeline import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
