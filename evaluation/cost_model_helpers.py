"""
Cost-model helpers for the frontier vs. local eval comparison.

Used by ``eval_harness.py`` (local amortized GPU cost) and by spreadsheet /
notebook cost narratives that consume ``eval_results.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRICING_CONFIG_PATH = Path(__file__).parent / "pricing.yaml"


def load_pricing_raw(path: Path | None = None) -> dict[str, Any]:
    with open(path or PRICING_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def frontier_cost_usd(
    input_per_million: float,
    output_per_million: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    return (input_tokens / 1_000_000) * input_per_million + (
        output_tokens / 1_000_000
    ) * output_per_million


def local_cost_per_call(gpu_hourly_rate_usd: float, latency_seconds: float) -> float:
    """Amortized $/call from wall-clock GPU occupancy."""
    if latency_seconds < 0:
        raise ValueError("latency_seconds must be non-negative")
    return gpu_hourly_rate_usd * (latency_seconds / 3600.0)


def local_cost_per_doc_at_throughput(
    gpu_hourly_rate_usd: float, docs_per_hour: float
) -> float:
    """Steady-state $/doc assuming sustained throughput (not per-call latency)."""
    if docs_per_hour <= 0:
        raise ValueError("docs_per_hour must be positive")
    return gpu_hourly_rate_usd / docs_per_hour
