"""Cost projections for the fixed-size RVL vision classification experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.utils.config import REPO_ROOT

DEFAULT_PRICING_PATH = REPO_ROOT / "evaluation" / "pricing.yaml"

# Observed averages from the Kimi K3 160-image Braintrust run (1024², max_tokens=500).
KIMI_K3_OBSERVED = {
    "prompt_tokens_avg": 1768.89,
    "completion_tokens_avg": 171.51,
    "prompt_cached_tokens_avg": 872.80,
    "completion_reasoning_tokens_avg": 156.02,
}

DEFAULT_SCALE_COUNTS: tuple[int, ...] = (160, 800, 25_000, 320_000)


@dataclass(frozen=True)
class ModelRates:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def cost_usd(
        self,
        *,
        prompt_tokens: float,
        completion_tokens: float,
        prompt_cached_tokens: float = 0.0,
    ) -> float:
        cached_rate = (
            self.cached_input_per_million
            if self.cached_input_per_million is not None
            else self.input_per_million
        )
        billable_prompt = max(0.0, prompt_tokens - prompt_cached_tokens)
        return (
            (billable_prompt / 1_000_000.0) * self.input_per_million
            + (prompt_cached_tokens / 1_000_000.0) * cached_rate
            + (completion_tokens / 1_000_000.0) * self.output_per_million
        )


def load_rates(path: Path | None = None) -> dict[str, ModelRates]:
    """Load per-model rates from ``evaluation/pricing.yaml`` (braintrust section)."""
    pricing_path = Path(path or DEFAULT_PRICING_PATH)
    raw = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
    bt = raw.get("braintrust_vision_models") or {}
    rates: dict[str, ModelRates] = {}
    for model_id, vals in bt.items():
        if not isinstance(vals, dict):
            continue
        rates[model_id] = ModelRates(
            input_per_million=float(vals["input_per_million"]),
            output_per_million=float(vals["output_per_million"]),
            cached_input_per_million=(
                float(vals["cached_input_per_million"])
                if vals.get("cached_input_per_million") is not None
                else None
            ),
        )
    # Fallbacks from frontier_models if braintrust block missing entries.
    frontier = raw.get("frontier_models") or {}
    aliases = {
        "anthropic/claude-sonnet-4.5": "anthropic",
        "anthropic/claude-opus-4.5": "anthropic",
        "openai/gpt-4o": "openai",
    }
    for model_id, key in aliases.items():
        if model_id in rates:
            continue
        block = frontier.get(key)
        if isinstance(block, dict):
            rates[model_id] = ModelRates(
                input_per_million=float(block["input_per_million"]),
                output_per_million=float(block["output_per_million"]),
            )
    return rates


def project_costs(
    *,
    model_id: str,
    n_images: int,
    prompt_tokens_avg: float = KIMI_K3_OBSERVED["prompt_tokens_avg"],
    completion_tokens_avg: float = KIMI_K3_OBSERVED["completion_tokens_avg"],
    prompt_cached_tokens_avg: float = KIMI_K3_OBSERVED["prompt_cached_tokens_avg"],
    rates: Mapping[str, ModelRates] | None = None,
) -> dict[str, Any]:
    """Project total tokens + USD for ``n_images`` at observed per-image averages."""
    table = rates or load_rates()
    if model_id not in table:
        raise KeyError(
            f"No pricing for {model_id!r}. Add it under braintrust_vision_models "
            f"in {DEFAULT_PRICING_PATH}"
        )
    model_rates = table[model_id]
    prompt_tokens = prompt_tokens_avg * n_images
    completion_tokens = completion_tokens_avg * n_images
    cached = prompt_cached_tokens_avg * n_images
    cost = model_rates.cost_usd(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cached_tokens=cached,
    )
    return {
        "model_id": model_id,
        "n_images": n_images,
        "prompt_tokens": round(prompt_tokens),
        "completion_tokens": round(completion_tokens),
        "prompt_cached_tokens": round(cached),
        "total_tokens": round(prompt_tokens + completion_tokens),
        "estimated_cost_usd": round(cost, 4),
        "rates": {
            "input_per_million": model_rates.input_per_million,
            "output_per_million": model_rates.output_per_million,
            "cached_input_per_million": model_rates.cached_input_per_million,
        },
    }


def project_scale_table(
    model_ids: Sequence[str] | None = None,
    counts: Sequence[int] = DEFAULT_SCALE_COUNTS,
    **token_avgs: float,
) -> list[dict[str, Any]]:
    rates = load_rates()
    ids = list(model_ids) if model_ids is not None else list(rates.keys())
    rows: list[dict[str, Any]] = []
    for model_id in ids:
        for n in counts:
            rows.append(
                project_costs(
                    model_id=model_id,
                    n_images=int(n),
                    rates=rates,
                    **token_avgs,
                )
            )
    return rows
