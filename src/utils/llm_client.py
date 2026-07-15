"""
Thin wrapper around OpenRouter (OpenAI-compatible chat completions API).

Used for:
  - synthetic data generation (Stage A / Stage B) via ``GenerationClient``
  - frontier-model eval baselines via ``OpenRouterClient``

Kept separate from inference code for the locally-hosted pipeline models —
these clients are tools for generation / benchmarking, not the deployed product.

OpenRouter lets you point --model at any provider's model (Anthropic,
OpenAI, open-weight models, etc.) through one API, which is convenient if
you want to A/B different generation-model choices without swapping SDKs.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import Config

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_openrouter_client(cfg: Config) -> OpenAI:
    if not cfg.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=cfg.openrouter_api_key,
        default_headers={
            # OpenRouter uses these for its public rankings/analytics —
            # optional but recommended, and harmless if left generic.
            "HTTP-Referer": cfg.openrouter_app_url or "https://github.com/",
            "X-Title": cfg.openrouter_app_name or "insurance-doc-ai",
        },
    )


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


class GenerationClient:
    def __init__(self, cfg: Config):
        self._client = _build_openrouter_client(cfg)
        self._model = cfg.generation_model
        self._max_retries = max(1, int(cfg.max_retries))

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        return self._generate_with_retry(system_prompt, user_prompt, max_tokens)

    def _generate_with_retry(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _once() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not response.choices:
                raise ValueError("Empty generation result (no choices) -- retrying.")

            choice = response.choices[0]

            # OpenRouter surfaces upstream provider errors inside a normal 200
            # response sometimes (e.g. provider overloaded) -- guard for that
            # instead of assuming message.content is always populated.
            finish_reason = getattr(choice, "finish_reason", None)
            content = (choice.message.content or "").strip()

            if not content:
                raise ValueError(
                    f"Empty generation result (finish_reason={finish_reason}) -- retrying."
                )

            return content

        return _once()


class OpenRouterClient:
    """
    OpenRouter chat-completions client used by the frontier vs. local eval harness.

    ``complete`` returns text plus token usage so the harness can compute $/call
    from ``evaluation/pricing.yaml``.

    The underlying HTTP client is constructed lazily so dry-run / import paths
    do not require ``OPENROUTER_API_KEY``.
    """

    def __init__(self, model: str, cfg: Config | None = None, **kwargs: Any):
        self.model = model
        self._cfg = cfg or Config.load()
        self._client: OpenAI | None = None
        self._max_retries = max(1, int(self._cfg.max_retries))
        self._system_prompt = kwargs.get(
            "system_prompt",
            "You are an insurance document analysis assistant. Follow the task instructions exactly.",
        )

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = _build_openrouter_client(self._cfg)
        return self._client

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        max_tokens = int(kwargs.get("max_tokens", 1024))
        system_prompt = kwargs.get("system_prompt", self._system_prompt)
        return self._complete_with_retry(prompt, max_tokens, system_prompt)

    def _complete_with_retry(
        self, prompt: str, max_tokens: int, system_prompt: str
    ) -> dict[str, Any]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _once() -> dict[str, Any]:
            response = self._ensure_client().chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            if not response.choices:
                raise ValueError("Empty completion result (no choices) -- retrying.")

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            content = (choice.message.content or "").strip()

            if not content:
                raise ValueError(
                    f"Empty completion result (finish_reason={finish_reason}) -- retrying."
                )

            return {
                "text": content,
                "usage": _extract_usage(response),
                "model": self.model,
                "finish_reason": finish_reason,
            }

        return _once()
