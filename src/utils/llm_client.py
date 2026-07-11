"""
Thin wrapper around OpenRouter (OpenAI-compatible chat completions API)
used only for synthetic data generation (Stage A / Stage B). Kept separate
from any inference code for the actual (locally-hosted, small) pipeline
models -- this client is a data-generation tool, not part of the deployed
product.

OpenRouter lets you point --model at any provider's model (Anthropic,
OpenAI, open-weight models, etc.) through one API, which is convenient if
you want to A/B different generation-model choices without swapping SDKs.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import Config

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GenerationClient:
    def __init__(self, cfg: Config):
        if not cfg.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=cfg.openrouter_api_key,
            default_headers={
                # OpenRouter uses these for its public rankings/analytics --
                # optional but recommended, and harmless if left generic.
                "HTTP-Referer": cfg.openrouter_app_url or "https://github.com/",
                "X-Title": cfg.openrouter_app_name or "insurance-doc-ai",
            },
        )
        self._model = cfg.generation_model
        self._max_retries = cfg.max_retries

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        return self._generate_with_retry(system_prompt, user_prompt, max_tokens)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _generate_with_retry(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
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
