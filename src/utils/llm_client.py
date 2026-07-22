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

When paid credits are exhausted (HTTP 402 / "requires more credits"), clients
automatically route subsequent calls to free OpenRouter models
(``openrouter/free`` and optional ``*:free`` fallbacks).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.config import Config

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free Models Router + a couple of commonly-available :free variants.
# Availability rotates; the router is the preferred first hop.
DEFAULT_FREE_FALLBACK_MODELS: tuple[str, ...] = (
    "openrouter/free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "openai/gpt-oss-20b:free",
)

# Process-wide sticky switch: once credits are known to be unavailable, skip
# paid models for the rest of the process (avoids N×402 storms during Stage A).
_CREDITS_UNAVAILABLE = False


def reset_credit_fallback_state() -> None:
    """Test helper — clear the process-wide free-routing sticky flag."""
    global _CREDITS_UNAVAILABLE
    _CREDITS_UNAVAILABLE = False


def _is_free_model(model: str) -> bool:
    slug = (model or "").strip().lower()
    return slug == "openrouter/free" or slug.endswith(":free")


def is_free_model(model: str) -> bool:
    """Public helper — True when ``model`` is a $0 OpenRouter free route."""
    return _is_free_model(model)


def parse_free_fallback_models(raw: str | None = None) -> tuple[str, ...]:
    """Parse comma-separated free fallback slugs from env or an explicit string."""
    text = (raw if raw is not None else os.getenv("OPENROUTER_FREE_FALLBACK_MODELS", "")).strip()
    if not text:
        return DEFAULT_FREE_FALLBACK_MODELS
    models = tuple(m.strip() for m in text.split(",") if m.strip())
    return models or DEFAULT_FREE_FALLBACK_MODELS


def prefer_free_models() -> bool:
    """True when OPENROUTER_PREFER_FREE / GENERATION_PREFER_FREE is set."""
    for key in ("OPENROUTER_PREFER_FREE", "GENERATION_PREFER_FREE"):
        val = os.getenv(key, "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
    return False


def is_credit_unavailable_error(exc: BaseException) -> bool:
    """Detect OpenRouter / provider errors that mean paid credits are exhausted."""
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True

    # openai.APIError and friends often nest the body on `.body` / `.message`
    parts: list[str] = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    message = getattr(exc, "message", None)
    if message is not None:
        parts.append(str(message))

    haystack = " ".join(parts).lower()
    needles = (
        "payment required",
        "requires more credits",
        "can only afford",
        "insufficient credits",
        "insufficient_quota",
        "credit balance",
        "out of credits",
        "upgrade to a paid account",
    )
    return any(n in haystack for n in needles)


def _is_retryable_transient(exc: BaseException) -> bool:
    """Retry rate limits / 5xx / empty responses — not credit exhaustion."""
    if is_credit_unavailable_error(exc):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    if isinstance(exc, ValueError):
        # Empty-choice guards inside _once raise ValueError and should retry.
        return True
    return False


def _mark_credits_unavailable(reason: BaseException | str) -> None:
    global _CREDITS_UNAVAILABLE
    if not _CREDITS_UNAVAILABLE:
        logger.warning(
            "OpenRouter paid credits unavailable (%s); routing to free models",
            reason if isinstance(reason, str) else type(reason).__name__,
        )
    _CREDITS_UNAVAILABLE = True


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
            "X-Title": cfg.openrouter_app_name or "smol-doc-analyzer",
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


def _chat_once(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if not response.choices:
        raise ValueError("Empty generation result (no choices) -- retrying.")

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    content = (choice.message.content or "").strip()
    if not content:
        raise ValueError(
            f"Empty generation result (finish_reason={finish_reason}) -- retrying."
        )
    return content, response


class GenerationClient:
    def __init__(self, cfg: Config):
        self._client = _build_openrouter_client(cfg)
        self._model = cfg.generation_model
        self._max_retries = max(1, int(cfg.max_retries))
        self._free_fallbacks = tuple(cfg.openrouter_free_fallback_models)
        self._prefer_free = bool(cfg.openrouter_prefer_free) or prefer_free_models()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        return self._generate_with_routing(system_prompt, user_prompt, max_tokens)

    def _model_candidates(self) -> list[str]:
        global _CREDITS_UNAVAILABLE
        if self._prefer_free or _CREDITS_UNAVAILABLE or _is_free_model(self._model):
            ordered = list(self._free_fallbacks)
            if _is_free_model(self._model) and self._model not in ordered:
                ordered.insert(0, self._model)
            # De-dupe while preserving order
            seen: set[str] = set()
            out: list[str] = []
            for m in ordered:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
            return out or list(DEFAULT_FREE_FALLBACK_MODELS)

        return [self._model, *self._free_fallbacks]

    def _generate_with_routing(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> str:
        errors: list[str] = []
        for model in self._model_candidates():
            try:
                return self._generate_with_retry(
                    model, system_prompt, user_prompt, max_tokens
                )
            except Exception as exc:
                if is_credit_unavailable_error(exc):
                    _mark_credits_unavailable(exc)
                    errors.append(f"{model}: credits unavailable")
                    continue
                # Non-credit failure on a free candidate: try next free model.
                if _is_free_model(model):
                    logger.warning(
                        "Free model %s failed (%s); trying next free fallback",
                        model,
                        exc,
                    )
                    errors.append(f"{model}: {exc}")
                    continue
                raise

        raise RuntimeError(
            "OpenRouter generation failed for all model candidates "
            f"({', '.join(errors) or 'no candidates'})"
        )

    def _generate_with_retry(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(_is_retryable_transient),
            reraise=True,
        )
        def _once() -> str:
            content, _response = _chat_once(
                self._client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            if model != self._model:
                logger.info("OpenRouter used free/fallback model %s", model)
            return content

        return _once()


class OpenRouterClient:
    """
    OpenRouter chat-completions client used by the frontier vs. local eval harness.

    ``complete`` returns text plus token usage so the harness can compute $/call
    from ``evaluation/pricing.yaml``.

    The underlying HTTP client is constructed lazily so dry-run / import paths
    do not require ``OPENROUTER_API_KEY``.

    On credit exhaustion, falls back to free OpenRouter models (same policy as
    ``GenerationClient``).
    """

    def __init__(self, model: str, cfg: Config | None = None, **kwargs: Any):
        self.model = model
        self._cfg = cfg or Config.load()
        self._client: OpenAI | None = None
        self._max_retries = max(1, int(self._cfg.max_retries))
        self._free_fallbacks: Sequence[str] = tuple(
            self._cfg.openrouter_free_fallback_models
        )
        self._prefer_free = bool(self._cfg.openrouter_prefer_free) or prefer_free_models()
        self._system_prompt = kwargs.get(
            "system_prompt",
            "You are an insurance document analysis assistant. Follow the task instructions exactly.",
        )

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = _build_openrouter_client(self._cfg)
        return self._client

    def _model_candidates(self) -> list[str]:
        global _CREDITS_UNAVAILABLE
        if self._prefer_free or _CREDITS_UNAVAILABLE or _is_free_model(self.model):
            ordered = list(self._free_fallbacks)
            if _is_free_model(self.model) and self.model not in ordered:
                ordered.insert(0, self.model)
            seen: set[str] = set()
            out: list[str] = []
            for m in ordered:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
            return out or list(DEFAULT_FREE_FALLBACK_MODELS)
        return [self.model, *self._free_fallbacks]

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        max_tokens = int(kwargs.get("max_tokens", 1024))
        system_prompt = kwargs.get("system_prompt", self._system_prompt)
        return self._complete_with_routing(prompt, max_tokens, system_prompt)

    def _complete_with_routing(
        self, prompt: str, max_tokens: int, system_prompt: str
    ) -> dict[str, Any]:
        errors: list[str] = []
        for model in self._model_candidates():
            try:
                return self._complete_with_retry(model, prompt, max_tokens, system_prompt)
            except Exception as exc:
                if is_credit_unavailable_error(exc):
                    _mark_credits_unavailable(exc)
                    errors.append(f"{model}: credits unavailable")
                    continue
                if _is_free_model(model):
                    logger.warning(
                        "Free model %s failed (%s); trying next free fallback",
                        model,
                        exc,
                    )
                    errors.append(f"{model}: {exc}")
                    continue
                raise

        raise RuntimeError(
            "OpenRouter completion failed for all model candidates "
            f"({', '.join(errors) or 'no candidates'})"
        )

    def _complete_with_retry(
        self, model: str, prompt: str, max_tokens: int, system_prompt: str
    ) -> dict[str, Any]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(_is_retryable_transient),
            reraise=True,
        )
        def _once() -> dict[str, Any]:
            content, response = _chat_once(
                self._ensure_client(),
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
            )
            used_model = getattr(response, "model", None) or model
            if model != self.model:
                logger.info("OpenRouter used free/fallback model %s", used_model)
            return {
                "text": content,
                "usage": _extract_usage(response),
                "model": used_model,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            }

        return _once()
