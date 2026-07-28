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

import base64
import io
import logging
import mimetypes
import os
from pathlib import Path
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


def is_free_model(model: str) -> bool:
    """True for OpenRouter free router / ``*:free`` model slugs."""
    slug = (model or "").strip().lower()
    return slug == "openrouter/free" or slug.endswith(":free")


# Back-compat alias used inside this module.
_is_free_model = is_free_model


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
    """Normalize OpenRouter / OpenAI usage, including reasoning + cache fields."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_cached_tokens": 0,
            "completion_reasoning_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
        prompt_tokens + completion_tokens
    )

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = 0
    reasoning = 0
    if prompt_details is not None:
        cached = int(
            getattr(prompt_details, "cached_tokens", None)
            or (prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else 0)
            or 0
        )
    if completion_details is not None:
        reasoning = int(
            getattr(completion_details, "reasoning_tokens", None)
            or (
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else 0
            )
            or 0
        )

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cached_tokens": cached,
        "completion_reasoning_tokens": reasoning,
        "total_tokens": total_tokens,
    }


def _extract_message_fields(message: Any) -> tuple[str, str | None]:
    """Return ``(content, reasoning)`` from an OpenRouter chat message.

    OpenRouter may place chain-of-thought in ``reasoning``, ``reasoning_content``,
    or nested ``reasoning_details``. Prefer the first non-empty string found.
    """
    content = (getattr(message, "content", None) or "").strip()
    reasoning: str | None = None
    for attr in ("reasoning", "reasoning_content"):
        raw = getattr(message, attr, None)
        if isinstance(raw, str) and raw.strip():
            reasoning = raw.strip()
            break
    if reasoning is None:
        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            chunks: list[str] = []
            for item in details:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None) or ""
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            if chunks:
                reasoning = "\n".join(chunks)
        elif isinstance(details, str) and details.strip():
            reasoning = details.strip()
    return content, reasoning


def encode_image_data_url_raw_png(path: Path) -> str:
    """Encode an on-disk PNG/JPEG as a data-URL without resizing (fixed-size sets)."""
    raw = Path(path).read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_image_data_url(
    image: Path | bytes | str,
    *,
    max_long_edge: int = 1280,
    jpeg_quality: int = 85,
) -> str:
    """Encode a page image as a ``data:image/...;base64,...`` URL for OpenRouter.

    TIFF/PNG/JPEG inputs are loaded via Pillow when available, resized so the
    long edge is at most ``max_long_edge``, and re-encoded as JPEG to control
    vision token cost. Falls back to raw base64 of the original bytes when
    Pillow is not installed (caller should prefer JPEG/PNG in that case).
    """
    raw: bytes
    if isinstance(image, (str, Path)):
        raw = Path(image).read_bytes()
        hint_name = str(image)
    else:
        raw = image
        hint_name = "image.bin"

    try:
        from PIL import Image
    except ImportError:
        mime = mimetypes.guess_type(hint_name)[0] or "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _chat_once(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    image_data_url: str | None = None,
    temperature: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, Any, str | None]:
    """Run one chat completion.

    Returns ``(content, response, reasoning)``. Reasoning models (Kimi K3,
    DeepSeek R1, o-series via OpenRouter) may populate ``reasoning`` while
    leaving ``content`` as the final answer — callers must budget ``max_tokens``
    high enough that reasoning does not consume the entire budget.
    """
    if image_data_url:
        user_content: Any = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        user_content = user_prompt

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise ValueError("Empty generation result (no choices) -- retrying.")

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    content, reasoning = _extract_message_fields(choice.message)
    if not content and not reasoning:
        raise ValueError(
            f"Empty generation result (finish_reason={finish_reason}) -- retrying."
        )
    # Some reasoning models burn the budget on hidden CoT and return empty content.
    # Surface that clearly so callers can raise max_tokens (Kimi K3 needs ≥500).
    if not content and reasoning:
        raise ValueError(
            "Empty content with non-empty reasoning "
            f"(finish_reason={finish_reason}); raise max_tokens so the model "
            "has budget left after reasoning."
        )
    if not content:
        raise ValueError(
            f"Empty generation result (finish_reason={finish_reason}) -- retrying."
        )
    return content, response, reasoning


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
            content, _response, _reasoning = _chat_once(
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
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body")
        reasoning = kwargs.get("reasoning")
        if reasoning is not None:
            body = dict(extra_body or {})
            body["reasoning"] = reasoning
            extra_body = body
        return self._complete_with_routing(
            prompt,
            max_tokens,
            system_prompt,
            image_data_url=None,
            temperature=temperature,
            extra_body=extra_body,
        )

    def complete_multimodal(
        self,
        prompt: str,
        *,
        image: Path | bytes | str | None = None,
        image_data_url: str | None = None,
        max_long_edge: int = 1280,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion with an optional page image (vision models).

        Provide either ``image`` (path/bytes) or a prebuilt ``image_data_url``.

        Optional kwargs:
          - ``temperature``: sampling temperature
          - ``reasoning``: OpenRouter reasoning config dict (effort / exclude / …)
          - ``extra_body``: merged into the OpenAI SDK ``extra_body``
          - ``preserve_square_png``: if True and ``image`` is a path, encode as
            a data-URL without long-edge JPEG resize (use for fixed 1024² sets)
        """
        max_tokens = int(kwargs.get("max_tokens", 1024))
        system_prompt = kwargs.get("system_prompt", self._system_prompt)
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body")
        reasoning = kwargs.get("reasoning")
        if reasoning is not None:
            body = dict(extra_body or {})
            body["reasoning"] = reasoning
            extra_body = body
        data_url = image_data_url
        if data_url is None and image is not None:
            if kwargs.get("preserve_square_png") and isinstance(image, (str, Path)):
                data_url = encode_image_data_url_raw_png(Path(image))
            else:
                data_url = encode_image_data_url(image, max_long_edge=max_long_edge)
        if data_url is None:
            raise ValueError("complete_multimodal requires image or image_data_url")
        return self._complete_with_routing(
            prompt,
            max_tokens,
            system_prompt,
            image_data_url=data_url,
            temperature=temperature,
            extra_body=extra_body,
        )

    def _complete_with_routing(
        self,
        prompt: str,
        max_tokens: int,
        system_prompt: str,
        *,
        image_data_url: str | None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for model in self._model_candidates():
            try:
                return self._complete_with_retry(
                    model,
                    prompt,
                    max_tokens,
                    system_prompt,
                    image_data_url=image_data_url,
                    temperature=temperature,
                    extra_body=extra_body,
                )
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
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        system_prompt: str,
        *,
        image_data_url: str | None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(_is_retryable_transient),
            reraise=True,
        )
        def _once() -> dict[str, Any]:
            content, response, reasoning = _chat_once(
                self._ensure_client(),
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
                image_data_url=image_data_url,
                temperature=temperature,
                extra_body=extra_body,
            )
            used_model = getattr(response, "model", None) or model
            if model != self.model:
                logger.info("OpenRouter used free/fallback model %s", used_model)
            return {
                "text": content,
                "reasoning": reasoning,
                "usage": _extract_usage(response),
                "model": used_model,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            }

        return _once()