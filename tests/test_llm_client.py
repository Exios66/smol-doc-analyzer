"""Unit tests for OpenRouter credit → free-model routing."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils import llm_client
from src.utils.config import Config
from src.utils.llm_client import (
    DEFAULT_FREE_FALLBACK_MODELS,
    DEFAULT_OLLAMA_FALLBACK_MODELS,
    GenerationClient,
    OpenRouterClient,
    is_credit_unavailable_error,
    is_openrouter_free_quota_error,
    parse_free_fallback_models,
    parse_ollama_fallback_models,
    reset_credit_fallback_state,
    reset_ollama_fallback_state,
)


class _FakeAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: object = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.message = message


@pytest.fixture(autouse=True)
def _reset_sticky():
    reset_credit_fallback_state()
    reset_ollama_fallback_state()
    yield
    reset_credit_fallback_state()
    reset_ollama_fallback_state()


def _cfg(**overrides) -> Config:
    base = Config.load()
    data = {f.name: getattr(base, f.name) for f in fields(base)}
    data.update(
        {
            "openrouter_api_key": "sk-or-v1-test",
            "generation_model": "anthropic/claude-sonnet-4.5",
            "max_retries": 1,
            "openrouter_free_fallback_models": DEFAULT_FREE_FALLBACK_MODELS,
            "openrouter_prefer_free": False,
            "ollama_base_url": "http://localhost:11434/v1",
            "ollama_fallback_models": DEFAULT_OLLAMA_FALLBACK_MODELS,
            "ollama_prefer": False,
            "ollama_fallback_enabled": True,
        }
    )
    data.update(overrides)
    return Config(**data)


def test_is_credit_unavailable_detects_402():
    assert is_credit_unavailable_error(_FakeAPIError("nope", status_code=402))


def test_is_credit_unavailable_detects_message_body():
    exc = _FakeAPIError(
        "Error code: 402",
        status_code=400,
        body={"error": {"message": "This request requires more credits, or fewer max_tokens."}},
    )
    assert is_credit_unavailable_error(exc)


def test_parse_free_fallback_models_default_and_custom(monkeypatch):
    monkeypatch.delenv("OPENROUTER_FREE_FALLBACK_MODELS", raising=False)
    assert parse_free_fallback_models()[0] == "openrouter/free"
    assert parse_free_fallback_models("a/b:free, c/d:free") == ("a/b:free", "c/d:free")


def test_generation_client_falls_back_to_free_on_402(monkeypatch):
    cfg = _cfg()
    client = GenerationClient(cfg)
    mock_openai = MagicMock()
    client._client = mock_openai

    paid_err = _FakeAPIError(
        "Payment Required",
        status_code=402,
        body={"error": {"message": "This request requires more credits"}},
    )

    free_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="  free model ok  "),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        model="meta-llama/llama-3.2-3b-instruct:free",
    )

    def _create(**kwargs):
        if kwargs["model"] == cfg.generation_model:
            raise paid_err
        return free_response

    mock_openai.chat.completions.create.side_effect = _create

    text = client.generate("sys", "user", max_tokens=64)
    assert text == "free model ok"

    models_called = [
        c.kwargs["model"] for c in mock_openai.chat.completions.create.call_args_list
    ]
    assert models_called[0] == cfg.generation_model
    assert models_called[1] == "openrouter/free"
    assert llm_client._CREDITS_UNAVAILABLE is True

    # Sticky: next call should skip paid model entirely
    mock_openai.chat.completions.create.reset_mock()
    mock_openai.chat.completions.create.side_effect = None
    mock_openai.chat.completions.create.return_value = free_response
    client.generate("sys", "user2", max_tokens=32)
    models_called = [
        c.kwargs["model"] for c in mock_openai.chat.completions.create.call_args_list
    ]
    assert cfg.generation_model not in models_called
    assert models_called[0] == "openrouter/free"


def test_generation_client_prefer_free_skips_paid(monkeypatch):
    cfg = _cfg(openrouter_prefer_free=True)
    client = GenerationClient(cfg)
    mock_openai = MagicMock()
    client._client = mock_openai
    mock_openai.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="prefer free"),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="openrouter/free",
    )
    assert client.generate("s", "u") == "prefer free"
    assert mock_openai.chat.completions.create.call_args.kwargs["model"] == "openrouter/free"


def test_config_loads_free_fallback_env(monkeypatch):
    monkeypatch.setenv(
        "OPENROUTER_FREE_FALLBACK_MODELS",
        "openrouter/free,foo/bar:free",
    )
    monkeypatch.setenv("OPENROUTER_PREFER_FREE", "1")
    cfg = Config.load()
    assert cfg.openrouter_free_fallback_models == ("openrouter/free", "foo/bar:free")
    assert cfg.openrouter_prefer_free is True


def test_is_openrouter_free_quota_error():
    exc = _FakeAPIError(
        "Rate limit exceeded: free-models-per-day",
        status_code=429,
        body={"error": {"message": "Rate limit exceeded: free-models-per-day"}},
    )
    assert is_openrouter_free_quota_error(exc)


def test_parse_ollama_fallback_models_default_and_custom(monkeypatch):
    monkeypatch.delenv("OLLAMA_FALLBACK_MODELS", raising=False)
    assert parse_ollama_fallback_models()[0] == "qwen3-vl:4b-instruct"
    assert parse_ollama_fallback_models("a:1, b:2") == ("a:1", "b:2")


def test_is_ollama_vision_model():
    from src.utils.llm_client import is_ollama_vision_model

    assert is_ollama_vision_model("qwen3-vl:4b-instruct")
    assert is_ollama_vision_model("minicpm-v:8b")
    assert is_ollama_vision_model("granite3.2-vision")
    assert not is_ollama_vision_model("llama3.1:8b-instruct-q4_K_M")


def test_openrouter_client_falls_back_to_ollama_on_free_quota(monkeypatch):
    cfg = _cfg(openrouter_prefer_free=True)
    client = OpenRouterClient(model="openrouter/free", cfg=cfg)
    mock_or = MagicMock()
    mock_ollama = MagicMock()
    client._client = mock_or
    client._ollama_client = mock_ollama

    quota_err = _FakeAPIError(
        "Rate limit exceeded: free-models-per-day",
        status_code=429,
        body={"error": {"message": "Rate limit exceeded: free-models-per-day"}},
    )

    ollama_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="letter"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        model="qwen3-vl:4b-instruct",
    )

    mock_or.chat.completions.create.side_effect = quota_err
    mock_ollama.chat.completions.create.return_value = ollama_response

    result = client.complete("classify this", max_tokens=32)
    assert result["text"] == "letter"
    assert result["model"] == "ollama/qwen3-vl:4b-instruct"
    assert llm_client._OLLAMA_ACTIVE is True
    assert mock_ollama.chat.completions.create.called


def test_prefer_ollama_skips_openrouter():
    cfg = _cfg(ollama_prefer=True)
    client = OpenRouterClient(model="moonshotai/kimi-k3", cfg=cfg)
    mock_or = MagicMock()
    mock_ollama = MagicMock()
    client._client = mock_or
    client._ollama_client = mock_ollama
    mock_ollama.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="invoice"),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="qwen3-vl:4b-instruct",
    )
    result = client.complete("x", max_tokens=16)
    assert result["text"] == "invoice"
    mock_or.chat.completions.create.assert_not_called()
    assert mock_ollama.chat.completions.create.call_args.kwargs["model"] == (
        "qwen3-vl:4b-instruct"
    )


def test_ollama_vision_passes_image_data_url():
    cfg = _cfg(ollama_prefer=True)
    client = OpenRouterClient(model="qwen3-vl:4b-instruct", cfg=cfg)
    mock_ollama = MagicMock()
    client._ollama_client = mock_ollama
    mock_ollama.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="memo"),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="qwen3-vl:4b-instruct",
    )
    data_url = "data:image/png;base64,aaaa"
    result = client.complete_multimodal(
        "classify",
        image_data_url=data_url,
        max_tokens=16,
    )
    assert result["text"] == "memo"
    content = mock_ollama.chat.completions.create.call_args.kwargs["messages"][1][
        "content"
    ]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
