"""Tests for Chloride Discord integration helpers (no live Discord required)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.discord_bot.formatters import compact_analysis, format_discord_summary
from src.discord_bot.runner import DEFAULT_CONFIG_DIR, _overlay_secrets, _resolve_config_dir
from src.discord_bot.tools import (
    _guess_kind,
    _safe_filename,
    analyze_insurance_document_impl,
)


def test_safe_filename_strips_path_and_unsafe_chars():
    assert _safe_filename("../../evil name!.pdf") == "evil_name_.pdf"
    assert _safe_filename("") == "attachment.bin"


def test_guess_kind_by_suffix_and_content_type(tmp_path: Path):
    assert _guess_kind(tmp_path / "a.pdf") == "pdf"
    assert _guess_kind(tmp_path / "a.PNG") == "image"
    assert _guess_kind(tmp_path / "a.bin", "application/pdf") == "pdf"
    assert _guess_kind(tmp_path / "a.bin", "image/png") == "image"
    assert _guess_kind(tmp_path / "a.txt") == "text"


def test_compact_analysis_and_discord_summary():
    result = {
        "record_id": "discord-1",
        "claim_id": "CLM-1",
        "classification": {"document_type": "loss_notice", "confidence": 0.91},
        "extraction": {
            "fields": {"claim_id": ["CLM-1"], "date_of_loss": ["2024-01-15"]},
            "fields_flat": {"claim_id": "CLM-1", "date_of_loss": "2024-01-15"},
        },
        "vision": {"refined_fields": {"loss_type": "collision"}},
        "summary": {"memo": "Short memo about the loss."},
        "memo": "Short memo about the loss.",
        "flags": ["low_confidence_extract"],
        "low_confidence": False,
        "markdown": {"approx_tokens": 120},
        "stages": [
            {
                "stage": "classify",
                "ok": True,
                "confidence": 0.91,
                "flags": [],
                "error": None,
            }
        ],
    }
    compact = compact_analysis(result)
    assert compact["document_type"] == "loss_notice"
    assert compact["fields"]["claim_id"] == "CLM-1"
    assert compact["fields"]["date_of_loss"] == "2024-01-15"
    assert compact["fields"]["loss_type"] == "collision"
    assert compact["memo"] == "Short memo about the loss."
    # No list wrappers in Discord-facing fields.
    assert not isinstance(compact["fields"]["claim_id"], list)

    text = format_discord_summary(compact)
    assert "## Document analysis" in text
    assert "loss_notice" in text
    assert "CLM-1" in text
    assert "Short memo" in text
    assert "low_confidence_extract" in text


def test_download_url_rejects_local_and_file_schemes():
    import asyncio

    from src.discord_bot.tools import _download_url, _validate_download_url

    with pytest.raises(ValueError, match="http"):
        _validate_download_url("/etc/passwd")
    with pytest.raises(ValueError, match="file://|Local"):
        _validate_download_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="localhost"):
        _validate_download_url("http://localhost/secret")

    async def _run():
        with pytest.raises(ValueError):
            await _download_url("file:///etc/passwd", Path("/tmp/should-not-exist.bin"))

    asyncio.run(_run())


def test_register_tools_public_api_delegates():
    from src.discord_bot import register_tools

    # Should not raise even without Chloride (ImportError swallowed).
    register_tools()



def test_overlay_secrets_fills_placeholders(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "discord-test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("DISCORD_USE_OPENROUTER", "1")
    raw = {
        "DISCORD_TOKEN": "Paste your Discord token here.",
        "AI_API_KEY": "Put your API key here.",
        "AI_MODEL_NAME": "google-gla:gemini-flash-latest",
        "AI_OPENAI_COMPATIBLE_BASE_URL": None,
    }
    out = _overlay_secrets(raw)
    assert out["DISCORD_TOKEN"] == "discord-test-token"
    assert out["AI_API_KEY"] == "or-test-key"
    assert out["AI_OPENAI_COMPATIBLE_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert out["AI_MODEL_NAME"] == "anthropic/claude-sonnet-4.5"


def test_example_config_yaml_parses():
    example = DEFAULT_CONFIG_DIR / "config.yaml.example"
    assert example.exists()
    data = yaml.full_load(example.read_text(encoding="utf-8"))
    assert "DISCORD_PREFIX" in data
    assert "tiers" in data
    default_tools = data["tiers"]["default"]["allowed_tools"]
    assert "analyze_insurance_document" in default_tools
    assert "save_note" in default_tools
    assert "transcribe_audio" in default_tools
    assert "vibe_control" in default_tools


def test_extra_tools_register_when_chloride_present():
    pytest.importorskip("coral")
    from coral.agent import agent

    from src.discord_bot.agent_extras import register_extra_tools
    from src.discord_bot.tools import register_tools

    register_tools()
    register_extra_tools()
    tool_names = set(agent._function_toolset.tools.keys())  # noqa: SLF001
    assert {
        "analyze_insurance_document",
        "save_note",
        "search_notes",
        "transcribe_audio",
        "vibe_control",
        "server_help",
    } <= tool_names


def test_resolve_config_dir_default():
    path = _resolve_config_dir(None)
    assert path.name == "smol-doc-analyzer"


def test_analyze_impl_on_text():
    import asyncio

    out = asyncio.run(
        analyze_insurance_document_impl(
            text=(
                "AUTOMOBILE LOSS NOTICE\n"
                "Claim Number: CLM-DISCORD-1\n"
                "Date of Loss: 2024-01-15\n"
                "Loss Type: collision\n"
            ),
            enable_vision=False,
            record_id="test-discord-text",
        )
    )
    assert out.get("ok") is True
    assert "discord_summary" in out
    assert out["analysis"]["record_id"] == "test-discord-text"


def test_post_webhook_requires_url(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    from src.discord_bot import webhook as wh

    # Ensure dotenv empty path doesn't revive a leaked env in CI.
    monkeypatch.setattr(wh, "webhook_url", lambda: "")
    out = wh.post_webhook("hi", url="")
    assert out["ok"] is False
    assert "not set" in out["error"]


def test_post_webhook_rejects_non_discord_url():
    from src.discord_bot.webhook import post_webhook

    out = post_webhook("hi", url="https://example.com/not-a-webhook")
    assert out["ok"] is False
    assert "does not look like" in out["error"]


def test_post_webhook_success_mocked(monkeypatch):
    from src.discord_bot import webhook as wh

    class _Resp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 204

    def fake_urlopen(req, timeout=30):
        assert req.full_url.endswith("/webhooks/1/abc")
        return _Resp()

    monkeypatch.setattr(wh.urllib.request, "urlopen", fake_urlopen)
    out = wh.post_webhook(
        "hello",
        url="https://discord.com/api/webhooks/1/abc",
    )
    assert out == {"ok": True, "status": 204}
