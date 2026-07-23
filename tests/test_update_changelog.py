"""Unit tests for changelog version bump helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "update_changelog", ROOT / "scripts" / "update_changelog.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)
Version = mod.Version


def test_version_roundtrip_beta():
    v0 = Version.parse("1.0.0b0")
    assert v0.display == "1.0.0-beta"
    assert v0.pep440 == "1.0.0b0"
    v1 = Version.parse("1.0.0-beta.1")
    assert v1.pep440 == "1.0.0b1"
    assert Version.parse("1.0.0-beta").pep440 == "1.0.0b0"


def test_bump_prerelease_and_semver():
    cur = Version.parse("1.0.0b0")
    nxt = mod._bump_version(cur, "prerelease")
    assert nxt.pep440 == "1.0.0b1"
    assert nxt.display == "1.0.0-beta.1"

    stable = Version.parse("1.0.0")
    assert mod._bump_version(stable, "minor").pep440 == "1.1.0"
    assert mod._bump_version(stable, "patch").pep440 == "1.0.1"
    assert mod._bump_version(stable, "major").pep440 == "2.0.0"


def test_auto_bump_on_beta_uses_prerelease():
    cur = Version.parse("1.0.0b0")
    commits = [("abc1234", "2026-07-23", "feat: add cool thing")]
    sections = {"Added": ["Add cool thing (abc1234)"]}
    kind, ver = mod._decide_bump(cur, commits, sections, "auto")
    assert kind == "prerelease"
    assert ver is not None
    assert ver.pep440 == "1.0.0b1"


def test_auto_bump_none_when_empty():
    cur = Version.parse("1.0.0b0")
    kind, ver = mod._decide_bump(cur, [], {}, "auto")
    assert kind is None and ver is None


def test_pin_rules_do_not_clobber_unrelated_text():
    old, new = Version.parse("1.0.0b0"), Version.parse("1.0.0b1")
    sample = 'version = "1.0.0b0"\n# historical 1.0.0-beta note stays if not matched\n'
    out = mod._apply_pin_rules(sample, old, new, ("pep440_quoted",))
    assert 'version = "1.0.0b1"' in out


def test_classify_breaking():
    assert mod._is_breaking("feat!: drop old API")
    assert not mod._is_breaking("feat: add widget")
