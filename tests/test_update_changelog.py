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


def test_version_stable_strips_prerelease():
    assert Version.parse("1.0.0b0").stable().pep440 == "1.0.0"
    assert Version.parse("1.0.0-beta").stable().display == "1.0.0"
    assert Version.parse("1.0.0").pep440 == "1.0.0"


def test_bump_patch_is_plus_one_patch():
    assert mod._bump_version(Version.parse("1.0.0"), "patch").pep440 == "1.0.1"
    assert mod._bump_version(Version.parse("1.0.0b0"), "patch").pep440 == "1.0.1"
    assert mod._bump_version(Version.parse("1.0.9"), "patch").pep440 == "1.0.10"


def test_bump_major_half_or_whole():
    # minor < 5 → X.5.0
    assert mod._bump_version(Version.parse("1.0.3"), "major").pep440 == "1.5.0"
    assert mod._bump_version(Version.parse("0.1.0"), "major").pep440 == "0.5.0"
    # minor >= 5 → (X+1).0.0
    assert mod._bump_version(Version.parse("0.7.0"), "major").pep440 == "1.0.0"
    assert mod._bump_version(Version.parse("1.5.2"), "major").pep440 == "2.0.0"
    assert mod._bump_version(Version.parse("1.5.0"), "major").pep440 == "2.0.0"


def test_auto_bump_defaults_to_patch():
    cur = Version.parse("1.0.0")
    commits = [("abc1234", "2026-07-23", "feat: add cool thing")]
    sections = {"Added": ["Add cool thing (abc1234)"]}
    kind, ver = mod._decide_bump(cur, commits, sections, "auto")
    assert kind == "patch"
    assert ver is not None
    assert ver.pep440 == "1.0.1"


def test_auto_bump_breaking_uses_major_milestone():
    cur = Version.parse("1.0.2")
    commits = [("abc1234", "2026-07-23", "feat!: rewrite API")]
    sections = {"Added": ["Rewrite API (abc1234)"]}
    kind, ver = mod._decide_bump(cur, commits, sections, "auto")
    assert kind == "major"
    assert ver is not None
    assert ver.pep440 == "1.5.0"


def test_auto_bump_none_when_empty():
    cur = Version.parse("1.0.0")
    kind, ver = mod._decide_bump(cur, [], {}, "auto")
    assert kind is None and ver is None


def test_pin_rules_do_not_clobber_unrelated_text():
    old, new = Version.parse("1.0.0"), Version.parse("1.0.1")
    sample = 'version = "1.0.0"\n# historical 1.0.0-beta note stays if not matched\n'
    out = mod._apply_pin_rules(sample, old, new, ("pep440_quoted",))
    assert 'version = "1.0.1"' in out
    assert "1.0.0-beta" in out


def test_classify_breaking():
    assert mod._is_breaking("feat!: drop old API")
    assert not mod._is_breaking("feat: add widget")
