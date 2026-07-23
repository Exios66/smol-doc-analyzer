#!/usr/bin/env python3
"""Update CHANGELOG.md from git history and incrementally bump the package version.

Manual:
  python scripts/update_changelog.py
  python scripts/update_changelog.py --dry-run
  python scripts/update_changelog.py --bump none     # Unreleased only
  python scripts/update_changelog.py --bump major    # → X.5.0 or (X+1).0.0

Scheduled (macOS LaunchAgent, Wednesday 23:00 America/Chicago):
  ./scripts/install_changelog_launchagent.sh

Version policy (``--bump auto``, default when writing):
  - Normal releases: **+0.0.1** (patch), e.g. ``1.0.0`` → ``1.0.1`` → ``1.0.2``
  - Major releases (``--bump major`` or breaking commits): jump to
    ``X.5.0`` if ``minor < 5``, else the next whole ``(X+1).0.0``
    (e.g. ``1.0.3`` → ``1.5.0``, ``1.5.2`` → ``2.0.0``, ``0.7.0`` → ``1.0.0``)
  - Prerelease pins (``1.0.0b0``) are normalized to the stable base first
  - Cuts ``[Unreleased]`` into ``## [X.Y.Z] — YYYY-MM-DD`` and syncs pins
  - Appends a row to ``data/changelog/version_log.jsonl``
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
STATE_PATH = REPO_ROOT / "data" / "changelog" / "last_run.txt"
VERSION_LOG_PATH = REPO_ROOT / "data" / "changelog" / "version_log.jsonl"
TZ_NAME = "America/Chicago"
GITHUB_COMPARE = "https://github.com/Exios66/smol-doc-analyzer/compare"
GITHUB_TAG = "https://github.com/Exios66/smol-doc-analyzer/releases/tag"

SECTION_ORDER = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
)

VERSION_HEADER_RE = re.compile(
    r"^## \[([^\]]+)\](?: — (\d{4}-\d{2}-\d{2}))?\s*$",
    re.MULTILINE,
)
UNRELEASED_RE = re.compile(
    r"(## \[Unreleased\]\s*\n)(.*?)(?=\n## \[|\Z)",
    re.DOTALL,
)
AUTO_MARKER_RE = re.compile(r"<!-- changelog-auto: .*? -->\n?")

# pep440: 1.0.0 / 1.0.0b0 / 1.0.0rc1 / 1.2.3b4
PEP440_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:(?P<pre_l>a|b|rc)(?P<pre_n>\d+))?$"
)

# Files where the *current* package version is authoritative (not historical prose).
VERSION_PIN_SPECS: tuple[tuple[Path, tuple[str, ...]], ...] = (
    (REPO_ROOT / "pyproject.toml", ("pep440_quoted",)),
    (REPO_ROOT / "src" / "__init__.py", ("pep440_dunder",)),
    (REPO_ROOT / "src" / "docie" / "serve.py", ("pep440_kw")),
    (REPO_ROOT / "src" / "discord_bot" / "webhook.py", ("pep440_ua",)),
    (REPO_ROOT / "src" / "discord_bot" / "tools.py", ("pep440_ua",)),
    (REPO_ROOT / "README.md", ("readme",)),
    (REPO_ROOT / "docs" / "usage.md", ("docs_current",)),
    (REPO_ROOT / "docs" / "about.qmd", ("docs_current",)),
    (REPO_ROOT / "docs" / "architecture.qmd", ("docs_current",)),
    (REPO_ROOT / "docs" / "plan" / "plan.md", ("docs_current",)),
    (REPO_ROOT / "discord" / "smol-doc-analyzer" / "README.md", ("docs_current",)),
)


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    pre_l: str | None = None  # a | b | rc
    pre_n: int | None = None

    @property
    def pep440(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_l is None:
            return base
        return f"{base}{self.pre_l}{self.pre_n or 0}"

    @property
    def display(self) -> str:
        """Keep-a-Changelog / human label. New bumps are plain X.Y.Z."""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_l is None:
            return base
        # Historical prerelease labels only (legacy pins / old changelog headers).
        label = {"a": "alpha", "b": "beta", "rc": "rc"}[self.pre_l]
        n = int(self.pre_n or 0)
        if self.pre_l == "b" and n == 0:
            return f"{base}-{label}"
        return f"{base}-{label}.{n}"

    def stable(self) -> "Version":
        """Drop prerelease suffix → ``1.0.0b0`` becomes ``1.0.0``."""
        return Version(self.major, self.minor, self.patch)

    @classmethod
    def parse(cls, text: str) -> "Version":
        raw = text.strip()
        # accept display forms
        m = re.match(
            r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
            r"(?:-(?P<label>alpha|beta|rc)(?:\.(?P<n>\d+))?)?$",
            raw,
            re.IGNORECASE,
        )
        if m:
            label = (m.group("label") or "").lower()
            pre_l = {"alpha": "a", "beta": "b", "rc": "rc"}.get(label)
            pre_n = None
            if pre_l:
                pre_n = int(m.group("n") or 0)
            return cls(
                int(m.group("major")),
                int(m.group("minor")),
                int(m.group("patch")),
                pre_l,
                pre_n,
            )
        m2 = PEP440_RE.match(raw)
        if not m2:
            raise ValueError(f"Unrecognized version: {text!r}")
        return cls(
            int(m2.group("major")),
            int(m2.group("minor")),
            int(m2.group("patch")),
            m2.group("pre_l"),
            int(m2.group("pre_n")) if m2.group("pre_n") is not None else None,
        )


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _now_central() -> datetime:
    return datetime.now(ZoneInfo(TZ_NAME))


def _read_current_pep440() -> Version:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return Version.parse(m.group(1))


def _latest_released_version(changelog: str) -> Version | None:
    for match in VERSION_HEADER_RE.finditer(changelog):
        name = match.group(1)
        if name.lower() == "unreleased":
            continue
        try:
            return Version.parse(name)
        except ValueError:
            continue
    return None


def _latest_released_date(changelog: str) -> str | None:
    for match in VERSION_HEADER_RE.finditer(changelog):
        name, date = match.group(1), match.group(2)
        if name.lower() == "unreleased":
            continue
        if date:
            return date
    return None


def _default_since(changelog: str) -> str:
    if STATE_PATH.exists():
        stamp = STATE_PATH.read_text(encoding="utf-8").strip().splitlines()[0]
        if stamp:
            return stamp
    released = _latest_released_date(changelog)
    if released:
        return released
    return _now_central().strftime("%Y-%m-%d")


def _classify(subject: str) -> str:
    lower = subject.strip().lower()
    if re.match(r"^(feat|feature)(\(|:|!)", lower):
        return "Added"
    if re.match(r"^(fix|bugfix)(\(|:|!)", lower):
        return "Fixed"
    if re.match(r"^(security|sec)(\(|:|!)", lower):
        return "Security"
    if re.match(r"^(deprecate|deprecated)(\(|:|!)", lower):
        return "Deprecated"
    if re.match(r"^(remove|removed|delete|drop)(\(|:|!)", lower):
        return "Removed"
    if re.match(r"^(docs?|doc)(\(|:|!)", lower):
        return "Changed"
    if re.match(r"^(refactor|perf|chore|build|ci|style|test)(\(|:|!)", lower):
        return "Changed"
    if lower.startswith(("add ", "added ", "create ", "created ", "introduce ")):
        return "Added"
    if lower.startswith(("fix ", "fixed ", "bugfix ", "hotfix ")):
        return "Fixed"
    if lower.startswith(("remove ", "removed ", "delete ", "drop ")):
        return "Removed"
    if lower.startswith(("deprecate ", "deprecated ")):
        return "Deprecated"
    if "security" in lower:
        return "Security"
    return "Changed"


def _is_breaking(subject: str) -> bool:
    s = subject.strip()
    if re.search(r"^(feat|fix|refactor|perf)(\([^)]*\))?!:", s, re.I):
        return True
    return "breaking change" in s.lower() or "breaking:" in s.lower()


def _clean_subject(subject: str) -> str:
    s = subject.strip()
    s = re.sub(
        r"^(feat|feature|fix|bugfix|docs?|doc|refactor|perf|chore|build|ci|"
        r"style|test|security|sec|remove|removed|deprecate|deprecated)"
        r"(\([^)]*\))?(!)?:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s.rstrip(".")


def _collect_commits(since: str) -> list[tuple[str, str, str]]:
    log = _run_git(
        [
            "log",
            f"--since={since}",
            "--date=short",
            "--pretty=format:%h%x09%ad%x09%s",
            "--no-merges",
        ]
    )
    rows: list[tuple[str, str, str]] = []
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        sha, date, subject = parts
        sub_l = subject.lower()
        if "changelog" in sub_l and any(
            w in sub_l for w in ("update", "sync", "auto", "refresh", "bump")
        ):
            continue
        if re.match(r"^v?\d+\.\d+\.\d+", subject):
            continue
        rows.append((sha, date, subject))
    return rows


def _parse_existing_unreleased(body: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in body.splitlines():
        heading = re.match(r"^###\s+(\w+)\s*$", line)
        if heading:
            current = heading.group(1)
            continue
        if current is None:
            continue
        if line.strip() in {"---", "***", "___"}:
            # Horizontal rules belong between sections, never inside a bullet.
            continue
        if line.startswith("- "):
            sections[current].append(line[2:].rstrip())
            continue
        if sections[current] and line.strip() and not line.startswith("#"):
            sections[current][-1] += "\n" + line.rstrip()
    return sections


def _normalize_bullet(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s*\([0-9a-f]{7,40}\)\s*$", "", first).lower()


def _format_bullet(item: str) -> list[str]:
    parts = item.splitlines() or [""]
    out = [f"- {parts[0]}"]
    for cont in parts[1:]:
        out.append(cont)
    return out


def _merge_sections(
    existing: dict[str, list[str]],
    commits: list[tuple[str, str, str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {k: list(v) for k, v in existing.items()}
    seen = {_normalize_bullet(b) for items in merged.values() for b in items}

    for sha, _date, subject in commits:
        section = _classify(subject)
        cleaned = _clean_subject(subject)
        bullet = f"{cleaned} ({sha})"
        key = _normalize_bullet(bullet)
        key_nosha = _normalize_bullet(cleaned)
        if key in seen or key_nosha in seen:
            continue
        if any(key_nosha[:48] in s or s[:48] in key_nosha for s in seen if len(s) > 20):
            continue
        merged.setdefault(section, []).append(bullet)
        seen.add(key)
        seen.add(key_nosha)
    return merged


def _sections_have_content(sections: dict[str, list[str]]) -> bool:
    return any(bool(v) for v in sections.values())


def _render_sections(sections: dict[str, list[str]], *, marker: str | None = None) -> str:
    lines: list[str] = []
    if marker:
        lines.extend([marker, ""])
    wrote = False
    for section in SECTION_ORDER:
        items = sections.get(section) or []
        if not items:
            continue
        wrote = True
        lines.append(f"### {section}")
        lines.append("")
        for item in items:
            lines.extend(_format_bullet(item))
        lines.append("")
    if not wrote:
        lines.extend(["_No notable commits since the last changelog update._", ""])
    return "\n".join(lines).rstrip() + "\n"


def _decide_bump(
    current: Version,
    commits: list[tuple[str, str, str]],
    sections: dict[str, list[str]],
    mode: str,
) -> tuple[str | None, Version | None]:
    """Return (kind, new_version) or (None, None) if no bump."""
    if mode == "none":
        return None, None
    if not commits and not _sections_have_content(sections):
        return None, None

    # Treat legacy beta/rc pins as their stable base (1.0.0b0 → 1.0.0).
    base = current.stable()
    breaking = any(_is_breaking(s) for _, _, s in commits)

    if mode == "auto":
        # Default cadence: +0.0.1. Breaking commits escalate to a major milestone.
        kind = "major" if breaking else "patch"
    elif mode == "prerelease":
        # Legacy alias — project no longer ships bN trains; map to patch.
        kind = "patch"
    else:
        kind = mode

    return kind, _bump_version(base, kind)


def _bump_version(current: Version, kind: str) -> Version:
    """Bump a *stable* version. ``major`` uses .5 / whole-number milestones."""
    base = current.stable()
    if kind == "patch":
        return Version(base.major, base.minor, base.patch + 1)
    if kind == "minor":
        return Version(base.major, base.minor + 1, 0)
    if kind == "major":
        # Half-step (X.5.0) or next whole major ((X+1).0.0).
        if base.minor < 5:
            return Version(base.major, 5, 0)
        return Version(base.major + 1, 0, 0)
    if kind == "prerelease":
        return _bump_version(base, "patch")
    raise ValueError(f"Unknown bump kind: {kind}")


def _milestone_blurb(sections: dict[str, list[str]], kind: str) -> str:
    n_added = len(sections.get("Added") or [])
    n_fixed = len(sections.get("Fixed") or []) + len(sections.get("Security") or [])
    n_changed = len(sections.get("Changed") or [])
    bits = []
    if n_added:
        bits.append(f"{n_added} addition(s)")
    if n_fixed:
        bits.append(f"{n_fixed} fix(es)")
    if n_changed:
        bits.append(f"{n_changed} change(s)")
    detail = ", ".join(bits) if bits else "maintenance"
    return f"Auto {kind} bump — {detail}"


def _update_version_map(changelog: str, new: Version, blurb: str, prev: Version) -> str:
    row = f"| `{new.display}` | {blurb} |"
    # Insert after header separator line of the version map table
    map_re = re.compile(
        r"(## Version map\n\n\| Version \| Milestone \|\n\|---------+\|-----------\|\n)",
        re.MULTILINE,
    )
    m = map_re.search(changelog)
    if m:
        # Avoid duplicate row
        if f"| `{new.display}` |" not in changelog:
            changelog = changelog[: m.end()] + row + "\n" + changelog[m.end() :]
    else:
        changelog = changelog.rstrip() + (
            f"\n\n## Version map\n\n"
            f"| Version | Milestone |\n|---------|-----------|\n{row}\n"
        )

    tag_line = f"[{new.display}]: {GITHUB_COMPARE}/v{prev.display}...v{new.display}"
    # Prefer tag URL for first of a line; compare is fine for incremental
    if new.pre_l and (new.pre_n or 0) == 0:
        tag_line = f"[{new.display}]: {GITHUB_TAG}/v{new.display}"
    if f"[{new.display}]:" not in changelog:
        # Insert near other link refs (after version map)
        changelog = changelog.rstrip() + "\n" + tag_line + "\n"
    return changelog


def _apply_pin_rules(text: str, old: Version, new: Version, rules: tuple[str, ...]) -> str:
    """Apply targeted version rewrites so historical changelog/plan text stays put."""
    out = text
    for rule in rules:
        if rule == "pep440_quoted":
            out = re.sub(
                rf'(?m)^(version\s*=\s*"){re.escape(old.pep440)}(")',
                rf"\g<1>{new.pep440}\g<2>",
                out,
            )
        elif rule == "pep440_dunder":
            out = re.sub(
                rf'(__version__\s*=\s*"){re.escape(old.pep440)}(")',
                rf"\g<1>{new.pep440}\g<2>",
                out,
            )
        elif rule == "pep440_kw":
            # FastAPI/kwargs style: version="1.0.0" or version = "1.0.0"
            out = re.sub(
                rf'(version\s*=\s*"){re.escape(old.pep440)}(")',
                rf"\g<1>{new.pep440}\g<2>",
                out,
            )
            out = out.replace(f'version="{old.pep440}"', f'version="{new.pep440}"')
        elif rule == "pep440_ua":
            out = out.replace(
                f"smol-doc-analyzer-discord-webhook/{old.pep440}",
                f"smol-doc-analyzer-discord-webhook/{new.pep440}",
            )
            out = out.replace(
                f"smol-doc-analyzer-discord-bot/{old.pep440}",
                f"smol-doc-analyzer-discord-bot/{new.pep440}",
            )
        elif rule == "readme":
            out = out.replace(
                f"smol--doc--analyzer-v{old.pep440}-",
                f"smol--doc--analyzer-v{new.pep440}-",
            )
            out = out.replace(
                f"version-{old.display.replace('-', '--')}-",
                f"version-{new.display.replace('-', '--')}-",
            )
            out = out.replace(
                f'alt="Version {old.display}"',
                f'alt="Version {new.display}"',
            )
            out = re.sub(
                rf"(\*\*Version:\*\*\s*\[`){re.escape(old.display)}"
                rf"(`\]\(CHANGELOG\.md\)\s*\(`){re.escape(old.pep440)}(`\))",
                rf"\g<1>{new.display}\g<2>{new.pep440}\g<3>",
                out,
            )
            out = re.sub(
                rf"(\*\*Package version:\*\*\s*`){re.escape(old.display)}"
                rf"(`\s*\(`){re.escape(old.pep440)}(`\))",
                rf"\g<1>{new.display}\g<2>{new.pep440}\g<3>",
                out,
            )
        elif rule == "docs_current":
            # Only lines that declare the *current* package version.
            out = re.sub(
                rf"(?m)^(\*\*(?:Current package )?Version:\*\*\s*`?)"
                rf"{re.escape(old.display)}(`?\s*\(`){re.escape(old.pep440)}(`\).*)$",
                rf"\g<1>{new.display}\g<2>{new.pep440}\g<3>",
                out,
            )
            out = re.sub(
                rf"(?m)^(\*\*Version:\*\*\s*`){re.escape(old.display)}(`\b.*)$",
                rf"\g<1>{new.display}\g<2>",
                out,
            )
            out = re.sub(
                rf"(package\s+`){re.escape(old.display)}(`)",
                rf"\g<1>{new.display}\g<2>",
                out,
                count=1,
            )
            out = re.sub(
                rf"(\*\*smol-doc-analyzer\*\*\s*\(package\s+`){re.escape(old.display)}(`\))",
                rf"\g<1>{new.display}\g<2>",
                out,
            )
            out = re.sub(
                rf"(smol-doc-analyzer\*\*\s*\(`){re.escape(old.display)}(`\))",
                rf"\g<1>{new.display}\g<2>",
                out,
            )
    return out


def _replace_version_pins(old: Version, new: Version, *, dry_run: bool) -> list[str]:
    """Replace current version pins in known files (targeted, not global)."""
    touched: list[str] = []
    for path, rules in VERSION_PIN_SPECS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = _apply_pin_rules(text, old, new, rules)
        if updated != text:
            touched.append(str(path.relative_to(REPO_ROOT)))
            if not dry_run:
                path.write_text(updated, encoding="utf-8")
    return touched


def _append_version_log(entry: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    VERSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VERSION_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def update_changelog(
    *,
    since: str | None = None,
    dry_run: bool = False,
    write_state: bool = True,
    bump: str = "auto",
) -> tuple[str, bool, dict]:
    if not CHANGELOG_PATH.exists():
        raise FileNotFoundError(f"Missing {CHANGELOG_PATH}")

    original = CHANGELOG_PATH.read_text(encoding="utf-8")
    since_val = since or _default_since(original)
    commits = _collect_commits(since_val)
    current = _read_current_pep440()

    match = UNRELEASED_RE.search(original)
    if not match:
        raise RuntimeError("CHANGELOG.md has no ## [Unreleased] section")

    existing_body = AUTO_MARKER_RE.sub("", match.group(2))
    # Keep-a-Changelog often puts a `---` rule after [Unreleased]; don't treat it
    # as a bullet continuation.
    existing_body = re.sub(r"\n(?:---\s*\n*)+$", "\n", existing_body)
    existing = _parse_existing_unreleased(existing_body)
    merged = _merge_sections(existing, commits)

    stamp = _now_central().strftime("%Y-%m-%d %H:%M %Z")
    marker = f"<!-- changelog-auto: {stamp} · {len(commits)} commit(s) scanned -->"
    kind, new_ver = _decide_bump(current, commits, merged, bump)

    meta: dict = {
        "since": since_val,
        "commits": len(commits),
        "bump": kind,
        "old_version": current.pep440,
        "old_display": current.display,
        "new_version": new_ver.pep440 if new_ver else None,
        "new_display": new_ver.display if new_ver else None,
        "files_touched": [],
    }

    if new_ver is not None and kind is not None and _sections_have_content(merged):
        # Cut a release: empty Unreleased + dated version section
        date = _now_central().date().isoformat()
        release_body = _render_sections(merged, marker=None)
        unreleased_body = (
            f"{marker}\n\n"
            "_Nothing yet — next scheduled / manual update will land here._\n"
        )
        release_block = f"## [{new_ver.display}] — {date}\n\n{release_body}"
        rest = original[match.end() :]
        rest = re.sub(r"^\s*---\s*\n*", "", rest)
        updated = (
            original[: match.start(1)]
            + "## [Unreleased]\n"
            + unreleased_body
            + "\n---\n\n"
            + release_block
            + "\n---\n"
            + rest
        )
        prev = _latest_released_version(original) or current
        blurb = _milestone_blurb(merged, kind)
        # Status note under new release
        status = (
            f"### Status\n\n"
            f"- Package version **`{new_ver.pep440}`** "
            f"(`{new_ver.display}`) — auto-bumped from "
            f"`{current.pep440}` ({kind})\n"
        )
        updated = updated.replace(
            release_block,
            release_block.rstrip() + "\n\n" + status + "\n",
            1,
        )
        updated = _update_version_map(updated, new_ver, blurb, prev)
        meta["files_touched"] = _replace_version_pins(current, new_ver, dry_run=dry_run)
        meta["files_touched"].insert(0, "CHANGELOG.md")
    else:
        # Refresh Unreleased only (no version cut)
        new_body = _render_sections(merged, marker=marker)
        updated = (
            original[: match.start(1)]
            + match.group(1)
            + new_body
            + "\n"
            + original[match.end() :]
        )
        if new_ver is not None and kind is not None:
            # Bump requested but nothing to cut — still sync pins if forced
            meta["files_touched"] = _replace_version_pins(
                current, new_ver, dry_run=dry_run
            )

    updated = re.sub(r"\n{3,}", "\n\n", updated)
    changed = updated != original or bool(meta["files_touched"])

    log_entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tz": TZ_NAME,
        "dry_run": dry_run,
        **meta,
    }

    if dry_run:
        _append_version_log(log_entry, dry_run=True)
        return updated, changed, meta

    if changed:
        CHANGELOG_PATH.write_text(updated, encoding="utf-8")

    _append_version_log(log_entry, dry_run=False)

    if write_state:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            _now_central().date().isoformat() + "\n",
            encoding="utf-8",
        )

    return updated, changed, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh CHANGELOG.md from git and incrementally bump the package "
            "version (+0.0.1 normally; major → X.5.0 or (X+1).0.0)."
        )
    )
    parser.add_argument(
        "--since",
        help="Git --since date (YYYY-MM-DD). Default: last run stamp or latest release date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results; do not write CHANGELOG, pins, or logs.",
    )
    parser.add_argument(
        "--print-unreleased",
        action="store_true",
        help="With --dry-run, print only the Unreleased / new release preview.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not update data/changelog/last_run.txt",
    )
    parser.add_argument(
        "--bump",
        choices=("auto", "major", "minor", "patch", "prerelease", "none"),
        default="auto",
        help=(
            "Version bump policy (default: auto = +0.0.1 patch; breaking → major). "
            "'major' jumps to X.5.0 or the next whole (X+1).0.0. "
            "'none' only refreshes [Unreleased]. "
            "'prerelease' is a deprecated alias for patch."
        ),
    )
    parser.add_argument(
        "--no-bump",
        action="store_true",
        help="Alias for --bump none",
    )
    args = parser.parse_args(argv)
    bump = "none" if args.no_bump else args.bump

    try:
        updated, changed, meta = update_changelog(
            since=args.since,
            dry_run=args.dry_run,
            write_state=not args.no_state and not args.dry_run,
            bump=bump,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        if args.print_unreleased:
            # Show Unreleased + the newest cut section if present
            m = UNRELEASED_RE.search(updated)
            print(m.group(0) if m else updated)
            if meta.get("new_display"):
                hdr = f"## [{meta['new_display']}]"
                idx = updated.find(hdr)
                if idx != -1:
                    rest = updated[idx:]
                    nxt = re.search(r"\n## \[", rest[len(hdr) :])
                    block = rest[: len(hdr) + nxt.start()] if nxt else rest
                    print("\n--- cut release preview ---\n")
                    print(block[:4000])
        else:
            print(updated)
        print(
            f"\n# dry-run: {'would change' if changed else 'no change'} "
            f"bump={meta.get('bump')} "
            f"{meta.get('old_version')} → {meta.get('new_version')} "
            f"(since={meta.get('since')})",
            file=sys.stderr,
        )
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "updated" if changed else "unchanged"
    print(f"[{stamp}] CHANGELOG.md {status}")
    if meta.get("bump"):
        print(
            f"version {meta['old_display']} ({meta['old_version']}) → "
            f"{meta['new_display']} ({meta['new_version']}) "
            f"[{meta['bump']}]"
        )
    else:
        print(f"version unchanged at {meta['old_display']} ({meta['old_version']})")
    if meta.get("files_touched"):
        print("pins: " + ", ".join(meta["files_touched"]))
    print(f"log → {VERSION_LOG_PATH}")
    if STATE_PATH.exists():
        print(f"last_run → {STATE_PATH.read_text(encoding='utf-8').strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
