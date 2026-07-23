#!/usr/bin/env python3
"""Update CHANGELOG.md [Unreleased] from git history.

Manual:
  python scripts/update_changelog.py
  python scripts/update_changelog.py --dry-run
  python scripts/update_changelog.py --since 2026-07-15

Scheduled (macOS LaunchAgent, Wednesday 23:00 America/Chicago):
  ./scripts/install_changelog_launchagent.sh
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
STATE_PATH = REPO_ROOT / "data" / "changelog" / "last_run.txt"
TZ_NAME = "America/Chicago"

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
        stamp = STATE_PATH.read_text(encoding="utf-8").strip()
        if stamp:
            return stamp
    released = _latest_released_date(changelog)
    if released:
        return released
    # Fall back to ~2 weeks
    return _now_central().strftime("%Y-%m-%d")


def _classify(subject: str) -> str:
    s = subject.strip()
    lower = s.lower()
    # conventional commits
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
    # heuristics on free-form subjects
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
    # Drop trailing PR refs like (#41) — keep them; they're useful
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s.rstrip(".")


def _collect_commits(since: str) -> list[tuple[str, str, str]]:
    """Return list of (sha, date, subject) after ``since`` (exclusive-ish)."""
    # Use since midnight on that date in Central time via git's --since
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
        # Skip pure changelog maintenance commits to avoid feedback loops
        sub_l = subject.lower()
        if "changelog" in sub_l and any(
            w in sub_l for w in ("update", "sync", "auto", "refresh")
        ):
            continue
        rows.append((sha, date, subject))
    return rows


def _parse_existing_unreleased(body: str) -> dict[str, list[str]]:
    """Parse Keep-a-Changelog bullets, preserving wrapped continuation lines."""
    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in body.splitlines():
        heading = re.match(r"^###\s+(\w+)\s*$", line)
        if heading:
            current = heading.group(1)
            continue
        if current is None:
            continue
        if line.startswith("- "):
            sections[current].append(line[2:].rstrip())
            continue
        # Continuation of a wrapped bullet (indented or plain prose line)
        if sections[current] and line.strip() and not line.startswith("#"):
            sections[current][-1] += "\n" + line.rstrip()
    return sections


def _normalize_bullet(text: str) -> str:
    # First line only; strip trailing "(abc1234)" auto suffixes for dedupe
    first = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s*\([0-9a-f]{7,40}\)\s*$", "", first).lower()


def _format_bullet(item: str) -> list[str]:
    """Render a bullet that may contain preserved newlines."""
    parts = item.splitlines() or [""]
    out = [f"- {parts[0]}"]
    for cont in parts[1:]:
        out.append(cont)
    return out


def _build_unreleased_body(
    existing: dict[str, list[str]],
    commits: list[tuple[str, str, str]],
) -> str:
    merged: dict[str, list[str]] = {k: list(v) for k, v in existing.items()}
    seen = {_normalize_bullet(b) for items in merged.values() for b in items}

    for sha, _date, subject in commits:
        section = _classify(subject)
        cleaned = _clean_subject(subject)
        bullet = f"{cleaned} ({sha})"
        key = _normalize_bullet(bullet)
        key_nosha = _normalize_bullet(cleaned)
        # Skip near-duplicates of hand-written bullets (substring match on first line)
        if key in seen or key_nosha in seen:
            continue
        if any(key_nosha[:48] in s or s[:48] in key_nosha for s in seen if len(s) > 20):
            continue
        merged.setdefault(section, []).append(bullet)
        seen.add(key)
        seen.add(key_nosha)

    stamp = _now_central().strftime("%Y-%m-%d %H:%M %Z")
    lines: list[str] = [
        f"<!-- changelog-auto: {stamp} · {len(commits)} commit(s) scanned -->",
        "",
    ]
    wrote_any = False
    for section in SECTION_ORDER:
        items = merged.get(section) or []
        if not items:
            continue
        wrote_any = True
        lines.append(f"### {section}")
        lines.append("")
        for item in items:
            lines.extend(_format_bullet(item))
        lines.append("")

    if not wrote_any:
        lines.extend(
            [
                "_No notable commits since the last changelog update._",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def update_changelog(
    *,
    since: str | None = None,
    dry_run: bool = False,
    write_state: bool = True,
) -> tuple[str, bool]:
    if not CHANGELOG_PATH.exists():
        raise FileNotFoundError(f"Missing {CHANGELOG_PATH}")

    original = CHANGELOG_PATH.read_text(encoding="utf-8")
    since_val = since or _default_since(original)
    commits = _collect_commits(since_val)

    match = UNRELEASED_RE.search(original)
    if not match:
        raise RuntimeError("CHANGELOG.md has no ## [Unreleased] section")

    existing_body = AUTO_MARKER_RE.sub("", match.group(2))
    existing = _parse_existing_unreleased(existing_body)
    new_body = _build_unreleased_body(existing, commits)
    updated = (
        original[: match.start(1)]
        + match.group(1)
        + new_body
        + "\n"
        + original[match.end() :]
    )
    # Collapse excessive blank lines around the splice
    updated = re.sub(r"\n{3,}", "\n\n", updated)

    changed = updated != original
    if dry_run:
        return updated, changed

    if changed:
        CHANGELOG_PATH.write_text(updated, encoding="utf-8")

    if write_state:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            _now_central().date().isoformat() + "\n",
            encoding="utf-8",
        )

    return updated, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh CHANGELOG.md [Unreleased] from git commits. "
            "Safe to run manually or via the Wednesday 11pm Central LaunchAgent."
        )
    )
    parser.add_argument(
        "--since",
        help="Git --since date (YYYY-MM-DD). Default: last run stamp or latest release date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the updated Unreleased section; do not write files.",
    )
    parser.add_argument(
        "--print-unreleased",
        action="store_true",
        help="With --dry-run, print only the Unreleased block.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not update data/changelog/last_run.txt",
    )
    args = parser.parse_args(argv)

    try:
        updated, changed = update_changelog(
            since=args.since,
            dry_run=args.dry_run,
            write_state=not args.no_state and not args.dry_run,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        if args.print_unreleased:
            m = UNRELEASED_RE.search(updated)
            print(m.group(0) if m else updated)
        else:
            print(updated)
        print(
            f"\n# dry-run: {'would change' if changed else 'no change'} "
            f"(since={args.since or _default_since(CHANGELOG_PATH.read_text(encoding='utf-8'))})",
            file=sys.stderr,
        )
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "updated" if changed else "unchanged"
    print(f"[{stamp}] CHANGELOG.md {status} → {CHANGELOG_PATH}")
    if STATE_PATH.exists():
        print(f"last_run → {STATE_PATH.read_text(encoding='utf-8').strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
