"""
Versioned prompt template loader.

Templates live under ``evaluation/prompts/`` (eval harness) and optionally
``src/prompts/``. Lookup order for ``load_prompt(name, version)``:

  1. ``{root}/{name}/{version}.txt``   (versioned directory layout)
  2. ``{root}/{name}.txt``             when version is ``latest`` (flat file)
  3. ``{root}/{name}/latest.txt``
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import REPO_ROOT

_PROMPT_ROOTS = (
    REPO_ROOT / "evaluation" / "prompts",
    REPO_ROOT / "src" / "prompts",
)


def load_prompt(name: str, version: str = "latest") -> str:
    """Load a prompt template by name and version string."""
    candidates: list[Path] = []
    for root in _PROMPT_ROOTS:
        candidates.append(root / name / f"{version}.txt")
        if version == "latest":
            candidates.append(root / f"{name}.txt")
            candidates.append(root / name / "latest.txt")
        else:
            candidates.append(root / f"{name}_{version}.txt")

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Prompt template '{name}' (version={version}) not found. Looked in: {searched}"
    )
