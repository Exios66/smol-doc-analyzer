"""Shared JSONL / profile helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def cache_safe_id(record_id: str) -> str:
    """Stable filesystem-safe id that avoids collisions across separators.

    Distinct IDs that only differ by ``::`` vs ``__`` (or path separators)
    must not map to the same filename.
    """
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:16]
    readable = (
        record_id.replace("::", "__")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )
    readable = re.sub(r"[^\w.%+-]+", "_", readable)[:80].strip("._") or "record"
    return f"{readable}__{digest}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))
