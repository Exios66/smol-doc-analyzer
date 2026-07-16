"""Guild-scoped note / transcript store for the Discord agent."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.config import REPO_ROOT

_LOCK = threading.Lock()


@dataclass
class Note:
    note_id: str
    guild_id: str
    channel_id: str
    author_id: str
    author_name: str
    kind: str  # note | transcript | reminder
    title: str
    body: str
    tags: list[str]
    created_at: float
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_db_path() -> Path:
    path = REPO_ROOT / "data" / "discord" / "notes.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class NotesStore:
    """SQLite-backed notes shared by slash commands and Chloride tools."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    guild_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_guild_created "
                "ON notes(guild_id, created_at DESC)"
            )
            conn.commit()

    def add(
        self,
        *,
        guild_id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        body: str,
        title: str = "",
        kind: str = "note",
        tags: Iterable[str] | None = None,
        source: str | None = None,
    ) -> Note:
        body = (body or "").strip()
        if not body:
            raise ValueError("Note body cannot be empty.")
        tag_list = [t.strip().lower() for t in (tags or []) if t and t.strip()]
        note = Note(
            note_id=uuid.uuid4().hex[:12],
            guild_id=str(guild_id or "dm"),
            channel_id=str(channel_id or "dm"),
            author_id=str(author_id),
            author_name=author_name or "unknown",
            kind=kind or "note",
            title=(title or "").strip() or _auto_title(body),
            body=body,
            tags=tag_list,
            created_at=time.time(),
            source=source,
        )
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notes (
                    note_id, guild_id, channel_id, author_id, author_name,
                    kind, title, body, tags, created_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    note.guild_id,
                    note.channel_id,
                    note.author_id,
                    note.author_name,
                    note.kind,
                    note.title,
                    note.body,
                    json.dumps(note.tags),
                    note.created_at,
                    note.source,
                ),
            )
            conn.commit()
        return note

    def list_recent(
        self,
        guild_id: str,
        *,
        kind: str | None = None,
        limit: int = 10,
    ) -> list[Note]:
        limit = max(1, min(int(limit), 50))
        sql = "SELECT * FROM notes WHERE guild_id = ?"
        params: list[Any] = [str(guild_id or "dm")]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with _LOCK, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_note(r) for r in rows]

    def search(self, guild_id: str, query: str, *, limit: int = 10) -> list[Note]:
        q = (query or "").strip().lower()
        if not q:
            return self.list_recent(guild_id, limit=limit)
        limit = max(1, min(int(limit), 50))
        like = f"%{q}%"
        with _LOCK, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM notes
                WHERE guild_id = ?
                  AND (
                    lower(title) LIKE ?
                    OR lower(body) LIKE ?
                    OR lower(tags) LIKE ?
                    OR lower(author_name) LIKE ?
                  )
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (str(guild_id or "dm"), like, like, like, like, limit),
            ).fetchall()
        return [_row_to_note(r) for r in rows]

    def get(self, note_id: str) -> Note | None:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE note_id = ?", (note_id,)
            ).fetchone()
        return _row_to_note(row) if row else None

    def delete(self, note_id: str, *, guild_id: str | None = None) -> bool:
        with _LOCK, self._connect() as conn:
            if guild_id is None:
                cur = conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            else:
                cur = conn.execute(
                    "DELETE FROM notes WHERE note_id = ? AND guild_id = ?",
                    (note_id, str(guild_id)),
                )
            conn.commit()
            return cur.rowcount > 0


def _auto_title(body: str) -> str:
    first = body.strip().splitlines()[0].strip()
    return (first[:72] + "…") if len(first) > 72 else first or "Untitled"


def _row_to_note(row: sqlite3.Row) -> Note:
    tags_raw = row["tags"] or "[]"
    try:
        tags = json.loads(tags_raw)
    except json.JSONDecodeError:
        tags = []
    return Note(
        note_id=row["note_id"],
        guild_id=row["guild_id"],
        channel_id=row["channel_id"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        kind=row["kind"],
        title=row["title"],
        body=row["body"],
        tags=list(tags),
        created_at=float(row["created_at"]),
        source=row["source"],
    )


def format_notes(notes: list[Note], *, heading: str = "Notes") -> str:
    if not notes:
        return f"## {heading}\n\n_(none)_"
    lines = [f"## {heading}", ""]
    for n in notes:
        tags = f" · tags: {', '.join(n.tags)}" if n.tags else ""
        lines.append(f"**`{n.note_id}`** · {n.kind} · {n.title}{tags}")
        preview = n.body if len(n.body) <= 280 else n.body[:277] + "…"
        lines.append(preview)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_STORE: NotesStore | None = None


def get_notes_store() -> NotesStore:
    global _STORE
    if _STORE is None:
        _STORE = NotesStore()
    return _STORE
