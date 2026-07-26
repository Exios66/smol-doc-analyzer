"""SQLite-backed queryable index over RVL-CDIP labels (and optional images)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.rvl_cdip.download import (
    DownloadResult,
    ensure_labels,
    iter_label_rows,
    label_file_paths,
)
from src.rvl_cdip.paths import (
    HF_DATASET_ID,
    LABEL_NAMES,
    assert_path_under_venv,
    default_db_path,
    images_dir,
    rvl_root,
)
from src.rvl_cdip.schema import DDL, SCHEMA_VERSION

_LOCK = threading.Lock()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj if obj is not None else {}, ensure_ascii=False)


class RvlCdipStore:
    """Queryable SQL house for RVL-CDIP document metadata.

    The database and all Hub downloads stay under ``.venv/rvl_cdip/``. Building
    the index uses only the small label files by default (~400k rows, ~17 MB
    download); image bytes are optional.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        path = Path(db_path) if db_path else default_db_path()
        # Default path is under .venv; custom paths (tests) may be elsewhere.
        if db_path is None:
            assert_path_under_venv(path)
        self.db_path = path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.executescript(DDL)
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            # Seed the 16 RVL-CDIP class labels.
            conn.executemany(
                "INSERT OR IGNORE INTO labels(label_id, name) VALUES (?, ?)",
                list(enumerate(LABEL_NAMES)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("dataset_id", HF_DATASET_ID),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("artifact_root", str(rvl_root())),
            )
            conn.commit()

    # ---------------------------------------------------------------- downloads
    def record_download(self, result: DownloadResult) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO download_events (
                    kind, remote_ref, local_path, bytes, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.kind,
                    result.remote_ref,
                    str(result.local_path),
                    result.bytes,
                    _json_dumps({**result.detail, "skipped": result.skipped}),
                    time.time(),
                ),
            )
            conn.commit()

    # -------------------------------------------------------------------- build
    def build_from_labels(
        self,
        *,
        force_download: bool = False,
        reset: bool = False,
        batch_size: int = 5_000,
    ) -> dict[str, Any]:
        """Download label files into ``.venv`` (if needed) and populate SQL.

        Does **not** download the image archive.
        """
        results = ensure_labels(force=force_download)
        for r in results:
            self.record_download(r)

        paths = label_file_paths()
        missing = [s for s, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"RVL-CDIP label files missing after download: {missing}"
            )

        if reset:
            with _LOCK, self._connect() as conn:
                conn.execute("DELETE FROM documents")
                conn.commit()

        counts: dict[str, int] = {}
        t0 = time.time()
        for split, path in paths.items():
            counts[split] = self._ingest_label_file(
                path, split=split, batch_size=batch_size
            )

        total = sum(counts.values())
        with _LOCK, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("document_count", str(total)),
            )
            conn.commit()

        return {
            "documents": total,
            "by_split": counts,
            "db_path": str(self.db_path),
            "source_labels": {k: str(v) for k, v in paths.items()},
            "elapsed_s": round(time.time() - t0, 3),
            "downloads": [
                {
                    "kind": r.kind,
                    "local_path": str(r.local_path),
                    "bytes": r.bytes,
                    "skipped": r.skipped,
                }
                for r in results
            ],
        }

    def _ingest_label_file(
        self, path: Path, *, split: str, batch_size: int
    ) -> int:
        now = time.time()
        img_root = images_dir()
        batch: list[tuple[Any, ...]] = []
        n = 0

        def flush(conn: sqlite3.Connection) -> None:
            if not batch:
                return
            conn.executemany(
                """
                INSERT INTO documents (
                    document_id, split, label_id, image_relpath, image_abspath,
                    source_line, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    split = excluded.split,
                    label_id = excluded.label_id,
                    image_relpath = excluded.image_relpath,
                    image_abspath = excluded.image_abspath,
                    source_line = excluded.source_line,
                    metadata_json = excluded.metadata_json
                """,
                batch,
            )
            batch.clear()

        with _LOCK, self._connect() as conn:
            for relpath, label_id, line_no in iter_label_rows(path, split=split):
                if label_id < 0 or label_id >= len(LABEL_NAMES):
                    raise ValueError(
                        f"Invalid label_id {label_id} in {path}#{line_no}"
                    )
                # document_id is stable across rebuilds: split + relpath
                document_id = f"{split}:{relpath}"
                abs_candidate = img_root / relpath
                image_abspath = str(abs_candidate) if abs_candidate.is_file() else None
                batch.append(
                    (
                        document_id,
                        split,
                        label_id,
                        relpath,
                        image_abspath,
                        line_no,
                        "{}",
                        now,
                    )
                )
                n += 1
                if len(batch) >= batch_size:
                    flush(conn)
            flush(conn)
            conn.commit()
        return n

    def refresh_image_paths(self) -> int:
        """Update ``image_abspath`` for rows whose files exist under source/images."""
        img_root = images_dir()
        updated = 0
        with _LOCK, self._connect() as conn:
            rows = conn.execute(
                "SELECT document_id, image_relpath FROM documents"
            ).fetchall()
            for row in rows:
                candidate = img_root / row["image_relpath"]
                if candidate.is_file():
                    conn.execute(
                        "UPDATE documents SET image_abspath = ? WHERE document_id = ?",
                        (str(candidate), row["document_id"]),
                    )
                    updated += 1
            conn.commit()
        return updated

    # ------------------------------------------------------------------- query
    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            by_split = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT split, COUNT(*) AS n
                    FROM documents
                    GROUP BY split
                    ORDER BY split
                    """
                )
            ]
            by_label = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT d.label_id, l.name AS label, COUNT(*) AS n
                    FROM documents d
                    JOIN labels l ON l.label_id = d.label_id
                    GROUP BY d.label_id
                    ORDER BY d.label_id
                    """
                )
            ]
            by_split_label = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT d.split, d.label_id, l.name AS label, COUNT(*) AS n
                    FROM documents d
                    JOIN labels l ON l.label_id = d.label_id
                    GROUP BY d.split, d.label_id
                    ORDER BY d.split, d.label_id
                    """
                )
            ]
            meta = {
                r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM schema_meta")
            }
            downloads = conn.execute(
                "SELECT COUNT(*) AS n FROM download_events"
            ).fetchone()["n"]
            with_images = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE image_abspath IS NOT NULL"
            ).fetchone()["n"]
        return {
            "documents": total,
            "with_image_abspath": with_images,
            "by_split": by_split,
            "by_label": by_label,
            "by_split_label": by_split_label,
            "schema_version": int(meta.get("schema_version", SCHEMA_VERSION)),
            "dataset_id": meta.get("dataset_id", HF_DATASET_ID),
            "artifact_root": meta.get("artifact_root"),
            "built_at": meta.get("built_at"),
            "download_events": downloads,
            "db_path": str(self.db_path),
        }

    def list_documents(
        self,
        *,
        split: str | None = None,
        label: str | int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if split:
            clauses.append("d.split = ?")
            params.append(split)
        if label is not None:
            if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
                clauses.append("d.label_id = ?")
                params.append(int(label))
            else:
                clauses.append("l.name = ?")
                params.append(str(label))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT d.document_id, d.split, d.label_id, l.name AS label,
                   d.image_relpath, d.image_abspath, d.source_line
            FROM documents d
            JOIN labels l ON l.label_id = d.label_id
            {where}
            ORDER BY d.split, d.label_id, d.source_line
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.*, l.name AS label
                FROM documents d
                JOIN labels l ON l.label_id = d.label_id
                WHERE d.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def query(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        max_rows: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Run a read-only SQL SELECT against the RVL-CDIP database."""
        stripped = sql.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed via RvlCdipStore.query()")
        if ";" in stripped:
            raise ValueError("Multiple SQL statements are not allowed")
        lowered = f" {stripped.lower()} "
        for bad in (" attach ", " pragma ", " insert ", " update ", " delete ", " drop ", " alter "):
            if bad in lowered:
                raise ValueError(f"Disallowed SQL construct in query: {bad.strip()}")
        with self._connect() as conn:
            cur = conn.execute(stripped, tuple(params or ()))
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
        return [dict(zip(cols, row)) for row in rows]

    def iter_documents(
        self,
        *,
        split: str | None = None,
        label_id: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if split:
            clauses.append("d.split = ?")
            params.append(split)
        if label_id is not None:
            clauses.append("d.label_id = ?")
            params.append(label_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT d.document_id, d.split, d.label_id, l.name AS label,
                   d.image_relpath, d.image_abspath
            FROM documents d
            JOIN labels l ON l.label_id = d.label_id
            {where}
            ORDER BY d.source_line
        """
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                yield dict(row)

    def labels(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT label_id, name FROM labels ORDER BY label_id"
                )
            ]
