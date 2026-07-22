"""SQLite-backed store for sample medical and salvage insurance documents."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.storage.schema import DDL, SCHEMA_VERSION
from src.storage.types import ClaimRecord, DocumentRecord, FieldRecord
from src.utils.config import REPO_ROOT
from src.utils.io import write_jsonl

_LOCK = threading.Lock()


def default_db_path() -> Path:
    try:
        from src.utils.config import Config

        path = Config.load().sample_corpus_db_path
    except Exception:
        path = REPO_ROOT / "data" / "sample_corpus" / "documents.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj if obj is not None else {}, ensure_ascii=False)


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


class DocumentStore:
    """Queryable corpus for synthetic medical bills and salvage documentation.

    Designed to house AmFam-style sample documents for analysis, evaluation,
    and fine-tuning without using proprietary insurer data.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_db_path()
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
            conn.commit()

    # ------------------------------------------------------------------ claims
    def upsert_claim(self, claim: ClaimRecord) -> ClaimRecord:
        now = claim.created_at or time.time()
        claim.created_at = now
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (
                    claim_id, application, carrier_name, state, date_of_loss,
                    loss_type, policy_number, insured_name, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    application = excluded.application,
                    carrier_name = excluded.carrier_name,
                    state = excluded.state,
                    date_of_loss = excluded.date_of_loss,
                    loss_type = excluded.loss_type,
                    policy_number = excluded.policy_number,
                    insured_name = excluded.insured_name,
                    metadata_json = excluded.metadata_json
                """,
                (
                    claim.claim_id,
                    claim.application,
                    claim.carrier_name,
                    claim.state,
                    claim.date_of_loss,
                    claim.loss_type,
                    claim.policy_number,
                    claim.insured_name,
                    _json_dumps(claim.metadata),
                    claim.created_at,
                ),
            )
            conn.commit()
        return claim

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
        return self._row_to_claim(row) if row else None

    # --------------------------------------------------------------- documents
    def upsert_document(self, doc: DocumentRecord) -> DocumentRecord:
        now = time.time()
        if not doc.created_at:
            doc.created_at = now
        doc.updated_at = now
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, claim_id, application, document_type, title, text,
                    source_kind, is_synthetic, split, source_path,
                    skeleton_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    claim_id = excluded.claim_id,
                    application = excluded.application,
                    document_type = excluded.document_type,
                    title = excluded.title,
                    text = excluded.text,
                    source_kind = excluded.source_kind,
                    is_synthetic = excluded.is_synthetic,
                    split = excluded.split,
                    source_path = excluded.source_path,
                    skeleton_json = excluded.skeleton_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    doc.document_id,
                    doc.claim_id,
                    doc.application,
                    doc.document_type,
                    doc.title,
                    doc.text,
                    doc.source_kind,
                    1 if doc.is_synthetic else 0,
                    doc.split,
                    doc.source_path,
                    _json_dumps(doc.skeleton),
                    _json_dumps(doc.metadata),
                    doc.created_at,
                    doc.updated_at,
                ),
            )
            if doc.fields:
                # Replace semantics per role: omitted keys must not linger as
                # stale gold labels after a corrective re-import. An empty
                # fields list leaves existing labels untouched (document-only
                # upserts).
                roles = {f.field_role for f in doc.fields}
                for role in roles:
                    conn.execute(
                        "DELETE FROM document_fields WHERE document_id = ? AND field_role = ?",
                        (doc.document_id, role),
                    )
                for f in doc.fields:
                    conn.execute(
                        """
                        INSERT INTO document_fields (
                            document_id, field_name, field_value, field_role,
                            confidence, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc.document_id,
                            f.field_name,
                            f.field_value,
                            f.field_role,
                            f.confidence,
                            now,
                        ),
                    )
            conn.commit()
        return doc

    def set_fields(
        self,
        document_id: str,
        fields: dict[str, str | None],
        *,
        role: str = "ground_truth",
        confidence: float | None = None,
    ) -> None:
        now = time.time()
        with _LOCK, self._connect() as conn:
            for name, value in fields.items():
                conn.execute(
                    """
                    INSERT INTO document_fields (
                        document_id, field_name, field_value, field_role,
                        confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, field_name, field_role) DO UPDATE SET
                        field_value = excluded.field_value,
                        confidence = excluded.confidence
                    """,
                    (document_id, name, value, role, confidence, now),
                )
            conn.commit()

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            fields = conn.execute(
                """
                SELECT field_name, field_value, field_role, confidence
                FROM document_fields
                WHERE document_id = ?
                ORDER BY field_id
                """,
                (document_id,),
            ).fetchall()
        return self._row_to_document(row, fields)

    def list_documents(
        self,
        *,
        application: str | None = None,
        document_type: str | None = None,
        claim_id: str | None = None,
        split: str | None = None,
        source_kind: str | None = None,
        limit: int | None = None,
    ) -> list[DocumentRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if application:
            clauses.append("application = ?")
            params.append(application)
        if document_type:
            clauses.append("document_type = ?")
            params.append(document_type)
        if claim_id:
            clauses.append("claim_id = ?")
            params.append(claim_id)
        if split:
            clauses.append("split = ?")
            params.append(split)
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(source_kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM documents {where} ORDER BY created_at, document_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            docs: list[DocumentRecord] = []
            for row in rows:
                fields = conn.execute(
                    """
                    SELECT field_name, field_value, field_role, confidence
                    FROM document_fields
                    WHERE document_id = ?
                    ORDER BY field_id
                    """,
                    (row["document_id"],),
                ).fetchall()
                docs.append(self._row_to_document(row, fields))
        return docs

    def iter_documents(self, **filters: Any) -> Iterator[DocumentRecord]:
        yield from self.list_documents(**filters)

    def count_documents(
        self,
        *,
        application: str | None = None,
        document_type: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if application:
            clauses.append("application = ?")
            params.append(application)
        if document_type:
            clauses.append("document_type = ?")
            params.append(document_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM documents {where}", params
            ).fetchone()
        return int(row["n"])

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            by_app = conn.execute(
                """
                SELECT application, document_type, COUNT(*) AS n
                FROM documents
                GROUP BY application, document_type
                ORDER BY application, document_type
                """
            ).fetchall()
            claims = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
            docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            fields = conn.execute(
                "SELECT COUNT(*) AS n FROM document_fields"
            ).fetchone()["n"]
        return {
            "db_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
            "claims": int(claims),
            "documents": int(docs),
            "fields": int(fields),
            "by_application_type": [
                {
                    "application": r["application"],
                    "document_type": r["document_type"],
                    "count": int(r["n"]),
                }
                for r in by_app
            ],
        }

    # -------------------------------------------------------------- provenance
    def add_provenance(
        self,
        *,
        stage: str,
        source: str,
        document_id: str | None = None,
        claim_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO provenance_events (
                    document_id, claim_id, stage, source, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    claim_id,
                    stage,
                    source,
                    _json_dumps(detail or {}),
                    time.time(),
                ),
            )
            conn.commit()

    # ----------------------------------------------------------- import/export
    def import_docie_jsonl(
        self,
        path: Path | str,
        *,
        source_kind: str = "imported_jsonl",
        default_application: str | None = None,
        overwrite: bool = True,
    ) -> int:
        """Import DICIE-style JSONL rows into the store."""
        from src.utils.io import iter_jsonl

        n = 0
        for row in iter_jsonl(path):
            document_id = str(row.get("record_id") or row.get("document_id") or "")
            if not document_id:
                continue
            if not overwrite and self.get_document(document_id) is not None:
                continue
            application = str(
                row.get("application") or default_application or "unknown"
            )
            document_type = str(row.get("document_type") or row.get("label") or "other")
            text = str(row.get("text") or "")
            claim_id = row.get("claim_id")
            gt = row.get("ground_truth_fields") or row.get("fields") or {}
            if not isinstance(gt, dict):
                gt = {}
            # Infer claim_id from ground truth when present.
            if not claim_id and gt.get("claim_id"):
                claim_id = gt.get("claim_id")
            if claim_id:
                self.upsert_claim(
                    ClaimRecord(
                        claim_id=str(claim_id),
                        application=application,
                        carrier_name=str(
                            (row.get("metadata") or {}).get("carrier_name")
                            or "Synthetic Carrier"
                        ),
                        metadata={"imported_from": str(path)},
                    )
                )
            fields = [
                FieldRecord(field_name=str(k), field_value=None if v is None else str(v))
                for k, v in gt.items()
            ]
            self.upsert_document(
                DocumentRecord(
                    document_id=document_id,
                    claim_id=str(claim_id) if claim_id else None,
                    application=application,
                    document_type=document_type,
                    text=text,
                    source_kind=source_kind,
                    is_synthetic=True,
                    split=row.get("split"),
                    source_path=str(path),
                    skeleton=row.get("skeleton") or {},
                    metadata=row.get("metadata") or {},
                    fields=fields,
                )
            )
            n += 1
        self.add_provenance(
            stage="import_docie_jsonl",
            source=str(path),
            detail={"imported": n, "source_kind": source_kind},
        )
        return n

    def export_jsonl(
        self,
        path: Path | str,
        *,
        format: str = "docie",
        application: str | None = None,
        document_type: str | None = None,
        split: str | None = None,
    ) -> int:
        """Export documents for DICIE eval, classification, or extraction training."""
        docs = self.list_documents(
            application=application,
            document_type=document_type,
            split=split,
        )
        if format == "docie":
            rows = [d.to_docie_row() for d in docs]
        elif format == "classification":
            rows = [d.to_classification_row() for d in docs]
        elif format == "extraction":
            rows = [d.to_extraction_row() for d in docs]
        else:
            raise ValueError(
                f"Unknown export format {format!r}; "
                "expected docie|classification|extraction"
            )
        return write_jsonl(path, rows)

    def bulk_upsert(
        self,
        documents: Iterable[DocumentRecord],
        *,
        claims: Iterable[ClaimRecord] | None = None,
    ) -> int:
        n = 0
        if claims:
            for claim in claims:
                self.upsert_claim(claim)
        for doc in documents:
            self.upsert_document(doc)
            n += 1
        return n

    # --------------------------------------------------------------- converters
    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            claim_id=row["claim_id"],
            application=row["application"],
            carrier_name=row["carrier_name"],
            state=row["state"],
            date_of_loss=row["date_of_loss"],
            loss_type=row["loss_type"],
            policy_number=row["policy_number"],
            insured_name=row["insured_name"],
            metadata=_json_loads(row["metadata_json"], {}),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _row_to_document(
        row: sqlite3.Row, field_rows: Iterable[sqlite3.Row]
    ) -> DocumentRecord:
        fields = [
            FieldRecord(
                field_name=f["field_name"],
                field_value=f["field_value"],
                field_role=f["field_role"],
                confidence=f["confidence"],
            )
            for f in field_rows
        ]
        return DocumentRecord(
            document_id=row["document_id"],
            claim_id=row["claim_id"],
            application=row["application"],
            document_type=row["document_type"],
            title=row["title"],
            text=row["text"],
            source_kind=row["source_kind"],
            is_synthetic=bool(row["is_synthetic"]),
            split=row["split"],
            source_path=row["source_path"],
            skeleton=_json_loads(row["skeleton_json"], {}),
            metadata=_json_loads(row["metadata_json"], {}),
            fields=fields,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
