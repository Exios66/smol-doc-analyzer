"""SQLite DDL for the queryable RVL-CDIP index."""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    label_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    label_id INTEGER NOT NULL,
    image_relpath TEXT NOT NULL,
    image_abspath TEXT,
    source_line INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    FOREIGN KEY (label_id) REFERENCES labels(label_id)
);

CREATE INDEX IF NOT EXISTS idx_rvl_documents_split
    ON documents(split);
CREATE INDEX IF NOT EXISTS idx_rvl_documents_label
    ON documents(label_id);
CREATE INDEX IF NOT EXISTS idx_rvl_documents_split_label
    ON documents(split, label_id);
CREATE INDEX IF NOT EXISTS idx_rvl_documents_relpath
    ON documents(image_relpath);

CREATE TABLE IF NOT EXISTS download_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    remote_ref TEXT NOT NULL,
    local_path TEXT NOT NULL,
    bytes INTEGER,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rvl_download_kind
    ON download_events(kind);
"""
