"""SQLite DDL for the sample medical + salvage document corpus."""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    application TEXT NOT NULL,
    carrier_name TEXT NOT NULL,
    state TEXT,
    date_of_loss TEXT,
    loss_type TEXT,
    policy_number TEXT,
    insured_name TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_application
    ON claims(application);
CREATE INDEX IF NOT EXISTS idx_claims_carrier
    ON claims(carrier_name);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    claim_id TEXT,
    application TEXT NOT NULL,
    document_type TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'synthetic_seed',
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    split TEXT,
    source_path TEXT,
    skeleton_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_application_type
    ON documents(application, document_type);
CREATE INDEX IF NOT EXISTS idx_documents_claim
    ON documents(claim_id);
CREATE INDEX IF NOT EXISTS idx_documents_split
    ON documents(split);
CREATE INDEX IF NOT EXISTS idx_documents_source_kind
    ON documents(source_kind);

CREATE TABLE IF NOT EXISTS document_fields (
    field_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_value TEXT,
    field_role TEXT NOT NULL DEFAULT 'ground_truth',
    confidence REAL,
    created_at REAL NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE (document_id, field_name, field_role)
);

CREATE INDEX IF NOT EXISTS idx_document_fields_name
    ON document_fields(field_name);
CREATE INDEX IF NOT EXISTS idx_document_fields_document
    ON document_fields(document_id);

CREATE TABLE IF NOT EXISTS document_pages (
    page_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    image_path TEXT,
    width INTEGER,
    height INTEGER,
    dpi INTEGER,
    ocr_text TEXT,
    words_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE (document_id, page_index)
);

CREATE TABLE IF NOT EXISTS provenance_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    claim_id TEXT,
    stage TEXT NOT NULL,
    source TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE SET NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_provenance_document
    ON provenance_events(document_id);
CREATE INDEX IF NOT EXISTS idx_provenance_stage
    ON provenance_events(stage);
"""
