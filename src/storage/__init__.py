"""Queryable sample-document corpus for medical bills and salvage claims."""

from src.storage.schema import SCHEMA_VERSION
from src.storage.store import DocumentStore
from src.storage.types import ClaimRecord, DocumentRecord, FieldRecord

__all__ = [
    "SCHEMA_VERSION",
    "DocumentStore",
    "ClaimRecord",
    "DocumentRecord",
    "FieldRecord",
]
