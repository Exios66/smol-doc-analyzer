"""Queryable sample-document corpus for medical bills and salvage claims."""

from src.storage.schema import SCHEMA_VERSION
from src.storage.store import DocumentStore
from src.storage.types import ClaimRecord, DocumentRecord, FieldRecord
from src.storage.training import (
    fit_tfidf_random_forest,
    prepare_both_applications,
    prepare_classification_dataset,
    prepare_extraction_dataset,
)

__all__ = [
    "SCHEMA_VERSION",
    "DocumentStore",
    "ClaimRecord",
    "DocumentRecord",
    "FieldRecord",
    "prepare_classification_dataset",
    "prepare_extraction_dataset",
    "prepare_both_applications",
    "fit_tfidf_random_forest",
]
