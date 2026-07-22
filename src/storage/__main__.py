"""CLI for the sample medical + salvage document corpus store.

Examples:

  # Initialize DB and seed realistic synthetic samples
  python -m src.storage seed --seed 42

  # Show corpus summary
  python -m src.storage summary

  # Export DICIE-compatible JSONL for training / eval
  python -m src.storage export --format docie --application salvage_claims \\
      --out data/sample_corpus/exports/salvage_docie.jsonl

  # Import existing DICIE fixtures / eval gold
  python -m src.storage import-jsonl --in tests/fixtures/sample_docie_documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.storage.sample_generator import generate_corpus
from src.storage.store import DocumentStore, default_db_path
from src.utils.config import Config, REPO_ROOT
from src.utils.io import write_json
from src.utils.provenance import ProvenanceRecord, log_provenance


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.storage",
        description=(
            "Queryable sample corpus for synthetic medical bills and salvage "
            "documentation (AmFam-style intake simulation)."
        ),
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite path (default: {default_db_path()})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="Generate and load synthetic sample documents")
    seed.add_argument("--seed", type=int, default=42)
    seed.add_argument("--medical-per-type", type=int, default=8)
    seed.add_argument("--salvage-per-type", type=int, default=8)
    seed.add_argument("--bundles-per-app", type=int, default=2)
    seed.add_argument(
        "--no-canonical",
        action="store_true",
        help="Skip embedding the canonical CI fixture documents",
    )
    seed.add_argument(
        "--also-export",
        action="store_true",
        help="Also write seed JSONL under data/sample_corpus/seeds/",
    )
    seed.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing DB file before seeding",
    )

    sub.add_parser("summary", help="Print corpus counts by application/type")

    lst = sub.add_parser("list", help="List documents (compact)")
    lst.add_argument("--application", choices=["medical_bills", "salvage_claims"])
    lst.add_argument("--document-type")
    lst.add_argument("--limit", type=int, default=50)

    show = sub.add_parser("show", help="Show one document by id")
    show.add_argument("document_id")

    exp = sub.add_parser("export", help="Export JSONL for training / DICIE")
    exp.add_argument(
        "--format",
        choices=["docie", "classification", "extraction"],
        default="docie",
    )
    exp.add_argument("--application", choices=["medical_bills", "salvage_claims"])
    exp.add_argument("--document-type")
    exp.add_argument("--split", choices=["train", "val", "test"])
    exp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path",
    )

    imp = sub.add_parser("import-jsonl", help="Import DICIE-style JSONL rows")
    imp.add_argument("--in", dest="inp", type=Path, required=True)
    imp.add_argument(
        "--source-kind",
        default="imported_jsonl",
        help="Provenance label for imported rows",
    )
    imp.add_argument("--application", default=None)
    imp.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip document_ids that already exist",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    db_path = args.db or default_db_path()

    if args.command == "seed":
        if args.reset and db_path.exists():
            db_path.unlink()
        store = DocumentStore(db_path)
        corpus = generate_corpus(
            seed=args.seed,
            medical_per_type=args.medical_per_type,
            salvage_per_type=args.salvage_per_type,
            bundles_per_app=args.bundles_per_app,
            include_canonical_fixtures=not args.no_canonical,
        )
        n = store.bulk_upsert(corpus.documents, claims=corpus.claims)
        store.add_provenance(
            stage="sample_corpus_seed",
            source="src.storage.sample_generator",
            detail={
                "seed": args.seed,
                "documents": n,
                "claims": len(corpus.claims),
                "medical_per_type": args.medical_per_type,
                "salvage_per_type": args.salvage_per_type,
                "bundles_per_app": args.bundles_per_app,
            },
        )
        log_provenance(
            Config.load().provenance_log_path,
            ProvenanceRecord(
                record_id=f"sample_corpus::seed::{args.seed}",
                stage="sample_corpus_seed",
                source="src.storage.sample_generator",
                prompt_version="sample_corpus_v1",
                model=None,
                extra={
                    "seed": args.seed,
                    "documents": n,
                    "claims": len(corpus.claims),
                    "db_path": str(db_path),
                },
            ),
        )
        if args.also_export:
            seed_dir = REPO_ROOT / "data" / "sample_corpus" / "seeds"
            seed_dir.mkdir(parents=True, exist_ok=True)
            out = seed_dir / f"seed_docie_s{args.seed}.jsonl"
            store.export_jsonl(out, format="docie")
            write_json(seed_dir / f"seed_summary_s{args.seed}.json", store.summary())
            print(f"Wrote seed export: {out}")
        print(json.dumps(store.summary(), indent=2))
        return 0

    store = DocumentStore(db_path)

    if args.command == "summary":
        print(json.dumps(store.summary(), indent=2))
        return 0

    if args.command == "list":
        docs = store.list_documents(
            application=args.application,
            document_type=args.document_type,
            limit=args.limit,
        )
        for d in docs:
            print(
                f"{d.document_id}\t{d.application}\t{d.document_type}\t"
                f"claim={d.claim_id}\tsplit={d.split}"
            )
        print(f"# {len(docs)} documents", file=sys.stderr)
        return 0

    if args.command == "show":
        doc = store.get_document(args.document_id)
        if doc is None:
            print(f"Document not found: {args.document_id}", file=sys.stderr)
            return 1
        print(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "export":
        n = store.export_jsonl(
            args.out,
            format=args.format,
            application=args.application,
            document_type=args.document_type,
            split=args.split,
        )
        store.add_provenance(
            stage="sample_corpus_export",
            source=str(args.out),
            detail={"format": args.format, "exported": n},
        )
        print(f"Exported {n} rows → {args.out}")
        return 0

    if args.command == "import-jsonl":
        n = store.import_docie_jsonl(
            args.inp,
            source_kind=args.source_kind,
            default_application=args.application,
            overwrite=not args.no_overwrite,
        )
        log_provenance(
            Config.load().provenance_log_path,
            ProvenanceRecord(
                record_id=f"sample_corpus::import::{Path(args.inp).name}",
                stage="sample_corpus_import",
                source=str(args.inp),
                prompt_version="sample_corpus_v1",
                model=None,
                extra={"imported": n, "db_path": str(db_path)},
            ),
        )
        print(f"Imported {n} rows from {args.inp}")
        print(json.dumps(store.summary(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
