"""CLI for the RVL-CDIP SQLite index.

Examples:

  # Download label files into .venv and build the SQL index (~400k rows)
  python -m src.rvl_cdip build

  # Inspect
  python -m src.rvl_cdip summary
  python -m src.rvl_cdip list --split train --label invoice --limit 5
  python -m src.rvl_cdip query "SELECT label, COUNT(*) AS n FROM documents d JOIN labels l ON l.label_id = d.label_id GROUP BY label ORDER BY n DESC"

  # Optional: download the ~38 GB image archive (explicit dual opt-in)
  python -m src.rvl_cdip download-images --preflight
  python -m src.rvl_cdip download-images \\
    --i-understand-large-download --confirm-writes-under-venv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.rvl_cdip.download import (
    download_images,
    download_labels,
    format_image_download_preflight,
    free_bytes,
    image_download_preflight,
)
from src.rvl_cdip.paths import (
    HF_DATASET_ID,
    IMAGE_ARCHIVE_BYTES_ESTIMATE,
    IMAGE_DOWNLOAD_MIN_FREE_BYTES,
    apply_hf_cache_env,
    default_db_path,
    rvl_root,
)
from src.rvl_cdip.store import RvlCdipStore
from src.utils.config import Config
from src.utils.provenance import ProvenanceRecord, log_provenance


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.rvl_cdip",
        description=(
            "Queryable SQL database for Hugging Face aharley/rvl_cdip. "
            "All downloads stay under .venv/rvl_cdip/."
        ),
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SQLite path (default: {default_db_path()})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    dl = sub.add_parser(
        "download-labels",
        help="Download train/test/val label files into .venv only (~17 MB)",
    )
    dl.add_argument("--force", action="store_true", help="Re-download even if present")

    img = sub.add_parser(
        "download-images",
        help=(
            f"Download rvl-cdip.tar.gz into .venv/rvl_cdip only "
            f"(~{IMAGE_ARCHIVE_BYTES_ESTIMATE / (1024**3):.0f} GB; dual opt-in)"
        ),
    )
    img.add_argument(
        "--preflight",
        action="store_true",
        help="Print paths + free-space check only (no download)",
    )
    img.add_argument(
        "--i-understand-large-download",
        action="store_true",
        help="Required acknowledgement that the archive is ~38 GB",
    )
    img.add_argument(
        "--confirm-writes-under-venv",
        action="store_true",
        help=(
            "Required acknowledgement that Hub cache + archive write only under "
            ".venv/rvl_cdip/ (not data/ or ~/.cache)"
        ),
    )
    img.add_argument(
        "--interactive-confirm",
        action="store_true",
        help=(
            "Prompt on stdin to type the ack phrase before downloading "
            "(still requires --i-understand-large-download)"
        ),
    )
    img.add_argument("--force", action="store_true")

    build = sub.add_parser(
        "build",
        help="Download labels (if needed) and populate the SQLite index",
    )
    build.add_argument("--force-download", action="store_true")
    build.add_argument(
        "--reset",
        action="store_true",
        help="Clear documents table before ingest",
    )

    sub.add_parser("summary", help="Print counts by split / label")

    lst = sub.add_parser("list", help="List documents (compact)")
    lst.add_argument("--split", choices=["train", "test", "validation"])
    lst.add_argument("--label", help="Class name or integer label id")
    lst.add_argument("--limit", type=int, default=20)
    lst.add_argument("--offset", type=int, default=0)

    show = sub.add_parser("show", help="Show one document by id")
    show.add_argument("document_id")

    q = sub.add_parser("query", help="Run a read-only SELECT against the DB")
    q.add_argument("sql", help="SELECT statement")
    q.add_argument("--max-rows", type=int, default=100)

    sub.add_parser("labels", help="List the 16 RVL-CDIP class labels")

    paths = sub.add_parser("paths", help="Show .venv artifact locations")
    paths.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON",
    )

    return p


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    apply_hf_cache_env()
    db_path = args.db or default_db_path()

    if args.command == "paths":
        info = {
            "dataset_id": HF_DATASET_ID,
            "rvl_root": str(rvl_root()),
            "db_path": str(db_path),
            "hf_env": apply_hf_cache_env(),
            "free_bytes": free_bytes(),
            "image_archive_estimate_bytes": IMAGE_ARCHIVE_BYTES_ESTIMATE,
            "image_download_min_free_bytes": IMAGE_DOWNLOAD_MIN_FREE_BYTES,
        }
        if args.as_json:
            _print_json(info)
        else:
            for k, v in info.items():
                print(f"{k}: {v}")
        return 0

    if args.command == "download-labels":
        results = download_labels(force=args.force)
        store = RvlCdipStore(db_path)
        for r in results:
            store.record_download(r)
        _print_json(
            [
                {
                    "kind": r.kind,
                    "remote_ref": r.remote_ref,
                    "local_path": str(r.local_path),
                    "bytes": r.bytes,
                    "skipped": r.skipped,
                }
                for r in results
            ]
        )
        return 0

    if args.command == "download-images":
        plan = image_download_preflight(force=args.force)
        if args.preflight:
            print(format_image_download_preflight(plan))
            _print_json(plan)
            return 0 if plan["enough_free_space"] or plan["archive_already_present"] else 2

        print(format_image_download_preflight(plan))
        print()

        confirmed = bool(args.confirm_writes_under_venv)
        if args.interactive_confirm:
            phrase = str(plan["confirmation_phrase"])
            print(f'Type exactly: {phrase}')
            try:
                typed = input("> ").strip()
            except EOFError:
                typed = ""
            if typed != phrase:
                print(
                    "Confirmation phrase did not match; aborting download.",
                    file=sys.stderr,
                )
                return 2
            confirmed = True

        if not args.i_understand_large_download:
            print(
                "Refusing to download rvl-cdip.tar.gz (~38 GB). "
                "Re-run with --i-understand-large-download after checking disk space.\n"
                "Also pass --confirm-writes-under-venv (or --interactive-confirm).",
                file=sys.stderr,
            )
            return 2
        if not confirmed:
            print(
                "Refusing to download without --confirm-writes-under-venv "
                "(writes stay under .venv/rvl_cdip only). "
                "Preview: python -m src.rvl_cdip download-images --preflight",
                file=sys.stderr,
            )
            return 2

        try:
            result = download_images(
                force=args.force,
                i_understand_large_download=True,
                confirm_writes_under_venv=True,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        store = RvlCdipStore(db_path)
        store.record_download(result)
        n = store.refresh_image_paths()
        _print_json(
            {
                "kind": result.kind,
                "local_path": str(result.local_path),
                "bytes": result.bytes,
                "skipped": result.skipped,
                "image_abspath_updated": n,
                "writes_only_under": plan["writes_only_under"],
                "detail": result.detail,
            }
        )
        return 0

    if args.command == "build":
        store = RvlCdipStore(db_path)
        stats = store.build_from_labels(
            force_download=args.force_download,
            reset=args.reset,
        )
        try:
            log_provenance(
                Config.load().provenance_log_path,
                ProvenanceRecord(
                    record_id="rvl_cdip::build",
                    stage="rvl_cdip_build",
                    source=HF_DATASET_ID,
                    prompt_version="rvl_cdip_sql_v1",
                    model=None,
                    extra={
                        "documents": stats["documents"],
                        "by_split": stats["by_split"],
                        "db_path": stats["db_path"],
                        "artifact_root": str(rvl_root()),
                    },
                ),
            )
        except Exception:  # noqa: BLE001 — provenance must not block build
            pass
        _print_json(stats)
        return 0

    store = RvlCdipStore(db_path)

    if args.command == "summary":
        _print_json(store.summary())
        return 0

    if args.command == "labels":
        _print_json(store.labels())
        return 0

    if args.command == "list":
        rows = store.list_documents(
            split=args.split,
            label=args.label,
            limit=args.limit,
            offset=args.offset,
        )
        _print_json(rows)
        return 0

    if args.command == "show":
        doc = store.get_document(args.document_id)
        if doc is None:
            print(f"document not found: {args.document_id}", file=sys.stderr)
            return 1
        _print_json(doc)
        return 0

    if args.command == "query":
        rows = store.query(args.sql, max_rows=args.max_rows)
        _print_json(rows)
        return 0

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
