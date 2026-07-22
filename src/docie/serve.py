"""
Lightweight FastAPI REST server for the DICIE pipeline (paper §VI).

Upstream systems POST PDF / image byte streams; the server runs Fig. 1
stages and returns the aggregated classification + extraction prediction.

Optional dependency: ``pip install fastapi uvicorn python-multipart``
"""

import argparse
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from src.docie.applications import list_applications
from src.docie.pipeline import DociePipeline
from src.utils.config import Config

logger = logging.getLogger(__name__)

_SAFE_RECORD_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_upload_record_id(record_id: str) -> str:
    """Collapse ``record_id`` to a basename-safe token (no path separators)."""
    raw = (record_id or "upload").strip() or "upload"
    # Reject absolute / parent-traversal attempts before basename cleanup.
    base = Path(raw).name
    cleaned = _SAFE_RECORD_ID.sub("_", base).strip("._") or "upload"
    return cleaned[:120]


def _safe_upload_path(tmp_dir: Path, record_id: str, suffix: str) -> Tuple[str, Path]:
    """Return ``(safe_record_id, path)`` guaranteed to stay under ``tmp_dir``."""
    safe_id = _sanitize_upload_record_id(record_id)
    # Normalize suffix to a simple extension token.
    safe_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix or "") else ".bin"
    root = tmp_dir.resolve()
    path = (root / f"{safe_id}{safe_suffix}").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Upload path escaped temp directory: {path}")
    return safe_id, path


def create_app(
    application: str = "salvage_claims",
    *,
    cfg: Optional[Config] = None,
) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for DICIE serving. "
            "Install with: pip install fastapi uvicorn python-multipart"
        ) from exc

    cfg = cfg or Config.load()
    pipelines: dict[str, DociePipeline] = {}

    def _pipe(name: str) -> DociePipeline:
        if name not in pipelines:
            pipelines[name] = DociePipeline(application=name, cfg=cfg)
        return pipelines[name]

    app = FastAPI(
        title="DICIE Document Pipeline",
        description=(
            "Document Image Classification and Information Extraction "
            "(Raj et al. Fig. 1): processing → classification → extraction → response"
        ),
        version="1.0.0b0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "applications": list_applications(),
            "default_application": application,
        }

    @app.post("/v1/predict")
    async def predict(
        file: UploadFile = File(...),
        application_name: str = Form(default=application),
        record_id: str = Form(default="upload"),
        response_only: bool = Form(default=True),
    ) -> JSONResponse:
        if application_name not in list_applications():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown application {application_name!r}",
            )
        suffix = Path(file.filename or "upload.bin").suffix.lower() or ".bin"
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        with tempfile.TemporaryDirectory(prefix="docie_upload_") as tmp:
            try:
                safe_id, path = _safe_upload_path(Path(tmp), record_id, suffix)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            path.write_bytes(raw)
            pipe = _pipe(application_name)
            if suffix == ".pdf":
                prediction = pipe.process(record_id=safe_id, pdf_path=path)
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
                prediction = pipe.process(record_id=safe_id, image_path=path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type {suffix!r}; upload PDF or image",
                )

        payload = (
            prediction.response_payload() if response_only else prediction.to_dict()
        )
        return JSONResponse(payload)

    @app.post("/v1/predict/text")
    async def predict_text(body: dict[str, Any]) -> JSONResponse:
        app_name = str(body.get("application") or application)
        if app_name not in list_applications():
            raise HTTPException(status_code=400, detail=f"Unknown application {app_name!r}")
        text = str(body.get("text") or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        record_id = str(body.get("record_id") or "adhoc")
        prediction = _pipe(app_name).process(record_id=record_id, text=text)
        response_only = bool(body.get("response_only", True))
        payload = (
            prediction.response_payload() if response_only else prediction.to_dict()
        )
        return JSONResponse(payload)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Serve the DICIE Fig. 1 pipeline via FastAPI")
    parser.add_argument("--application", "-a", default="salvage_claims", choices=list_applications())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "uvicorn is required. Install with: pip install fastapi uvicorn python-multipart"
        ) from exc
    app = create_app(application=args.application)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
