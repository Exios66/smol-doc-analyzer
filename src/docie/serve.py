"""
Lightweight FastAPI REST server for the DICIE pipeline (paper §VI).

Upstream systems POST PDF / image byte streams; the server runs Fig. 1
stages and returns the aggregated classification + extraction prediction.

Optional dependency: ``pip install fastapi uvicorn python-multipart``
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

from src.docie.applications import list_applications
from src.docie.pipeline import DociePipeline
from src.utils.config import Config

logger = logging.getLogger(__name__)


def create_app(
    application: str = "salvage_claims",
    *,
    cfg: Config | None = None,
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
        version="0.5.0",
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
            path = Path(tmp) / f"{record_id}{suffix}"
            path.write_bytes(raw)
            pipe = _pipe(application_name)
            if suffix == ".pdf":
                prediction = pipe.process(record_id=record_id, pdf_path=path)
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
                prediction = pipe.process(record_id=record_id, image_path=path)
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
