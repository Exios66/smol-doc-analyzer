"""Tests for the RVL-CDIP SQLite index (downloads mocked; no Hub I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rvl_cdip import paths as rvl_paths
from src.rvl_cdip.download import DownloadResult, iter_label_rows
from src.rvl_cdip.store import RvlCdipStore


def _write_label_file(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{rel} {label}" for rel, label in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def venv_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RVL-CDIP artifact roots at a fake ``.venv`` under tmp_path."""
    fake_repo = tmp_path / "repo"
    fake_venv = fake_repo / ".venv"
    fake_venv.mkdir(parents=True)
    monkeypatch.setattr(rvl_paths, "REPO_ROOT", fake_repo)

    # Bypass Config so default_db_path / rvl_root use the monkeypatched REPO_ROOT.
    monkeypatch.setattr(
        "src.rvl_cdip.paths.default_db_path",
        lambda: fake_venv / "rvl_cdip" / "rvl_cdip.db",
    )
    monkeypatch.setattr(
        "src.rvl_cdip.store.default_db_path",
        lambda: fake_venv / "rvl_cdip" / "rvl_cdip.db",
    )
    return fake_repo


def test_assert_path_under_venv(venv_layout: Path):
    inside = venv_layout / ".venv" / "rvl_cdip" / "x"
    inside.parent.mkdir(parents=True)
    assert rvl_paths.assert_path_under_venv(inside) == inside.resolve()
    with pytest.raises(RuntimeError, match="must stay under"):
        rvl_paths.assert_path_under_venv(venv_layout / "data" / "escape.db")


def test_iter_label_rows_parses(tmp_path: Path):
    path = tmp_path / "train.txt"
    _write_label_file(
        path,
        [
            ("imagesa/a/b/c/abc00c00/000.tif", 11),
            ("imagesq/q/o/e/qoe09e00/500.tif", 15),
        ],
    )
    rows = list(iter_label_rows(path, split="train"))
    assert rows == [
        ("imagesa/a/b/c/abc00c00/000.tif", 11, 1),
        ("imagesq/q/o/e/qoe09e00/500.tif", 15, 2),
    ]


def test_build_from_local_labels(venv_layout: Path, monkeypatch: pytest.MonkeyPatch):
    labels = rvl_paths.labels_dir()
    _write_label_file(
        labels / "train.txt",
        [
            ("a/doc1.tif", 0),
            ("a/doc2.tif", 11),
            ("a/doc3.tif", 15),
        ],
    )
    _write_label_file(
        labels / "test.txt",
        [("b/doc4.tif", 1)],
    )
    _write_label_file(
        labels / "val.txt",
        [("c/doc5.tif", 2)],
    )

    monkeypatch.setattr(
        "src.rvl_cdip.store.ensure_labels",
        lambda force=False: [
            DownloadResult(
                kind="labels",
                remote_ref="local",
                local_path=labels / "train.txt",
                bytes=10,
                skipped=True,
            )
        ],
    )

    store = RvlCdipStore()
    stats = store.build_from_labels(reset=True)
    assert stats["documents"] == 5
    assert stats["by_split"]["train"] == 3
    assert stats["by_split"]["test"] == 1
    assert stats["by_split"]["validation"] == 1

    summary = store.summary()
    assert summary["documents"] == 5
    assert summary["schema_version"] == 1
    assert summary["dataset_id"] == "aharley/rvl_cdip"

    invoices = store.list_documents(label="invoice", limit=10)
    assert len(invoices) == 1
    assert invoices[0]["label_id"] == 11
    assert invoices[0]["split"] == "train"

    doc = store.get_document("train:a/doc2.tif")
    assert doc is not None
    assert doc["label"] == "invoice"

    rows = store.query(
        "SELECT l.name AS label, COUNT(*) AS n "
        "FROM documents d JOIN labels l ON l.label_id = d.label_id "
        "GROUP BY l.name ORDER BY n DESC"
    )
    assert rows[0]["n"] >= 1
    assert {r["label"] for r in rows} >= {"letter", "invoice", "memo", "form", "email"}


def test_query_rejects_writes(venv_layout: Path):
    store = RvlCdipStore()
    with pytest.raises(ValueError, match="Only SELECT"):
        store.query("DELETE FROM documents")
    with pytest.raises(ValueError, match="Multiple SQL statements"):
        store.query("SELECT 1; DROP TABLE documents")
    with pytest.raises(ValueError, match="Disallowed"):
        store.query("SELECT * FROM documents WHERE 1=1 DROP TABLE documents")


def test_download_images_requires_opt_in(venv_layout: Path):
    from src.rvl_cdip.download import download_images

    with pytest.raises(RuntimeError, match="Refusing to download"):
        download_images(i_understand_large_download=False)


def test_download_images_requires_venv_confirm(venv_layout: Path):
    from src.rvl_cdip.download import download_images

    with pytest.raises(RuntimeError, match="confirm-writes-under-venv"):
        download_images(
            i_understand_large_download=True,
            confirm_writes_under_venv=False,
        )


def test_image_download_preflight(venv_layout: Path):
    from src.rvl_cdip.download import (
        format_image_download_preflight,
        image_download_preflight,
    )

    plan = image_download_preflight()
    assert ".venv" in plan["writes_only_under"]
    assert "rvl_cdip" in plan["writes_only_under"]
    assert plan["confirmation_phrase"] == "writes only under .venv/rvl_cdip"
    text = format_image_download_preflight(plan)
    assert "writes only under" in text
    assert "Free space now" in text


def test_cli_download_images_preflight(venv_layout: Path, capsys: pytest.CaptureFixture[str]):
    from src.rvl_cdip.__main__ import main

    code = main(["download-images", "--preflight"])
    out = capsys.readouterr().out
    assert "writes_only_under" in out or ".venv" in out
    assert code in (0, 2)  # 2 if free space insufficient on tiny tmp volumes


def test_cli_paths(venv_layout: Path, capsys: pytest.CaptureFixture[str]):
    from src.rvl_cdip.__main__ import main

    assert main(["paths", "--json"]) == 0
    out = capsys.readouterr().out
    assert "aharley/rvl_cdip" in out
    assert ".venv" in out


def test_config_rvl_defaults():
    from src.utils.config import Config, REPO_ROOT

    cfg = Config.load()
    assert cfg.rvl_cdip_db_path == REPO_ROOT / ".venv" / "rvl_cdip" / "rvl_cdip.db"
    assert cfg.rvl_cdip_root == REPO_ROOT / ".venv" / "rvl_cdip"
