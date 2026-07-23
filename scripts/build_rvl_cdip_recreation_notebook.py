#!/usr/bin/env python3
"""Generate notebooks/rvl_cdip_recreation_sampling.ipynb and docs symlink."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "rvl_cdip_recreation_sampling.ipynb"
DOCS_LINK = REPO / "docs" / "notebooks" / "rvl_cdip_recreation_sampling.ipynb"
CELLS_JSON = Path(__file__).with_name("rvl_cdip_recreation_nb_cells.json")


def main() -> None:
    payload = json.loads(CELLS_JSON.read_text())
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    cells = []
    for item in payload:
        if item["cell_type"] == "markdown":
            cells.append(nbf.v4.new_markdown_cell(item["source"]))
        else:
            cells.append(nbf.v4.new_code_cell(item["source"]))
    nb["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)

    DOCS_LINK.parent.mkdir(parents=True, exist_ok=True)
    if DOCS_LINK.is_symlink() or DOCS_LINK.exists():
        DOCS_LINK.unlink()
    DOCS_LINK.symlink_to(Path("../../notebooks") / OUT.name)
    print("linked", DOCS_LINK, "->", DOCS_LINK.resolve())


if __name__ == "__main__":
    main()
