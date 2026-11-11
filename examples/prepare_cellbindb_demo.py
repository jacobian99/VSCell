#!/usr/bin/env python3
"""Prepare the four-image local CellBinDB demo from an existing local copy.

The raw files are intentionally gitignored until dataset redistribution terms
are confirmed. This script performs no download.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SAMPLES = (
    ("DAPI", "Y96476V7-x7516_y10406_w256_h256"),
    ("DAPI", "prostate-x9465_y7872_w256_h256"),
    ("ssDNA", "HH799999623YI_X5-x15842_y7830_w256_h256"),
    ("DAPI", "liver-x14224_y6778_w256_h256"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellbindb-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "cellbindb_demo",
    )
    args = parser.parse_args()
    rows = {row["sample_id"]: row for row in csv.DictReader(args.index.open())}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for modality, sample_id in SAMPLES:
        row = rows[sample_id]
        source = args.cellbindb_root / modality / sample_id / row["img_file"]
        destination = args.output / f"{sample_id}.tif"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        manifest.append(
            {
                "image_id": sample_id,
                "file": destination.name,
                "modality": modality,
                "target": "nuclear",
                "species": row["species"],
                "tissue": row["tissue"],
                "source": "CellBinDB local copy",
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
