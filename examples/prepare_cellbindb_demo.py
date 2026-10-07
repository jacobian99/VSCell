#!/usr/bin/env python3
"""Prepare the five-image local ssDNA demo from an existing CellBinDB copy.

The raw files are intentionally gitignored until dataset redistribution terms
are confirmed. This script performs no download.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SAMPLE_IDS = (
    "HH799999269YO_X8-x15353_y3153_w256_h256",  # diseased kidney
    "HH799999623YI_X5-x15842_y7830_w256_h256",  # heart
    "X98659Z5-x17139_y15021_w256_h256",         # uterine moult
    "HH799999455GI_W7-x8065_y12660_w256_h256",  # liver
    "HH799999164YO_W4-x15272_y15492_w256_h256", # brain
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellbindb-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ssdna_demo",
    )
    args = parser.parse_args()
    rows = {row["sample_id"]: row for row in csv.DictReader(args.index.open())}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in SAMPLE_IDS:
        row = rows[sample_id]
        source = args.cellbindb_root / "ssDNA" / sample_id / row["img_file"]
        destination = args.output / f"{sample_id}.tif"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        manifest.append(
            {
                "image_id": sample_id,
                "file": destination.name,
                "modality": "ssDNA",
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
