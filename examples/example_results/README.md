# Curated example results

These five 256×256 CellBinDB crops let a visitor inspect the input image,
ground-truth annotation, VSCell output, and each candidate model before running
VSCell. They are static display assets, not inputs to the package. The raw
reference image is shown in `reference/`; every other directory contains that
same reference with a colorized instance-overlay for the named result.

`comparison/` contains the compact seven-panel versions used below. The
individual display crops are organized by result source:

```text
comparison/     # REF, GT, VSCell, and four candidate models in one panel
reference/      # reference microscopy crop
ground_truth/   # annotation overlay, evaluation only
vscell/         # VSCell final stitched output
cellpose/       # Cellpose candidate
stardist/       # StarDist candidate
cellsam/        # CellSAM candidate
cellpose_sam/   # Cellpose-SAM candidate
```

The displayed score is the post-hoc held-out metric
`P = (F1@0.50 + mean matched-cell IoU) / 2`. Ground truth was **not** supplied
to any VSCell inference, ranking, consensus, or stitching step; it was used
only after inference to prepare this visual comparison. These selected examples
are illustrative, not an aggregate benchmark. VSCell is the row winner in
these selected examples; this does not claim it always outperforms every
individual candidate model.

| Crop | Modality | VSCell P | Per-image best single model | Delta |
|---|---|---:|---:|---:|
| Y96476V7 (7516, 10406) | DAPI | 0.8528 | Cellpose, 0.8347 | +0.0181 |
| Y96476U4 (24456, 14466) | DAPI | 0.6611 | Cellpose-SAM, 0.6567 | +0.0044 |
| liver (14224, 6778) | DAPI | 0.6708 | CellSAM, 0.6678 | +0.0030 |
| X5 (15842, 7830) | ssDNA | 0.8210 | Cellpose-SAM, 0.7791 | +0.0419 |
| Y96476V7 (14776, 4316) | DAPI | 0.8280 | Cellpose, 0.8254 | +0.0026 |

## DAPI Y96476V7 (7516, 10406)

![DAPI Y96476V7 comparison](comparison/01_dapi_y96476v7_7516_10406.png)

## DAPI Y96476U4 (24456, 14466)

![DAPI Y96476U4 comparison](comparison/02_dapi_y96476u4_24456_14466.png)

## DAPI liver (14224, 6778)

![DAPI liver comparison](comparison/03_dapi_liver_14224_6778.png)

## ssDNA X5 (15842, 7830)

![ssDNA X5 comparison](comparison/04_ssdna_x5_15842_7830.png)

## DAPI Y96476V7 (14776, 4316)

![DAPI Y96476V7 comparison](comparison/05_dapi_y96476v7_14776_4316.png)

## Data reference

These approved CellBinDB-derived display crops and annotations should be cited
as: Shi C, Fan J, Deng Z, *et al.* CellBinDB: a large-scale multimodal
annotated dataset for cell segmentation with benchmarking of universal models.
*GigaScience*. 2025;14:giaf069.
[https://doi.org/10.1093/gigascience/giaf069](https://doi.org/10.1093/gigascience/giaf069)
