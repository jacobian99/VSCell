# Candidate segmentation masks

If candidate masks already exist, create one subdirectory per model here:

```text
segmentation_results/
├── cellpose/
│   └── image01_masks.tif
├── stardist/
│   └── image01_masks.tif
└── my_model/
    └── image01_masks.tif
```

This directory is optional when VSCell will submit its built-in segmentation
backends instead.
