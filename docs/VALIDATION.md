# Validation record

The release candidate is validated with the GT-free workflow described in this
package.

## Automated package checks

- The package test suite covers concurrent teacher ranking, guided local
  completion, Slurm detach/resume,
  secure temporary/existing key handling, target-filtered modality selection, student-training
  preflight constraints, monitored one-GPU student train/predict orchestration,
  modern and legacy candidate-folder discovery, all four registered raw-image
  backends, all eight packaged modality descriptions, alternative VLM routing,
  and VLM Slurm rendering.
- The wheel built and installed successfully in a clean target directory.
- The installed CLI reported `VSCell 0.1.0.dev0`.
- The wheel contained the MIT license and third-party notices.
- Distributable files contained no credentials, third-party weights, or
  executable hard-coded HPC paths. The only included microscopy visuals are the
  five approved 256×256 CellBinDB-derived display panels under
  `examples/example_results/`; their provenance, citation, and post-hoc-only
  evaluation role are documented beside the assets.

## No-API workflow

The no-API integration check uses local CellBinDB-derived images without
copying them into the distributable package. Cellpose, StarDist, CellSAM, and
Cellpose-SAM each produce one full-image candidate mask per image; every mask
must have the source shape and integer instance labels.

Workspace preparation produced candidate-derived adaptive crop grids, with one
reference panel and one candidate panel per model for every planned patch.

No hosted VLM inference is run, no API key is accessed, and no image is
transmitted. Deterministic synthetic rankings, explicitly marked as software
validation data rather than scientific results, exercise Solution 08 and global
stitching. The check requires contiguous global IDs, no label spanning
disconnected components, and an audit recording that segmentation ground truth
was not used.

## Optional student smoke test

A synthetic teacher/student split verifies checkpoint creation, source-image
grouped training/validation, prediction, consensus, and final stitching. It
requires contiguous connected instance labels and records that segmentation
ground truth was not used.

This validates software integration and output invariants. Because the rankings
were synthetic, it is not an evaluation of segmentation accuracy or VLM ranking
quality.

## Hosted-VLM workflow

After the candidate-derived grid is available, a hosted teacher run ranks the
selected patch fraction three times per patch with valid anonymous candidate
permutations. The recorded usage, rankings, consensus output, final mask, and
audit are checked for complete patch coverage and connected globally relabelled
instances. This validates the hosted adapter and concurrent end-to-end
execution; it is not a segmentation-accuracy benchmark.

## Issues found and corrected during validation

- Added an authorized local CellSAM-weight path override for offline/shared
  caches, without redistributing the weight.
- Made rendered Slurm log paths absolute and created the log directory before
  job submission.
- Split disconnected islands emitted under one upstream label into distinct
  instances during global relabeling.
- Added bounded concurrent teacher ranking and a secret-free CPU Slurm renderer;
  only the main process writes resume-safe JSONL records.
- Replaced the default all-extras installation example with a provider-specific
  lightweight installation to avoid unnecessary PyTorch/CUDA downloads.
- Added a guided `vscell run` interface with explicit preference prompts,
  preflight confirmation, resumable state, visible Slurm/progress monitoring,
  deliberate detach behavior, and automatic finalization. Core segmentation,
  ranking, consensus, and stitching algorithms were not changed.
- Added a monitored one-GPU Slurm stage for student training and remaining-patch
  prediction. Its combined ranking file is validated by finalization before
  consensus and stitching; the student architecture and loss were not changed.
