# Validation record

The local release candidate was validated on 2026-10-25 before any GitHub
publication.

## Automated package checks

- All 44 tests passed in the environment containing PyTorch, including
  concurrent teacher ranking, guided local completion, Slurm detach/resume,
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

## Five-image no-API workflow

Five local 256×256 CellBinDB ssDNA images were used without copying them into
the distributable package. Cellpose, StarDist, CellSAM, and Cellpose-SAM were
each run once per full image through separate Slurm jobs. All 20 candidate masks
had the expected shape and integer instance labels.

Workspace preparation produced:

- 80 adaptive 4×4 crops;
- 320 candidate-mask crops;
- 80 reference panels and 320 candidate panels.

No hosted VLM inference was run, no API key was accessed, and no image was
transmitted. Deterministic synthetic rankings, explicitly marked as software
validation data and not scientific results, exercised Solution 08 and global
stitching. All five final masks were produced with contiguous global IDs and no
label spanning disconnected components. The final audit recorded that
segmentation ground truth was not used.

## Optional student smoke test

A 25%/75% split used 20 synthetic teacher-ranked crops and 60 student-ranked
crops. A one-epoch CPU smoke test verified checkpoint creation, source-image
grouped training/validation, prediction, consensus, and final stitching. It
produced all five final masks with contiguous connected instance labels and
recorded that segmentation ground truth was not used.

This validates software integration and output invariants. Because the rankings
were synthetic, it is not an evaluation of segmentation accuracy or VLM ranking
quality.

## Real hosted-VLM workflow

On 2026-10-07, one of the five 256×256 ssDNA images and its four previously
generated full-image candidate masks were processed through a real GPT-5.5
teacher run. The normal 4×4 grid produced 16 patches and 48 hosted calls. A
16-worker CPU Slurm job completed all rankings in 3 minutes 1 second, with 16
unique patch records, three repetitions per patch, and valid candidate
permutations. Recorded usage was 28,128 input tokens and 96,104 output tokens.

Consensus and stitching completed in 2.17 seconds. The final 256×256 `uint16`
mask contained 54 contiguous instance IDs, with every label connected under the
package's documented 8-connectivity. The final audit recorded that segmentation
ground truth was not used. This run validates the hosted adapter and concurrent
end-to-end execution; it is still not a segmentation-accuracy benchmark.

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
