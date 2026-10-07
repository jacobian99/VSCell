# Method and code provenance

VSCell is primarily a packaging and integration effort. The following validated
experiment components were ported into package-native modules so the public
package does not depend on absolute paths into historical experiment folders.

| Packaged component | Historical source | Preserved behavior |
|---|---|---|
| Full-image model adapters | `z_exp_0514_CellBinDB/model_inf/shared/` and `z_exp_0514_CellBinDB_bench/model_inf/shared/` | Backend-specific preprocessing, pretrained variants, full-image inference |
| Candidate environment separation | `z_exp_0515_CellBinDB_oracle/model_inf/gen_slurm.py` | Classic Cellpose, PyTorch/CellSAM, and TensorFlow/StarDist isolation |
| Local padding | `z_exp_0718_summary_k5_TN_CB/common/crops.py` | Historical `PAD=0.10`, interpreted as 10% total side enlargement |
| VLM prompts and routing | `z_exp_0911_Claude_GPT_gtFREE_rerun/prompt_router.py` and `prompt_sources/` | Target-aware descriptions, anonymous candidate panels, strict ranking format |
| Repeated ranking | `z_exp_0911_Claude_GPT_gtFREE_rerun/analyze_results.py` | Three independently shuffled rankings and deterministic Borda aggregation |
| Consensus ensemble | `z_exp_0815_AoD_test/general_resolution_solutions/08_anchor_con_selection/` | Final Solution 08 anchor consensus, residual selection/replacement, exact thresholds |
| Boundary refinement | `z_exp_0718_summary_k5_TN_CB/common/fusion.py` | Two-other-peer corroborating boundary refinement |
| Student ranker | `z_exp_0912_ distillation/scripts/cb_ranker.py` and `train_cb_ranker.py` | Identity-blind shared candidate encoder, set context, pairwise/listwise loss, source-image-grouped validation |
| Historical cost planning | GPT-5.5 smoke-100 audit plus Claude/GPT-6 usage JSON | Empirical cost per teacher-ranked patch, including three repetitions |

`vscell.workflow` and the guided `vscell run` command are user-interface and
orchestration code only. They call the same packaged backend, ranking, student,
consensus, and stitching functions as the individual commands; they do not
alter model settings, prompts, repetitions, thresholds, or mask-selection
logic.

## Intentional GT-free package changes

- Paper crop selection used GT-selected neighborhoods. The package replaces
  this with the user-approved image-size-distribution grid and never accepts a
  GT mask at inference time.
- The paper used dataset-specific candidate lists. The package exposes one
  consistent four-backend set: Cellpose, StarDist, CellSAM, and Cellpose-SAM.
- Stable SHA-256-derived random seeds replace Python's process-salted `hash()`
  for reproducible anonymous candidate shuffles across machines.
- Full-image stitching is new glue around the local validated method: cells
  spanning multiple grid cores come from the image-level Borda winner, while
  non-overlap cells come from the local Solution 08 mask.
- Third-party model weights are not vendored. Upstream download/access paths
  and licensing remain authoritative.

## Frozen Solution 08 defaults

- Anchor support IoU: 0.50
- Residual complete-link IoU: 0.50
- Residual support: two distinct peer models
- Anchor-null threshold: 0.10
- Boundary refinement match IoU: 0.50
- Boundary refinement support: two other peer models
- Minimum instance area: 1 pixel for the GT-free full-image Figure 2 setting
